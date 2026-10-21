"""The fp8-KV prefill patch on vLLM 0.30 (DESIGN rev 2 section 5, T15a), CPU part: the installed flash_attn backend
passes the patch's source check; routing (what goes to the block-dequantized fp16 path, what stays on the stock
kernel); the LSE merge against float64 attention with the kernel's [H, Q] LSE layout; register() refusals for the
two routes the port drops (EXL3_FP8KV_MIXED, EXL3_ONEDNN_ATTN). The GPU half is tests/xpu/t15a_fp8kv.py."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("vllm", reason="needs the vLLM image")

_HERE = Path(__file__).resolve().parent


def _run(script: str, *args: str, env: dict | None = None, drop: tuple = ()) -> dict:
    penv = {k: v for k, v in dict(os.environ, PYTHONDONTWRITEBYTECODE="1", **(env or {})).items() if k not in drop}
    p = subprocess.run([sys.executable, str(_HERE / script), *args], env=penv, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, text=True, timeout=900)
    tag = "FP8KV_PROBE " if script == "fp8kv_probe.py" else "REGISTER_PROBE "
    lines = [ln for ln in p.stdout.splitlines() if ln.startswith(tag)]
    assert lines, p.stdout[-4000:]
    res = json.loads(lines[-1][len(tag):])
    res["log"] = p.stdout
    return res


_CLEAN = ("EXL3_FP8KV_MIXED", "EXL3_ONEDNN_ATTN", "EXL3_FP8KV_CHECK", "EXL3_VLLM_PATCHES")


def test_sources_match_vllm030():
    r = _run("fp8kv_probe.py", "sources", drop=_CLEAN)
    assert r["ok"], r.get("trace") or r
    s = r["result"]
    assert s["problems"] == [] and s["applied"] is True and s["patched"] and s["keeps_stock"], s
    assert s["again"] is False, s          # idempotent: a second call does not wrap twice


@pytest.fixture(scope="module")
def routing():
    r = _run("fp8kv_probe.py", "routing", drop=_CLEAN)
    assert r["ok"], r.get("trace") or r
    return {c["name"]: c for c in r["result"]["cases"]}


ROUTED = ["base_routed", "q64_l4096_routed", "q4096_l4096_routed", "fp16_query_routed", "scales_passed",
          "padding_rows_untouched", "tp1_shape_routed"]
STOCK = ["kv_auto", "q63", "l4095", "short_prompt_256", "decode_q1", "two_sequences", "tokens_ne_query",
         "causal_tensor", "causal_false", "cascade", "mm_prefix", "rswa", "sliding_window", "soft_cap", "md_none",
         "fp8_query", "output_scale", "alibi", "sinks", "dcp2", "pcp2", "encoder_only", "capturing"]


@pytest.mark.parametrize("name", ROUTED)
def test_routed(routing, name):
    c = routing[name]
    assert c["error"] is None and c["route"] == "routed", c
    assert c["out_written"] and c["rows_after_untouched"], c
    i = c["info"]
    want_q = "torch.float16" if name == "fp16_query_routed" else "torch.bfloat16"
    assert i["q_dtype"] == want_q and i["kc_dtype"] == "torch.float8_e4m3fn" and i["pages"] == 4125, i
    assert i["out_dtype"] == "torch.bfloat16", i         # written straight into vLLM's bf16 output


def test_routed_arguments(routing):
    b = routing["base_routed"]["info"]
    assert b["seq_len"] == 8192 and b["q_shape"] == [2112, 8, 256] and b["kc_shape"] == [330, 64, 1, 256], b
    assert abs(b["scale"] - 256 ** -0.5) < 1e-12 and (b["ks"], b["vs"]) == (1.0, 1.0), b
    s = routing["scales_passed"]["info"]
    assert (s["ks"], s["vs"]) == (0.5, 2.0), s
    t = routing["tp1_shape_routed"]["info"]
    assert t["q_shape"] == [2112, 32, 256] and t["kc_shape"] == [330, 64, 2, 256], t
    assert routing["q64_l4096_routed"]["info"]["seq_len"] == 4096


@pytest.mark.parametrize("name", STOCK)
def test_stock(routing, name):
    c = routing[name]
    assert c["error"] is None and c["route"] == "stock", c


def test_check_catches_upper_bound(routing):
    c = routing["check_catches_upper_bound"]
    assert c["route"] is None and c["error"] and "differ from the sequence's seq_len 8000" in c["error"], c


def test_paged_route_vs_float64():
    """The served route: one paged causal call over the gathered, dequantized sequence."""
    r = _run("fp8kv_probe.py", "paged", drop=_CLEAN)
    assert r["ok"], r.get("trace") or r
    for c in r["result"]["cases"]:
        call = c["call"]
        assert c["calls"] == 1 and call["bt_identity"] and call["bt_dtype"] == "torch.int32", c
        assert call["causal"] is True and call["mix"] is False and call["cu_k"] is None, c
        assert call["seqused"] == [c["L"]] and call["cu_q"] == [0, c["Q"]] and call["max_q"] == c["Q"], c
        assert call["max_k"] == c["L"] and call["k_dtype"] == call["q_dtype"] == c["q_dtype"], c
        assert call["k_shape"][0] == (c["L"] + 63) // 64, c          # only the sequence's kernel blocks
        # bf16 output: within 1.5x the bf16 rounding floor of the float64 result (dequantization is exact)
        assert c["rel"] <= 1.5 * c["bf16_floor"] + 1e-4, c


def test_merge_vs_float64():
    r = _run("fp8kv_probe.py", "merge", drop=_CLEAN)
    assert r["ok"], r.get("trace") or r
    for c in r["result"]["cases"]:
        assert c["error"] is None, c
        # fp16 partial outputs: within 2x the fp16 rounding floor of the result (+ a little for the merge)
        assert c["rel"] <= 2 * c["fp16_floor"] + 2e-4, c
    assert sorted({c["parts"] for c in r["result"]["cases"]}) == [1, 2, 3, 5, 6], r["result"]["cases"]
    w = r["result"]["wrong_layout"]
    assert w["raised"] and "expected [H, Q]" in w["raised"], w


@pytest.mark.parametrize("env,needle", [({"EXL3_FP8KV_MIXED": "1"}, "EXL3_FP8KV_MIXED=1 is not supported"),
                                        ({"EXL3_ONEDNN_ATTN": "1"}, "EXL3_ONEDNN_ATTN=1 is not validated")])
def test_register_refuses_dropped_routes(env, needle):
    r = _run("register_probe.py", "--xpu-platform", env=dict(env, EXL3_VLLM_PATCHES="fp8kv"))
    assert not r["ok"] and "EXL3_VLLM_PATCHES requests 'fp8kv' but the patch did not apply" in r["error"], r
    assert needle in r["log"], r["log"][-3000:]


def test_apply_all_refused():
    r = subprocess.run([sys.executable, "-c", "import sys; sys.path[:0] = [sys.argv[1]]\n"
                        "from exl3xpu import vllm_patches\n"
                        "try:\n    vllm_patches.apply_all()\nexcept RuntimeError as e:\n    print('REFUSED', e)",
                        str(_HERE.parents[1] / "xpu")], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                       timeout=600, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
    assert "REFUSED" in r.stdout, r.stdout[-3000:]
