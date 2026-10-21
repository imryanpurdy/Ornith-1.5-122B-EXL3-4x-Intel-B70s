"""What the plugin refuses, on real vLLM 0.30 layers of Ornith-mini (plugin_harness.py units), and what register()
does in a fresh process for each relevant environment (register_probe.py):

  routing: a LinearBase without EXL3 storage outside in_proj_ba / visual.* raises (language model and mtp.); an
      EXL3 embedding raises; a fused module with mixed or partly missing storage raises; routed experts that are
      not all EXL3 with one K raise; in_proj_ba and visual.* get UnquantizedLinearMethod.
  dense loaders: tuple shard ids other than (0, 1, 2), a matrix delivered under the wrong shard id, a codebook
      marker that is not mul1, a trellis of another K; process_weights_after_loading raises on a missing shard,
      a component loaded twice, a missing codebook marker, a K the kernels lack (no silent Triton fallback).
  routed-expert loader: unexpected names, out-of-range ids, an expert after it was packed, a duplicate tensor,
      a bad marker; remote experts under EP are dropped and counted; process raises on missing local experts.
  EXL3XPU_LOADED: counts that differ from the checkpoint index raise (index with 4 extra local expert tensors).
  register(): registers 'exl3' and installs the graph guard; EXL3_VLLM_PATCHES default empty, 'fp8kv' applies
      patch_fp8kv_prefill only, align_sync / gdn_mask_index / unknown names are refused or ignored with a log line;
      EXL3_DRAFT_VOCAB and EXL3_CAPTURE_GUARD != raise abort the process."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import run_harness

pytest.importorskip("vllm", reason="needs the vLLM image")

UNITS = {
    "route_unknown_linear_raises", "route_unknown_mtp_linear_raises", "route_exl3_embedding_raises",
    "route_mixed_fused_raises", "route_missing_expert_raises", "route_in_proj_ba_unquantized",
    "route_visual_unquantized", "loader_tuple_shard_23_refused", "loader_wrong_matrix_for_shard_refused",
    "loader_bad_marker_refused", "loader_wrong_K_refused", "loader_complete_layer_processes",
    "process_missing_shard_raises", "process_duplicate_raises", "process_missing_marker_raises",
    "process_unsupported_K_refuses_fallback", "moe_unexpected_name_refused", "moe_expert_out_of_range_refused",
    "moe_remote_expert_dropped", "moe_local_expert_packed", "moe_after_complete_refused", "moe_duplicate_refused",
    "moe_bad_marker_refused", "moe_process_missing_raises", "loaded_count_mismatch_raises",
}


@pytest.fixture(scope="module")
def units(mini_dir, lib_dir):
    return {u["name"]: u for u in run_harness("units", mini_dir, [], lib=lib_dir)}


@pytest.mark.parametrize("name", sorted(UNITS))
def test_refusal(units, name):
    assert name in units, sorted(units)
    assert units[name]["ok"], units[name]["detail"]


def _probe(env: dict) -> dict:
    p = subprocess.run([sys.executable, str(Path(__file__).with_name("register_probe.py")), "--xpu-platform"],
                       env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1", **env), stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, text=True, timeout=600)
    lines = [ln for ln in p.stdout.splitlines() if ln.startswith("REGISTER_PROBE ")]
    assert lines, p.stdout[-4000:]
    res = json.loads(lines[-1][len("REGISTER_PROBE "):])
    res["log"] = p.stdout
    return res


def test_register_default():
    r = _probe({"EXL3_VLLM_PATCHES": ""})
    assert r["ok"] and r["registered"] and r["patches"] == {} and r["fp8kv_patched"] is False, r
    assert r["guard"]["G2"] and r["guard"]["G3a"] and r["guard"]["G3b"], r["guard"]     # G1/G4 apply on import
    assert "EXL3XPU_GRAPH_GUARD installed=" in r["log"] and "EXL3XPU_PLUGIN registered quantization=exl3" in r["log"]
    assert r["log"].index("EXL3XPU_GRAPH_GUARD") < r["log"].index("EXL3XPU_PLUGIN registered")


def test_register_fp8kv_only():
    r = _probe({"EXL3_VLLM_PATCHES": "fp8kv"})
    assert r["ok"] and r["fp8kv_patched"] is True and "patches=fp8kv:applied" in r["log"], r


def test_register_refuses_other_patches():
    r = _probe({"EXL3_VLLM_PATCHES": "patch_align_sync,gdn_mask_index,xpu_block_size,bogus"})
    assert r["ok"] and r["fp8kv_patched"] is False, r
    assert ("patches=align_sync:refused,gdn_mask_index:refused,xpu_block_size:refused,bogus:unknown"
            in r["log"]), r["log"][-3000:]


@pytest.mark.parametrize("env,needle", [({"EXL3_DRAFT_VOCAB": "/nonexistent.json"}, "EXL3_DRAFT_VOCAB"),
                                        ({"EXL3_CAPTURE_GUARD": "log"}, "EXL3_CAPTURE_GUARD")])
def test_register_aborts(env, needle):
    r = _probe(env)
    assert not r["ok"] and needle in r["error"], r
