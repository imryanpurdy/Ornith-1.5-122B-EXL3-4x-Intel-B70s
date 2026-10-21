"""CPU probe for the fp8-KV prefill patch (xpu/exl3xpu/vllm_patches.py patch_fp8kv_prefill, fp8kv_prefill.py) on
vLLM 0.30's real FlashAttentionImpl / FlashAttentionMetadata classes. Runs in its own interpreter (vLLM binds its
platform at import; the XPU platform is installed first, as plugin_harness.py does) and prints one JSON line
"FP8KV_PROBE {...}".

  sources  the installed flash_attn backend passes the patch's source check, and the patch applies to the real class
  routing  the stock forward is replaced by a recorder BEFORE the patch is applied (the source check is bypassed for
           that stub), prefill_attention_paged by a recorder; routed cases plus every condition that must fall
           through to the stock kernel; EXL3_FP8KV_CHECK=1 catches a max_seq_len that is not the sequence's length
  paged    the real prefill_attention_paged (the served route) on a CPU fp8 cache, with vllm_xpu_kernels'
           flash_attn_varlen_func replaced by a float64 paged attention that also checks the call (identity block
           table, seqused_k, causal, prefill-only, K/V dtype = q dtype): gather, dequantization, scales vs float64
  merge    the real prefill_attention (trellis' block merge, kept but not served) on a CPU fp8 cache with its kernel
           call replaced by a float64 CPU attention that returns the LSE in the kernel's [H, Q] layout: block
           splitting, page offsets and the LSE merge vs one float64 causal pass; an LSE in [Q, H] layout raises
"""
from __future__ import annotations

import json
import math
import os
import sys
import traceback
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
for p in (_ROOT / "core" / "src", _ROOT / "xpu"):
    sys.path.insert(0, str(p))

import torch  # noqa: E402

import vllm.platforms as _P  # noqa: E402
from vllm.platforms.xpu import XPUPlatform  # noqa: E402

XPUPlatform.current_device = classmethod(lambda cls: torch.device("cpu"))
_P._current_platform = XPUPlatform()

D = 256
FP8 = torch.float8_e4m3fn


def sources():
    from vllm.v1.attention.backends import flash_attn as fa
    from exl3xpu import vllm_patches
    problems = vllm_patches._fp8kv_source_problems(fa)
    stock = fa.FlashAttentionImpl.forward
    applied = vllm_patches.patch_fp8kv_prefill()
    return {"problems": problems, "applied": applied,
            "patched": bool(getattr(fa.FlashAttentionImpl, "_exl3_fp8kv_patched", False)),
            "keeps_stock": fa.FlashAttentionImpl._exl3_fp8kv_orig_forward is stock,
            "again": vllm_patches.patch_fp8kv_prefill()}


def routing():
    os.environ["EXL3_FP8KV_CHECK"] = "1"
    from vllm.v1.attention.backends import flash_attn as fa
    from exl3xpu import fp8kv_prefill, vllm_patches
    calls = []

    def stub(self, layer, query, key, value, kv_cache, attn_metadata, output, output_scale=None,
             output_block_scale=None):
        calls.append(("stock", None))
        return output

    fa.FlashAttentionImpl.forward = stub
    vllm_patches._fp8kv_source_problems = lambda m: []
    assert vllm_patches.patch_fp8kv_prefill()

    def rec(q, key_cache, value_cache, pages, seq_len, ks, vs, scale, out, dtype=None):
        calls.append(("routed", {"q_dtype": str(q.dtype), "q_shape": list(q.shape), "seq_len": seq_len, "ks": ks,
                                 "vs": vs, "scale": scale, "kc_dtype": str(key_cache.dtype),
                                 "kc_shape": list(key_cache.shape), "pages": int(pages.shape[0]),
                                 "out_dtype": str(out.dtype)}))
        out.fill_(1.5)
        return out

    fp8kv_prefill.prefill_attention_paged = rec

    class Layer(torch.nn.Module):
        pass

    def impl(hq=8, hk=1, **kw):
        args = dict(num_heads=hq, head_size=D, scale=D ** -0.5, num_kv_heads=hk, alibi_slopes=None,
                    sliding_window=None, kv_cache_dtype="fp8", logits_soft_cap=None,
                    attn_type=fa.AttentionType.DECODER, kv_sharing_target_layer_name=None, sinks=None)
        args.update(kw)
        return fa.FlashAttentionImpl(**args)

    def md(seqs=((8192, 2112),), **over):
        qlens = [q for _, q in seqs]
        qsl = torch.tensor([0] + torch.tensor(qlens).cumsum(0).tolist(), dtype=torch.int32)
        m = fa.FlashAttentionMetadata(
            num_actual_tokens=sum(qlens), max_query_len=max(qlens), query_start_loc=qsl,
            max_seq_len=max(s for s, _ in seqs), seq_lens=torch.tensor([s for s, _ in seqs], dtype=torch.int32),
            block_table=torch.zeros(len(seqs), 4125, dtype=torch.int32), slot_mapping=torch.zeros(sum(qlens),
                                                                                                  dtype=torch.int64),
            use_cascade=False, common_prefix_len=0, cu_prefix_query_lens=None, prefix_kv_lens=None,
            suffix_kv_lens=None, causal=True)
        for k, v in over.items():
            setattr(m, k, v)
        return m

    from vllm.model_executor.layers.attention.attention import set_default_quant_scales

    def run(name, im=None, m=None, qdtype=torch.bfloat16, layer_scales=(1.0, 1.0), extra_tokens=0, **fw):
        im = im or impl()
        m = m if m is not None else md()
        n = (m.num_actual_tokens if m is not None else 64) + extra_tokens
        layer = Layer()
        set_default_quant_scales(layer, register_buffer=True)
        layer._k_scale_float, layer._v_scale_float = layer_scales
        q = torch.zeros(n, im.num_heads, D, dtype=qdtype)
        kv = torch.zeros(10 * 33, im.num_kv_heads, 64, 2 * D, dtype=torch.uint8)
        out = torch.zeros(n, im.num_heads, D, dtype=torch.bfloat16)
        calls.clear()
        err = None
        try:
            im.forward(layer, q, q[:, :im.num_kv_heads], q[:, :im.num_kv_heads], kv, m, out, **fw)
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {e}"
        r = {"name": name, "route": calls[0][0] if calls else None, "info": calls[0][1] if calls else None,
             "error": err}
        if r["route"] == "routed":
            nq = m.num_actual_tokens
            r["out_written"] = bool((out[:nq] == 1.5).all())
            r["rows_after_untouched"] = bool((out[nq:] == 0).all())
        return r

    res = []
    res.append(run("base_routed"))
    res.append(run("q64_l4096_routed", m=md(((4096, 64),))))
    res.append(run("q4096_l4096_routed", m=md(((4096, 4096),))))
    res.append(run("fp16_query_routed", qdtype=torch.float16))
    res.append(run("scales_passed", layer_scales=(0.5, 2.0)))
    res.append(run("padding_rows_untouched", extra_tokens=16))
    res.append(run("tp1_shape_routed", im=impl(32, 2)))
    for name, kw in [
        ("kv_auto", dict(im=impl(kv_cache_dtype="auto"))),
        ("q63", dict(m=md(((8192, 63),)))),
        ("l4095", dict(m=md(((4095, 2112),)))),
        ("short_prompt_256", dict(m=md(((256, 256),)))),
        ("decode_q1", dict(m=md(((8192, 1),)))),
        ("two_sequences", dict(m=md(((8192, 2112), (5000, 1000))))),
        ("tokens_ne_query", dict(m=md(num_actual_tokens=2113))),
        ("causal_tensor", dict(m=md(causal=torch.ones(1, dtype=torch.int32)))),
        ("causal_false", dict(m=md(causal=False))),
        ("cascade", dict(m=md(use_cascade=True))),
        ("mm_prefix", dict(m=md(mm_prefix_query_range_tensor=torch.zeros(2112, 2, dtype=torch.int32)))),
        ("rswa", dict(m=md(rswa_prefix_lens=torch.zeros(1, dtype=torch.int32)))),
        ("sliding_window", dict(im=impl(sliding_window=1024))),
        ("soft_cap", dict(im=impl(logits_soft_cap=30.0))),
        ("md_none", dict(m=None)),
        ("fp8_query", dict(qdtype=FP8)),
        ("output_scale", dict(output_scale=torch.ones(1))),
    ]:
        if name == "md_none":
            im = impl()
            layer = Layer()
            set_default_quant_scales(layer, register_buffer=True)
            calls.clear()
            q = torch.zeros(64, 8, D, dtype=torch.bfloat16)
            im.forward(layer, q, q[:, :1], q[:, :1], torch.zeros(1), None, torch.zeros(64, 8, D))
            res.append({"name": name, "route": calls[0][0] if calls else None, "error": None})
            continue
        res.append(run(name, **kw))
    # instance attributes the constructor cannot set on XPU
    for name, attr, val in [("alibi", "alibi_slopes", torch.ones(8)), ("sinks", "sinks", torch.ones(8)),
                            ("dcp2", "dcp_world_size", 2), ("pcp2", "pcp_world_size", 2),
                            ("encoder_only", "attn_type", fa.AttentionType.ENCODER_ONLY)]:
        im = impl()
        setattr(im, attr, val)
        res.append(run(name, im=im))
    # graph capture in progress
    real_avail, real_cap = torch.xpu.is_available, torch.xpu.is_current_stream_capturing
    torch.xpu.is_available = lambda: True
    torch.xpu.is_current_stream_capturing = lambda: True
    try:
        res.append(run("capturing"))
    finally:
        torch.xpu.is_available, torch.xpu.is_current_stream_capturing = real_avail, real_cap
    # EXL3_FP8KV_CHECK: max_seq_len that is not the sequence's length
    bad = md()
    bad.seq_lens = torch.tensor([8000], dtype=torch.int32)
    res.append(run("check_catches_upper_bound", m=bad))
    return {"cases": res, "routed_total": vllm_patches.FP8KV_STATS["routed"]}


def _ref_attention(q64, k64, v64, scale, causal):
    Q, hq, _ = q64.shape
    L, hk, _ = k64.shape
    g = hq // hk
    kk = k64.repeat_interleave(g, dim=1)
    vv = v64.repeat_interleave(g, dim=1)
    S = torch.einsum("qhd,khd->hqk", q64, kk) * scale
    if causal:
        rows = torch.arange(Q)[:, None] + (L - Q)
        S = S.masked_fill(torch.arange(L)[None, :] > rows, float("-inf"))
    lse = torch.logsumexp(S, -1)                                             # [H, Q]
    return torch.einsum("hqk,khd->qhd", torch.softmax(S, -1), vv), lse


def paged():
    import vllm_xpu_kernels.flash_attn_interface as fai
    from exl3xpu import fp8kv_prefill
    seen = []

    def fake(q, k, v, max_seqlen_q, cu_seqlens_q, max_seqlen_k, cu_seqlens_k=None, seqused_k=None,
             softmax_scale=None, causal=False, block_table=None, out=None, is_mix_batch=True, **kw):
        nb = k.shape[0]
        seen.append({"q_dtype": str(q.dtype), "k_dtype": str(k.dtype), "causal": causal, "mix": is_mix_batch,
                     "bt_identity": bool(torch.equal(block_table, torch.arange(nb, dtype=torch.int32)[None])),
                     "bt_dtype": str(block_table.dtype), "seqused": seqused_k.tolist(),
                     "cu_q": cu_seqlens_q.tolist(), "max_q": max_seqlen_q, "max_k": max_seqlen_k,
                     "cu_k": None if cu_seqlens_k is None else cu_seqlens_k.tolist(), "k_shape": list(k.shape)})
        L = int(seqused_k[0])
        kf = k.flatten(0, 1)[:L].double()
        vf = v.flatten(0, 1)[:L].double()
        o, _ = _ref_attention(q.double(), kf, vf, softmax_scale, causal)
        out.copy_(o.to(out.dtype))
        return out

    fai.flash_attn_varlen_func = fake
    gen = torch.Generator().manual_seed(11)
    res = []
    BS = 64
    for hq, hk, L, Q, ks, vs, qdt in [(8, 1, 4096, 4096, 1.0, 1.0, torch.bfloat16),
                                      (8, 1, 5000, 777, 1.0, 1.0, torch.bfloat16),
                                      (8, 1, 4160, 64, 0.5, 2.0, torch.bfloat16),
                                      (32, 2, 4500, 300, 1.0, 1.0, torch.bfloat16),
                                      (16, 1, 4100, 100, 1.0, 1.0, torch.float16)]:
        npages = (L + BS - 1) // BS + 7
        kc = (torch.randn(npages, BS, hk, D, generator=gen) * 3).to(FP8)
        vc = (torch.randn(npages, BS, hk, D, generator=gen) * 3).to(FP8)
        pages = torch.randperm(npages, generator=gen)[: (L + BS - 1) // BS + 2].to(torch.int32)   # row wider than L
        q = torch.randn(Q, hq, D, generator=gen).to(qdt)
        out = torch.empty(Q, hq, D, dtype=torch.bfloat16)
        kfull = kc.index_select(0, pages.long()).flatten(0, 1)[:L].double() * ks
        vfull = vc.index_select(0, pages.long()).flatten(0, 1)[:L].double() * vs
        ref, _ = _ref_attention(q.double(), kfull, vfull, D ** -0.5, True)
        seen.clear()
        fp8kv_prefill.prefill_attention_paged(q, kc, vc, pages, L, ks, vs, D ** -0.5, out)
        rel = float((out.double() - ref).norm() / ref.norm())
        floor = float((ref.to(torch.bfloat16).double() - ref).norm() / ref.norm())
        res.append({"hq": hq, "hk": hk, "L": L, "Q": Q, "ks": ks, "vs": vs, "q_dtype": str(qdt), "rel": rel,
                    "bf16_floor": floor, "call": seen[0] if seen else None, "calls": len(seen)})
    return {"cases": res}


def merge():
    from exl3xpu import fp8kv_prefill
    layout = {"lse": "hq"}

    def fake_fa(q, k, v, causal, scale):
        o, lse = _ref_attention(q.double(), k.double(), v.double(), scale, causal)
        lse = lse.float() if layout["lse"] == "hq" else lse.transpose(0, 1).contiguous().float()
        return o.to(torch.float16), lse

    fp8kv_prefill._fa = fake_fa
    gen = torch.Generator().manual_seed(7)
    res = []
    BS = 64
    for hq, hk, L, Q, kb, ks, vs in [(8, 1, 1500, 300, 256, 1.0, 1.0), (8, 1, 1500, 300, 1000, 1.0, 1.0),
                                     (8, 1, 777, 777, 256, 1.0, 1.0), (8, 1, 4096, 64, 1024, 1.0, 1.0),
                                     (16, 1, 2000, 333, 512, 0.75, 1.25), (32, 2, 1300, 200, 300, 1.0, 1.0),
                                     (8, 1, 1024 + 64, 64, 1024, 1.0, 1.0)]:
        npages = (L + BS - 1) // BS + 5
        kc = (torch.randn(npages, BS, hk, D, generator=gen) * 2).to(FP8)
        vc = (torch.randn(npages, BS, hk, D, generator=gen) * 2).to(FP8)
        pages = torch.randperm(npages, generator=gen)[: (L + BS - 1) // BS].to(torch.int32)
        q = torch.randn(Q, hq, D, generator=gen).to(torch.float16)
        out = torch.empty(Q, hq, D, dtype=torch.float16)
        kfull = kc.index_select(0, pages.long()).flatten(0, 1)[:L].double() * ks
        vfull = vc.index_select(0, pages.long()).flatten(0, 1)[:L].double() * vs
        ref, _ = _ref_attention(q.double(), kfull, vfull, D ** -0.5, True)
        err = None
        try:
            fp8kv_prefill.prefill_attention(q, kc, vc, pages, L, ks, vs, D ** -0.5, out, kb=kb)
            rel = float((out.double() - ref).norm() / ref.norm())
        except Exception as e:  # noqa: BLE001
            rel, err = None, f"{type(e).__name__}: {e}"
        floor = float((ref.to(torch.float16).double() - ref).norm() / ref.norm())
        res.append({"hq": hq, "hk": hk, "L": L, "Q": Q, "kb": kb, "parts": fp8kv_prefill.num_parts(L, Q, kb),
                    "rel": rel, "fp16_floor": floor, "error": err})
    # an LSE delivered as [Q, H] (the pre-0.14 layout the original merge assumed) must raise, not broadcast
    layout["lse"] = "qh"
    hq, hk, L, Q = 8, 1, 1500, 300
    npages = (L + BS - 1) // BS + 1
    kc = torch.randn(npages, BS, hk, D, generator=gen).to(FP8)
    vc = torch.randn(npages, BS, hk, D, generator=gen).to(FP8)
    pages = torch.arange((L + BS - 1) // BS, dtype=torch.int32)
    q = torch.randn(Q, hq, D, generator=gen).to(torch.float16)
    out = torch.empty(Q, hq, D, dtype=torch.float16)
    try:
        fp8kv_prefill.prefill_attention(q, kc, vc, pages, L, 1.0, 1.0, D ** -0.5, out, kb=512)
        wrong = {"raised": None}
    except Exception as e:  # noqa: BLE001
        wrong = {"raised": f"{type(e).__name__}: {e}"}
    return {"cases": res, "wrong_layout": wrong}


if __name__ == "__main__":
    what = sys.argv[1]
    try:
        out = {"ok": True, "result": {"sources": sources, "routing": routing, "paged": paged, "merge": merge}[what]()}
    except Exception as e:  # noqa: BLE001
        out = {"ok": False, "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-3000:]}
    print("FP8KV_PROBE " + json.dumps(out, default=str))
