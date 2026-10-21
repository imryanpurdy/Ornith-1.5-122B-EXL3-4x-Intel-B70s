"""Which per-card attention shapes of Ornith the stock vLLM 0.30 FLASH_ATTN path can run on prod's vllm_xpu_kernels,
for BF16 and FP8 KV caches, on ONE XPU card (an M2c / M3 planning check found by T15a).

Ornith has 32 q heads and 2 KV heads (head_dim 256): TP4 = 8 q / 1 kv per card (GQA 8), TP2 = 16 / 1 (GQA 16),
TP1 = 32 / 2 (GQA 16). For each shape and cache dtype: one decode step (1 query over 8,192 keys) and one prefill
chunk (512 queries over 8,192 keys) through FlashAttentionImpl.forward as shipped (no patch), on the same vLLM cache
views as tests/xpu/t15a_fp8kv.py. Records an error, how many calls fell back to vllm_xpu_kernels' PyTorch
reference attention, the missing kernel configuration, and the accuracy vs float64 when it ran.

Usage: python tests/xpu/t15a_gqa16.py --out DIR
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import t15a_fp8kv as T  # noqa: E402  (the fp8kv patch is never applied here: setup() is not called)
import torch  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--stop-file", default=None)
    ap.add_argument("--threads", type=int, default=16)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    from vllm.v1.attention.backends import flash_attn as fa
    import vllm_xpu_kernels.flash_attn_interface as fai
    from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheTensor, KVQuantMode, create_kv_cache_views
    from vllm.v1.kv_cache_layout import KVCacheLayout
    fb = fai._fallback_varlen_attn
    cnt = {"n": 0}

    def counting(*args, **kw):
        cnt["n"] += 1
        return fb(*args, **kw)

    fai._fallback_varlen_attn = counting
    dev = torch.device("xpu", 0)
    gen = torch.Generator().manual_seed(5)
    rows = []
    for kvdt in ("fp8", "auto"):
        for shape in ("tp4", "tp2", "tp1"):
            for L, Q in ((8192, 1), (8192, 512)):
                if a.stop_file and os.path.exists(a.stop_file):
                    sys.exit(3)
                hq, hk = T.SHAPES[shape]
                impl = fa.FlashAttentionImpl(num_heads=hq, head_size=T.D, scale=T.SCALE, num_kv_heads=hk,
                                             alibi_slopes=None, sliding_window=None, kv_cache_dtype=kvdt,
                                             logits_soft_cap=None, attn_type=fa.AttentionType.DECODER,
                                             kv_sharing_target_layer_name=None, sinks=None)
                layer = T.make_layer(dev)
                if kvdt == "fp8":
                    cache = T.Cache(hk, L + T.PAGE, dev, gen)
                    view = cache.view
                    page_tokens = T.PAGE
                else:   # BF16 KV: same construction, bf16 storage (the BF16 attention page is 1,088 tokens)
                    page_tokens = 1088
                    spec = FullAttentionSpec(block_size=page_tokens, num_kv_heads=hk, head_size=T.D,
                                             dtype=torch.bfloat16, kv_quant_mode=KVQuantMode.NONE)
                    nblocks = 2 * ((L + page_tokens - 1) // page_tokens) + 3
                    size = nblocks * spec.page_size_bytes
                    raw = torch.zeros(size, dtype=torch.int8, device=dev)
                    kvt = KVCacheTensor(size=size, layers=["l"], layer_stride=size, block_stride=spec.page_size_bytes)
                    view = create_kv_cache_views(raw, spec, nblocks, KVCacheLayout.LBHNC, kvt, kernel_block_size=64)[0]
                    free = (torch.randperm(nblocks - 1, generator=gen) + 1).tolist()
                n_pages = (L + page_tokens - 1) // page_tokens
                kpp = page_tokens // 64
                kb_ids = [b * kpp + k for b in free[:n_pages] for k in range(kpp)] if kvdt != "fp8" else None
                if kvdt == "fp8":
                    row = cache.alloc(L)
                else:
                    row = torch.zeros(T.BT_WIDTH, dtype=torch.int32)
                    row[:len(kb_ids)] = torch.tensor(kb_ids, dtype=torch.int32)
                    row = row.to(dev)
                qb, kb, vb = T.gen_qkv(L, Q, hq, hk, "gauss", gen, dev)
                T.write_kv(impl, layer, view, kb, vb, row, dev)
                md = T.make_metadata(fa, [(L, Q, row)], dev)
                out = torch.full((Q, hq, T.D), float("nan"), dtype=torch.bfloat16, device=dev)
                c0 = cnt["n"]
                t0 = time.perf_counter()
                err = None
                try:
                    impl.forward(layer, qb, kb[L - Q:], vb[L - Q:], view, md, out)
                    T.sync()
                except Exception as e:  # noqa: BLE001
                    err = f"{type(e).__name__}: {str(e).splitlines()[0][:200]}"
                dt = (time.perf_counter() - t0) * 1e3
                r = {"kv": kvdt, "shape": shape, "hq": hq, "hk": hk, "L": L, "Q": Q, "error": err,
                     "fallback_calls": cnt["n"] - c0, "ms": round(dt, 2)}
                if err is None:
                    kc, vc = view.transpose(1, 2).split(T.D, dim=-1)
                    nkb = (L + 63) // 64
                    idx = row[:nkb].long()
                    if kvdt == "fp8":
                        k64, v64 = cache.readback(row, L)
                    else:
                        k64 = kc.index_select(0, idx).flatten(0, 1)[:L].double().cpu()
                        v64 = vc.index_select(0, idx).flatten(0, 1)[:L].double().cpu()
                    ref, _ = T.ref_attention(qb.double().cpu(), k64, v64, T.SCALE)
                    r["rel"] = T.errors(out, ref)
                    r["bf16_floor"] = T.bf16_floor(ref)
                rows.append(r)
                print("GQA", json.dumps(r), flush=True)
    miss = [c.split("\n\n")[2].strip() if c.count("\n\n") > 2 else c[:120] for c in T.missing_kernel_configs()]
    print("MISSING_KERNEL_CONFIGS", json.dumps(miss), flush=True)
    with open(os.path.join(a.out, "gqa16.json"), "w") as f:
        json.dump({"rows": rows, "missing": miss}, f, indent=1)
    print("T15A_GQA_DONE", flush=True)


if __name__ == "__main__":
    main()
