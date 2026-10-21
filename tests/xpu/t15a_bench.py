"""T15a timing on ONE XPU card (indicative: other jobs share the CPUs): vLLM 0.30's stock FP8 prefill attention vs the
two fp8-KV prefill routes in exl3xpu.fp8kv_prefill, on vLLM's real cache layout (tests/xpu/t15a_fp8kv.py builders):

  stock      FlashAttentionImpl.forward as shipped (paged fp8 K/V, bf16 q, mixed-batch FA2 call)
  paged16    prefill_attention_paged: gather + dequantize the sequence to bf16 (or fp16), one paged causal FA call
  kern16     the paged FA call alone on an already-dequantized bf16 copy (no gather)
  merge      prefill_attention (trellis' KB-block merge); on vllm_xpu_kernels 0.1.14.1 its non-paged head_dim-256 calls
             fall back to PyTorch attention, so it is timed only up to 8,192 keys (its fallback holds H x Q x KB fp32)

Also the accuracy of paged16 vs the float64 reference on two cases. Usage:
  python tests/xpu/t15a_bench.py --out DIR [--stop-file F] [--reps N]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import t15a_fp8kv as T  # noqa: E402  (sets EXL3_VLLM_PATCHES=fp8kv and the import paths)
import torch  # noqa: E402


def med(fn, reps, warm=2):
    for _ in range(warm):
        fn()
    T.sync()
    ts = []
    for _ in range(reps):
        T.sync()
        t0 = time.perf_counter()
        fn()
        T.sync()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts) // 2] * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--stop-file", default=None)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--threads", type=int, default=16)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)
    fa, stock_forward, vp, fp8kv = T.setup()
    dev = torch.device("xpu", 0)
    gen = torch.Generator().manual_seed(99)
    rows = []
    grid = [("tp4", L, Q) for L in (8192, 32768, 131072, 262144) for Q in (64, 2112, 4096)]
    grid += [("tp1", 32768, 2112), ("tp1", 131072, 2112)]
    for shape, L, Q in grid:
        if a.stop_file and os.path.exists(a.stop_file):
            print("STOPPED", flush=True)
            sys.exit(3)
        hq, hk = T.SHAPES[shape]
        impl = T.make_impl(fa, hq, hk)
        layer = T.make_layer(dev)
        cache = T.Cache(hk, L + T.PAGE, dev, gen)
        qb, kb, vb = T.gen_qkv(L, Q, hq, hk, "peaky", gen, dev)
        row = cache.alloc(L)
        T.write_kv(impl, layer, cache.view, kb, vb, row, dev)
        md = T.make_metadata(fa, [(L, Q, row)], dev)
        out = torch.empty(Q, hq, T.D, dtype=torch.bfloat16, device=dev)
        kc, vc = cache.view.transpose(1, 2).split(T.D, dim=-1)
        kc, vc = kc.view(T.FP8), vc.view(T.FP8)
        r = {"shape": shape, "L": L, "Q": Q}
        miss0 = set(T.missing_kernel_configs())
        r["stock_ms"] = med(lambda: stock_forward(impl, layer, qb, kb[L - Q:], vb[L - Q:], cache.view, md, out), a.reps)
        r["paged_bf16_ms"] = med(lambda: fp8kv.prefill_attention_paged(qb, kc, vc, row, L, 1.0, 1.0, T.SCALE, out),
                                 a.reps)
        q16 = qb.to(torch.float16)
        o16 = torch.empty(Q, hq, T.D, dtype=torch.float16, device=dev)
        r["paged_fp16_ms"] = med(lambda: fp8kv.prefill_attention_paged(q16, kc, vc, row, L, 1.0, 1.0, T.SCALE, o16),
                                 a.reps)
        nb = (L + 63) // 64
        kd = kc.index_select(0, row[:nb]).to(torch.bfloat16)
        vd = vc.index_select(0, row[:nb]).to(torch.bfloat16)
        from vllm_xpu_kernels.flash_attn_interface import flash_attn_varlen_func
        bt = torch.arange(nb, dtype=torch.int32, device=dev).unsqueeze(0)
        cq = torch.tensor([0, Q], dtype=torch.int32, device=dev)
        sk = torch.tensor([L], dtype=torch.int32, device=dev)
        r["kern_bf16_ms"] = med(lambda: flash_attn_varlen_func(qb, kd, vd, Q, cq, L, seqused_k=sk, block_table=bt,
                                                              causal=True, softmax_scale=T.SCALE, out=out,
                                                              is_mix_batch=False), a.reps)
        r["gather_dequant_ms"] = med(lambda: (kc.index_select(0, row[:nb]).to(torch.bfloat16),
                                              vc.index_select(0, row[:nb]).to(torch.bfloat16)), a.reps)
        if L <= 8192:
            r["merge_ms"] = med(lambda: fp8kv.prefill_attention(q16, kc, vc, row, L, 1.0, 1.0, T.SCALE, o16), 2,
                                warm=1)
        r["missing_kernel_configs_new"] = sorted(set(T.missing_kernel_configs()) - miss0)
        r["stock_over_paged_bf16"] = r["stock_ms"] / r["paged_bf16_ms"]
        r["tflops_stock"] = 4 * Q * (L - Q / 2) * T.D * hq / (r["stock_ms"] * 1e-3) / 1e12
        r["tflops_kern_bf16"] = 4 * Q * (L - Q / 2) * T.D * hq / (r["kern_bf16_ms"] * 1e-3) / 1e12
        r["peak_mem_gib"] = torch.xpu.max_memory_allocated() / 2 ** 30
        torch.xpu.reset_peak_memory_stats()
        rows.append(r)
        print("BENCH", json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
        del cache, kd, vd, kb, vb, qb
        torch.xpu.empty_cache()
    # accuracy of the paged route vs float64 on the stored values
    acc = []
    for shape, L, Q in (("tp4", 32768, 2112), ("tp1", 8192, 1000)):
        hq, hk = T.SHAPES[shape]
        impl = T.make_impl(fa, hq, hk)
        layer = T.make_layer(dev)
        cache = T.Cache(hk, L + T.PAGE, dev, gen)
        qb, kb, vb = T.gen_qkv(L, Q, hq, hk, "peaky", gen, dev)
        row = cache.alloc(L)
        T.write_kv(impl, layer, cache.view, kb, vb, row, dev)
        kc, vc = cache.view.transpose(1, 2).split(T.D, dim=-1)
        out = torch.empty(Q, hq, T.D, dtype=torch.bfloat16, device=dev)
        fp8kv.prefill_attention_paged(qb, kc.view(T.FP8), vc.view(T.FP8), row, L, 1.0, 1.0, T.SCALE, out)
        T.sync()
        k64, v64 = cache.readback(row, L)
        ref, _ = T.ref_attention(qb.double().cpu(), k64, v64, T.SCALE)
        e = T.errors(out, ref)
        acc.append({"shape": shape, "L": L, "Q": Q, "paged_bf16": e, "bf16_floor": T.bf16_floor(ref)})
        print("ACC", json.dumps(acc[-1]), flush=True)
    with open(os.path.join(a.out, "bench.json"), "w") as f:
        json.dump({"rows": rows, "acc": acc}, f, indent=1)
    print("T15A_BENCH_DONE", flush=True)


if __name__ == "__main__":
    main()
