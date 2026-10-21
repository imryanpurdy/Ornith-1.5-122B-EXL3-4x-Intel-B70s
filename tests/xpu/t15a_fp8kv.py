"""T15a (DESIGN rev 2 section 5.4) on ONE XPU card: the fp8-KV prefill route (EXL3_VLLM_PATCHES=fp8kv) against
vLLM 0.30's stock FP8 flash attention and a float64 reference, through vLLM's own pieces:

  - the cache is vLLM's: one int8 backing buffer viewed by vllm.v1.kv_cache_interface.create_kv_cache_views
    (FullAttentionSpec, fp8 = uint8 storage, 2,112-token manager pages split into 64-token kernel blocks, LBHNC
    layout as the prod engine logs), manager block 0 reserved, pages taken in shuffled order, the block table in
    kernel-block units and as wide as a 262,144-token request (125 pages x 33 kernel blocks), trailing entries 0;
  - every byte outside the sequence's pages is 0x7F (an e4m3fn NaN), the sequence's pages are zeroed first (vLLM's
    kv_block_zeroer does that on allocation), so a read of a wrong page or past the sequence shows up as NaN;
  - K/V are written by FlashAttentionImpl.do_kv_cache_update (reshape_and_cache_flash, fp8, scale 1.0) in
    2,112-token chunks from a strided bf16 buffer, as the attention layer passes them;
  - "stock" is FlashAttentionImpl.forward as vLLM 0.30 ships it, "patched" is the class method after
    vllm_plugin.apply_patches() with EXL3_VLLM_PATCHES=fp8kv (the plugin's own selection path; routed calls go to
    fp8kv_prefill.prefill_attention_paged), both on FlashAttentionMetadata filled as
    FlashAttentionMetadataBuilder.build fills it for a non-DCP, non-cascade batch; "merge" (some cases) is
    trellis' block-merge prefill_attention called directly, with the LSE fix; it is not served on 0.30;
  - the reference is float64 attention on the fp8 values actually stored in the cache (read back by block table),
    so it isolates the attention kernels from the fp8 rounding of K/V (that rounding is reported separately).

Patch verdict per case ("ok"): routed cases move FP8KV_STATS["routed"] by exactly one per call, hit no
vllm_xpu_kernels PyTorch fallback, are finite, deterministic and within rel <= 2 * floor + 1e-3 (rel = ||out - ref||
/ ||ref|| over all rows and heads, floor = ||bf16(ref) - ref|| / ||ref||); every other case leaves the counter alone
and equals stock bit for bit (or raises stock's own error). What the stock kernel does (fallbacks, errors, accuracy)
is recorded separately as findings. Also: the kernel's LSE layout and units, the KV write path byte for byte against
torch's clamp + cast (incl. saturation, NaN/Inf inputs, -1 padding slots), and (with --model) a hard bound on
max|K| / max|q| from Ornith's q_norm / k_norm weights.

Usage (prod image, exactly one card visible via ZE_AFFINITY_MASK):
  python tests/xpu/t15a_fp8kv.py --out DIR [--quick] [--stop-file F] [--model CHECKPOINT_DIR] [--threads N]
Writes DIR/results.jsonl and DIR/summary.json; prints one line per case and T15A_PASS / T15A_FAIL at the end.
The stop file is polled between cases (never inside a kernel); a stopped run exits 3.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "core" / "src", ROOT / "xpu"):
    sys.path.insert(0, str(_p))

os.environ["EXL3_VLLM_PATCHES"] = "fp8kv"
os.environ["EXL3_FP8KV_CHECK"] = "1"
for _v in ("EXL3_FP8KV_MIXED", "EXL3_ONEDNN_ATTN"):
    os.environ.pop(_v, None)

import torch  # noqa: E402

D = 256
PAGE = 2112                      # FP8 attention page (tokens) at TP4, DESIGN section 5.6
KBLK = 64                        # FLASH_ATTN kernel block on XPU ("Setting kv cache block size to 64")
MAX_MODEL_LEN = 262144
BT_WIDTH = math.ceil(MAX_MODEL_LEN / PAGE) * (PAGE // KBLK)     # 125 pages x 33 kernel blocks
SCALE = D ** -0.5
FP8 = torch.float8_e4m3fn
NAN_BYTE = 0x7F                  # e4m3fn NaN
SHAPES = {"tp4": (8, 1), "tp2": (16, 1), "tp1": (32, 2)}        # (q heads, kv heads) per card; Ornith 32 q / 2 kv

STOP = None
FALLBACK = {"n": 0}              # calls of vllm_xpu_kernels' PyTorch reference attention (kernel not compiled)


def stopped() -> bool:
    return bool(STOP) and os.path.exists(STOP)


def sync():
    torch.xpu.synchronize()


# ------------------------------------------------------------------------------------------------ vLLM pieces
def setup():
    from vllm.v1.attention.backends import flash_attn as fa
    stock_forward = fa.FlashAttentionImpl.forward
    from exl3xpu import fp8kv_prefill, vllm_patches, vllm_plugin
    res = vllm_plugin.apply_patches()
    if res != {"fp8kv": "applied"} or fa.FlashAttentionImpl.forward is stock_forward:
        raise SystemExit(f"fp8kv patch not applied: {res}")
    if fa.FlashAttentionImpl._exl3_fp8kv_orig_forward is not stock_forward:
        raise SystemExit("patched class does not keep the stock forward")
    import vllm_xpu_kernels.flash_attn_interface as fai
    fb = fai._fallback_varlen_attn

    def counting(*args, **kw):
        FALLBACK["n"] += 1
        return fb(*args, **kw)

    fai._fallback_varlen_attn = counting      # flash_attn_varlen_func looks it up at call time
    return fa, stock_forward, vllm_patches, fp8kv_prefill


def make_impl(fa, hq, hk):
    return fa.FlashAttentionImpl(num_heads=hq, head_size=D, scale=SCALE, num_kv_heads=hk, alibi_slopes=None,
                                 sliding_window=None, kv_cache_dtype="fp8", logits_soft_cap=None,
                                 attn_type=fa.AttentionType.DECODER, kv_sharing_target_layer_name=None, sinks=None)


def make_layer(dev, k_scale=1.0, v_scale=1.0):
    from vllm.model_executor.layers.attention.attention import set_default_quant_scales
    layer = torch.nn.Module()
    set_default_quant_scales(layer, register_buffer=True)
    layer.to(dev)
    layer._k_scale.fill_(k_scale)
    layer._v_scale.fill_(v_scale)
    layer._k_scale_float, layer._v_scale_float = float(k_scale), float(v_scale)
    return layer


class Cache:
    """vLLM's view of one attention layer's fp8 KV cache; pages handed out in shuffled order."""

    def __init__(self, hk, max_tokens, dev, gen):
        from vllm.v1.kv_cache_interface import (FullAttentionSpec, KVCacheTensor, KVQuantMode,
                                                create_kv_cache_views)
        from vllm.v1.kv_cache_layout import KVCacheLayout
        self.hk, self.dev = hk, dev
        self.spec = FullAttentionSpec(block_size=PAGE, num_kv_heads=hk, head_size=D, dtype=torch.uint8,
                                      kv_quant_mode=KVQuantMode.FP8_PER_TENSOR)
        self.page_bytes = self.spec.page_size_bytes
        need = math.ceil(max_tokens / PAGE)
        self.nblocks = 2 * need + 3
        size = self.nblocks * self.page_bytes
        self.raw = torch.full((size,), NAN_BYTE, dtype=torch.int8, device=dev)
        kvt = KVCacheTensor(size=size, layers=["model.layers.3.self_attn.attn"], layer_stride=size,
                            block_stride=self.page_bytes)
        self.view = create_kv_cache_views(self.raw, self.spec, self.nblocks, KVCacheLayout.LBHNC, kvt,
                                          kernel_block_size=KBLK)[0]
        assert tuple(self.view.shape) == (self.nblocks * (PAGE // KBLK), hk, KBLK, 2 * D), self.view.shape
        self.free = (torch.randperm(self.nblocks - 1, generator=gen) + 1).tolist()   # block 0 = null block

    def layout(self):
        return {"shape": list(self.view.shape), "stride": list(self.view.stride()), "dtype": str(self.view.dtype),
                "page_bytes": self.page_bytes, "manager_blocks": self.nblocks}

    def alloc(self, tokens):
        """-> block-table row (int32, kernel-block units, BT_WIDTH wide) for a sequence of `tokens` tokens."""
        n = math.ceil(tokens / PAGE)
        pages, self.free = self.free[:n], self.free[n:]
        for b in pages:                                    # vLLM zeroes freshly allocated blocks
            self.raw[b * self.page_bytes:(b + 1) * self.page_bytes].zero_()
        kb = [b * (PAGE // KBLK) + k for b in pages for k in range(PAGE // KBLK)]
        row = torch.zeros(BT_WIDTH, dtype=torch.int32)
        row[:len(kb)] = torch.tensor(kb, dtype=torch.int32)
        return row.to(self.dev)

    def readback(self, bt_row, length):
        """fp8 K and V of tokens [0, length) of one sequence -> float64 CPU [length, hk, D] each."""
        kc, vc = self.view.transpose(1, 2).split(D, dim=-1)              # uint8 [B, 64, hk, D]
        nkb = math.ceil(length / KBLK)
        idx = bt_row[:nkb].long()
        k = kc.index_select(0, idx).flatten(0, 1)[:length].contiguous().view(FP8)
        v = vc.index_select(0, idx).flatten(0, 1)[:length].contiguous().view(FP8)
        return k.float().cpu().double(), v.float().cpu().double()


def slots_for(bt_row, start, end, dev):
    pos = torch.arange(start, end, device=dev)
    return (bt_row.long()[pos // KBLK] * KBLK + pos % KBLK).to(torch.int64)


def make_metadata(fa, seqs, dev):
    """seqs: list of (seq_len, q_len, bt_row) -> FlashAttentionMetadata as build() fills it (no DCP, no cascade)."""
    q_lens = [q for _, q, _ in seqs]
    qsl = torch.tensor([0] + list(torch.tensor(q_lens).cumsum(0).tolist()), dtype=torch.int32, device=dev)
    seq_lens = torch.tensor([s for s, _, _ in seqs], dtype=torch.int32, device=dev)
    bt = torch.stack([row for _, _, row in seqs])
    slot = torch.cat([slots_for(row, s - q, s, dev) for s, q, row in seqs])
    return fa.FlashAttentionMetadata(
        num_actual_tokens=sum(q_lens), max_query_len=max(q_lens), query_start_loc=qsl,
        max_seq_len=max(s for s, _, _ in seqs), seq_lens=seq_lens, block_table=bt, slot_mapping=slot,
        max_dcp_context_kv_len=0, dcp_context_kv_lens=None, num_decode_reqs=0, num_prefill_reqs=0,
        num_decode_tokens=0, num_prefill_tokens=0, use_cascade=False, common_prefix_len=0,
        scheduler_metadata=None, cu_prefix_query_lens=None, prefix_kv_lens=None, suffix_kv_lens=None,
        prefix_scheduler_metadata=None, max_num_splits=0, causal=True)


# ------------------------------------------------------------------------------------------------ data
def gen_qkv(L, Q, hq, hk, regime, gen, dev):
    """bf16 q for the last Q positions and K/V for all L positions, the K/V as strided views of one wider buffer
    (as the attention layer passes slices of its qkv projection)."""
    big = torch.zeros(L, D + 2 * hk * D, dtype=torch.bfloat16, device=dev)
    q = torch.randn(Q, hq, D, generator=gen)
    k = torch.randn(L, hk, D, generator=gen)
    v = torch.randn(L, hk, D, generator=gen)
    if regime == "peaky":
        ch = torch.ones(D)
        ch[:64] = 3.0                                       # the 64 rotary dims (partial_rotary_factor 0.25)
        ch[torch.randperm(D, generator=gen)[:4]] = 8.0      # a few outlier channels
        k = k * ch
        v = v * torch.exp(torch.randn(L, 1, 1, generator=gen))          # per-token lognormal V scale
        v = v.clamp(-300, 300)
        g = hq // hk
        nn = min(16, Q)
        rows = torch.randperm(Q, generator=gen)[:nn]
        prefix = L - Q
        for r in rows.tolist():                             # needles: a far key that one head of the group locks on
            h = int(torch.randint(hq, (1,), generator=gen))
            j = int(torch.randint(max(1, min(prefix, 30000)) if prefix > 0 else prefix + r + 1, (1,),
                                  generator=gen))
            k[j, h // g] = q[r, h] * 1.5
    big[:, D: D + hk * D] = k.reshape(L, hk * D).to(dev, torch.bfloat16)
    big[:, D + hk * D:] = v.reshape(L, hk * D).to(dev, torch.bfloat16)
    kb = big[:, D: D + hk * D].view(L, hk, D)
    vb = big[:, D + hk * D:].view(L, hk, D)
    qb = q.to(dev, torch.bfloat16)
    return qb, kb, vb


def write_kv(impl, layer, cache_view, kb, vb, bt_row, dev, start=0, end=None):
    end = kb.shape[0] if end is None else end
    for s in range(start, end, PAGE):
        e = min(s + PAGE, end)
        impl.do_kv_cache_update(layer, kb[s:e], vb[s:e], cache_view, slots_for(bt_row, s, e, dev))


# ------------------------------------------------------------------------------------------------ reference
def ref_attention(q64, k64, v64, scale, max_elems=1 << 26):
    """float64 causal attention, q [Q, Hq, D] at the end of k/v [L, Hk, D] (bottom-right aligned) -> [Q, Hq, D]."""
    Q, hq, _ = q64.shape
    L, hk, _ = k64.shape
    g = hq // hk
    prefix = L - Q
    out = torch.empty(Q, hq, D, dtype=torch.float64)
    lse = torch.empty(hq, Q, dtype=torch.float64)
    cols = torch.arange(L)
    qc = max(1, max_elems // (g * L))
    for kv in range(hk):
        K, V = k64[:, kv], v64[:, kv]
        for s in range(0, Q, qc):
            e = min(s + qc, Q)
            qq = q64[s:e, kv * g:(kv + 1) * g].transpose(0, 1).reshape(g * (e - s), D)     # [g*n, D]
            S = (qq @ K.T) * scale
            rows = (prefix + torch.arange(s, e)).repeat(g)
            S.masked_fill_(cols[None, :] > rows[:, None], float("-inf"))
            m = S.amax(-1, keepdim=True)
            P = torch.exp(S - m)
            z = P.sum(-1, keepdim=True)
            o = (P @ V) / z
            out[s:e, kv * g:(kv + 1) * g] = o.view(g, e - s, D).transpose(0, 1)
            lse[kv * g:(kv + 1) * g, s:e] = (m + torch.log(z)).view(g, e - s)
    return out, lse


def errors(out, ref):
    o = out.double().cpu()
    nan = int((~torch.isfinite(o)).sum())
    if nan:
        return {"nonfinite": nan, "rel": float("inf"), "max_abs": float("inf")}
    e = o - ref
    return {"nonfinite": 0, "rel": float(e.norm() / ref.norm()), "max_abs": float(e.abs().max())}


def bf16_floor(ref):
    return float((ref.to(torch.bfloat16).double() - ref).norm() / ref.norm())


# ------------------------------------------------------------------------------------------------ cases
def timed(fn, reps):
    ts = []
    for _ in range(reps):
        sync()
        t0 = time.perf_counter()
        fn()
        sync()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts) // 2]


def missing_kernel_configs():
    from vllm_xpu_kernels import flash_attn_interface as fai
    return sorted(getattr(fai, "_warned_missing_configs", set()))


def _try(fn):
    try:
        fn()
        sync()
        return None
    except Exception as e:  # noqa: BLE001
        return f"{type(e).__name__}: {str(e).splitlines()[0][:300]}"


def run_case(ctx, shape, seqs_spec, regime, gen, k_scale=1.0, v_scale=1.0, reps=3, fp8_vs_bf16=False,
             merge=False):
    """seqs_spec: list of (L, Q). Returns a result dict. The patch verdict ("ok") is separate from what the stock
    kernel does (stock_* fields): a stock fallback or crash is a finding about prod's kernels, not about the patch,
    as long as the patched forward falls through to exactly the same thing."""
    fa, stock_forward, vp, fp8kv = ctx["fa"], ctx["stock"], ctx["vp"], ctx["fp8kv"]
    dev = ctx["dev"]
    hq, hk = SHAPES[shape]
    impl = make_impl(fa, hq, hk)
    layer = make_layer(dev, k_scale, v_scale)
    cache = Cache(hk, sum(L for L, _ in seqs_spec) + PAGE, dev, gen)
    seqs, qs, refs, fp8_info = [], [], [], []
    for L, Q in seqs_spec:
        qb, kb, vb = gen_qkv(L, Q, hq, hk, regime, gen, dev)
        row = cache.alloc(L)
        write_kv(impl, layer, cache.view, kb, vb, row, dev)
        seqs.append((L, Q, row))
        qs.append(qb)
        refs.append((qb, kb, vb, row))
    sync()
    md = make_metadata(fa, seqs, dev)
    q_all = torch.cat(qs)
    n = q_all.shape[0]
    k_new = torch.cat([kb[L - Q:] for (L, Q), (_, kb, _, _) in zip(seqs_spec, refs)])
    v_new = torch.cat([vb[L - Q:] for (L, Q), (_, _, vb, _) in zip(seqs_spec, refs)])
    out_s1 = torch.full((n, hq, D), float("nan"), dtype=torch.bfloat16, device=dev)
    out_s2 = torch.full_like(out_s1, float("nan"))
    out_p = torch.full_like(out_s1, float("nan"))
    stock = lambda o: stock_forward(impl, layer, q_all, k_new, v_new, cache.view, md, o)   # noqa: E731
    patched = lambda o: impl.forward(layer, q_all, k_new, v_new, cache.view, md, o)        # noqa: E731
    m0, f0 = set(missing_kernel_configs()), FALLBACK["n"]
    stock_err = _try(lambda: stock(out_s1)) or _try(lambda: stock(out_s2))
    m1, f1 = set(missing_kernel_configs()), FALLBACK["n"]
    stock_det = stock_err is None and bool(torch.equal(out_s1, out_s2))
    r0 = vp.FP8KV_STATS["routed"]
    patched_err = _try(lambda: patched(out_p))
    routed = vp.FP8KV_STATS["routed"] - r0
    m2, f2 = set(missing_kernel_configs()), FALLBACK["n"]
    out_p1 = out_p.clone()
    t_stock = timed(lambda: stock(out_s2), reps) if stock_err is None else None
    r1 = vp.FP8KV_STATS["routed"]
    t_patch = timed(lambda: patched(out_p), reps) if patched_err is None else None
    routed_t = vp.FP8KV_STATS["routed"] - r1
    sync()
    patched_det = patched_err is None and bool(torch.equal(out_p, out_p1))
    # trellis' block-merge route called directly (not served on 0.30): accuracy, time, kernel fallback
    merge_info = None
    if merge:
        (L, Q), (qb, _, _, row) = seqs_spec[0], refs[0]
        kc, vc = cache.view.transpose(1, 2).split(D, dim=-1)
        kc, vc = kc.view(FP8), vc.view(FP8)
        q16 = qb.to(torch.float16)
        om = torch.empty(Q, hq, D, dtype=torch.float16, device=dev)
        mm0, mf0 = set(missing_kernel_configs()), FALLBACK["n"]
        merr = _try(lambda: fp8kv.prefill_attention(q16, kc, vc, row, L, k_scale, v_scale, SCALE, om))
        mf1 = FALLBACK["n"]
        tm = timed(lambda: fp8kv.prefill_attention(q16, kc, vc, row, L, k_scale, v_scale, SCALE, om), 1) \
            if merr is None else None
        merge_info = {"error": merr, "t_ms": tm * 1e3 if tm else None, "parts": fp8kv.num_parts(L, Q),
                      "fallback_calls": mf1 - mf0,
                      "fallback_configs": [c.split("\n\n")[2].strip() if c.count("\n\n") > 2 else c[:80]
                                           for c in sorted(set(missing_kernel_configs()) - mm0)], "out": om}
    # reference per sequence on the stored fp8 values
    o0 = 0
    ref_cat, outs_s, outs_p = [], [], []
    for (L, Q), (qb, kb, vb, row) in zip(seqs_spec, refs):
        k64, v64 = cache.readback(row, L)
        ref, _ = ref_attention(qb.double().cpu(), k64 * k_scale, v64 * v_scale, SCALE)
        ref_cat.append(ref)
        outs_s.append(out_s1[o0:o0 + Q])
        outs_p.append(out_p[o0:o0 + Q])
        if fp8_vs_bf16:
            ref16, _ = ref_attention(qb.double().cpu(), kb.double().cpu(), vb.double().cpu(), SCALE)
            fp8_info.append({"L": L, "Q": Q, "rel_fp8kv_vs_bf16kv": float((ref - ref16).norm() / ref16.norm())})
        o0 += Q
    ref = torch.cat(ref_cat)
    es = errors(torch.cat(outs_s), ref) if stock_err is None else None
    ep = errors(torch.cat(outs_p), ref) if patched_err is None else None
    if merge_info is not None:
        merge_info["err"] = errors(merge_info.pop("out"), ref) if merge_info["error"] is None else None
        merge_info.pop("out", None)
    floor = bf16_floor(ref)
    sp = (float((out_p.double() - out_s1.double()).norm() / out_s1.double().norm())
          if stock_err is None and patched_err is None else None)
    eligible_expected = (len(seqs_spec) == 1 and seqs_spec[0][1] >= 64 and seqs_spec[0][0] >= 4096)
    tol = 2 * floor + 1e-3
    short = lambda c: c.split("\n\n")[2].strip() if c.count("\n\n") > 2 else c[:80]   # noqa: E731
    res = {
        "shape": shape, "hq": hq, "hk": hk, "seqs": seqs_spec, "regime": regime, "k_scale": k_scale,
        "v_scale": v_scale, "routed_expected": eligible_expected, "routed_calls": routed + routed_t,
        "bf16_floor": floor, "tol": tol,
        "stock_error": stock_err, "stock_fallback_configs": [short(c) for c in sorted(m1 - m0)],
        "stock_fallback_calls": f1 - f0, "patched_fallback_calls": f2 - f1,
        "stock_deterministic": stock_det, "stock": es,
        "stock_within_tol": es is not None and es["nonfinite"] == 0 and es["rel"] <= tol,
        "patched_error": patched_err, "patched_fallback_configs": [short(c) for c in sorted(m2 - m1)],
        "patched_deterministic": patched_det, "patched": ep, "rel_patched_vs_stock": sp,
        "bit_equal_patched_stock": stock_err is None and patched_err is None and bool(torch.equal(out_p, out_s1)),
        "t_stock_ms": t_stock * 1e3 if t_stock else None, "t_patched_ms": t_patch * 1e3 if t_patch else None,
        "fp8_info": fp8_info, "merge": merge_info,
    }
    if eligible_expected:
        ok = (patched_err is None and routed == 1 and routed_t == reps and res["patched_fallback_configs"] == []
              and res["patched_fallback_calls"] == 0 and ep["nonfinite"] == 0 and ep["rel"] <= tol and patched_det)
    else:
        ok = routed == 0 and routed_t == 0 and res["patched_fallback_configs"] == [] if stock_err is None else \
            routed == 0 and routed_t == 0
        if stock_err is None:
            ok = ok and res["bit_equal_patched_stock"] and stock_det
        else:   # falls through to the same stock failure
            ok = ok and patched_err == stock_err
    res["ok"] = bool(ok)
    return res


def lse_check(ctx, gen):
    """The kernel's LSE: layout [H, Q] and natural-log units of scale * q.k (what the merge assumes). Also runs
    the pre-port merge (LSE stacked as if [Q, H]) on two real parts to show how it fails on this kernel."""
    from exl3xpu.fp8kv_prefill import _fa
    dev = ctx["dev"]
    out = {}
    for hq, hk, Q, L, causal in ((8, 1, 300, 1000, False), (8, 1, 777, 777, True), (32, 2, 64, 4096, False)):
        q = torch.randn(Q, hq, D, generator=gen).to(dev, torch.float16)
        k = torch.randn(L, hk, D, generator=gen).to(dev, torch.float16)
        v = torch.randn(L, hk, D, generator=gen).to(dev, torch.float16)
        fb0 = FALLBACK["n"]
        o, lse = _fa(q, k, v, causal, SCALE)
        sync()
        fbc = FALLBACK["n"] - fb0
        if causal:
            ref_o, ref_lse = ref_attention(q.double().cpu(), k.double().cpu(), v.double().cpu(), SCALE)
        else:
            ref_o, ref_lse = _ref_noncausal(q.double().cpu(), k.double().cpu(), v.double().cpu())
        out[f"hq{hq}_hk{hk}_Q{Q}_L{L}_{'causal' if causal else 'full'}"] = {
            "lse_shape": list(lse.shape), "lse_dtype": str(lse.dtype),
            "lse_max_abs_err": (float((lse.double().cpu() - ref_lse).abs().max()) if list(lse.shape) == [hq, Q]
                                else None),
            "out_rel": float((o.double().cpu() - ref_o).norm() / ref_o.norm()),
            "fallback_calls": fbc,
        }
    # the pre-port merge on two parts (prefix + causal) of a Q != H chunk
    hq, hk, Q, L = 8, 1, 300, 1300
    q = torch.randn(Q, hq, D, generator=gen).to(dev, torch.float16)
    k = torch.randn(L, hk, D, generator=gen).to(dev, torch.float16)
    v = torch.randn(L, hk, D, generator=gen).to(dev, torch.float16)
    parts = [_fa(q, k[:L - Q], v[:L - Q], False, SCALE), _fa(q, k[L - Q:], v[L - Q:], True, SCALE)]
    try:
        lses = torch.stack([p[1] for p in parts])
        m = lses.amax(0)
        w = torch.exp(lses - m)
        num = sum(p[0].float() * wi.unsqueeze(-1) for p, wi in zip(parts, w))
        res = (num / w.sum(0).unsqueeze(-1))
        sync()
        ref_o, _ = ref_attention(q.double().cpu(), k.double().cpu(), v.double().cpu(), SCALE)
        out["old_merge_q300_h8"] = {"raised": None, "out_shape": list(res.shape)}
    except Exception as e:  # noqa: BLE001
        out["old_merge_q300_h8"] = {"raised": f"{type(e).__name__}: {str(e)[:200]}"}
    return out


def _ref_noncausal(q64, k64, v64):
    Q, hq, _ = q64.shape
    L, hk, _ = k64.shape
    g = hq // hk
    kk = k64.repeat_interleave(g, dim=1)
    vv = v64.repeat_interleave(g, dim=1)
    S = torch.einsum("qhd,khd->hqk", q64, kk) * SCALE
    lse = torch.logsumexp(S, -1)
    o = torch.einsum("hqk,khd->qhd", torch.softmax(S, -1), vv)
    return o, lse


def write_path_check(ctx, gen):
    """reshape_and_cache_flash (fp8, scale 1.0) through do_kv_cache_update vs torch clamp(+-448) + cast to e4m3fn,
    byte for byte, from strided bf16 K/V in scattered slot order with -1 padding slots mixed in. Specials (exact
    fp8 values, rounding midpoints, subnormals, values past 448, +-inf, NaN) sit in token 0's K, head 0."""
    fa, dev = ctx["fa"], ctx["dev"]
    res = {}
    for hk in (1, 2):
        hq = 8 * hk
        impl = make_impl(fa, hq, hk)
        layer = make_layer(dev)
        cache = Cache(hk, 3 * PAGE, dev, gen)
        N = 3 * PAGE - 100
        row = cache.alloc(N)
        x = torch.randn(N, 2, hk, D, generator=gen) * torch.exp(torch.randn(N, 1, 1, 1, generator=gen) * 1.5)
        specials = torch.tensor([0.0, -0.0, 448.0, -448.0, 449.0, 464.0, -480.0, 500.0, 1000.0, -1e4, 65280.0,
                                 2.0 ** -9, 2.0 ** -10, 3 * 2.0 ** -10, 1e-3, 240.0, 232.0, 248.0, 0.5 + 1 / 32,
                                 float("inf"), float("-inf"), float("nan")])
        x.view(-1)[:specials.numel()] = specials
        xb = x.to(torch.bfloat16)
        big = torch.zeros(N + 8, D + 2 * hk * D, dtype=torch.bfloat16)
        big[:N, D: D + hk * D] = xb[:, 0].reshape(N, hk * D)
        big[:N, D + hk * D:] = xb[:, 1].reshape(N, hk * D)
        big[N:, D:] = 100.0                                                   # 8 padding rows, slot -1
        big = big.to(dev)
        kb = big[:, D: D + hk * D].view(N + 8, hk, D)
        vb = big[:, D + hk * D:].view(N + 8, hk, D)
        slots = torch.cat([slots_for(row, 0, N, dev), torch.full((8,), -1, dtype=torch.int64, device=dev)])
        perm = torch.randperm(N + 8, generator=gen).to(dev)                   # scattered order, padding mixed in
        impl.do_kv_cache_update(layer, kb[perm], vb[perm], cache.view, slots[perm])
        sync()
        kc, vc = cache.view.transpose(1, 2).split(D, dim=-1)
        nkb = math.ceil(N / KBLK)
        idx = row[:nkb].long()
        got_k = kc.index_select(0, idx).flatten(0, 1).cpu()                   # uint8 [nkb*64, hk, D]
        got_v = vc.index_select(0, idx).flatten(0, 1).cpu()
        want = torch.clamp(xb.float(), -448.0, 448.0).to(FP8).view(torch.uint8)    # torch: NaN stays NaN (0x7f)
        nan_in = torch.isnan(xb.float())
        mism_k = (got_k[:N] != want[:, 0]) & ~nan_in[:, 0]
        mism_v = (got_v[:N] != want[:, 1]) & ~nan_in[:, 1]
        flat_got = got_k[:N].reshape(-1)
        sp = {}
        for i, sv in enumerate(specials.tolist()):
            bt = int(flat_got[i])
            fv = float(torch.tensor([bt], dtype=torch.uint8).view(FP8).float())
            sp[repr(sv)] = {"byte": f"0x{bt:02x}", "value": fv, "torch_byte": f"0x{int(want.reshape(-1)[i]):02x}"}
        # bytes outside the sequence's pages must still be the 0x7F fill (nothing written elsewhere)
        pages = sorted(set((idx // (PAGE // KBLK)).tolist()))
        rawc = cache.raw.view(torch.uint8).cpu()
        keep = torch.ones(rawc.numel(), dtype=torch.bool)
        for b in pages:
            keep[b * cache.page_bytes:(b + 1) * cache.page_bytes] = False
        outside_ok = bool((rawc[keep] == NAN_BYTE).all())
        r = {"tokens": N, "elements": int(2 * N * hk * D), "mismatch_k": int(mism_k.sum()),
             "mismatch_v": int(mism_v.sum()), "nan_inputs": int(nan_in.sum()),
             "tail_after_seq_zero": bool((got_k[N:] == 0).all() and (got_v[N:] == 0).all()),
             "outside_pages_untouched": outside_ok, "specials_k": sp}
        r["ok"] = r["mismatch_k"] == 0 and r["mismatch_v"] == 0 and r["tail_after_seq_zero"] and outside_ok
        res[f"hk{hk}"] = r
    return res


def norm_bounds(model_dir):
    """Hard bounds from the checkpoint: GemmaRMSNorm y = x / rms(x) * (1 + w) gives |y_i| <= sqrt(D) * |1 + w_i|;
    partial NeoX RoPE rotates pairs, so max|K| (pre-cache, any token) <= sqrt(D) * max_i |1 + w_i| per layer."""
    from safetensors import safe_open
    idx = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))["weight_map"]
    out = {}
    names = sorted(k for k in idx if k.endswith(("self_attn.k_norm.weight", "self_attn.q_norm.weight")))
    for name in names:
        with safe_open(os.path.join(model_dir, idx[name]), "pt") as f:
            w = f.get_tensor(name).float()
        g = (1.0 + w).abs()
        out[name] = {"max_abs_1_plus_w": float(g.max()), "bound": float(math.sqrt(D) * g.max()),
                     "mean_abs_1_plus_w": float(g.mean())}
    return out


# ------------------------------------------------------------------------------------------------ plan
def plan(quick):
    S = []
    add = lambda shape, seqs, regime="peaky", **kw: S.append((shape, seqs, regime, kw))   # noqa: E731
    # routed: one sequence, >= 64 queries, >= 4,096 keys
    add("tp4", [(4096, 4096)], "gauss", merge=True)        # L = Q, both thresholds at their edge
    add("tp4", [(4096, 64)], "gauss", merge=True)          # Q at its edge
    add("tp4", [(4096, 2112)], merge=True)
    add("tp4", [(8192, 2112)], fp8_vs_bf16=True, merge=True)
    add("tp4", [(8192, 4096)], "gauss", fp8_vs_bf16=True)
    add("tp4", [(8192, 1000)], merge=True)                 # unaligned prefix (7,192)
    add("tp4", [(32768, 2112)], fp8_vs_bf16=True)          # 32K keys
    add("tp4", [(33768, 1024)], merge=True)                # DESIGN 5.4 (merge: 2 parts)
    add("tp4", [(33768, 512)], "gauss", merge=True)        # merge: 3 parts
    add("tp4", [(33768, 64)], merge=True)                  # merge: 3 parts
    add("tp4", [(34880, 2112)], "gauss")                   # prefix exactly 32,768
    add("tp4", [(100003, 1234)])                           # unaligned
    add("tp4", [(131072, 2112)])                           # 128K keys
    add("tp4", [(131072, 64)], "gauss")
    add("tp4", [(131072, 4096)])
    if not quick:
        add("tp4", [(262144, 2112)])                       # max_model_len
    add("tp4", [(8192, 2112)], k_scale=0.5, v_scale=2.0, merge=True)   # non-unit scales (the 5.3 fallback)
    # stock fall-through: short prompts and everything outside the route
    for n in (16, 63, 64, 256, 1000, 2112, 4095):
        add("tp4", [(n, n)], "gauss")
    add("tp4", [(8192, 63)])                               # Q below 64
    add("tp4", [(4095, 2112)])                             # keys below 4,096
    add("tp4", [(8192, 1)], "gauss")                       # decode
    add("tp4", [(8192, 4)], "gauss")                       # MTP k=3 verify shape
    add("tp4", [(8192, 16)], "gauss")
    add("tp4", [(8192, 2112), (5000, 1000)])               # two sequences
    # other per-card shapes: TP2 (16 q / 1 kv) and TP1 (32 q / 2 kv, GQA 16 as in the config)
    for shape in ("tp2", "tp1"):
        add(shape, [(4096, 4096)], "gauss")
        add(shape, [(8192, 1000)])
        add(shape, [(33768, 64)])
        add(shape, [(33768, 512)], "gauss")
        add(shape, [(131072, 2112)])
        for n in (64, 256, 2112):
            add(shape, [(n, n)], "gauss")
        add(shape, [(8192, 63)])
        add(shape, [(8192, 1)], "gauss")
        add(shape, [(8192, 4)], "gauss")
        add(shape, [(8192, 2112), (5000, 1000)])
    return S


def main():
    global STOP
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--stop-file", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--only", default=None, help="comma list of case indexes")
    a = ap.parse_args()
    STOP = a.stop_file
    torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)
    rj = open(os.path.join(a.out, "results.jsonl"), "a")

    def emit(kind, obj):
        rec = {"kind": kind, **obj}
        rj.write(json.dumps(rec) + "\n")
        rj.flush()
        return rec

    import vllm
    import vllm_xpu_kernels
    dev = torch.device("xpu", 0)
    env = {"torch": torch.__version__, "vllm": vllm.__version__,
           "vllm_xpu_kernels": getattr(vllm_xpu_kernels, "__version__", "?"),
           "device": torch.xpu.get_device_name(0), "device_count": torch.xpu.device_count(),
           "ZE_AFFINITY_MASK": os.environ.get("ZE_AFFINITY_MASK")}
    print("ENV", json.dumps(env), flush=True)
    emit("env", env)
    fa, stock_forward, vp, fp8kv = setup()
    ctx = {"fa": fa, "stock": stock_forward, "vp": vp, "fp8kv": fp8kv, "dev": dev}
    gen = torch.Generator().manual_seed(a.seed)
    allok = True
    summary = {"env": env, "cases": [], "fail": []}

    if a.model:
        nb = norm_bounds(a.model)
        emit("norm_bounds", {"bounds": nb})
        kb_ = max(v["bound"] for k, v in nb.items() if "k_norm" in k)
        qb_ = max(v["bound"] for k, v in nb.items() if "q_norm" in k)
        summary["k_bound_max"], summary["q_bound_max"] = kb_, qb_
        print(f"NORM_BOUNDS max|K| <= {kb_:.2f}  max|q| <= {qb_:.2f}  over {len(nb)} norms", flush=True)

    lc = lse_check(ctx, gen)
    emit("lse", {"checks": lc})
    lse_ok = all(v["lse_max_abs_err"] is not None and v["lse_max_abs_err"] < 1e-2 and v["out_rel"] < 2e-3
                 for k, v in lc.items() if not k.startswith("old_merge"))
    print("LSE", json.dumps(lc), "OK" if lse_ok else "FAIL", flush=True)
    allok &= lse_ok
    summary["lse_ok"] = lse_ok

    wp = write_path_check(ctx, gen)
    emit("write_path", {"checks": wp})
    wp_ok = all(v["ok"] for v in wp.values())
    print("WRITE", json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "specials_k"} for k, v in wp.items()}),
          "OK" if wp_ok else "FAIL", flush=True)
    print("WRITE_SPECIALS", json.dumps(wp["hk1"]["specials_k"]), flush=True)
    allok &= wp_ok
    summary["write_ok"] = wp_ok

    cases = plan(a.quick)
    only = set(int(x) for x in a.only.split(",")) if a.only else None
    for i, (shape, seqs, regime, kw) in enumerate(cases):
        if only is not None and i not in only:
            continue
        if stopped():
            print("STOPPED by stop file before case", i, flush=True)
            rj.close()
            sys.exit(3)
        t0 = time.time()
        try:
            r = run_case(ctx, shape, seqs, regime, gen, **kw)
        except Exception as e:  # noqa: BLE001
            import traceback
            r = {"shape": shape, "seqs": seqs, "regime": regime, "ok": False,
                 "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-3000:]}
        r["index"] = i
        r["wall_s"] = round(time.time() - t0, 1)
        emit("case", r)
        allok &= r["ok"]
        summary["cases"].append(r)
        if not r["ok"]:
            summary["fail"].append(i)
        if "error" in r:
            print(f"CASE {i:2d} {shape} {seqs} {regime}: ERROR {r['error']}", flush=True)
            continue
        f = lambda e: "-" if e is None else (f"{e['rel']:.2e}" if e["nonfinite"] == 0 else f"NONFINITE{e['nonfinite']}")  # noqa: E731
        t = lambda x: "-" if x is None else f"{x:.2f}"   # noqa: E731
        mg = r.get("merge")
        mgs = "" if not mg else (f" merge={f(mg['err'])} t_merge={t(mg['t_ms'])} parts={mg['parts']}"
                                 f" merge_fb_calls={mg['fallback_calls']}")
        print(f"CASE {i:2d} {shape} {str(seqs):26s} {regime:5s} routed={r['routed_calls']}"
              f" floor={r['bf16_floor']:.2e} stock={f(r['stock'])} patched={f(r['patched'])}"
              f" eq={int(r['bit_equal_patched_stock'])} t_stock={t(r['t_stock_ms'])} t_patch={t(r['t_patched_ms'])}"
              f"{mgs} stock_fb_calls={r['stock_fallback_calls']} patch_fb_calls={r['patched_fallback_calls']}"
              f" stock_err={(r['stock_error'] or '-')[:60]}"
              f" {'OK' if r['ok'] else 'FAIL'} ({r['wall_s']}s)", flush=True)
    summary["routed_total"] = vp.FP8KV_STATS["routed"]
    summary["fallback_calls_total"] = FALLBACK["n"]
    summary["missing_kernel_configs"] = [c.split("\n\n")[2].strip() if c.count("\n\n") > 2 else c[:120]
                                         for c in missing_kernel_configs()]
    print("MISSING_KERNEL_CONFIGS", json.dumps(summary["missing_kernel_configs"]), flush=True)
    summary["pass"] = bool(allok)
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1, default=str)
    rj.close()
    print("T15A_PASS" if allok else "T15A_FAIL", flush=True)
    sys.exit(0 if allok else 1)


if __name__ == "__main__":
    main()
