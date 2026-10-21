"""M2a, part 2: the slice plan on REAL tensors of the Ornith-1.5-122B EXL3 checkpoint (EXL3_MODEL_DIR, read-only;
only the needed tensors are read, by byte range). For every dense matrix kind, the MTP layer and routed experts
(MoE TP narrowing), at TP 1, 2, 4 and every rank:
  1. the per-rank slices are contiguous, have the planned shapes, and scatter back to the original tensors
     exactly (torch.equal), with every element covered exactly `replicas` times (2 for k/v at TP4, else 1);
  2. the per-rank reconstructed weight equals the matching block of the TP1 reconstructed weight, bit for bit
     (exact float64 reference, exact_ref.py, on trellis_core.reference's fp16 decode);
  3. row splits: the per-rank partial products sum to the TP1 product (float64), and with each partial rounded to
     bf16 (what HadOutKernel writes) and summed in fp32 then rounded once, the error is within the elementwise
     bound sum_r u|y_r| + u|sum| (u = 2^-8) and its rms is <= 1.6x the TP1 bf16 floor (DESIGN 2.3: one extra bf16
     rounding per partial, ~1.4x the floor).
"""
from __future__ import annotations

import pytest
import torch

from exact_ref import weight_exact
from exl3xpu import tp
from results import record
from trellis_core.reference import CB_MUL1, weight_orig

P = "model.language_model.layers."
MUL1 = -2082680531
REAL = [
    P + "3.self_attn.q_proj", P + "3.self_attn.k_proj", P + "3.self_attn.v_proj", P + "3.self_attn.o_proj",
    P + "47.self_attn.q_proj", P + "47.self_attn.k_proj", P + "47.self_attn.v_proj", P + "47.self_attn.o_proj",
    P + "0.linear_attn.in_proj_qkv", P + "0.linear_attn.in_proj_z", P + "0.linear_attn.out_proj",
    P + "46.linear_attn.in_proj_qkv", P + "46.linear_attn.in_proj_z", P + "46.linear_attn.out_proj",
    P + "0.mlp.shared_expert.gate_proj", P + "0.mlp.shared_expert.up_proj", P + "0.mlp.shared_expert.down_proj",
    P + "47.mlp.shared_expert.gate_proj", P + "47.mlp.shared_expert.up_proj", P + "47.mlp.shared_expert.down_proj",
    "lm_head",
    "mtp.fc", "mtp.layers.0.self_attn.q_proj", "mtp.layers.0.self_attn.k_proj", "mtp.layers.0.self_attn.v_proj",
    "mtp.layers.0.self_attn.o_proj", "mtp.layers.0.mlp.shared_expert.gate_proj",
    "mtp.layers.0.mlp.shared_expert.up_proj", "mtp.layers.0.mlp.shared_expert.down_proj",
    P + "0.mlp.experts.7.gate_proj", P + "0.mlp.experts.7.up_proj", P + "0.mlp.experts.7.down_proj",
    P + "3.mlp.experts.72.gate_proj", "mtp.layers.0.mlp.experts.0.down_proj",
]
U_BF16 = 2.0 ** -8


def _short(m: str) -> str:
    return m.replace(P, "L")


def _block(W: torch.Tensor, p: tp.SlicePlan) -> torch.Tensor:
    rows = W.narrow(0, *p.rows)
    return torch.cat([rows.narrow(1, a, s) for a, s in p.cols], dim=1) if len(p.cols) > 1 else rows.narrow(1, *p.cols[0])


class _Rebuild:
    """Scatter per-rank slices back into checkpoint coordinates, checking that replicas agree."""

    def __init__(self, t):
        self.t = t
        self.buf = {k: torch.zeros_like(v) for k, v in t.items() if k != "mul1"}
        kt, nt = t["trellis"].shape[:2]
        self.cnt_rows = torch.zeros(kt, dtype=torch.int64)     # trellis tile rows / suh blocks
        self.cnt_cols = torch.zeros(nt, dtype=torch.int64)

    def _put(self, part, dim, idx, src):
        buf = self.buf[part]
        cnt = self.cnt_cols if (part == "svh" or (part == "trellis" and dim == 1)) else self.cnt_rows
        tile_idx = idx // 16 if part != "trellis" else idx
        seen = cnt.index_select(0, tile_idx) > 0
        if bool(seen.any()):
            sel = seen.nonzero().flatten()
            assert torch.equal(buf.index_select(dim, idx[sel]), src.index_select(dim, sel)), f"replica mismatch {part}"
        buf.index_copy_(dim, idx, src)

    def add(self, p: tp.SlicePlan, s: dict):
        ci, ri = p.col_index(), p.row_index()
        if p.splits_rows:
            assert torch.equal(s["svh"], self.t["svh"])        # svh whole on every rank
            ti = ri[::16] // 16
            self._put("trellis", 0, ti, s["trellis"])
            self._put("suh", 0, ri, s["suh"])
            self.buf["svh"] = self.t["svh"].clone()
            self.cnt_rows.index_add_(0, ti, torch.ones_like(ti))
            self.cnt_cols += 1
        else:
            assert torch.equal(s["suh"], self.t["suh"])        # suh whole on every rank
            tj = ci[::16] // 16
            self._put("trellis", 1, tj, s["trellis"])
            self._put("svh", 0, ci, s["svh"])
            self.buf["suh"] = self.t["suh"].clone()
            self.cnt_cols.index_add_(0, tj, torch.ones_like(tj))
            self.cnt_rows += 1

    def check(self, replicas: int, tp_size: int):
        split = self.cnt_rows if not torch.all(self.cnt_rows == tp_size) else self.cnt_cols
        assert int(split.min()) == replicas and int(split.max()) == replicas, (int(split.min()), int(split.max()))
        for part, v in self.buf.items():
            assert torch.equal(v, self.t[part]), f"{part} does not reassemble"


@pytest.fixture(scope="module")
def cfg(ckpt):
    c = ckpt.config()
    assert c["layer_types"][3] == "full_attention" and c["layer_types"][47] == "full_attention"
    assert c["layer_types"][0] == "linear_attention" and c["layer_types"][46] == "linear_attention"
    return c


@pytest.mark.parametrize("mod", REAL, ids=[_short(m) for m in REAL])
def test_real_slices(ckpt, cfg, mod):
    t = ckpt.exl3(mod)
    kt, nt, words = t["trellis"].shape
    k, n, K = kt * 16, nt * 16, words // 16
    assert t["trellis"].dtype == torch.int16 and t["suh"].shape == (k,) and t["svh"].shape == (n,)
    assert int(t["mul1"]) == MUL1
    W = weight_exact(t["trellis"], t["suh"], t["svh"])
    if k * n <= 3072 * 1024:     # orientation cross-check against trellis_core.reference's fp32 path
        Wr = weight_orig(t["trellis"], t["suh"], t["svh"], K, CB_MUL1).double()
        scale = float(W.abs().max()) or 1.0
        assert float((Wr - W).abs().max()) <= 1e-5 * scale
    kind, _ = tp.classify(mod)
    is_row = kind == "row" or (kind == "expert" and mod.endswith("down_proj"))
    g = torch.Generator().manual_seed(1234)
    x = torch.randn(64, k, generator=g).to(torch.bfloat16) if is_row else None
    y = x.double() @ W if is_row else None
    summary = []
    for tp_size in (1, 2, 4):
        rb = _Rebuild(t)
        parts = []
        for r in range(tp_size):
            p = tp.qwen35_plan(mod, k, n, cfg, tp_rank=r, tp_size=tp_size)
            s = p.narrow_all(t)
            for part, v in s.items():
                assert v.is_contiguous(), part
                assert tuple(v.shape) == p.local_shapes(K)[part], (part, tuple(v.shape))
            rb.add(p, s)
            Wl = weight_exact(s["trellis"], s["suh"], s["svh"])
            assert torch.equal(Wl, _block(W, p)), f"rank {r}/{tp_size}: per-rank weight != TP1 block"
            if is_row:
                parts.append(x.double().narrow(1, *p.rows) @ Wl)
            del Wl, s
        replicas = 2 if (kind == "qkv" and tp.classify(mod)[1] in ("k", "v") and tp_size == 4) else 1
        rb.check(replicas, tp_size)
        if is_row and tp_size > 1:
            summary.append(_row_numerics(mod, tp_size, y, parts))
    record(f"real {_short(mod)}", f"K={K} {k}x{n} slices exact at TP1/2/4" + (" | " + " | ".join(summary)
                                                                               if summary else ""))


def _rms(a: torch.Tensor) -> float:
    return float(a.pow(2).mean().sqrt())


def _row_numerics(mod, tp_size, y, parts) -> str:
    S = torch.stack(parts).sum(0)
    exact_rel = float((S - y).abs().max() / y.abs().max())
    assert exact_rel <= 1e-12, exact_rel
    ref = _rms(y)
    f1 = _rms(y.to(torch.bfloat16).double() - y) / ref                       # TP1: one bf16 rounding of y
    pb = [p_.to(torch.bfloat16) for p_ in parts]                              # HadOutKernel bf16 partials
    s32 = torch.stack([p_.float() for p_ in pb]).sum(0)                       # all-reduce, fp32 accumulate
    out = s32.to(torch.bfloat16).double()
    err = (out - y).abs()
    bound = U_BF16 * torch.stack([p_.abs() for p_ in parts]).sum(0) + U_BF16 * s32.double().abs() \
        + tp_size * 2.0 ** -24 * torch.stack([p_.double().abs() for p_ in pb]).sum(0)
    assert bool((err <= bound * (1 + 1e-9)).all()), float((err - bound).max())
    fa = _rms(out - y) / ref
    assert fa <= 1.6 * f1, (fa, f1)
    ring = pb[0]
    for p_ in pb[1:]:                                                         # ring all-reduce in bf16
        ring = (ring.float() + p_.float()).to(torch.bfloat16)
    fb = _rms(ring.double() - y) / ref
    vs_tp1 = _rms(out - y.to(torch.bfloat16).double()) / ref
    return (f"TP{tp_size}: sum exact {exact_rel:.1e}, floor {f1:.2e}, fp32-AR {fa:.2e} ({fa / f1:.2f}x), "
            f"bf16-ring {fb:.2e} ({fb / f1:.2f}x), vs TP1-out {vs_tp1:.2e}")
