"""M2b (DESIGN rev 2, section 7): the K=4 grouped MoE kernel and the vLLM MoE method on ONE B70, on Ornith-mini
routed experts. Needs a card; the CPU workflow only runs its --cpu-dryrun (plumbing) mode.

  python tests/xpu/m2_moe_parity.py --model <ornith-mini dir> --layers 0,3 --out <run dir>/m2b.json
        [--checks i,ii,iii,iiib,iiic,iv,v,vi,vii,viii,ix,x,xi,det] [--dump-dir <EXL3_DUMP_DIR of an M2c run>]
  python tests/xpu/m2_moe_parity.py --model <ornith-mini dir> --cpu-dryrun --out <file>      # no device

Run inside the serving image with the lane environment (prod driver/L0 env, the plugin's dist on PYTHONPATH or this
repo's xpu/ + core/src), one card visible. EXL3_MOE_LIB / EXL3_LIB select the built _moe.so / _C.so.

Checks (DESIGN section 7, M2b table; x is bf16 with sigma 0.5 unless noted, routing from the layer's real router):
  i     SGLang compute path: ExpertStore (host USM, zero-copy, no slots) -> moe_forward
  ii    the vLLM method: one RoutedExperts layer built by FusedMoEFactory under set_current_vllm_config (TP1), loaded
        through its load_weights, process_weights_after_loading (moe_info, plan check, variant warmup), routing from
        vLLM's router with an is_padding mask; pass: i == ii bit for bit (loader / placement parity only)
  iii   i and ii vs the fp32 oracle (trellis_core.reference.weight_orig per selected expert), M grid of section 7:
        rms_rel <= 2 x floor_bf16 + 1e-3 (floor = rms_rel of the oracle rounded to the output dtype)
  iiib  every compiled K=4 (MR, P, RB) forced with moe_set_mr / moe_set_splits, I in {1024, 512, 256} (experts
        sliced for MoE-TP rank 0), M in {1, 4, 5, 64, 65, 1024}: the iii criterion on every variant
  iiic  MoE-TP sum: sum over ranks of moe_forward(x, ids, w, ptrs_r, I/tp, 4, 256) vs the TP1 oracle, tp in {2, 4}:
        <= 2 x floor_bf16 + 1e-3 (bf16 x), <= 1e-3 (fp16 x)
  iv    second oracle: per-expert loop over the dense exl3xpu_C.linear on zero-copy views of the blob (gate|up as one
        2-group linear, down); oracle and kernel each within 2 x floor + 1e-3 of the fp32 oracle / of each other
  v     routing only to layer 3's dead experts (72, 166, 180, 252): output exactly 0
  vi    EP exact: per EP4 rank r, moe_forward(x, emap_r[ids], w, ptrs[64r:+64], I, 4, 64) == moe_forward(x, ids,
        w * (ids // 64 == r), ptrs, I, 4, 256) (torch.equal); sum of partials vs the oracle as iii (bf16), <= 1e-3 (fp16)
  vii   graphs: capture [the method's apply on a TP1 layer, on an EP4-rank-1 layer, a dense EXL3 linear] at
        M in {1,2,4,8,12,16,24,32,64} with ALL ids -1 (as vLLM captures), replay with real x / ids (EP ids with -1,
        pair mode included): replay == eager bit for bit
  viii  microbench: us per MoE layer per forced (MR, P, RB) for EP4 (I=1024, real routing, ~3/4 remote) and TP4
        (I=256), M in {1,2,4,8,16,32,64,256,1024}, moe_info's per-kernel limits; best plan per M -> splits.json
  ix    padding: rows with id -1 / weight 0 and all-padding batches, TP1 and an EP map, M in {1,4,32,1024}: exactly 0
  x     the method's chunked apply (M in {1025, 2048, 4096}) vs one unchunked moe_forward: bit-equal (else iii)
  xi    replay the MoE inputs an M2c run dumped (EXL3_DUMP_LAYER / EXL3_DUMP_DIR) through the iii oracle; record the
        oracle's max |g|, |u|, |x * suh| per layer: finite and < 32768 (fp16 intermediates as exllamav3)
  det   dense determinism probe: TP4 shapes 2048->3072, 3072->{4096, 5120, 62080} at M in {129, 256, 512}, 8 calls
        each (INFO: identity tests stay <= 128 tokens)

--cpu-dryrun: no device. moe_forward is replaced by tests/cpu/moe_ref.BlobKernel (float64 interpreter of the kernel
contract over the same packed blobs and pointer tables), the dense linear by trellis_core.reference.linear_forward;
the device-only checks (i, vii, viii, det) are skipped. It proves the harness's slicing, packing, routing, oracle
and bookkeeping, not the kernels. Exit status: 0 iff every check that ran passed.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "core" / "src", ROOT / "xpu", ROOT / "tests" / "cpu"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True, help="Ornith-mini directory (read-only)")
ap.add_argument("--layers", default="0,3")
ap.add_argument("--checks", default="i,ii,iii,iiib,iiic,iv,v,vi,vii,viii,ix,x,xi,det")
ap.add_argument("--out", required=True, help="JSON results file; splits.json goes next to it")
ap.add_argument("--dump-dir", default="", help="xi: the EXL3_DUMP_DIR of an M2c run")
ap.add_argument("--bench-iters", type=int, default=50)
ap.add_argument("--cpu-dryrun", action="store_true")
ap.add_argument("--m-grid", default="1,2,3,4,5,8,16,32,64,65,128,129,255,256,257,1024,2048,4096")
ap.add_argument("--expert-subset", type=int, default=-1,
                help="route only to N experts spread over all EP ranks (incl. the dead ones); default all, 24 in dry runs")
args = ap.parse_args()

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

DRY = args.cpu_dryrun
if DRY:
    import vllm.platforms as _P
    from vllm.platforms.xpu import XPUPlatform
    XPUPlatform.current_device = classmethod(lambda cls: torch.device("cpu"))
    _P._current_platform = XPUPlatform()

from st_range import Checkpoint  # noqa: E402
from trellis_core import reference as ref  # noqa: E402
from exl3xpu import loadcheck, moe_vllm, tp  # noqa: E402
from exl3xpu.moe_offload import ExpertStore, pack_expert, s64, ops as moe_ops  # noqa: E402

DEV = torch.device("cpu") if DRY else torch.device("xpu", 0)
H, K, E, TOPK, IFULL = 3072, 4, 256, 8, 1024
DEAD = {3: (72, 166, 180, 252)}
ROLES = ("gate_proj", "up_proj", "down_proj")
ckpt = Checkpoint(args.model)
cfg_text = ckpt.config()
X = moe_ops()
CHECKS = set(args.checks.split(","))
RESULTS: dict[str, list] = {}
OUTDIR = Path(args.out).resolve().parent
OUTDIR.mkdir(parents=True, exist_ok=True)


def rec(check: str, **kw) -> dict:
    RESULTS.setdefault(check, []).append(kw)
    print(check, {k: v for k, v in kw.items() if not isinstance(v, (list, dict)) or len(str(v)) < 200}, flush=True)
    return kw


def sync():
    if not DRY:
        torch.xpu.synchronize()


# ------------------------------------------------------------------------------------------------ kernel shim
if DRY:
    from moe_ref import BlobKernel
    _BK = BlobKernel()

    def KFN(x, ids, w, ptrs, I, K_, E_):
        return _BK(x, ids, w, ptrs, I, K_, E_).to(x.dtype)

    def register_arena(t):
        _BK.add_arena(t)
else:
    KFN = X.moe_forward

    def register_arena(t):
        pass


# --------------------------------------------------------------------------------------------------- weights
def expert_tensors(L: int, e: int) -> dict:
    base = f"model.language_model.layers.{L}.mlp.experts.{e}"
    return {r: {p: ckpt.get(f"{base}.{r}.{p}") for p in ("trellis", "suh", "svh")} for r in ROLES}


def narrow_expert(t: dict, tp_size: int, rank: int, L: int, e: int) -> dict:
    out = {}
    for r in ROLES:
        key = f"layers.{L}.experts.{e}.{r}"
        plan = tp.plan_row(key, IFULL, H, rank, tp_size) if r == "down_proj" else tp.plan_col(key, H, IFULL, rank,
                                                                                               tp_size)
        out[r] = plan.narrow_all(t[r])
    return out


class Arena:
    """All 256 experts of layer L (MoE-TP rank slice when tp_size > 1) packed into one device arena."""

    def __init__(self, L: int, tp_size: int = 1, rank: int = 0):
        self.L, self.tp_size, self.rank = L, tp_size, rank
        self.I = IFULL // tp_size
        self.blob = int(X.blob_bytes(H, self.I, K))
        assert self.blob == moe_vllm.blob_bytes(H, self.I, K)
        self.data = torch.empty((E, self.blob), dtype=torch.uint8, device=DEV)
        stage = torch.empty(self.blob, dtype=torch.uint8, pin_memory=not DRY)
        for e in range(E):
            p = narrow_expert(expert_tensors(L, e), tp_size, rank, L, e)
            pack_expert(p["gate_proj"], p["up_proj"], p["down_proj"], K, out=stage)
            self.data[e].copy_(stage)
        register_arena(self.data)
        self.ptrs = (s64(self.data.data_ptr()) + torch.arange(E, dtype=torch.int64) * self.blob).to(DEV)


_ARENAS: dict = {}


def arena(L, tp_size=1, rank=0) -> Arena:
    k = (L, tp_size, rank)
    if k not in _ARENAS:
        t0 = time.time()
        _ARENAS[k] = Arena(L, tp_size, rank)
        print(f"arena L{L} tp{tp_size} r{rank}: {E} x {_ARENAS[k].blob} B in {time.time() - t0:.1f}s", flush=True)
    return _ARENAS[k]


# ------------------------------------------------------------------------------------------- routing, oracle
def router_weight(L: int) -> torch.Tensor:
    return ckpt.get(f"model.language_model.layers.{L}.mlp.gate.weight").float()


SUBSET = args.expert_subset if args.expert_subset >= 0 else (24 if DRY else 0)


def _subset_mask(L: int) -> torch.Tensor | None:
    if not SUBSET:
        return None
    keep = set(range(0, E, max(1, E // SUBSET))) | set(DEAD.get(L, ())) | {E // 4 - 1, E // 4, E - 1}
    m = torch.full((E,), float("-inf"))
    m[sorted(keep)] = 0
    return m


def pad_rows(M: int) -> torch.Tensor:
    """vLLM padding rows of a test batch: every 7th row from row 2 (so M = 1, 2 are all real rows and every larger
    prefix of a batch has some), written as id -1 / weight 0 like XPU topk_softmax does for is_padding rows."""
    return torch.arange(M) % 7 == 2


def make_batch(L: int, M: int, seed: int, dtype=torch.bfloat16, pad: bool = False):
    g = torch.Generator().manual_seed(seed)
    x = (torch.randn((M, H), generator=g) * 0.5).to(dtype)
    logits = x.float() @ router_weight(L).t()
    sm = _subset_mask(L)
    if sm is not None:
        logits = logits + sm
    w, ids = torch.topk(torch.softmax(logits, -1), TOPK, dim=-1)
    if cfg_text.get("norm_topk_prob", True):
        w = w / w.sum(-1, keepdim=True)
    ids, w = ids.to(torch.int32), w.float()
    if pad:
        m = pad_rows(M)
        ids[m] = -1
        w[m] = 0
    return x.to(DEV), ids.to(DEV), w.to(DEV)


def oracle(x, ids, w, L, tp_size=1, rank=0, stats=None) -> torch.Tensor:
    """fp32 sum_k w * down(silu(x gate) * (x up)) over GLOBAL ids (-1 skipped), weights decoded per expert on the fly
    (all experts of a layer in fp32 would be ~9 GiB); MoE-TP rank slices when tp_size > 1."""
    out = torch.zeros((x.shape[0], H), dtype=torch.float32, device=DEV)
    xf = x.float()
    flat = ids.flatten().long().cpu()
    wc = w.cpu()
    for e in sorted(set(flat[flat >= 0].tolist())):
        p = narrow_expert(expert_tensors(L, e), tp_size, rank, L, e)
        W = {r: ref.weight_orig(p[r]["trellis"].to(DEV), p[r]["suh"].to(DEV), p[r]["svh"].to(DEV), K, 2) for r in ROLES}
        pos = (flat == e).nonzero().flatten()
        rows, ks = pos // ids.shape[1], pos % ids.shape[1]
        xs = xf[rows.to(DEV)]
        gt, up = xs @ W["gate_proj"], xs @ W["up_proj"]
        if stats is not None:
            stats["g"] = max(stats.get("g", 0.0), gt.abs().max().item())
            stats["u"] = max(stats.get("u", 0.0), up.abs().max().item())
            stats["xsuh"] = max(stats.get("xsuh", 0.0), (xs * p["gate_proj"]["suh"].to(DEV).float()).abs().max().item())
        d = (F.silu(gt) * up) @ W["down_proj"]
        out.index_add_(0, rows.to(DEV), d * wc[rows, ks].to(DEV).unsqueeze(1))
    return out


def rms_rel(y, r) -> float:
    den = r.float().pow(2).mean().sqrt().item()
    num = (y.float() - r.float()).pow(2).mean().sqrt().item()
    return num / den if den else num


def floor_of(r, dtype) -> float:
    return rms_rel(r.to(dtype).float(), r)


def crit(y, r, dtype=torch.bfloat16) -> tuple[bool, float, float]:
    e, f = rms_rel(y, r), floor_of(r, dtype)
    return e <= 2 * f + 1e-3, e, f


def emap_for(ep: int, r: int) -> torch.Tensor:
    m = torch.full((E,), -1, dtype=torch.int32)
    n = E // ep
    m[r * n:(r + 1) * n] = torch.arange(n, dtype=torch.int32)
    return m.to(DEV)


def fake_layer(a: Arena, ep: int = 1, r: int = 0, dtype=torch.bfloat16):
    """What Exl3MoEMethod.process_weights_after_loading leaves on a layer, for moe_vllm.moe_apply."""
    import types
    if ep == 1:
        return types.SimpleNamespace(exl3_kernel=KFN, exl3_ptrs=a.ptrs, exl3_I=a.I, exl3_K=K, exl3_E=E, exl3_emap=None,
                                     exl3_layer_idx=str(a.L))
    n = E // ep
    return types.SimpleNamespace(exl3_kernel=KFN, exl3_ptrs=a.ptrs[r * n:(r + 1) * n], exl3_I=a.I, exl3_K=K,
                                 exl3_E=n, exl3_emap=emap_for(ep, r), exl3_layer_idx=str(a.L))


_VARIANTS: dict = {}


def variants() -> tuple[list, list]:
    """Compiled K=4 gate|up (MR, P) and down (MR, P) variants, from moe_info (called once: on a device it JIT-builds
    every kernel and records the per-kernel work-group limits moe_plan honours)."""
    if not _VARIANTS:
        info = X.moe_info(-1 if DRY else (DEV.index or 0))
        F_, n = info[0], info[1]
        rows = [info[2 + i * F_: 2 + (i + 1) * F_] for i in range(n)]
        _VARIANTS["rows"] = rows
        _VARIANTS["gu"] = sorted({(r[2], r[3]) for r in rows if r[0] == 0 and r[1] == K})
        _VARIANTS["dn"] = sorted({(r[2], r[3]) for r in rows if r[0] == 1 and r[1] == K})
    return _VARIANTS["gu"], _VARIANTS["dn"]


def plan_list(I: int, M: int) -> list[tuple]:
    """Every forced (MR, gu_P, dn_P) the planner accepts for (H, I, 4, M), as [MR, gu, dn, RB] plans."""
    gu, dn = variants()
    plans = set()
    for mr in sorted({m for m, _ in gu}):
        for _, gp in [v for v in gu if v[0] == mr]:
            for _, dp in [v for v in dn if v[0] == mr]:
                X.moe_set_mr(mr)
                X.moe_set_splits(gp, dp)
                p = tuple(int(v) for v in X.moe_plan(H, I, K, M))
                if any(p) and p[:3] == (mr, gp, dp) or (any(p) and M <= 4 and p[1:3] == (gp, dp)):
                    plans.add(p)
    X.moe_set_mr(0)
    X.moe_set_splits(0, 0)
    return sorted(plans)


def force(p):
    X.moe_set_mr(p[0])
    X.moe_set_splits(p[1], p[2])


def unforce():
    X.moe_set_mr(0)
    X.moe_set_splits(0, 0)


# -------------------------------------------------------------------------------------------- the vLLM layer
def vllm_layer(L: int):
    """One RoutedExperts layer as vLLM builds it (FusedMoEFactory, TP1, the plugin's Exl3MoEMethod)."""
    from exl3xpu import vllm_plugin
    import vllm.distributed.parallel_state as ps
    from vllm.config import set_current_vllm_config
    from vllm.engine.arg_utils import EngineArgs
    from vllm.model_executor.layers.fused_moe import FusedMoEFactory
    from vllm.utils.torch_utils import set_default_torch_dtype
    vllm_plugin.register()

    class G:
        def __init__(s):
            s.rank_in_group, s.world_size, s.ranks, s.rank, s.local_rank = 0, 1, [0], 0, 0
            s.is_first_rank = s.is_last_rank = True
            s.device_group = s.cpu_group = type("PG", (), {"size": lambda self: 1, "rank": lambda self: 0})()
            s.unique_name = "m2b"
    for n in ("_TP", "_PP", "_DP", "_EP", "_PCP", "_DCP", "_EPLB", "_ETP", "_WORLD", "_INNER_DP_WORLD"):
        setattr(ps, n, G())
    vc = EngineArgs(model=args.model, quantization="exl3", dtype="bfloat16", tensor_parallel_size=1,
                    max_model_len=4096, max_num_batched_tokens=4096, enforce_eager=True, skip_tokenizer_init=True,
                    language_model_only=True).create_engine_config()
    with set_current_vllm_config(vc), set_default_torch_dtype(torch.bfloat16), DEV:
        runner = FusedMoEFactory(num_experts=E, top_k=TOPK, hidden_size=H, intermediate_size=IFULL,
                                 renormalize=cfg_text.get("norm_topk_prob", True), quant_config=vc.quant_config,
                                 prefix=f"language_model.model.layers.{L}.mlp.experts")
    re_ = runner.routed_experts

    def weights():
        for e in range(E):
            for r in ROLES:
                for p in ("mul1", "suh", "svh", "trellis"):
                    yield f"{e}.{r}.{p}", ckpt.get(f"model.language_model.layers.{L}.mlp.experts.{e}.{r}.{p}")
    loaded = list(runner.load_weights(weights()))
    # a lone layer is its own load batch: its expected EXL3XPU_LOADED counts are its own 256 x 12 tensors
    loadcheck.expected_counts = lambda b, role, par: {"dense": 0, "experts": E * 12}
    with set_current_vllm_config(vc):
        re_.quant_method.process_weights_after_loading(re_)
    if DRY:
        register_arena(re_.exl3_arena)
        re_.exl3_kernel = KFN
    assert len(loaded) == E * 12 and loadcheck.CLOSED[-1]["counts"]["experts"] == E * 12
    return runner, re_


def vllm_route(runner, x, L: int):
    """vLLM's router on the real gate weights, with vLLM's padding mask (VLLM_MOE_SKIP_PADDING) for the first rows."""
    from vllm.model_executor.layers.fused_moe.router import fused_topk_router as ftr
    logits = (x.float() @ router_weight(L).to(DEV).t()).to(x.dtype)
    pad = pad_rows(x.shape[0]).to(DEV)
    orig = ftr._get_padding_mask
    ftr._get_padding_mask = lambda n: pad[:n]
    try:
        w, ids = runner.router.select_experts(hidden_states=x, router_logits=logits)
    finally:
        ftr._get_padding_mask = orig
    return w, ids


# ---------------------------------------------------------------------------------------------------- checks
def check_i_ii_iii(L):
    runner = re_ = store = None
    if "ii" in CHECKS or "iii" in CHECKS:
        runner, re_ = vllm_layer(L)
    if "i" in CHECKS and not DRY:
        store = ExpertStore(H, IFULL, K, E, 0, DEV, max_layers=1)
        h = store.add_layer(f"L{L}")
        for e in range(E):
            t = expert_tensors(L, e)
            pack_expert(t["gate_proj"], t["up_proj"], t["down_proj"], K, out=h[e])
    grid = [int(v) for v in args.m_grid.split(",")]
    Mmax = max(grid)
    x, ids_t, w_t = make_batch(L, Mmax, 10 + L, pad=True)
    if re_ is not None and not DRY:
        w, ids = vllm_route(runner, x, L)
        rec("ii_router", layer=L, ok=torch.equal(ids.cpu() < 0, pad_rows(Mmax).unsqueeze(1).expand(-1, TOPK))
            and bool((w.cpu()[pad_rows(Mmax)] == 0).all()), what="vLLM router writes -1 / 0 exactly on padding rows")
    else:
        ids, w = ids_t, w_t
    sync()
    r_full = oracle(x, ids, w, L)
    for M in grid:
        xm, im, wm = x[:M].contiguous(), ids[:M].contiguous(), w[:M].contiguous()
        out = {}
        if re_ is not None:
            out["ii"] = re_.quant_method.apply(re_, xm, wm, im)
        if store is not None:
            out["i"] = store.forward(f"L{L}", xm, im, wm)
        sync()
        if "i" in out and "ii" in out:
            rec("ii", layer=L, M=M, ok=torch.equal(out["i"], out["ii"]), what="i == ii bit for bit")
        if "iii" in CHECKS:
            for k_, y in out.items():
                ok, e, f = crit(y, r_full[:M])
                pad0 = bool((y[pad_rows(M).to(DEV)] == 0).all())
                rec("iii", layer=L, M=M, path=k_, ok=ok and pad0, rms_rel=e, floor=f, padding_rows_zero=pad0)
    return runner, re_


def check_iiib(L):
    Ms = (1, 4, 5, 64, 65, 1024)
    x, ids, w = make_batch(L, max(Ms), 20 + L, pad=True)
    for tpn in (1, 2, 4):
        a = arena(L, tpn, 0)
        r = oracle(x, ids, w, L, tpn, 0)
        for M in Ms:
            for p in plan_list(a.I, M):
                force(p)
                try:
                    y = KFN(x[:M].contiguous(), ids[:M].contiguous(), w[:M].contiguous(), a.ptrs, a.I, K, E)
                    sync()
                finally:
                    unforce()
                ok, e, f = crit(y, r[:M])
                rec("iiib", layer=L, I=a.I, M=M, plan=list(p), ok=ok, rms_rel=e, floor=f)


def check_iiic(L):
    Ms = (1, 5, 64, 256, 1024)
    for dtype, tol in ((torch.bfloat16, None), (torch.float16, 1e-3)):
        x, ids, w = make_batch(L, max(Ms), 30 + L, dtype=dtype, pad=True)
        r = oracle(x, ids, w, L)
        for tpn in (2, 4):
            for M in Ms:
                parts = [KFN(x[:M].contiguous(), ids[:M].contiguous(), w[:M].contiguous(), arena(L, tpn, rk).ptrs,
                             IFULL // tpn, K, E) for rk in range(tpn)]
                sync()
                s = sum(p.float() for p in parts)
                e, f = rms_rel(s, r[:M]), floor_of(r[:M], dtype)
                ok = e <= (2 * f + 1e-3 if tol is None else tol)
                rec("iiic", layer=L, dtype=str(dtype), tp=tpn, M=M, ok=ok, rms_rel=e, floor=f)


def _blob_views(blob: torch.Tensor, I: int):
    GU = H // 16 * (2 * I) // 16 * 128
    DN = I // 16 * H // 16 * 128
    o = GU + DN
    f16 = blob[o:].view(torch.float16)
    v = {"gu_tr": blob[:GU].view(torch.int16).view(H // 16, 2 * I // 16, 64),
         "dn_tr": blob[GU:GU + DN].view(torch.int16).view(I // 16, H // 16, 64),
         "suh2": f16[:2 * H].view(2, H), "svh2": f16[2 * H:2 * H + 2 * I],
         "suh_d": f16[2 * H + 2 * I:2 * H + 3 * I].view(1, I), "svh_d": f16[2 * H + 3 * I:3 * H + 3 * I]}
    return v


def _dense(x, tr, suh, svh, bounds):
    sonb = torch.empty(svh.shape[0] // 128, dtype=torch.int32)
    for gi in range(len(bounds) - 1):
        sonb[bounds[gi] // 128:bounds[gi + 1] // 128] = gi
    if DRY:
        ys = [ref.linear_forward(x.float(), tr[:, b0 // 16:b1 // 16].contiguous(), suh[gi], svh[b0:b1], K, 2)
              for gi, (b0, b1) in enumerate(zip(bounds, bounds[1:]))]
        return torch.cat(ys, 1).to(x.dtype)
    from exl3xpu import ops
    return torch.ops.exl3xpu_C.linear(x, tr, suh, svh, sonb.to(DEV), list(bounds), K, 2, ops.SMALL_M_MAX,
                                      ops.RECON_SLICE_N)


def check_iv(L):
    a = arena(L)
    for M in (1, 64, 256):
        x, ids, w = make_batch(L, M, 40 + L, pad=True)
        y_iv = torch.zeros((M, H), dtype=torch.float32, device=DEV)
        flat = ids.flatten().long().cpu()
        for e in sorted(set(flat[flat >= 0].tolist())):
            v = _blob_views(a.data[e], a.I)
            pos = (flat == e).nonzero().flatten()
            rows, ks = pos // TOPK, pos % TOPK
            xs = x[rows.to(DEV)].contiguous()
            gu = _dense(xs, v["gu_tr"], v["suh2"], v["svh2"], [0, a.I, 2 * a.I]).float()
            act = (F.silu(gu[:, :a.I]) * gu[:, a.I:]).half()
            d = _dense(act, v["dn_tr"], v["suh_d"], v["svh_d"], [0, H]).float()
            y_iv.index_add_(0, rows.to(DEV), d * w.cpu()[rows, ks].to(DEV).unsqueeze(1))
        y_k = KFN(x, ids, w, a.ptrs, a.I, K, E)
        sync()
        r = oracle(x, ids, w, L)
        ok1, e1, f = crit(y_iv, r)
        ok2, e2, _ = crit(y_k, y_iv)
        rec("iv", layer=L, M=M, ok=ok1 and ok2, dense_vs_oracle=e1, kernel_vs_dense=e2, floor=f)


def check_v(L):
    if L not in DEAD:
        return
    a = arena(L)
    for M in (1, 5, 64, 1024):
        x, _, _ = make_batch(L, M, 50 + L)
        ids = torch.full((M, TOPK), -1, dtype=torch.int32)
        ids[:, :4] = torch.tensor(DEAD[L], dtype=torch.int32)
        w = torch.full((M, TOPK), 0.25)
        w[:, 4:] = 0
        y = KFN(x, ids.to(DEV), w.to(DEV), a.ptrs, a.I, K, E)
        sync()
        rec("v", layer=L, M=M, ok=bool((y == 0).all()), nonzero=int((y != 0).sum()))


def check_vi(L):
    a = arena(L)
    for dtype in (torch.bfloat16, torch.float16):
        for M in (1, 4, 5, 64, 65, 256, 1024):
            x, ids, w = make_batch(L, M, 60 + L + M, dtype=dtype, pad=True)
            parts, exact = [], True
            for r in range(4):
                lay = fake_layer(a, 4, r)
                p_local = moe_vllm.moe_apply(lay, x, w, ids)
                p_glob = KFN(x, ids, w * ((torch.div(ids, 64, rounding_mode="floor") == r).float()), a.ptrs, a.I, K, E)
                sync()
                exact = exact and torch.equal(p_local, p_glob)
                parts.append(p_local)
            s = sum(p.float() for p in parts)
            r_ = oracle(x, ids, w, L)
            e, f = rms_rel(s, r_), floor_of(r_, dtype)
            ok = exact and (e <= 2 * f + 1e-3 if dtype == torch.bfloat16 else e <= 1e-3)
            rec("vi", layer=L, dtype=str(dtype), M=M, ok=ok, ep_exact=exact, sum_rms_rel=e, floor=f)


def check_vii(L):
    a = arena(L)
    tr = ckpt.exl3(f"model.language_model.layers.{L}.mlp.shared_expert.gate_proj")
    Kd = tr["trellis"].shape[-1] // 16
    dt_tr, dt_suh, dt_svh = tr["trellis"].to(DEV), tr["suh"].to(DEV).view(1, -1), tr["svh"].to(DEV)
    n = dt_svh.shape[0]
    sonb = torch.zeros(n // 128, dtype=torch.int32, device=DEV)
    from exl3xpu import ops
    for M in (1, 2, 4, 8, 12, 16, 24, 32, 64):
        layers = {"tp1": fake_layer(a), "ep4r1": fake_layer(a, 4, 1)}
        xs = torch.zeros((M, H), dtype=torch.bfloat16, device=DEV)
        ids_s = torch.full((M, TOPK), -1, dtype=torch.int32, device=DEV)
        w_s = torch.zeros((M, TOPK), dtype=torch.float32, device=DEV)
        outs = {}

        def body():
            for k_, lay in layers.items():
                outs[k_] = moe_vllm.moe_apply(lay, xs, w_s, ids_s)
            outs["dense"] = torch.ops.exl3xpu_C.linear(xs, dt_tr, dt_suh, dt_svh, sonb, [0, n], Kd, 2, ops.SMALL_M_MAX,
                                                       ops.RECON_SLICE_N)
        s = torch.xpu.Stream()
        s.wait_stream(torch.xpu.current_stream())
        with torch.xpu.stream(s):
            for _ in range(3):
                body()
        torch.xpu.current_stream().wait_stream(s)
        sync()
        g = torch.xpu.XPUGraph()
        with torch.xpu.graph(g):
            body()
        sync()
        cap = dict(outs)                      # the graph's static outputs (eager runs rebind `outs`)
        bad = 0
        for seed in range(5):
            x, ids, w = make_batch(L, M, 70 + seed, pad=True)
            xs.copy_(x)
            ids_s.copy_(ids)
            w_s.copy_(w)
            g.replay()
            sync()
            got = {k_: v.clone() for k_, v in cap.items()}
            body()
            sync()
            bad += sum(not torch.equal(got[k_], outs[k_]) for k_ in got)
        rec("vii", layer=L, M=M, ok=bad == 0, replay_mismatches=bad)


def _bench(fn, iters):
    for _ in range(3):
        fn()
    sync()
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    sync()
    return (time.perf_counter() - t) / iters * 1e6


def check_viii(L):
    variants()
    rec("viii_info", layer=L, ok=True, rows=len(_VARIANTS["rows"]), table=_VARIANTS["rows"])
    best: dict = {}
    for name, a, lay_ep in (("ep4", arena(L), 4), ("tp4", arena(L, 4, 0), 1)):
        lay = fake_layer(a, 4, 0) if lay_ep == 4 else fake_layer(a)
        for M in (1, 2, 4, 8, 16, 32, 64, 256, 1024):
            x, ids, w = make_batch(L, M, 80 + M)
            local = moe_vllm.map_ids(lay, ids)
            default = tuple(int(v) for v in X.moe_plan(H, a.I, K, M))
            res = []
            for p in plan_list(a.I, M):
                force(p)
                try:
                    us = _bench(lambda: KFN(x, local, w, lay.exl3_ptrs, a.I, K, lay.exl3_E), args.bench_iters)
                finally:
                    unforce()
                res.append((us, list(p)))
                rec("viii", layer=L, split=name, M=M, plan=list(p), us=round(us, 1), default=list(p) == list(default),
                    ok=True, remote_frac=round(float((local < 0).float().mean()), 3))
            if res:
                b = min(res)
                best.setdefault(name, {})[M] = {"plan": b[1], "us": round(b[0], 1), "default": list(default)}
    (OUTDIR / f"splits-L{L}.json").write_text(json.dumps(best, indent=1))


def check_ix(L):
    a = arena(L)
    for M in (1, 4, 32, 1024):
        for name, lay in (("tp1", fake_layer(a)), ("ep4r1", fake_layer(a, 4, 1))):
            x, ids, w = make_batch(L, M, 90 + M)
            half = torch.arange(M) % 2 == 1                     # every other row padding (M = 1: none)
            ids[half.to(DEV)] = -1
            w[half.to(DEV)] = 0
            y = moe_vllm.moe_apply(lay, x, w, ids)
            ids_all = torch.full_like(ids, -1)
            y0 = moe_vllm.moe_apply(lay, x, torch.zeros_like(w), ids_all)
            sync()
            ok = bool((y[half.to(DEV)] == 0).all()) and bool((y0 == 0).all()) and bool(torch.isfinite(y).all())
            rec("ix", layer=L, M=M, layer_kind=name, ok=ok)


def check_x(L):
    a = arena(L)
    lay = fake_layer(a)
    for M in (1025, 2048, 4096):
        x, ids, w = make_batch(L, M, 100 + M, pad=True)
        y_c = moe_vllm.moe_apply(lay, x, w, ids)
        y_u = KFN(x, ids, w, a.ptrs, a.I, K, E)
        sync()
        eq = torch.equal(y_c, y_u)
        ok = eq or crit(y_c, oracle(x, ids, w, L))[0]
        rec("x", layer=L, M=M, ok=ok, bit_equal=eq)


def check_xi():
    files = sorted(glob.glob(os.path.join(args.dump_dir, "L*_r*_s*.pt"))) if args.dump_dir else []
    if not files:
        rec("xi", ok=False, error=f"no dumps in {args.dump_dir!r}")
        return
    for fpath in files:
        d = torch.load(fpath)
        L = int(re.match(r"L(\d+)_", os.path.basename(fpath)).group(1))
        tp_size, rank = d["moe_tp"]
        a = arena(L, tp_size, rank)
        x, local, w = d["x"].to(DEV), d["local_ids"].to(DEV), d["topk_weights"].float().to(DEV)
        if d["use_ep"]:
            ep, er = d["ep"]
            n = E // ep
            ptrs = a.ptrs[er * n:(er + 1) * n]
            g_ids = torch.where(local >= 0, local + er * n, local)
        else:
            ptrs, n, g_ids = a.ptrs, E, local
        y = KFN(x, local.to(torch.int32), w, ptrs, a.I, K, n)
        sync()
        stats = {}
        r = oracle(x, g_ids, w, L, tp_size, rank, stats=stats)
        ok, e, f = crit(y, r)
        same = torch.equal(y.cpu(), d["out"])
        fin = all(v == v and v < 32768 for v in stats.values())
        rec("xi", file=os.path.basename(fpath), layer=L, ok=ok and fin, rms_rel=e, floor=f, replay_equal_dump=same,
            max_g=stats.get("g"), max_u=stats.get("u"), max_x_suh=stats.get("xsuh"))


def check_det():
    from exl3xpu import ops
    shapes = {"o_proj 2048->3072": ("model.language_model.layers.3.self_attn.o_proj", "row"),
              "q_proj 3072->4096": ("model.language_model.layers.3.self_attn.q_proj", "col"),
              "in_proj_qkvz 3072->5120": (None, "qkvz"), "lm_head 3072->62080": ("lm_head", "col")}
    for label, (mod, kind) in shapes.items():
        if kind == "qkvz":
            pq = tp.qwen35_plan("model.language_model.layers.0.linear_attn.in_proj_qkv", H, 12288, cfg_text,
                                tp_rank=0, tp_size=4)
            pz = tp.qwen35_plan("model.language_model.layers.0.linear_attn.in_proj_z", H, 8192, cfg_text,
                                tp_rank=0, tp_size=4)
            fq = ckpt.exl3("model.language_model.layers.0.linear_attn.in_proj_qkv")
            fz = ckpt.exl3("model.language_model.layers.0.linear_attn.in_proj_z")
            tr = torch.cat([pq.narrow("trellis", fq["trellis"]), pz.narrow("trellis", fz["trellis"])], 1)
            suh = torch.stack([fq["suh"], fz["suh"]])
            svh = torch.cat([pq.narrow("svh", fq["svh"]), pz.narrow("svh", fz["svh"])])
            bounds = [0, pq.local_n, pq.local_n + pz.local_n]
            Kd = fq["trellis"].shape[-1] // 16
        else:
            full = ckpt.exl3(mod)
            k_, n_ = full["suh"].shape[0], full["svh"].shape[0]
            p = tp.qwen35_plan(mod, k_, n_, cfg_text, tp_rank=0, tp_size=4)
            tr, svh = p.narrow("trellis", full["trellis"]), p.narrow("svh", full["svh"])
            suh = p.narrow("suh", full["suh"]).view(1, -1)
            bounds = [0, p.local_n]
            Kd = full["trellis"].shape[-1] // 16
        sonb = torch.empty(svh.shape[0] // 128, dtype=torch.int32)
        for gi in range(len(bounds) - 1):
            sonb[bounds[gi] // 128:bounds[gi + 1] // 128] = gi
        tr, suh, svh, sonb = tr.to(DEV), suh.to(DEV), svh.to(DEV), sonb.to(DEV)
        for M in (129, 256, 512):
            x = (torch.randn((M, suh.shape[1]), generator=torch.Generator().manual_seed(M)) * 0.5).to(torch.bfloat16)
            x = x.to(DEV)
            ys = [torch.ops.exl3xpu_C.linear(x, tr, suh, svh, sonb, bounds, Kd, 2, ops.SMALL_M_MAX, ops.RECON_SLICE_N)
                  for _ in range(8)]
            sync()
            same = sum(torch.equal(ys[0], y) for y in ys[1:])
            rec("det", shape=label, M=M, ok=True, identical_repeats=f"{same}/7")


# ------------------------------------------------------------------------------------------------------- main
def main() -> int:
    layers = [int(v) for v in args.layers.split(",")]
    device_only = {"i", "vii", "viii", "det"}
    skipped = sorted(CHECKS & device_only) if DRY else []
    t0 = time.time()
    for L in layers:
        if CHECKS & {"i", "ii", "iii"}:
            check_i_ii_iii(L)
        for name, fn in (("iiib", check_iiib), ("iiic", check_iiic), ("iv", check_iv), ("v", check_v),
                         ("vi", check_vi), ("vii", check_vii), ("viii", check_viii), ("ix", check_ix),
                         ("x", check_x)):
            if name in CHECKS and name not in skipped:
                fn(L)
    if "xi" in CHECKS and args.dump_dir:
        check_xi()
    if "det" in CHECKS and "det" not in skipped:
        check_det()
    fails = {k: sum(1 for r in v if not r.get("ok", True)) for k, v in RESULTS.items()}
    summary = {"model": os.path.basename(args.model.rstrip("/")), "dryrun": DRY, "layers": layers,
               "checks": sorted(CHECKS), "skipped_device_only": skipped, "seconds": round(time.time() - t0, 1),
               "fails": fails, "pass": not any(fails.values())}
    Path(args.out).write_text(json.dumps({"summary": summary, "results": RESULTS}, indent=1, default=str))
    print("M2B", json.dumps(summary), flush=True)
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
