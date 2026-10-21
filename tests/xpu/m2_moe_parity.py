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
        emu criterion (see emu_crit): rms_rel vs the fp64 oracle <= 1.5 x the emulated fp16 schedule's own error
        on the same rows + 2e-4, on the first min(M, 8) rows
  iiib  every compiled K=4 (MR, P, RB) forced with moe_set_mr / moe_set_splits, I in {1024, 512, 256} (experts
        sliced for MoE-TP rank 0), M in {1, 4, 5, 64, 65, 1024}: the iii (emu) criterion on every variant
  iiic  MoE-TP sum: sum over ranks of moe_forward(x, ids, w, ptrs_r, I/tp, 4, 256) vs the TP1 oracle, tp in {2, 4}:
        the iii (emu) criterion on the summed partials
  iv    second oracle: per-expert loop over the dense exl3xpu_C.linear on zero-copy views of the blob (gate|up as one
        2-group linear, down); oracle and kernel each within 2 x floor + 1e-3 of the fp32 oracle / of each other
  v     routing only to layer 3's dead experts (72, 166, 180, 252): output exactly 0
  vi    EP exact: per EP4 rank r, moe_forward(x, emap_r[ids], w, ptrs[64r:+64], I, 4, 64) == moe_forward(x, ids,
        w * (ids // 64 == r), ptrs, I, 4, 256) (torch.equal); sum of partials vs the oracle by the iii (emu) criterion
  vii   graphs: capture [the method's apply on a TP1 layer, on an EP4-rank-1 layer, a dense EXL3 linear] at
        M in {1,2,4,8,12,16,24,32,64} with ALL ids -1 (as vLLM captures), replay with real x / ids (EP ids with -1,
        pair mode included): replay == eager bit for bit
  viii  microbench: us per MoE layer per forced (MR, P, RB) for EP4 (I=1024, real routing, ~3/4 remote) and TP4
        (I=256), M in {1,2,4,8,16,32,64,256,1024}, moe_info's per-kernel limits; best plan per M -> splits.json
  ix    padding: rows with id -1 / weight 0 and all-padding batches, TP1 and an EP map, M in {1,4,32,1024}: exactly 0
  x     the method's chunked apply (M in {1025, 2048, 4096}) vs one unchunked moe_forward: bit-equal (else iii)
  xi    replay the MoE inputs an M2c run dumped (EXL3_DUMP_LAYER / EXL3_DUMP_DIR) through the iii oracle; record the
        oracle's max |g|, |u|, |x * suh| per layer: finite and < 32768 (fp16 intermediates as exllamav3); every file
        must hold routed (non-padding) rows and at least one file must route to a local expert
  det   dense determinism probe: TP4 shapes 2048->3072, 3072->{4096, 5120, 62080} at M in {129, 256, 512}, 8 calls
        each (INFO: identity tests stay <= 128 tokens)
  info  (runs first) moe_info(device): per-kernel work-group limit and private memory under the runtime JIT, every
        K=4 kernel T <= limit and private memory 0; seconds for the call (the JIT of every GEMV kernel); moe_plan for
        the three Ornith shapes at M = 1..1024 is the same before and after the device limits are known
  warm  the load-time variant warmup (loadcheck.warmup's MoE launches for I = 1024 / 512 / 256, plus dense EXL3 linears
        on mini tensors at loadcheck.DENSE_WARM_M): seconds per first and second launch = JIT cost (INFO)
  scratch  exl3xpu_C::exl3_reserve_scratch grows the reconstruct scratch once (memory_allocated delta), a later
        reconstruct-path linear allocates nothing persistent and matches the oracle; the contiguity TORCH_CHECKs of
        linear / exl3_reconstruct / moe_forward raise on non-contiguous operands and the device works afterwards;
        moe_forward on row-strided x equals it on the contiguous copy
  svh   layer 1 expert 162 (the checkpoint's largest gate / up svh, 1902 / 2772 at channel 1019) forced into every
        row at x scales 0.5 / 1 / 2: finite, (iii) criterion; the oracle's max |g|, |u|, |act| recorded
  dead122  --model122: every dead routed expert of the 122B (13, layers 3-13) in one arena with 3 live experts of
        the same layers: rows routed only to dead experts give exact 0, rows with live experts meet (iii)

Oracle (iii and every check that compares with one): float64 on the CPU. W_inner is decoded by tests/cpu/exl3_ref.py
(a transcription of exllamav3's CUDA decoder, no trellis-serve code), cross-checked bit for bit against
trellis_core.reference.reconstruct_inner on sampled experts; the two Hadamards are exact +-1 transforms in float64
(tests/cpu/exact_ref.fwht128_): y = H(H(x * suh) @ W_inner) * svh / 128. MoE-TP rank partials use the 128-aligned
column / row blocks of the same expert.
--stop-file (or M2B_STOP_FILE): checked between kernel launches; when it exists the run syncs, writes its partial
results and exits 3 (never kill a container with kernels in flight).

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
ap.add_argument("--checks", default="info,i,ii,iii,iiib,iiic,iv,v,vi,vii,ix,x,xi,scratch,svh,warm,viii,det")
ap.add_argument("--model122", default="", help="dead122: the 122B checkpoint directory (read-only)")
ap.add_argument("--stop-file", default=os.environ.get("M2B_STOP_FILE", ""))
ap.add_argument("--threads", type=int, default=int(os.environ.get("M2B_CPU_THREADS", "12")),
                help="CPU threads of the float64 oracle")
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

torch.set_num_threads(max(1, args.threads))

import exl3_ref  # noqa: E402  (tests/cpu: exllamav3 transcription, the oracle's decoder)
from exact_ref import fwht128_  # noqa: E402
from st_range import Checkpoint  # noqa: E402
from trellis_core import reference as ref  # noqa: E402
from exl3xpu import loadcheck, moe_vllm, tp  # noqa: E402
from exl3xpu.moe_offload import ExpertStore, pack_expert, s64, ops as moe_ops  # noqa: E402

DEV = torch.device("cpu") if DRY else torch.device("xpu", 0)
if not DRY:
    from exl3xpu import ops as _dense_ops
    if not _dense_ops._get_esimd():          # loads _C.so ($EXL3_LIB): torch.ops.exl3xpu_C.* below need it
        raise SystemExit("m2b: exl3xpu_C (_C.so) did not load")
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


STARTED = time.time()


def write_results(extra: dict | None = None) -> dict:
    fails = {k: sum(1 for r in v if not r.get("ok", True)) for k, v in RESULTS.items()}
    summary = {"model": os.path.basename(args.model.rstrip("/")), "dryrun": DRY, "layers": args.layers,
               "checks": sorted(CHECKS), "seconds": round(time.time() - STARTED, 1), "fails": fails,
               "pass": not any(fails.values())}
    summary.update(extra or {})
    Path(args.out).write_text(json.dumps({"summary": summary, "results": RESULTS}, indent=1, default=str))
    return summary


def poll_stop() -> None:
    """Abort path between kernel launches: sync, write what we have, exit 3."""
    if args.stop_file and os.path.exists(args.stop_file):
        sync()
        s = write_results({"aborted": f"stop file {args.stop_file}", "pass": False})
        print("M2B", json.dumps(s), flush=True)
        sys.exit(3)


def rec(check: str, **kw) -> dict:
    RESULTS.setdefault(check, []).append(kw)
    print(check, {k: v for k, v in kw.items() if not isinstance(v, (list, dict)) or len(str(v)) < 200}, flush=True)
    poll_stop()
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


def decode_expert(t: dict) -> dict:
    """{role: {trellis, suh, svh}} -> {role: (W_inner fp16 [k, n] by exl3_ref, suh f64, svh f64)}."""
    return {r: (exl3_ref.reconstruct_inner(t[r]["trellis"], K), t[r]["suh"].to(torch.float64),
                t[r]["svh"].to(torch.float64)) for r in ROLES}


def lin64(x64: torch.Tensor, wi: torch.Tensor, suh: torch.Tensor, svh: torch.Tensor) -> torch.Tensor:
    """float64 x @ W_orig with W_orig = diag(suh) H W_inner H diag(svh) / 128 (H = +-1 Sylvester, 128-blocks)."""
    t = (x64 * suh).contiguous()
    fwht128_(t, 1)
    y = (t @ wi.to(torch.float64)).contiguous()
    fwht128_(y, 1)
    return y * (svh / 128.0)


def expert64(D: dict, xs: torch.Tensor, tp_size: int = 1, rank: int = 0, stats: dict | None = None) -> torch.Tensor:
    """float64 down(silu(x gate) * (x up)) of one decoded expert; MoE-TP rank partial (I-slice) when tp_size > 1."""
    g = lin64(xs, *D["gate_proj"])
    u = lin64(xs, *D["up_proj"])
    a = F.silu(g) * u
    wi, suh, svh = D["down_proj"]
    if tp_size > 1:
        ip = wi.shape[0] // tp_size
        sl = slice(rank * ip, (rank + 1) * ip)
        a, wi, suh = a[:, sl].contiguous(), wi[sl], suh[sl]
    if stats is not None:
        for k_, v in (("g", g), ("u", u), ("act", a), ("act_suh_d", a * suh),
                      ("x_suh", xs * D["gate_proj"][1]), ("x_suh_u", xs * D["up_proj"][1])):
            if v.numel():
                stats[k_] = max(stats.get(k_, 0.0), float(v.abs().max()))
    return lin64(a, wi, suh, svh)


class Oracle:
    """Decoded experts of one mini layer at a time (W_inner fp16: 18.9 MB per expert, 4.8 GB for a full layer)."""

    def __init__(self):
        self.L = None
        self.cache: dict = {}
        self.xcheck: dict = {}

    def expert(self, L: int, e: int) -> dict:
        if L != self.L:
            self.cache.clear()
            self.L = L
        if e not in self.cache:
            t = expert_tensors(L, e)
            D = decode_expert(t)
            if e % 64 == 0 or e in DEAD.get(L, ()) or (L, e) in XCHECK_EXTRA:
                # the trellis-serve decoder must agree bit for bit (two independent transcriptions)
                eq = all(torch.equal(D[r][0], ref.reconstruct_inner(t[r]["trellis"], K, 2)) for r in ROLES)
                self.xcheck[(L, e)] = eq
                rec("oracle_xcheck", layer=L, expert=e, ok=eq, what="exl3_ref W_inner == trellis_core W_inner")
            self.cache[e] = D
        return self.cache[e]


XCHECK_EXTRA = {(1, 162)}
ORACLE = Oracle()


def oracle(x, ids, w, L, tp_size=1, rank=0, stats=None, expert_fn=None) -> torch.Tensor:
    """float64 CPU sum_k w * expert(x) over GLOBAL ids (-1 skipped); MoE-TP rank partial when tp_size > 1.
    expert_fn(e) -> decoded expert (default: mini layer L through ORACLE)."""
    xf = x.detach().to("cpu", torch.float64)
    ids_c = ids.detach().cpu().long()
    wc = w.detach().cpu().to(torch.float64)
    out = torch.zeros((x.shape[0], x.shape[1]), dtype=torch.float64)
    flat = ids_c.flatten()
    for e in sorted(set(flat[flat >= 0].tolist())):
        D = expert_fn(e) if expert_fn is not None else ORACLE.expert(L, e)
        pos = (flat == e).nonzero().flatten()
        rows, ks = pos // ids_c.shape[1], pos % ids_c.shape[1]
        d = expert64(D, xf[rows], tp_size, rank, stats)
        out.index_add_(0, rows, d * wc[rows, ks].unsqueeze(1))
    return out


def rms_rel(y, r) -> float:
    y64 = y.detach().to("cpu", torch.float64)
    r64 = r.detach().to("cpu", torch.float64)
    den = r64.pow(2).mean().sqrt().item()
    num = (y64 - r64).pow(2).mean().sqrt().item()
    return num / den if den else num


def floor_of(r, dtype) -> float:
    r64 = r.detach().to("cpu", torch.float64)
    return rms_rel(r64.to(dtype), r64)


def crit(y, r, dtype=torch.bfloat16) -> tuple[bool, float, float]:
    e, f = rms_rel(y, r), floor_of(r, dtype)
    return e <= 2 * f + 1e-3, e, f


def emu_moe(x, ids, w, L, dtype, tp_size=1, ranks=None, emap=None, prefix=8) -> torch.Tensor:
    """CPU emulation of the kernel's fp16 rounding schedule (emulate_expert) for the first `prefix` rows of a
    batch: per-rank partials cast to `dtype` (as the kernel outputs them) then summed in float64.
    ranks=None: one full-width (or MoE-TP rank-0 at tp_size>1) partial. emap: expert -> owning EP rank.
    Models iii/iiib (single partial), iiic (MoE-TP partials summed), vi (EP local partials summed)."""
    n = min(prefix, x.shape[0])
    xc, ic, wc = x[:n].cpu(), ids[:n].cpu().long(), w[:n].cpu().to(torch.float64)
    total = torch.zeros((n, H), dtype=torch.float64)
    for rk in (range(1) if ranks is None else range(ranks)):
        part = torch.zeros((n, H), dtype=torch.float64)
        for m in range(n):
            for k_ in range(ic.shape[1]):
                e = int(ic[m, k_])
                if e < 0 or (emap is not None and int(emap[e]) != rk):
                    continue
                part[m] += wc[m, k_] * emulate_expert(ORACLE.expert(L, e), xc[m:m + 1], True, tp_size, rk)[0]
        total += part.to(dtype).to(torch.float64)
    return total


def emu_crit(y, r, emu, dtype, prefix=None):
    """Criterion per notes/job2/m2b-layer0-fp8-diagnosis.md: rms_rel(y, oracle) <= 1.5 x emu_floor + 2e-4, where
    emu_floor is the CPU emulation's OWN error vs the fp64 oracle on the same rows/dtype/slice — the intrinsic
    cost of the kernel's fp16 schedule. Replaces the output-cast floor and the flat fp16 1e-3, which the layer-0
    ill-conditioning analysis showed are anchored to the wrong error term. emu_gate (kernel vs emulation) is
    recorded as INFO only: the emulation is schedule-faithful on the emu-check cells but NOT across forced
    MR/P/RB plans and other batches (3e-4 to 2e-3 schedule deviations, run main2), so it cannot gate."""
    if prefix is not None:
        y, r, emu = y[:prefix], r[:prefix], emu[:prefix]
    e, g, f = rms_rel(y, r), rms_rel(y, emu.to(dtype)), rms_rel(emu.to(dtype), r)
    return e <= 1.5 * f + 2e-4, e, f, g


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
    emu8 = emu_moe(x, ids, w, L, torch.bfloat16) if "iii" in CHECKS else None
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
                ok, e, f, g = emu_crit(y, r_full[:M], emu8, torch.bfloat16, prefix=min(M, 8))
                pad0 = bool((y[pad_rows(M).to(DEV)] == 0).all())
                rec("iii", layer=L, M=M, path=k_, ok=ok and pad0, rms_rel=e, floor=f, emu_gate=g,
                    padding_rows_zero=pad0)
    return runner, re_


def check_iiib(L):
    Ms = (1, 4, 5, 64, 65, 1024)
    x, ids, w = make_batch(L, max(Ms), 20 + L, pad=True)
    for tpn in (1, 2, 4):
        a = arena(L, tpn, 0)
        r = oracle(x, ids, w, L, tpn, 0)
        emu8 = emu_moe(x, ids, w, L, torch.bfloat16, tp_size=tpn)
        for M in Ms:
            for p in plan_list(a.I, M):
                force(p)
                try:
                    y = KFN(x[:M].contiguous(), ids[:M].contiguous(), w[:M].contiguous(), a.ptrs, a.I, K, E)
                    sync()
                finally:
                    unforce()
                ok, e, f, g = emu_crit(y, r[:M], emu8, torch.bfloat16, prefix=min(M, 8))
                rec("iiib", layer=L, I=a.I, M=M, plan=list(p), ok=ok, rms_rel=e, floor=f, emu_gate=g)


def check_iiic(L):
    Ms = (1, 5, 64, 256, 1024)
    for dtype in (torch.bfloat16, torch.float16):
        x, ids, w = make_batch(L, max(Ms), 30 + L, dtype=dtype, pad=True)
        r = oracle(x, ids, w, L)
        for tpn in (2, 4):
            emu8 = emu_moe(x, ids, w, L, dtype, tp_size=tpn, ranks=tpn)
            for M in Ms:
                parts = [KFN(x[:M].contiguous(), ids[:M].contiguous(), w[:M].contiguous(), arena(L, tpn, rk).ptrs,
                             IFULL // tpn, K, E) for rk in range(tpn)]
                sync()
                s = sum(p.float() for p in parts)
                ok, e, f, g = emu_crit(s, r[:M], emu8, dtype, prefix=min(M, 8))
                rec("iiic", layer=L, dtype=str(dtype), tp=tpn, M=M, ok=ok, rms_rel=e, floor=f, emu_gate=g)


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
            emu8 = emu_moe(x, ids, w, L, dtype, ranks=4, emap=torch.arange(E) // 64, prefix=min(M, 8))
            ok, e, f, g = emu_crit(s, r_, emu8, dtype, prefix=min(M, 8))
            ok = exact and ok
            rec("vi", layer=L, dtype=str(dtype), M=M, ok=ok, ep_exact=exact, sum_rms_rel=e, floor=f, emu_gate=g)


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
    local_pairs_total = 0
    for fpath in files:
        d = torch.load(fpath)
        L = int(re.match(r"L(\d+)_", os.path.basename(fpath)).group(1))
        # an all-padding batch (vLLM's profile / capture warmup) routes nowhere: y = oracle = 0 and the |g| / |u|
        # statistics are empty, so it would pass without evidence. Such a file fails; a file whose routed rows all
        # go to other EP ranks is a valid zero check, but at least one file must exercise a local expert.
        real_rows = int((d["topk_ids"] >= 0).any(-1).sum())
        local_pairs = int((d["local_ids"] >= 0).sum())
        local_pairs_total += local_pairs
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
        rec("xi", file=os.path.basename(fpath), layer=L, ok=ok and fin and real_rows > 0, rms_rel=e, floor=f,
            replay_equal_dump=same, real_rows=real_rows, local_pairs=local_pairs,
            max_g=stats.get("g"), max_u=stats.get("u"), max_x_suh=stats.get("x_suh"),
            max_act=stats.get("act"))
    if local_pairs_total == 0:
        rec("xi", ok=False, error="no dumped batch routes to a local expert (nothing was checked)")


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


# ------------------------------------------------------------------------------- device limits, JIT, scratch
SHAPES_I = (1024, 512, 256)            # EP / whole expert, MoE-TP2, MoE-TP4


def _plans_all(I: int, max_m: int = 1024) -> list[tuple]:
    unforce()
    return [tuple(int(v) for v in X.moe_plan(H, I, K, m)) for m in range(1, max_m + 1)]


INFO_FIELDS = ("kind", "K", "MR", "P", "RB", "dtype", "T", "GRF", "SLM", "wg_limit", "private_mem")


def check_info():
    """moe_info(device): per-kernel limits under the runtime JIT; the planner's picks do not move."""
    before = {I: _plans_all(I) for I in SHAPES_I}
    t0 = time.perf_counter()
    info = X.moe_info(DEV.index or 0)
    secs = time.perf_counter() - t0
    F_, n = info[0], info[1]
    rows = [list(info[2 + i * F_: 2 + (i + 1) * F_]) for i in range(n)]
    _VARIANTS["rows"] = rows
    _VARIANTS["gu"] = sorted({(r[2], r[3]) for r in rows if r[0] == 0 and r[1] == K})
    _VARIANTS["dn"] = sorted({(r[2], r[3]) for r in rows if r[0] == 1 and r[1] == K})
    after = {I: _plans_all(I) for I in SHAPES_I}
    moved = {I: [m + 1 for m in range(1024) if before[I][m] != after[I][m]] for I in SHAPES_I}
    k4 = [dict(zip(INFO_FIELDS, r)) for r in rows if r[1] == 4]
    bad = [r for r in k4 if not (r["wg_limit"] >= r["T"] and r["private_mem"] == 0)]
    lim = {}
    for r in rows:
        lim.setdefault(f"K{r[1]}_grf{r[7]}", set()).add(r[9])
    k3_over = [dict(zip(INFO_FIELDS, r)) for r in rows if r[1] == 3 and r[6] > r[9]]
    rec("info", ok=not bad and not any(moved.values()) and len(k4) == 49, seconds=round(secs, 2), kernels=n,
        k4_kernels=len(k4), k4_bad=bad, wg_limits={k: sorted(v) for k, v in lim.items()},
        private_mem_nonzero=[dict(zip(INFO_FIELDS, r)) for r in rows if r[10] != 0],
        plans_moved_by_limits={I: v[:8] for I, v in moved.items()}, k3_variants_over_limit=k3_over,
        table=[dict(zip(INFO_FIELDS, r)) for r in rows])


def _zero_arena(I: int, n: int = 8):
    blob = int(X.blob_bytes(H, I, K))
    data = torch.zeros((n, blob), dtype=torch.uint8, device=DEV)
    return data, (s64(data.data_ptr()) + torch.arange(n, dtype=torch.int64) * blob).to(DEV)


def _timed(fn) -> float:
    sync()
    t = time.perf_counter()
    fn()
    sync()
    return time.perf_counter() - t


def _dense_layers() -> list[tuple]:
    """Mini dense EXL3 linears as vLLM fuses them at TP1: (label, trellis, suh [S, k], svh, shard_of_nb, bounds, K)."""
    def fused(label, mods):
        ts = [ckpt.exl3(m) for m in mods]
        tr = torch.cat([t["trellis"] for t in ts], 1)
        suh = torch.stack([t["suh"] for t in ts])
        svh = torch.cat([t["svh"] for t in ts])
        bounds = [0]
        for t in ts:
            bounds.append(bounds[-1] + t["svh"].shape[0])
        sonb = torch.empty(svh.shape[0] // 128, dtype=torch.int32)
        for gi in range(len(bounds) - 1):
            sonb[bounds[gi] // 128:bounds[gi + 1] // 128] = gi
        return (label, tr.to(DEV), suh.to(DEV), svh.to(DEV), sonb.to(DEV), bounds, tr.shape[-1] // 16)
    p0, p3 = "model.language_model.layers.0", "model.language_model.layers.3"
    return [fused("gdn in_proj_qkvz", [f"{p0}.linear_attn.in_proj_qkv", f"{p0}.linear_attn.in_proj_z"]),
            fused("gdn out_proj", [f"{p0}.linear_attn.out_proj"]),
            fused("shared gate_up", [f"{p0}.mlp.shared_expert.gate_proj", f"{p0}.mlp.shared_expert.up_proj"]),
            fused("shared down", [f"{p0}.mlp.shared_expert.down_proj"]),
            fused("attn qkv", [f"{p3}.self_attn.q_proj", f"{p3}.self_attn.k_proj", f"{p3}.self_attn.v_proj"]),
            fused("attn o_proj", [f"{p3}.self_attn.o_proj"]),
            fused("lm_head", ["lm_head"])]


def check_warm():
    """The load-time variant warmup (loadcheck.warmup's launches): first vs second launch per variant = JIT cost.
    INFO; run it in a fresh process (and with an empty SYCL cache) for the cold number."""
    from exl3xpu import loadcheck, ops
    tot = {"moe_first": 0.0, "moe_second": 0.0, "dense_first": 0.0, "dense_second": 0.0}
    for I in SHAPES_I:
        data, ptrs = _zero_arena(I)
        plans: dict = {}
        for m, p in enumerate(_plans_all(I), 1):
            plans.setdefault(p, m)
        for p, m in sorted(plans.items(), key=lambda kv: kv[1]):
            x = torch.zeros((m, H), dtype=torch.bfloat16, device=DEV)
            ids = (torch.arange(m * TOPK, dtype=torch.int32, device=DEV) % data.shape[0]).view(m, TOPK)
            w = torch.zeros((m, TOPK), dtype=torch.float32, device=DEV)
            t1 = _timed(lambda: X.moe_forward(x, ids, w, ptrs, I, K, data.shape[0]))
            t2 = _timed(lambda: X.moe_forward(x, ids, w, ptrs, I, K, data.shape[0]))
            tot["moe_first"] += t1
            tot["moe_second"] += t2
            rec("warm", kind="moe", I=I, plan=list(p), M=m, first_s=round(t1, 4), second_s=round(t2, 4), ok=True)
        del data, ptrs
    for label, tr, suh, svh, sonb, bounds, Kd in _dense_layers():
        k = suh.shape[1]
        for m in loadcheck.DENSE_WARM_M:
            x = torch.zeros((m, k), dtype=torch.bfloat16, device=DEV)
            f = lambda: torch.ops.exl3xpu_C.linear(x, tr, suh, svh, sonb, bounds, Kd, 2, ops.SMALL_M_MAX,  # noqa: E731
                                                   ops.RECON_SLICE_N)
            t1, t2 = _timed(f), _timed(f)
            tot["dense_first"] += t1
            tot["dense_second"] += t2
            rec("warm", kind="dense", tensor=label, K=Kd, M=m, first_s=round(t1, 4), second_s=round(t2, 4), ok=True)
    rec("warm_total", ok=True, **{k: round(v, 2) for k, v in tot.items()},
        moe_jit_s=round(tot["moe_first"] - tot["moe_second"], 2),
        dense_jit_s=round(tot["dense_first"] - tot["dense_second"], 2))


def _expect_raise(fn, want: str) -> tuple[bool, str]:
    try:
        fn()
        sync()
    except RuntimeError as ex:      # TORCH_CHECK -> RuntimeError (c10::Error)
        return want in str(ex), str(ex).splitlines()[0][:200]
    return False, "no exception"


def check_scratch():
    """exl3_reserve_scratch and the contiguity checks on the device. Run it before any other dense-linear check in
    the process: the reconstruct scratch is a process-wide buffer and the first measurement needs it unallocated."""
    from exl3xpu import ops
    C = torch.ops.exl3xpu_C
    dev_i = DEV.index or 0
    t = ckpt.exl3("model.language_model.layers.0.linear_attn.in_proj_qkv")         # K5 3072 -> 12288
    tr, svh = t["trellis"].to(DEV), t["svh"].to(DEV)
    suh = t["suh"].to(DEV).view(1, -1)
    k, n, Kd = suh.shape[1], svh.shape[0], t["trellis"].shape[-1] // 16
    sonb = torch.zeros(n // 128, dtype=torch.int32, device=DEV)
    M = ops.SMALL_M_MAX + 72                                                          # reconstruct path
    x = (torch.randn((M, k), generator=torch.Generator().manual_seed(5)) * 0.5).to(torch.bfloat16).to(DEV)
    lin = lambda xx=x, trr=tr, su=suh, sv=svh, so=sonb: C.linear(xx, trr, su, sv, so, [0, sv.shape[0]], Kd, 2,  # noqa
                                                                  ops.SMALL_M_MAX, ops.RECON_SLICE_N)
    mem = lambda: (sync(), torch.xpu.memory_allocated(DEV))[1]                       # noqa: E731
    a0 = mem()
    y1 = lin()
    a1 = mem()
    first_growth = a1 - a0 - y1.numel() * 2
    numel = H * ops.RECON_SLICE_N
    C.exl3_reserve_scratch(dev_i, numel)
    a2 = mem()
    y2 = lin()
    a3 = mem()
    C.exl3_reserve_scratch(dev_i, 1000)
    a4 = mem()
    # a reconstruct-path linear that needs the whole reserved slice (attention q_proj+gate, 3072 -> 16384)
    tq = ckpt.exl3("model.language_model.layers.3.self_attn.q_proj")
    trq, svq, suq = tq["trellis"].to(DEV), tq["svh"].to(DEV), tq["suh"].to(DEV).view(1, -1)
    soq = torch.zeros(svq.shape[0] // 128, dtype=torch.int32, device=DEV)
    a5 = mem()
    y3 = lin(x, trq, suq, svq, soq)
    a6 = mem()
    Wq = (exl3_ref.reconstruct_inner(t["trellis"], Kd), t["suh"].to(torch.float64), t["svh"].to(torch.float64))
    r = lin64(x.cpu().to(torch.float64), *Wq)
    ok2, e2, f2 = crit(y2, r)
    Wq3 = (exl3_ref.reconstruct_inner(tq["trellis"], Kd), tq["suh"].to(torch.float64), tq["svh"].to(torch.float64))
    ok3, e3, f3 = crit(y3, lin64(x.cpu().to(torch.float64), *Wq3))
    rec("scratch", ok=(first_growth == k * n * 2 and a2 - a1 == (numel - k * n) * 2 and a3 - a2 == y2.numel() * 2
                       and a4 == a3 and a6 - a5 == y3.numel() * 2 and ok2 and ok3),
        first_call_persistent_B=first_growth, expect_first_B=k * n * 2, reserve_delta_B=a2 - a1,
        expect_reserve_delta_B=(numel - k * n) * 2, after_reserve_call_delta_B=a3 - a2, out_B=y2.numel() * 2,
        shrink_request_delta_B=a4 - a3, full_slice_call_delta_B=a6 - a5, full_slice_out_B=y3.numel() * 2,
        rms_rel_after_reserve=e2, floor=f2, rms_rel_full_slice=e3, floor_full_slice=f3,
        equal_before_after_reserve=torch.equal(y1, y2))
    # contiguity TORCH_CHECKs (host side, before any launch)
    half = n // 2
    cases = {
        "linear trellis": lambda: lin(x, tr[:, :half // 16], suh, svh[:half], sonb[:half // 128]),
        "linear suh": lambda: lin(x, tr, torch.zeros((1, 2 * k), dtype=torch.float16, device=DEV)[:, ::2], svh, sonb),
        "linear svh": lambda: lin(x, tr, suh, torch.stack([svh, svh], 1)[:, 0], sonb),
        "linear shard_of_nb": lambda: lin(x, tr, suh, svh, torch.zeros(2 * (n // 128), dtype=torch.int32,
                                                                         device=DEV)[::2]),
        "exl3_reconstruct trellis": lambda: C.exl3_reconstruct(tr[:, :half // 16], torch.empty(
            (k, half), dtype=torch.float16, device=DEV), 0, Kd, 2),
    }
    for name, fn in cases.items():
        ok, msg = _expect_raise(fn, "contiguous")
        rec("contig", case=name, ok=ok, message=msg)
    a = arena(0)
    xm, ids, w = make_batch(0, 64, 7, pad=True)
    y_ref = KFN(xm, ids, w, a.ptrs, a.I, K, E)
    sync()
    moe_cases = {
        "moe_forward ids": (lambda: KFN(xm, ids.t().contiguous().t(), w, a.ptrs, a.I, K, E), "is_contiguous"),
        "moe_forward weights": (lambda: KFN(xm, ids, w.t().contiguous().t(), a.ptrs, a.I, K, E), "is_contiguous"),
        "moe_forward ptrs": (lambda: KFN(xm, ids, w, torch.stack([a.ptrs, a.ptrs], 1)[:, 0], a.I, K, E),
                             "is_contiguous"),
        "moe_forward x stride(1)": (lambda: KFN(xm.t().contiguous().t(), ids, w, a.ptrs, a.I, K, E), "stride(1)"),
    }
    for name, (fn, want) in moe_cases.items():
        ok, msg = _expect_raise(fn, want)
        rec("contig", case=name, ok=ok, message=msg)
    # the device still computes the same after the refused calls; row-strided x (stride(0) != H) is supported
    y_after = KFN(xm, ids, w, a.ptrs, a.I, K, E)
    xbig = torch.zeros((128, H), dtype=torch.bfloat16, device=DEV)
    xbig[::2] = xm
    y_strided = KFN(xbig[::2], ids, w, a.ptrs, a.I, K, E)
    sync()
    rec("contig", case="device after refusals; strided rows", ok=torch.equal(y_ref, y_after) and
        torch.equal(y_ref, y_strided), after_equal=torch.equal(y_ref, y_after), strided_equal=torch.equal(y_ref, y_strided))


def _small_arena(tensors: list[dict]):
    blob = int(X.blob_bytes(H, IFULL, K))
    data = torch.empty((len(tensors), blob), dtype=torch.uint8, device=DEV)
    stage = torch.empty(blob, dtype=torch.uint8, pin_memory=not DRY)
    for i, t in enumerate(tensors):
        pack_expert(t["gate_proj"], t["up_proj"], t["down_proj"], K, out=stage)
        data[i].copy_(stage)
        sync()
    register_arena(data)
    return data, (s64(data.data_ptr()) + torch.arange(len(tensors), dtype=torch.int64) * blob).to(DEV)


def check_svh():
    """Layer 1 expert 162: gate / up svh reach 1902 / 2772 (channel 1019), the largest in the checkpoint. The kernel
    keeps g, u, SiLU(g) * u and act * suh_d in fp16 (as exllamav3): record the oracle's peaks, require finite output
    and the (iii) criterion for x ~ N(0, s^2), s in {0.5, 1, 2}."""
    L, e = 1, 162
    others = [0, 100, 200]
    tens = [expert_tensors(L, v) for v in [e] + others]
    data, ptrs = _small_arena(tens)
    dec = {i: ORACLE.expert(L, v) for i, v in enumerate([e] + others)}
    svg, svu = tens[0]["gate_proj"]["svh"].float(), tens[0]["up_proj"]["svh"].float()
    rec("svh_expert", ok=True, layer=L, expert=e, max_svh_gate=float(svg.abs().max()), ch_gate=int(svg.abs().argmax()),
        max_svh_up=float(svu.abs().max()), ch_up=int(svu.abs().argmax()))
    for s in (0.5, 1.0, 2.0):
        for M in (1, 4, 5, 64, 256):
            g = torch.Generator().manual_seed(int(1000 * s) + M)
            x = (torch.randn((M, H), generator=g) * s).to(torch.bfloat16)
            ids = torch.full((M, TOPK), -1, dtype=torch.int32)
            ids[:, 0] = 0
            ids[:, 1:4] = torch.tensor([1, 2, 3], dtype=torch.int32)
            w = torch.zeros((M, TOPK))
            w[:, :4] = torch.tensor([0.4, 0.2, 0.2, 0.2])
            y = KFN(x.to(DEV), ids.to(DEV), w.to(DEV), ptrs, IFULL, K, len(tens))
            sync()
            st: dict = {}
            r = oracle(x, ids, w, L, stats=st, expert_fn=lambda i: dec[i])
            ok, err, fl = crit(y, r)
            fin = bool(torch.isfinite(y).all())
            rec("svh", layer=L, expert=e, sigma=s, M=M, ok=ok and fin, finite=fin, rms_rel=err, floor=fl,
                **{f"max_{k_}": round(v, 2) for k_, v in st.items()})
    del data, ptrs


_RS128 = float(torch.tensor(1.0 / 128 ** 0.5, dtype=torch.float32))      # kRsqrt128 (fp32 constant)


def _f16(t: torch.Tensor) -> torch.Tensor:
    return t.to(torch.float16).to(torch.float64)


def _fwht(t: torch.Tensor) -> torch.Tensor:
    t = t.to(torch.float64).contiguous().clone()
    fwht128_(t, t.dim() - 1)
    return t


def _gemv_emu(xh: torch.Tensor, wi: torch.Tensor, fma: bool = True) -> torch.Tensor:
    """The GEMV kernels' accumulation (GemvKernel<4>::step): per 16-row tile-row, each output column's 16 products
    are summed in 4 fp16 accumulators over the rows {2q, 2q+1, 2q+8, 2q+9} (q = lane % 4, in that order, starting
    from 0), then folded into fp32 (here float64). xh fp16-valued [n, k], wi fp16 W_inner [k, N] -> [n, N]."""
    n, k = xh.shape
    W = wi.to(torch.float64).view(k // 16, 16, -1)
    X = xh.view(n, k // 16, 16)
    out = torch.zeros((n, W.shape[-1]), dtype=torch.float64)
    for q in range(4):
        h = torch.zeros((n, k // 16, W.shape[-1]), dtype=torch.float64)
        for row in (2 * q, 2 * q + 1, 2 * q + 8, 2 * q + 9):
            p = X[:, :, row].unsqueeze(-1) * W[:, row, :].unsqueeze(0)
            h = _f16(h + p) if fma else _f16(h + _f16(p))
        out += h.sum(1)
    return out


def emulate_expert(D: dict, x: torch.Tensor, fma: bool = True, tp_size: int = 1, rank: int = 0) -> torch.Tensor:
    """float64 emulation of GateUpKernel<4> + DownKernel<4> for one expert with weight 1, with the kernels' fp16
    roundings: fp16(x*suh), fp16(H(.)/sqrt128), fp16 tile-row accumulators, fp16 g / u after svh, fp16 SiLU(g)*u,
    fp16 act*suh_d, fp16 abuf; fp32 (here float64) everywhere else. Returns float64 [n, H] before the output cast."""
    x64 = x.to(torch.float64)
    gu = {}
    for r in ("gate_proj", "up_proj"):
        wi, suh, svh = D[r]
        xh = _f16(_fwht(_f16(x64 * suh)) * _RS128)
        acc = _gemv_emu(xh, wi, fma)
        gu[r] = _f16(_fwht(acc) * _RS128 * svh)
    g, u = gu["gate_proj"], gu["up_proj"]
    act = _f16(g / (1.0 + torch.exp(-g)) * u)
    wi, suh, svh = D["down_proj"]
    if tp_size > 1:                     # MoE-TP rank partial: 128-aligned I-slice (tile-rows stay whole)
        ip = wi.shape[0] // tp_size
        sl = slice(rank * ip, (rank + 1) * ip)
        act, wi, suh = act[:, sl].contiguous(), wi[sl], suh[sl]
    act = _f16(act * suh)
    ab = _f16(_fwht(act) * _RS128)
    acc = _gemv_emu(ab, wi, fma)
    return _fwht(acc) * _RS128 * svh


def check_emu():
    """Where the kernel's error comes from: per expert (weight 1, single-expert rows), kernel vs exact float64, the
    float64 emulation of the kernel's fp16 roundings vs exact, and kernel vs emulation. INFO + a sanity bound."""
    L = 0
    for e in (0, 17, 100, 200, 255):
        D = ORACLE.expert(L, e)
        data, ptrs = _small_arena([expert_tensors(L, e)])
        for dt in (torch.float16, torch.bfloat16):
            g = torch.Generator().manual_seed(500 + e)
            x = (torch.randn((8, H), generator=g) * 0.5).to(dt)
            ids = torch.full((8, TOPK), -1, dtype=torch.int32)
            ids[:, 0] = 0
            w = torch.zeros((8, TOPK))
            w[:, 0] = 1.0
            for M in (1, 8):
                y = KFN(x[:M].to(DEV), ids[:M].to(DEV), w[:M].to(DEV), ptrs, IFULL, K, 1)
                sync()
                exact = expert64(D, x[:M].to(torch.float64))
                emu = emulate_expert(D, x[:M])
                emu_mul = emulate_expert(D, x[:M], fma=False)
                k_ex, em_ex = rms_rel(y, exact), rms_rel(emu.to(dt), exact)
                k_em, k_em2 = rms_rel(y, emu.to(dt)), rms_rel(y, emu_mul.to(dt))
                rec("emu", layer=L, expert=e, dtype=str(dt), M=M, ok=k_ex <= 1.5 * em_ex + 2e-4,
                    kernel_vs_exact=k_ex, emu_vs_exact=em_ex, emu_unrounded_vs_exact=rms_rel(emu, exact),
                    kernel_vs_emu_fma=k_em, kernel_vs_emu_mul=k_em2, floor_out=floor_of(exact, dt),
                    kernel_eq_emu_fma_frac=float((y.cpu().to(torch.float64) == emu.to(dt).to(torch.float64))
                                                 .float().mean()))
        del data, ptrs
    # the rows of the iii-b batch (layer 0, seed 20) at TP1 / MoE-TP rank-0 slices, full 8-expert sums
    x, ids, w = make_batch(L, 8, 20 + L, pad=True)
    xc, ic, wc = x.cpu(), ids.cpu().long(), w.cpu().to(torch.float64)
    for tpn in (1, 2, 4):
        a = arena(L, tpn, 0)
        y = KFN(x, ids, w, a.ptrs, a.I, K, E)
        sync()
        exact = oracle(x, ids, w, L, tpn, 0)
        emu = torch.zeros_like(exact)
        for m in range(xc.shape[0]):
            for k_ in range(TOPK):
                if ic[m, k_] >= 0:
                    emu[m] += wc[m, k_] * emulate_expert(ORACLE.expert(L, int(ic[m, k_])), xc[m:m + 1], True, tpn, 0)[0]
        for m in (0, 1, 3, 4):
            rec("emu_row", layer=L, tp=tpn, row=m, ok=True, kernel_vs_exact=rms_rel(y[m], exact[m]),
                emu_vs_exact=rms_rel(emu[m].to(x.dtype), exact[m]), floor=floor_of(exact[m], x.dtype),
                kernel_vs_emu=rms_rel(y[m], emu[m].to(x.dtype)))


DEAD122 = [(3, 72), (3, 166), (3, 180), (3, 252), (4, 62), (4, 235), (4, 239), (5, 210), (6, 188), (7, 241), (8, 82),
           (10, 6), (13, 40)]
LIVE122 = [(4, 61), (10, 7), (13, 41)]


def check_dead122():
    """Every dead routed expert of the 122B (all-zero gate / up / down svh) plus 3 live experts of the same layers in
    one 16-expert arena: dead-only rows give exact zeros; rows that mix in live experts meet (iii)."""
    if not args.model122:
        rec("dead122", ok=False, error="--model122 not given")
        return
    c122 = Checkpoint(args.model122)

    def tens(L, e):
        b = f"model.language_model.layers.{L}.mlp.experts.{e}"
        return {r: {p: c122.get(f"{b}.{r}.{p}") for p in ("trellis", "suh", "svh")} for r in ROLES}
    tl = [tens(L, e) for L, e in DEAD122 + LIVE122]
    nd = len(DEAD122)
    zero = [all(bool((t[r]["svh"] == 0).all()) for r in ROLES) for t in tl]
    rec("dead122_list", ok=all(zero[:nd]) and not any(zero[nd:]), dead_all_zero_svh=zero[:nd],
        live_zero=zero[nd:])
    data, ptrs = _small_arena(tl)
    dec = {i: decode_expert(t) for i, t in enumerate(tl)}
    for M in (1, 4, 5, 64, 1024):
        g = torch.Generator().manual_seed(77 + M)
        x = (torch.randn((M, H), generator=g) * 0.5).to(torch.bfloat16)
        ids = torch.stack([torch.randperm(nd, generator=g)[:TOPK] for _ in range(M)]).to(torch.int32)
        w = torch.softmax(torch.randn((M, TOPK), generator=g), -1)
        mixed = torch.arange(M) % 3 == 1                              # these rows also route to a live expert
        ids[mixed, -1] = nd + (torch.arange(M)[mixed] % len(LIVE122)).to(torch.int32)
        y = KFN(x.to(DEV), ids.to(DEV), w.to(DEV), ptrs, IFULL, K, len(tl))
        sync()
        r = oracle(x, ids, w, None, expert_fn=lambda i: dec[i])
        dead_rows = ~mixed
        z = bool((y.cpu()[dead_rows] == 0).all())
        oracle_zero = bool((r[dead_rows] == 0).all())
        ok_m, err, fl = (True, 0.0, 0.0) if not mixed.any() else crit(y.cpu()[mixed], r[mixed])
        rec("dead122", M=M, ok=z and oracle_zero and ok_m, dead_rows_exact_zero=z, oracle_dead_rows_zero=oracle_zero,
            mixed_rows=int(mixed.sum()), rms_rel_mixed=err, floor=fl)
    del data, ptrs


# ------------------------------------------------------------------------------------------------------- main
def main() -> int:
    layers = [int(v) for v in args.layers.split(",")]
    device_only = {"i", "vii", "viii", "det", "info", "warm", "scratch"}
    skipped = sorted(CHECKS & device_only) if DRY else []

    def run(name, fn, *a, gate=None):
        if not (CHECKS & (gate or {name})) or name in skipped:
            return
        print(f"=== {name} {a} t={time.time() - STARTED:.0f}s", flush=True)
        try:
            fn(*a)
        except SystemExit:
            raise
        except Exception as ex:          # noqa: BLE001  (record, keep going unless the device is gone)
            import traceback
            rec(f"{name}_error", ok=False, args=str(a), error=traceback.format_exc()[-3000:])
            write_results({"skipped_device_only": skipped})
            if "UR_RESULT_ERROR" in str(ex) or "DEVICE_LOST" in str(ex).upper():
                raise
        write_results({"skipped_device_only": skipped})

    run("info", check_info)
    run("scratch", check_scratch)
    for L in layers:
        run("i_ii_iii", check_i_ii_iii, L, gate={"i", "ii", "iii"})
        for name, fn in (("iiib", check_iiib), ("iiic", check_iiic), ("iv", check_iv), ("v", check_v),
                         ("vi", check_vi), ("vii", check_vii), ("ix", check_ix), ("x", check_x)):
            run(name, fn, L)
    run("svh", check_svh)
    run("dead122", check_dead122)
    run("emu", check_emu)
    if "xi" in CHECKS and args.dump_dir:
        run("xi", check_xi)
    run("warm", check_warm)
    for L in layers:
        run("viii", check_viii, L)
    run("det", check_det)
    summary = write_results({"skipped_device_only": skipped,
                             "xi": "not run (needs --dump-dir from M2c)" if "xi" in CHECKS and not args.dump_dir
                             else "run" if "xi" in CHECKS else "not selected"})
    print("M2B", json.dumps(summary), flush=True)
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
