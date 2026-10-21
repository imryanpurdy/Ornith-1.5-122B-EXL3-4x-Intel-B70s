"""Per-process bookkeeping of EXL3 layers from model construction to the end of weight post-processing.

A batch is every EXL3 layer (dense linear or routed-expert block) created since the previous batch closed. vLLM
builds, loads and post-processes the target model first and then, with --speculative-config, the MTP drafter, so
the target is one batch and the drafter the next. When the last layer of a batch finishes
process_weights_after_loading, the batch closes, on every rank:

  1. EXL3XPU_LOADED rank=<r> dense=<n> experts=<n> model=<target|drafter> expected=<d>/<e> moe=<ep|tp><size> ...
     n = EXL3 tensors (trellis/suh/svh/codebook marker) that reached this rank's EXL3 loaders; non-local experts
     under EP are dropped and not counted. Expected values come from the checkpoint index through
     tp.expected_loaded_counts (the drafter count includes its own lm_head copy: +36 dense on the 122B, not +32).
     A mismatch raises: vLLM checks nothing on quantized loads.
  2. On an XPU device: exl3xpu_C::exl3_reserve_scratch with the largest reconstruct slice of the batch, so no
     serving call allocates it.
  3. On an XPU device: the variant warmup. Every dense kernel path (per K, codebook, local shape and dtype at the
     M values of DENSE_WARM_M) and every MoE plan moe_plan() returns for M = 1..1024 is launched once, so no kernel
     JIT-compiles while serving.

The layers are held only until their batch closes (the drafter's own lm_head is replaced by the target's right
after its batch closes and must be freed then).
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

import torch

logger = logging.getLogger("vllm.exl3xpu.loadcheck")

# Dense M values of DESIGN section 3.4 item 8: vector / DPAS / MR switches and the small-M -> reconstruct switch.
DENSE_WARM_M = (1, 2, 3, 4, 8, 9, 16, 32, 33, 64, 65, 128, 129)
MOE_PLAN_MAX_M = 1024          # apply() runs larger batches in chunks of this many rows


class LoadCheckError(RuntimeError):
    pass


class Batch:
    def __init__(self, seq: int):
        self.seq = seq
        self.dense: list = []
        self.moe: list = []
        self.processed = 0
        self.counts = {"dense": 0, "experts": 0, "experts_dropped": 0}
        self.closed = False

    @property
    def size(self) -> int:
        return len(self.dense) + len(self.moe)


_open: Batch | None = None
_seq = 0
CLOSED: list[dict[str, Any]] = []     # one summary per closed batch (tests and debugging)
_warmed: set = set()                  # warmup keys already launched in this process
_moe_info_done: set = set()           # devices moe_info() was called for
_plan_cache: dict = {}                # (H, I, K, devices with moe_info) -> moe_plans() result


def reset_for_tests() -> None:
    global _open, _seq
    _open, _seq = None, 0
    CLOSED.clear()
    _warmed.clear()
    _moe_info_done.clear()
    _plan_cache.clear()


def add_layer(layer, kind: str) -> Batch:
    """Called from create_weights. kind: 'dense' or 'moe'."""
    global _open, _seq
    if _open is not None and _open.processed:
        raise LoadCheckError(f"exl3: {layer.exl3_prefix}: a new EXL3 layer was created while {_open.processed} of "
                             f"{_open.size} layers of the previous model were post-processed")
    if _open is None:
        _seq += 1
        _open = Batch(_seq)
    (_open.dense if kind == "dense" else _open.moe).append(layer)
    layer.exl3_batch = _open
    return _open


def layer_done(layer) -> None:
    """Called at the end of process_weights_after_loading; closes the batch after its last layer."""
    global _open
    b = getattr(layer, "exl3_batch", None)
    if b is None or b.closed:
        raise LoadCheckError(f"exl3: {getattr(layer, 'exl3_prefix', layer)}: process_weights_after_loading ran "
                             "twice or for a layer that was never created")
    if getattr(layer, "exl3_done", False):
        raise LoadCheckError(f"exl3: {layer.exl3_prefix}: process_weights_after_loading ran twice")
    layer.exl3_done = True
    b.processed += 1
    if b.processed == b.size:
        if _open is b:
            _open = None
        close(b)


def _role(b: Batch) -> str:
    return "drafter" if any(l.exl3_prefix.startswith("mtp.") or ".mtp." in f".{l.exl3_prefix}"
                            for l in b.dense + b.moe) else "target"


def _one(values, what: str):
    vals = sorted(set(values))
    if len(vals) != 1:
        raise LoadCheckError(f"exl3: layers of one model disagree on {what}: {vals}")
    return vals[0]


def _tp_group() -> tuple[int, int]:
    from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size
    return int(get_tensor_model_parallel_rank()), int(get_tensor_model_parallel_world_size())


def _parallel(b: Batch) -> dict[str, Any]:
    tp_rank, tp_size = _tp_group()
    odd = sorted({(l.exl3_prefix, l.exl3_tp_rank, l.exl3_tp_size) for l in b.dense
                  if (l.exl3_tp_rank, l.exl3_tp_size) not in ((tp_rank, tp_size), (0, 1))})
    if odd:
        raise LoadCheckError(f"exl3: layers split differently from the TP group (rank {tp_rank} of {tp_size}): "
                             f"{odd[:4]}")
    p = {"tp_size": tp_size, "tp_rank": tp_rank, "moe": None, "num_experts": 1}
    if b.moe:
        p["num_experts"] = _one([l.exl3_global_E for l in b.moe], "global expert count")
        use_ep = _one([l.exl3_use_ep for l in b.moe], "expert parallel")
        if use_ep:
            p["moe"] = ("ep", _one([l.exl3_ep_size for l in b.moe], "ep_size"),
                        _one([l.exl3_ep_rank for l in b.moe], "ep_rank"))
        else:
            p["moe"] = ("tp", _one([l.exl3_moe_tp_size for l in b.moe], "moe tp_size"),
                        _one([l.exl3_moe_tp_rank for l in b.moe], "moe tp_rank"))
    return p


def _index_names(b: Batch) -> list[str]:
    cfgs = {id(l.exl3_config): l.exl3_config for l in b.dense + b.moe}
    paths = {c.index_path for c in cfgs.values()}
    if len(paths) != 1 or None in paths:
        raise LoadCheckError(f"exl3: cannot check the loaded tensor counts without the checkpoint index "
                             f"(model.safetensors.index.json); index paths seen: {sorted(map(str, paths))}")
    with open(paths.pop()) as f:
        return list(json.load(f)["weight_map"])


def expected_counts(b: Batch, role: str, par: dict[str, Any]) -> dict[str, int]:
    from . import tp
    names = _index_names(b)
    if par["moe"] is None:
        mode, size, rank = "tp", par["tp_size"], par["tp_rank"]
    else:
        mode, size, rank = par["moe"]
        if mode == "tp":            # every rank narrows every expert; the dense split is the attention TP
            size, rank = par["tp_size"], par["tp_rank"]
    exp = tp.expected_loaded_counts(names, num_experts=par["num_experts"], moe_mode=mode, tp_size=size,
                                    tp_rank=rank, with_mtp=role == "drafter")
    return exp[role]


def _check_drafter_shares_lm_head() -> None:
    """vLLM's V2 MTP speculator replaces the drafter's lm_head with the target's after loading. If the drafter
    class ever declared its own lm_head, _should_share would compare draft.weight, which an EXL3 lm_head does not
    have (tp-guard review): refuse that configuration instead of failing inside vLLM."""
    try:
        from vllm.model_executor.models import qwen3_5_mtp as m
    except Exception:        # noqa: BLE001  (other drafters are not EXL3 targets of this plugin)
        return
    for name in ("Qwen3_5MTP", "Qwen3_5MoeMTP"):
        cls = getattr(m, name, None)
        if cls is not None and getattr(cls, "has_own_lm_head", False):
            raise LoadCheckError(f"exl3: {name}.has_own_lm_head is set; the drafter's EXL3 lm_head has no .weight to "
                                 "compare with the target's, so vLLM's lm_head sharing would fail")


def close(b: Batch) -> None:
    b.closed = True
    role = _role(b)
    try:
        if role == "drafter":
            _check_drafter_shares_lm_head()
        par = _parallel(b)
        try:
            from vllm.distributed import get_pp_group
            pp = get_pp_group().world_size
        except Exception:    # noqa: BLE001
            pp = 1
        if pp != 1:
            raise LoadCheckError(f"exl3: pipeline parallel ({pp}) is not supported; the load counts assume PP=1")
        exp = expected_counts(b, role, par)
        moe = "-" if par["moe"] is None else f"{par['moe'][0]}{par['moe'][1]}"
        line = (f"EXL3XPU_LOADED rank={par['tp_rank']} dense={b.counts['dense']} experts={b.counts['experts']} "
                f"model={role} expected={exp['dense']}/{exp['experts']} moe={moe} tp={par['tp_size']} "
                f"layers={len(b.dense)}+{len(b.moe)} dropped={b.counts['experts_dropped']}")
        logger.info(line)
        summary = {"seq": b.seq, "role": role, "rank": par["tp_rank"], "tp_size": par["tp_size"], "moe": par["moe"],
                   "counts": dict(b.counts), "expected": exp, "layers": (len(b.dense), len(b.moe)), "line": line}
        CLOSED.append(summary)
        if exp["dense"] != b.counts["dense"] or exp["experts"] != b.counts["experts"]:
            raise LoadCheckError(f"exl3: {line}: loaded EXL3 tensor counts differ from the checkpoint index "
                                 f"(dense {b.counts['dense']} vs {exp['dense']}, experts {b.counts['experts']} vs "
                                 f"{exp['experts']})")
        dev = _device(b)
        if dev is not None and dev.type == "xpu":
            reserve_scratch(b, dev)
            summary["warmup"] = warmup(b, dev)
    finally:
        # drop the layer references (the drafter's own lm_head must be freed when vLLM replaces it)
        for l in b.dense + b.moe:
            l.__dict__.pop("exl3_batch", None)
        b.dense, b.moe = [], []


def _device(b: Batch):
    for l in b.dense:
        return l.trellis.device
    for l in b.moe:
        return l.exl3_arena.device
    return None


def _dev_index(dev) -> int:
    return dev.index if dev.index is not None else torch.xpu.current_device()


# ------------------------------------------------------------------------------------------ scratch and warmup
def scratch_numel(layers) -> int:
    """Largest fp16 reconstruct slice (k_local x min(slice_n, group width)) any dense layer can request."""
    from . import ops
    best = 0
    for l in layers:
        k = l.trellis.shape[0] * 16
        bounds = l.exl3_bounds
        widest = max(b1 - b0 for b0, b1 in zip(bounds, bounds[1:]))
        best = max(best, k * min(ops.RECON_SLICE_N, widest))
    return best


def reserve_scratch(b: Batch, dev) -> None:
    n = scratch_numel([l for l in b.dense if l.exl3_cpp])
    if n:
        torch.ops.exl3xpu_C.exl3_reserve_scratch(_dev_index(dev), n)
        logger.info("EXL3XPU_SCRATCH rank=%s numel=%d (%.1f MiB)", getattr(b.dense[0], "exl3_tp_rank", "?"), n,
                    n * 2 / 2**20)


def moe_info_once(dev) -> None:
    """moe_info(device) reports per-kernel work-group limits that moe_plan() then honours; call it before any plan
    is checked or warmed (moe-k4 review), once per device and process."""
    from .moe_offload import ops as moe_ops
    idx = _dev_index(dev) if dev.type == "xpu" else -1
    if idx in _moe_info_done:
        return
    rows = moe_ops().moe_info(idx)
    _moe_info_done.add(idx)
    if idx >= 0:
        logger.info("EXL3XPU_MOE_INFO device=%d kernels=%d", idx, rows[1] if len(rows) > 1 else -1)


def moe_plans(H: int, I: int, K: int, max_m: int = MOE_PLAN_MAX_M) -> dict[tuple, int]:
    """{plan: smallest M with that plan} for M = 1..max_m (overrides reset first). Cached per shape once moe_info
    has reported the device limits the planner honours."""
    from .moe_offload import ops as moe_ops
    ck = (H, I, K, max_m, frozenset(_moe_info_done))
    if ck in _plan_cache:
        return dict(_plan_cache[ck])
    X = moe_ops()
    X.moe_set_mr(0)
    X.moe_set_splits(0, 0)
    plans: dict[tuple, int] = {}
    for m in range(1, max_m + 1):
        p = tuple(int(v) for v in X.moe_plan(H, I, K, m))
        plans.setdefault(p, m)
    _plan_cache[ck] = dict(plans)
    return plans


def warmup(b: Batch, dev) -> dict[str, Any]:
    t0 = time.time()
    n_dense = n_moe = 0
    keys = []
    with torch.no_grad():
        for l in b.dense:
            k, n = l.trellis.shape[0] * 16, l.svh.shape[0]
            key = ("dense", l.exl3_K, l.exl3_cb, k, n, tuple(l.exl3_bounds), l.exl3_dtype, str(dev), l.exl3_cpp)
            if key in _warmed:
                continue
            _warmed.add(key)
            keys.append(key)
            for m in DENSE_WARM_M:
                x = torch.zeros((m, k), dtype=l.exl3_dtype, device=dev)
                l.quant_method.apply(l, x)
                n_dense += 1
        if b.moe:
            moe_info_once(dev)
        from .moe_offload import ops as moe_ops
        for l in b.moe:
            key = ("moe", l.exl3_H, l.exl3_I, l.exl3_K, l.exl3_E, l.exl3_dtype, str(dev))
            if key in _warmed:
                continue
            _warmed.add(key)
            keys.append(key)
            X = moe_ops()
            for plan, m in sorted(moe_plans(l.exl3_H, l.exl3_I, l.exl3_K).items(), key=lambda kv: kv[1]):
                if not any(plan):
                    raise LoadCheckError(f"exl3: {l.exl3_prefix}: no MoE kernel plan at M={m}")
                x = torch.zeros((m, l.exl3_H), dtype=l.exl3_dtype, device=dev)
                ids = (torch.arange(m * l.exl3_topk, dtype=torch.int32, device=dev) % l.exl3_E).view(m, l.exl3_topk)
                w = torch.zeros((m, l.exl3_topk), dtype=torch.float32, device=dev)
                X.moe_forward(x, ids, w, l.exl3_ptrs, l.exl3_I, l.exl3_K, l.exl3_E)
                n_moe += 1
        torch.xpu.synchronize(dev)
    res = {"dense_launches": n_dense, "moe_launches": n_moe, "keys": len(keys), "seconds": round(time.time() - t0, 2)}
    logger.info("EXL3XPU_WARMUP_VARIANTS rank=%s dense=%d moe=%d shapes=%d seconds=%.1f",
                getattr((b.dense + b.moe)[0], "exl3_tp_rank", "?"), n_dense, n_moe, len(keys), res["seconds"])
    return res
