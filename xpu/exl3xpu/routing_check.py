"""EXL3_ROUTING_CHECK: test-only rank-identity check of the MoE router's topk_ids (DESIGN M3 row 'routing
identical on all ranks').

Why: MoE-TP keeps every expert on every rank (narrowed to I/tp), so the router (gate) is replicated and every
rank must pick the same top-k experts for the same tokens. A rank that picks differently still produces a
well-formed partial output, and the single oneCCL all-reduce at the end of the MoE block then sums *different*
expert results: the failure is silent in the logits and only shows up as an accuracy gap or as a rank-dependent
token stream. This hook makes it visible: under TP > 1 in eager mode it all-gathers each MoE layer's topk_ids
over the TP group for the first 64 model steps and compares every rank's ids with the local ones.

What it does: with EXL3_ROUTING_CHECK=1, register() installs this module. It wraps
exl3xpu.moe_vllm.Exl3MoEMethod.apply, the single point where the router's topk_ids reach the routed-expert
kernel and where the layer index is available to group calls into model steps: apply() runs once per MoE layer
per forward, so a step boundary is the layer index wrapping back (li <= previous li), and a step also closes as
soon as its highest layer index (the model's last MoE layer) is seen again in a later step. topk_ids there are
the *logical* global ids the router picked (no EPLB on this engine), so the
comparison is of routing decisions, not of the per-rank local id mapping that map_ids() derives from them.

Comm: the all-gather goes through vLLM's TP group when it is initialized (get_tp_group().all_gather: the same
oneCCL group and backend the engine already uses for its TP collectives), else torch.distributed.all_gather on
the default group. Active only when the TP group has more than one rank (a pure-DP run has TP size 1 and its DP
ranks legitimately pick differently) and the run is eager (no torch.compile trace, no XPU graph capture): a
collective recorded into a graph would deadlock or replay stale ids. Zero
overhead when EXL3_ROUTING_CHECK is unset: install() returns before importing or patching anything.

Lockstep: every rank runs the same layers, so every rank calls the same number of collectives. After the first
mismatch the verdict is logged at once, but the hook keeps gathering (without comparing) until the 64-step
budget is exhausted, so one rank stopping early can never leave its peers blocked in a collective.

Output: per-step detail on the first mismatch, then the verdict when the budget is exhausted or at the first
mismatch:
    EXL3XPU_ROUTING_CHECK mismatch step=3 layer=7 rank=0 rows=2/8 first_row=1 local=[...] r0=[...] r1=[...]
    EXL3XPU_ROUTING_CHECK steps=64 identical=64 pass=True
pass is False on any mismatch and on a run that ends before the budget (atexit line).
"""
from __future__ import annotations

import atexit
import logging
import os
import re
import threading

import torch

logger = logging.getLogger("vllm.exl3xpu.routing_check")

ENV = "EXL3_ROUTING_CHECK"
STEPS = 64
MARK = "_exl3_routing_check_patched"


class State:
    """Per-rank counters of the check (one process, one engine)."""

    def __init__(self) -> None:
        self.step = 0               # current model step (1-based; a step is one pass over the MoE layers)
        self.prev_layer = None      # layer index of the previous apply() call, for the wrap detection
        self.max_layer = None       # highest layer index seen: the model's last MoE layer
        self.folded = 0             # last step whose result is already in identical/mismatch_steps
        self.cur_bad = False        # the current step has seen a mismatched row
        self.rows_in_step = 0       # rows compared in the current step (0 = the step counted nothing)
        self.identical_steps = 0
        self.mismatch_steps = 0
        self.rows_total = 0
        self.rows_mismatched = 0
        self.detail: list[str] = []  # the first-mismatch lines
        self.verdict: str | None = None
        self.error: str | None = None
        self.done = False           # the budget is exhausted (or a gather failed): no more collectives


STATE = State()
_lock = threading.Lock()
_installed = False
_atexit = False


# ------------------------------------------------------------------------------------------------ environment
def armed() -> bool:
    """The env gate alone (EXL3_ROUTING_CHECK=1)."""
    return os.environ.get(ENV, "0") == "1"


def eager() -> bool:
    """False while tracing (a collective would be captured) or while capturing an XPU graph (no collective)."""
    try:
        if torch._dynamo.is_compiling():
            return False
    except Exception:      # noqa: BLE001
        pass
    try:
        if torch.xpu.is_available() and torch.xpu.is_current_stream_capturing():
            return False
    except Exception:      # noqa: BLE001
        pass
    return True


def active() -> bool:
    """Armed, tensor-parallel over more than one rank, and eager: the three conditions the check needs."""
    if not armed():
        return False
    try:
        if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            return False
        if tp_world_size() <= 1:
            return False
    except Exception:      # noqa: BLE001
        return False
    return eager()


def tp_world_size() -> int:
    """The TP group's size: the group the router is replicated over. Falls back to the world size when vLLM's TP
    group is not available (a pure-DP run has TP size 1, and DP ranks legitimately pick differently)."""
    g = _tp_group()
    if g is None:
        return int(torch.distributed.get_world_size())
    return int(getattr(g, "world_size", 1))


def rank() -> int:
    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return int(torch.distributed.get_rank())
    except Exception:      # noqa: BLE001
        pass
    return 0


def layer_index(layer) -> int | None:
    """The decoder layer number of an apply() call; None for mtp/non-decoder layers (they never open a step)."""
    v = getattr(layer, "exl3_layer_idx", None)
    if v is not None:
        try:
            return int(v)
        except (TypeError, ValueError):
            pass
    prefix = getattr(layer, "exl3_prefix", None) or getattr(layer, "prefix", "") or ""
    m = re.search(r"layers\.(\d+)", str(prefix))
    return int(m.group(1)) if m else None


# --------------------------------------------------------------------------------------------------- the comm
def _tp_group():
    try:
        from vllm.distributed import get_tp_group
        return get_tp_group()
    except Exception:      # noqa: BLE001
        return None


def all_gather(t: torch.Tensor) -> torch.Tensor:
    """Every rank's copy of t, stacked [world, *t.shape] in rank order, over the engine's TP comm backend."""
    t = t.contiguous()
    g = _tp_group()
    ws = int(getattr(g, "world_size", 1)) if g is not None else 1
    if ws > 1:
        try:
            out = g.all_gather(t, dim=0)
            if tuple(out.shape) == (ws, *t.shape):
                return out
            if out.numel() == ws * t.numel() and out.shape[0] == ws * t.shape[0]:
                return out.reshape(ws, *t.shape)
        except Exception:      # noqa: BLE001 - fall through to the default group (same backend, same ranks here)
            pass
    n = int(torch.distributed.get_world_size())
    parts = [torch.empty_like(t) for _ in range(n)]
    torch.distributed.all_gather(parts, t)
    return torch.stack(parts)


def compare(stacked: torch.Tensor, local: torch.Tensor) -> tuple[int, int, int]:
    """(rows compared, rows where any rank differs from local, first mismatched row or -1)."""
    w = stacked.shape[0]
    flat = stacked.reshape(w, -1)
    bad = (flat != local.detach().reshape(-1).unsqueeze(0)).any(dim=0)      # [numel]
    rows = bad.reshape(local.shape).any(dim=-1) if local.dim() >= 2 else bad
    nbad = int(rows.sum().item())
    first = int(torch.nonzero(rows, as_tuple=False)[0].item()) if nbad else -1
    return int(rows.numel()), nbad, first


def _row(t: torch.Tensor, i: int) -> list[int]:
    r = t[i] if t.dim() >= 2 else t[i:i + 1]
    return [int(x) for x in r.detach().reshape(-1).cpu().tolist()]


def detail_line(step: int, li: int | None, r: int, stacked: torch.Tensor, local: torch.Tensor,
                nbad: int, nrows: int, first: int) -> str:
    """One line with the first mismatched row as every rank picked it, local rank first."""
    parts = [f"r{i}={_row(stacked[i], first)}" for i in range(stacked.shape[0])]
    return (f"EXL3XPU_ROUTING_CHECK mismatch step={step} layer={li} rank={r} rows={nbad}/{nrows} "
            f"first_row={first} local={_row(local, first)} " + " ".join(parts))


# ----------------------------------------------------------------------------------------------------- state
def _fold_locked(quiet: bool) -> None:
    """Close the current step: count it identical or mismatched (once per step). `quiet` = a verdict was already
    reported, so the step only has to advance the budget without touching the counters."""
    if STATE.step > STATE.folded and (STATE.rows_in_step or quiet):
        STATE.folded = STATE.step
        if not quiet:
            if STATE.cur_bad:
                STATE.mismatch_steps += 1
            else:
                STATE.identical_steps += 1
        STATE.cur_bad = False


def _verdict_locked(exhausted: bool) -> None:
    if STATE.verdict is not None:       # the first-mismatch verdict stands; exhaustion only closes the hook
        STATE.done = True
        return
    steps = STEPS if exhausted else STATE.step
    identical = STATE.identical_steps
    ok = exhausted and identical == STEPS and STATE.mismatch_steps == 0 and STATE.error is None
    STATE.verdict = f"EXL3XPU_ROUTING_CHECK steps={steps} identical={identical} pass={ok}"
    if exhausted:
        STATE.done = True
    logger.info("%s", STATE.verdict)


def observe(layer, topk_ids, r: int | None = None) -> bool:
    """All-gather one MoE layer's topk_ids and compare with the local ones; False when the check is not running."""
    if not active():
        return False
    if topk_ids is None or not torch.is_tensor(topk_ids) or topk_ids.numel() == 0:
        return False
    li = layer_index(layer)
    with _lock:
        if STATE.done:
            return False
        if STATE.prev_layer is None or (li is not None and li <= STATE.prev_layer):
            _fold_locked(quiet=STATE.verdict is not None)      # the layer sequence wrapped: the step ended
            STATE.step += 1
            STATE.rows_in_step = 0
        prev_max = STATE.max_layer
        if li is not None:
            STATE.prev_layer = li
            if prev_max is None or li > prev_max:
                STATE.max_layer = li
        if STATE.step > STEPS:
            _verdict_locked(exhausted=True)
            return False
        step, quiet = STATE.step, STATE.verdict is not None
    r = rank() if r is None else int(r)
    local = topk_ids.detach()
    try:
        stacked = all_gather(local)
    except Exception as e:     # noqa: BLE001 - a broken collective must not look like a routing mismatch
        with _lock:
            STATE.error = f"{type(e).__name__}: {e}"
            _verdict_locked(exhausted=False)
            STATE.done = True
        logger.warning("EXL3XPU_ROUTING_CHECK all-gather failed, check disabled: %s", STATE.error)
        return False
    if not quiet:
        nrows, nbad, first = compare(stacked, local)
        with _lock:
            STATE.rows_total += nrows
            STATE.rows_mismatched += nbad
            STATE.rows_in_step += nrows
            if nbad:
                STATE.cur_bad = True
                STATE.detail.append(detail_line(step, li, r, stacked, local, nbad, nrows, first))
                if STATE.verdict is None:
                    logger.info("%s", STATE.detail[-1])
                    _verdict_locked(exhausted=False)     # first mismatch: report it now, keep gathering
    with _lock:
        # the highest layer index seen is the model's last MoE layer: seeing it again closes the step, so the
        # verdict lands at the end of step 64 instead of only when a 65th step starts.
        if li is not None and prev_max is not None and li == prev_max and STATE.step > 1:
            _fold_locked(quiet=STATE.verdict is not None)
        if STATE.folded >= STEPS:
            _verdict_locked(exhausted=True)
    return True


def verdict() -> bool:
    with _lock:
        return bool(STATE.verdict) and STATE.verdict.endswith("pass=True")


def report() -> list[str]:
    """The first-mismatch lines, then the verdict (or an 'incomplete' line for a run that ended early)."""
    with _lock:
        lines = list(STATE.detail)
        if STATE.error:
            lines.append(f"EXL3XPU_ROUTING_CHECK error={STATE.error}")
        if STATE.verdict:
            lines.append(STATE.verdict)
        elif STATE.step:
            lines.append(f"EXL3XPU_ROUTING_CHECK incomplete steps={STATE.folded} of {STEPS} "
                         f"identical={STATE.identical_steps}")
    return lines


def flush() -> list[str]:
    """atexit (and tests): fold the step still in progress, then log everything that was collected."""
    with _lock:
        if STATE.verdict is None:
            _fold_locked(quiet=False)
    lines = report()
    with _lock:
        if STATE.verdict is None and STATE.folded:
            lines.append(f"EXL3XPU_ROUTING_CHECK steps={STATE.folded} identical={STATE.identical_steps} "
                         f"pass=False")
    prev, logging.raiseExceptions = logging.raiseExceptions, False   # at exit the log stream may be closed
    try:
        for ln in lines:
            logger.info("%s", ln)
    finally:
        logging.raiseExceptions = prev
    return lines


def reset() -> None:
    global STATE
    with _lock:
        STATE = State()


# ---------------------------------------------------------------------------------------------------- install
def install() -> bool:
    """Wrap Exl3MoEMethod.apply when EXL3_ROUTING_CHECK=1 (idempotent). False when the env is unset or the
    MoE method is not importable."""
    global _installed, _atexit
    if not armed():
        return False
    if _installed:
        return True
    try:
        from . import moe_vllm
    except Exception as e:     # noqa: BLE001
        logger.warning("EXL3XPU_ROUTING_CHECK install failed: moe_vllm not importable (%s)", e)
        return False
    cls = moe_vllm.Exl3MoEMethod
    cur = cls.apply
    if not getattr(cur, MARK, False):
        def apply(self, *args, **kwargs):
            try:
                layer = kwargs.get("layer", args[0] if args else None)
                topk_ids = kwargs.get("topk_ids", args[3] if len(args) > 3 else None)
                if layer is not None and topk_ids is not None:
                    observe(layer, topk_ids)
            except Exception:      # noqa: BLE001 - a check must never break serving
                pass
            return cls._exl3_routing_check_orig_apply(self, *args, **kwargs)

        setattr(apply, MARK, True)
        cls._exl3_routing_check_orig_apply = cur
        cls.apply = apply
    _installed = True
    if not _atexit:
        atexit.register(flush)
        _atexit = True
    logger.info("EXL3XPU_ROUTING_CHECK installed: MoE router topk_ids all-gathered over the TP group for the "
                "first %d eager steps under TP > 1, pass if every rank picks identically", STEPS)
    return True


def uninstall() -> None:
    """Restore the stock Exl3MoEMethod.apply (tests, and a re-install after reset())."""
    global _installed
    try:
        from . import moe_vllm
        cls = moe_vllm.Exl3MoEMethod
        if getattr(cls.apply, MARK, False):
            cls.apply = cls._exl3_routing_check_orig_apply
            del cls._exl3_routing_check_orig_apply
    except Exception:      # noqa: BLE001
        pass
    _installed = False


def installed() -> bool:
    return bool(_installed)
