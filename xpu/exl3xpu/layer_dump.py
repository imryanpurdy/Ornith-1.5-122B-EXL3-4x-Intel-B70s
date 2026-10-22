"""EXL3_LAYER_DUMP=<dir>: test-only per-layer hidden-state dump (DESIGN M3 row 'per-layer hidden states').

Why: the M3 row reads 'decoder-layer forward hooks on hidden_states and residual, 16 prompts vs TP1 | layer 0
sets the scale: pass = each layer's rel-RMS <= 3x layer 0's, and layer 0 <= 5e-3'. rel-RMS is measured offline
from the tensors this hook writes. The residual stream entering layer L is layer L-1's output, so with every
layer's output stored, rel-RMS(L) = rms(out_L - out_{L-1}) / rms(out_{L-1}) for L > 0 and layer 0 is taken
against the embedding output. Two runs (e.g. TP2 against TP1) are then compared layer by layer, step by step,
from the same files.

What it does: with EXL3_LAYER_DUMP=<dir>, register() installs this module. install() puts in one *global*
forward pre-hook (torch.nn.modules.module.register_module_forward_pre_hook) purely for discovery: the first
module whose forward runs is the model, its named_modules() are walked once for decoder layers (names matching
`layers.<N>`, no mtp, at least one child module), and each decoder layer then gets its own torch forward hook
(nn.Module.register_forward_hook) - forward is never monkeypatched. The discovery pre-hook is removed as soon
as the walk is done, so nothing global stays installed.

Capture: per layer call the output is split into hidden_states and the residual stream. A layer returning a
tuple (hidden_states, residual) gives both; a layer returning a single tensor gives that tensor for both,
because in this model's decoder layers the returned hidden_states *is* the post-layer residual (residual =
hidden_states after the layer's add). Tensors are detached to CPU in their own dtype.

Budget: the first STEPS=16 forward passes, one pass being one sweep over the decoder layers (a pass ends when
the layer index wraps back). The pass that ends at the model's highest layer index of pass 16 removes every
hook and writes the files, so a long run pays nothing after the first 16 passes; a 17th pass would find no
hooks at all. atexit writes whatever was captured when a run ends earlier than the budget.

Eager only: a forward hook cannot be replayed out of an XPU graph (moving a captured tensor to CPU inside a
capture is illegal), so install() refuses when graphs are enabled - torch.xpu.is_current_stream_capturing(), or
vLLM's compilation config asking for a cudagraph mode other than NONE - and logs why; the hook body also skips
a call while torch.compile is tracing or a graph is being captured, in case the mode changes after install.

Output: one file per layer per rank, <dir>/layer{L}_rank{R}.pt, a dict
    {"hidden": stacked, "residual": stacked, "meta": {"layer", "rank", "dtype", "steps", "stacked",
                                                      "shape", "name"}}
where stacked is [steps, tokens, hidden] when every captured step has the same token count (else a list of
per-step tensors with meta['stacked'] = False: prefill chunks differ in length). Rank comes from
torch.distributed when initialized, else 0, so a 2-card run writes layer{L}_rank0.pt and layer{L}_rank1.pt side
by side for the offline comparison. One summary line at shutdown:
    EXL3XPU_LAYER_DUMP layers=62 steps=16 dir=/dumps files=62
"""
from __future__ import annotations

import atexit
import logging
import os
import re
import threading

import torch

logger = logging.getLogger("vllm.exl3xpu.layer_dump")

ENV = "EXL3_LAYER_DUMP"
STEPS = 16
LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)$")
MTP_RE = re.compile(r"(?:^|\.)mtp(?:\.|$)")
GRAPH_MODES_OFF = {"", "none", "cudagraphmode.none", "0", "n/a"}
# per-layer cap on the captured bytes (16 passes of a long prefill would otherwise hold gigabytes per layer);
# EXL3_LAYER_DUMP_MAX_BYTES overrides it, 0 disables the cap.
MAX_BYTES = int(os.environ.get("EXL3_LAYER_DUMP_MAX_BYTES", str(512 << 20)))


class State:
    """Per-rank state of the dump (one process, one engine)."""

    def __init__(self) -> None:
        self.step = 0                  # current forward pass (1-based)
        self.prev_layer: int | None = None   # layer index of the previous hook call, for the wrap detection
        self.max_layer: int | None = None    # highest layer index seen: the model's last decoder layer
        self.layers: dict[int, dict[str, list]] = {}   # layer -> {"hidden": [...], "residual": [...]}
        self.names: dict[int, str] = {}     # layer -> module name it was attached to
        self.handles: list = []             # per-layer forward-hook handles
        self.discovery = None               # the global pre-hook handle, until the layers are attached
        self.calls = 0                      # layer calls stored
        self.skipped_bytes = 0              # bytes dropped by the per-layer cap
        self.done = False                   # the budget is spent (or the files are written): nothing more
        self.saved: list[str] = []          # paths written by the last save()
        self.error: str | None = None
        self.graphs_refused = False         # install() refused because graphs are enabled


STATE = State()
_lock = threading.RLock()
_installed = False
_atexit = False


# ------------------------------------------------------------------------------------------------ environment
def armed() -> bool:
    """EXL3_LAYER_DUMP holds a directory path ('1' and '0' are not directories, so they are not armed)."""
    v = os.environ.get(ENV, "").strip()
    return bool(v) and v not in ("0", "1")


def dump_dir() -> str:
    return os.environ.get(ENV, "").strip()


def capturing() -> bool:
    """An XPU graph is being captured right now."""
    try:
        return bool(torch.xpu.is_available() and torch.xpu.is_current_stream_capturing())
    except Exception:      # noqa: BLE001
        return False


def cudagraph_mode() -> str | None:
    """vLLM's compilation config cudagraph mode, when one exists (None before the engine is built)."""
    try:
        from vllm.config import get_current_vllm_config
        cfg = get_current_vllm_config()
        cc = getattr(cfg, "compilation_config", None)
        m = getattr(cc, "cudagraph_mode", None)
        if m is None:
            return None
        return str(getattr(m, "value", m))
    except Exception:      # noqa: BLE001
        return None


def graphs_enabled() -> bool:
    """Capturing now, or a cudagraph mode other than NONE: a forward hook would be captured or replayed."""
    if capturing():
        return True
    m = cudagraph_mode()
    return m is not None and m.strip().lower() not in GRAPH_MODES_OFF


def eager() -> bool:
    """False while torch.compile is tracing or a graph is being captured: no tensor may leave the device."""
    try:
        if torch._dynamo.is_compiling():
            return False
    except Exception:      # noqa: BLE001
        pass
    return not capturing()


def rank() -> int:
    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return int(torch.distributed.get_rank())
    except Exception:      # noqa: BLE001
        pass
    return 0


# --------------------------------------------------------------------------------------------------- capture
def split_output(output):
    """(hidden_states, residual) of a decoder layer's return: a tuple gives both, a single tensor gives it twice."""
    if isinstance(output, (tuple, list)):
        if not output:
            return None, None
        h = output[0] if torch.is_tensor(output[0]) else None
        r = output[1] if len(output) > 1 and torch.is_tensor(output[1]) else h
        return h, r
    if torch.is_tensor(output):
        return output, output
    return None, None


def _cpu(t: torch.Tensor) -> torch.Tensor:
    return t.detach().to("cpu")


def record(li: int, output, r: int | None = None) -> bool:
    """Store one layer call's hidden_states and residual; False when the call is not captured."""
    if not eager():
        return False
    h, res = split_output(output)
    if h is None:
        return False
    with _lock:
        if STATE.done:
            return False
        if STATE.prev_layer is None or li <= STATE.prev_layer:
            STATE.step += 1                  # the layer sequence wrapped: a new forward pass started
        STATE.prev_layer = li
        if STATE.max_layer is None or li > STATE.max_layer:
            STATE.max_layer = li
        over = STATE.step > STEPS
        if not over:
            slot = STATE.layers.setdefault(li, {"hidden": [], "residual": []})
            add = int(h.numel()) * h.element_size() + int(res.numel()) * res.element_size()
            held = sum(int(t.numel()) * t.element_size() for t in slot["hidden"])
            if MAX_BYTES and held + add > MAX_BYTES:
                STATE.skipped_bytes += add
                return False
            hc = _cpu(h)
            rc = hc if res is h else _cpu(res)
            slot["hidden"].append(hc)
            slot["residual"].append(rc)
            STATE.calls += 1
        spent = STATE.step == STEPS and li == STATE.max_layer     # the last layer of pass STEPS has run
    if over:
        uninstall()                          # a 17th pass: the budget is spent, take the hooks off
        return False
    if spent:
        uninstall()
    return not over


def _make_hook(li: int):
    def hook(module, args, output):
        try:
            record(li, output)
        except Exception:      # noqa: BLE001 - a measurement must never break serving
            pass
    return hook


# ---------------------------------------------------------------------------------------------------- layers
def decoder_layers(model) -> list[tuple[int, torch.nn.Module, str]]:
    """(index, module, name) for every decoder layer of `model`: `layers.<N>` names, no mtp, with children."""
    out = []
    for name, m in model.named_modules():
        if MTP_RE.search(name):
            continue
        hit = LAYER_RE.search(name)
        if hit is None:
            continue
        if not any(True for _ in m.children()):
            continue
        out.append((int(hit.group(1)), m, name))
    out.sort(key=lambda t: t[0])
    return out


def attach(model) -> int:
    """Attach one forward hook per decoder layer of `model` (idempotent per layer); returns the hook count."""
    with _lock:
        if STATE.done:
            return len(STATE.handles)
        for li, m, name in decoder_layers(model):
            if li in STATE.names:
                continue
            STATE.names[li] = name
            STATE.layers.setdefault(li, {"hidden": [], "residual": []})
            STATE.handles.append(m.register_forward_hook(_make_hook(li)))
        _drop_discovery()
        return len(STATE.handles)


def _discover(module, args):
    """Global pre-hook, discovery only: the first module whose forward runs is the model, so attach from it."""
    try:
        if STATE.done:
            _drop_discovery()
            return
        if decoder_layers(module):
            attach(module)
    except Exception:      # noqa: BLE001
        pass


def _drop_discovery() -> None:
    if STATE.discovery is not None:
        try:
            STATE.discovery.remove()
        except Exception:      # noqa: BLE001
            pass
        STATE.discovery = None


def discovered() -> list[int]:
    return sorted(STATE.names)


def discovery_installed() -> bool:
    return STATE.discovery is not None


# ------------------------------------------------------------------------------------------------------ files
def _stacked(items):
    """[steps, ...] when every step has the same shape, else the per-step list (meta['stacked'] = False)."""
    if not items:
        return None, True
    if len({tuple(t.shape) for t in items}) == 1:
        return torch.stack(items), True
    return list(items), False


def save(path_dir: str | None = None) -> list[str]:
    """Write one layer{L}_rank{R}.pt per captured layer; returns the paths (idempotent until reset())."""
    with _lock:
        d = path_dir or dump_dir()
        if not d:
            return []
        r = rank()
        paths: list[str] = []
        try:
            os.makedirs(d, exist_ok=True)
            for li in sorted(STATE.layers):
                slot = STATE.layers[li]
                if not slot["hidden"]:
                    continue
                hs, stacked = _stacked(slot["hidden"])
                rs, _ = _stacked(slot["residual"])
                meta = {"layer": int(li), "rank": int(r), "dtype": str(slot["hidden"][0].dtype),
                        "steps": len(slot["hidden"]), "stacked": bool(stacked),
                        "shape": list(slot["hidden"][0].shape), "name": STATE.names.get(li, "")}
                p = os.path.join(d, f"layer{li}_rank{r}.pt")
                tmp = p + ".tmp"
                torch.save({"hidden": hs, "residual": rs, "meta": meta}, tmp)
                os.replace(tmp, p)                     # never leave a half-written file behind
                paths.append(p)
        except Exception as e:     # noqa: BLE001 - a dump must never break serving
            STATE.error = f"{type(e).__name__}: {e}"
            logger.warning("EXL3XPU_LAYER_DUMP save failed under %r: %s", d, STATE.error)
        STATE.saved = paths
        return paths


def summary_lines() -> list[str]:
    with _lock:
        n = sum(1 for v in STATE.layers.values() if v["hidden"])
        lines = [f"EXL3XPU_LAYER_DUMP layers={n} steps={min(STATE.step, STEPS)} "
                 f"dir={dump_dir() or '<unset>'} files={len(STATE.saved)}"]
        if STATE.skipped_bytes:
            lines.append(f"EXL3XPU_LAYER_DUMP capped: {STATE.skipped_bytes} bytes dropped over "
                         f"{MAX_BYTES} bytes per layer")
        if STATE.error:
            lines.append(f"EXL3XPU_LAYER_DUMP error={STATE.error}")
    return lines


def flush() -> list[str]:
    """atexit (and tests): write whatever was captured, then log the summary and every file path."""
    with _lock:
        if not STATE.saved and any(v["hidden"] for v in STATE.layers.values()):
            save()
        lines = summary_lines()
        lines += [f"EXL3XPU_LAYER_DUMP file={p}" for p in STATE.saved]
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
    """Install the per-layer dump when EXL3_LAYER_DUMP=<dir> (idempotent). False when unset, when graphs are
    enabled, or when the dump directory is not writable."""
    global _installed, _atexit
    if not armed():
        return False
    if _installed:
        return True
    if graphs_enabled():
        STATE.graphs_refused = True
        why = "a capture is in progress" if capturing() else f"cudagraph_mode={cudagraph_mode()}"
        logger.warning("EXL3XPU_LAYER_DUMP not installed: XPU graphs are enabled (%s); this hook needs an eager "
                       "run (--enforce-eager), a forward hook cannot be captured into a graph", why)
        return False
    d = dump_dir()
    try:
        os.makedirs(d, exist_ok=True)
    except Exception as e:     # noqa: BLE001
        logger.warning("EXL3XPU_LAYER_DUMP not installed: dump dir %r is not writable (%s)", d, e)
        return False
    STATE.discovery = torch.nn.modules.module.register_module_forward_pre_hook(_discover)
    _installed = True
    if not _atexit:
        atexit.register(flush)
        _atexit = True
    logger.info("EXL3XPU_LAYER_DUMP installed: per-layer hidden_states and residual stream for the first %d "
                "eager forward passes, one layer{L}_rank{R}.pt per layer per rank under %s", STEPS, d)
    return True


def uninstall() -> None:
    """Remove every hook and write the files captured so far (the budget's end, tests, atexit)."""
    global _installed
    with _lock:
        for h in STATE.handles:
            try:
                h.remove()
            except Exception:      # noqa: BLE001
                pass
        STATE.handles = []
        _drop_discovery()
        _installed = False
        STATE.done = True
        if not STATE.saved and any(v["hidden"] for v in STATE.layers.values()):
            save()


def installed() -> bool:
    return bool(_installed)
