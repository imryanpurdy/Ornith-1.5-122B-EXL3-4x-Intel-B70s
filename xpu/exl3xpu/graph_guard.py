"""Never capture an XPU graph while serving (DESIGN rev 2, sections 4.2 and 4.3).

Installed by the plugin's register() in every vLLM process. Each hook is applied when its target module is
imported (immediately if it already is), so importing this module never pulls in vLLM's worker code.

  G1  Worker.compile_or_warm_up_model (vllm.v1.worker.gpu_worker; XPUWorker inherits it). After it returns, per
      rank: torch.xpu.is_current_stream_capturing() must be False, every XPUGraph.capture_begin must have a
      matching capture_end, and torch.xpu.synchronize() must succeed (it raises on a queue that is still
      recording). Any failure logs EXL3XPU_OPEN_CAPTURE_AT_WARMUP and fails the boot. Otherwise it logs
      EXL3XPU_WARMUP_DONE rank= runner= captures= graphs= and enters SERVING; it also calls
      vllm.compilation.monitor.set_cudagraph_capturing_enabled(False) (only CUDAGraphWrapper and the breakable
      wrapper read that flag; it is a no-op for FULL_DECODE_ONLY's direct capture).
  G2  torch.xpu.graphs.graph.__enter__ (vLLM aliases torch.cuda.graph / CUDAGraph to the XPU classes,
      vllm/v1/worker/xpu_model_runner.py _torch_cuda_wrapper). While SERVING it logs
      EXL3XPU_CAPTURE_AFTER_WARMUP and raises BEFORE the original __enter__ runs its synchronize(), empty_cache(),
      stream switch and capture_begin, so no stream context or recording is left open. XPUGraph.capture_begin /
      capture_end are wrapped to count captures exactly (and capture_begin also refuses while SERVING, for a
      direct call that bypasses graph.__enter__).
  G3  The capture miss paths of CUDAGraphWrapper.__call__ (vllm.compilation.cuda_graph) and
      BreakableCUDAGraphWrapper._capture (vllm.compilation.breakable_cudagraph): when capturing is disabled, run
      the runnable eagerly and log EXL3XPU_EAGER_MISS <descriptor> once per descriptor, instead of vLLM's raise.
      Never on the FULL_DECODE_ONLY path (V2 dispatches uncaptured shapes to NONE); covers piecewise/encoder.
  G4  The V2 runner's GPUModelRunner.execute_model (vllm.v1.worker.gpu.model_runner; XPUModelRunnerV2 inherits
      it). At entry: if torch.xpu.is_current_stream_capturing(), log EXL3XPU_OPEN_CAPTURE and raise.

A G2 or G4 trip kills the engine (EngineCore treats a worker exception as fatal). That is intended.

Self-test (M2c, one card): EXL3_GUARD_SELFTEST=1 makes the first real request after warmup attempt one
torch.xpu.graph capture inside execute_model. Expected: EXL3XPU_CAPTURE_AFTER_WARMUP, then EngineDeadError and a
clean container exit. If the capture is not blocked it logs EXL3XPU_GUARD_SELFTEST_FAILED and raises anyway.
"""
from __future__ import annotations

import functools
import importlib.abc
import importlib.machinery
import logging
import os
import sys
import threading
import traceback
from typing import Any, Callable

logger = logging.getLogger("vllm.exl3xpu.graph_guard")

# ------------------------------------------------------------------------------------------------------- state
SERVING = False                 # set by G1 at the end of warmup, per process (= per rank)
RANK: Any = None                # worker rank recorded by G1
COUNTS = {"enter": 0, "begin": 0, "end": 0, "blocked": 0, "eager_miss": 0, "open_capture": 0}
SELFTEST = False                # EXL3_GUARD_SELFTEST=1, read by install()
_selftest_done = False
_eager_miss_seen: set = set()
_installed: dict[str, bool] = {}
_lock = threading.Lock()
_MARK = "_exl3_graph_guard"

HOOK_MODULES = {
    "G1": "vllm.v1.worker.gpu_worker",
    "G2": "torch.xpu.graphs",
    "G3a": "vllm.compilation.cuda_graph",
    "G3b": "vllm.compilation.breakable_cudagraph",
    "G4": "vllm.v1.worker.gpu.model_runner",
}


class GraphGuardError(RuntimeError):
    """An XPU graph capture was attempted, or left open, outside warmup."""


def capture_count() -> int:
    return COUNTS["begin"]


def reset_for_tests() -> None:
    global SERVING, RANK, _selftest_done
    SERVING, RANK, _selftest_done = False, None, False
    for k in COUNTS:
        COUNTS[k] = 0
    _eager_miss_seen.clear()


def _short_stack(skip: int = 2, depth: int = 8) -> str:
    frames = traceback.extract_stack()[:-skip][-depth:]
    return " <- ".join(f"{os.path.basename(f.filename)}:{f.lineno}:{f.name}" for f in reversed(frames))


def _is_patched(fn) -> bool:
    return bool(getattr(fn, _MARK, False))


def _mark(fn, orig):
    setattr(fn, _MARK, True)
    fn.__wrapped__ = orig
    return fn


# ---------------------------------------------------------------------------------------------------- G2 hooks
def _patch_torch_xpu_graphs(mod) -> None:
    graph_cls = getattr(mod, "graph", None)
    xgraph = getattr(mod, "XPUGraph", None)
    if graph_cls is None or xgraph is None:
        logger.warning("EXL3XPU_GRAPH_GUARD G2 not installed: %s has no graph/XPUGraph", mod.__name__)
        return
    orig_enter = graph_cls.__enter__
    if not _is_patched(orig_enter):
        def __enter__(self):
            if SERVING:
                COUNTS["blocked"] += 1
                logger.error("EXL3XPU_CAPTURE_AFTER_WARMUP rank=%s via=graph.__enter__ stack=%s", RANK, _short_stack())
                raise GraphGuardError("EXL3XPU_CAPTURE_AFTER_WARMUP: XPU graph capture attempted while serving "
                                      f"(rank {RANK}); refused before synchronize/stream switch")
            COUNTS["enter"] += 1
            return orig_enter(self)
        graph_cls.__enter__ = _mark(functools.update_wrapper(__enter__, orig_enter), orig_enter)

    orig_begin = xgraph.capture_begin
    if not _is_patched(orig_begin):
        def capture_begin(self, *args, **kwargs):
            if SERVING:
                COUNTS["blocked"] += 1
                logger.error("EXL3XPU_CAPTURE_AFTER_WARMUP rank=%s via=XPUGraph.capture_begin stack=%s", RANK,
                             _short_stack())
                raise GraphGuardError(f"EXL3XPU_CAPTURE_AFTER_WARMUP: XPUGraph.capture_begin while serving (rank {RANK})")
            out = orig_begin(self, *args, **kwargs)
            COUNTS["begin"] += 1
            return out
        xgraph.capture_begin = _mark(functools.update_wrapper(capture_begin, orig_begin), orig_begin)

    orig_end = xgraph.capture_end
    if not _is_patched(orig_end):
        def capture_end(self, *args, **kwargs):
            out = orig_end(self, *args, **kwargs)
            COUNTS["end"] += 1
            return out
        xgraph.capture_end = _mark(functools.update_wrapper(capture_end, orig_end), orig_end)
    _installed["G2"] = True


# ---------------------------------------------------------------------------------------------------- G1 hooks
def _graphs_of(mgr) -> int:
    g = getattr(mgr, "graphs", None)
    if isinstance(g, dict):
        return len(g)
    bg = getattr(mgr, "budget_graphs", None)            # EncoderCudaGraphManager
    if isinstance(bg, dict):
        return sum(len(v) for v in bg.values() if isinstance(v, dict))
    return 0


def count_graphs(runner) -> tuple[int, dict[str, int]]:
    """Captured graphs held by the runner's, the speculator's and the encoder's graph managers, plus any
    CUDAGraphWrapper entries (V1, piecewise). Returns (total, per-source detail). Breakable-wrapper captures
    (one capture_begin per segment) are not counted here, so they show up as a captures/graphs mismatch; that
    wrapper is only used for PIECEWISE without torch.compile, which the Ornith lane never runs."""
    detail: dict[str, int] = {}
    seen: set[int] = set()

    def add(label, mgr):
        if mgr is None or id(mgr) in seen:
            return
        seen.add(id(mgr))
        detail[label] = _graphs_of(mgr)

    if runner is not None:
        add("model", getattr(runner, "cudagraph_manager", None))
        spec = getattr(runner, "speculator", None)
        if spec is not None:
            for name, val in sorted(vars(spec).items()):
                if type(val).__name__.endswith("CudaGraphManager"):
                    add(f"spec.{name}", val)
        enc = getattr(getattr(runner, "model_state", None), "encoder_runner", None)
        add("encoder", getattr(enc, "cudagraph_manager", None))
    cg = sys.modules.get("vllm.compilation.cuda_graph")
    if cg is not None and hasattr(cg, "CUDAGraphWrapper"):
        n = 0
        for w in list(getattr(cg.CUDAGraphWrapper, "_all_instances", ()) or ()):
            n += sum(1 for e in getattr(w, "concrete_cudagraph_entries", {}).values()
                     if getattr(e, "cudagraph", None) is not None)
        if n:
            detail["wrappers"] = n
    return sum(detail.values()), detail


def _open_capture_reason() -> str | None:
    import torch
    try:
        if torch.xpu.is_current_stream_capturing():
            return "current stream is capturing"
    except Exception as e:                                          # noqa: BLE001
        return f"is_current_stream_capturing raised {e!r}"
    open_n = COUNTS["begin"] - COUNTS["end"]
    if open_n:
        return f"{open_n} XPUGraph.capture_begin without capture_end"
    try:
        torch.xpu.synchronize()
    except Exception as e:                                          # noqa: BLE001
        return f"torch.xpu.synchronize raised {e!r}"
    return None


def after_warmup(worker) -> None:
    """G1 body, run after compile_or_warm_up_model returned on one rank."""
    global SERVING, RANK
    RANK = getattr(worker, "rank", None)
    reason = _open_capture_reason()
    if reason is not None:
        COUNTS["open_capture"] += 1
        logger.error("EXL3XPU_OPEN_CAPTURE_AT_WARMUP rank=%s reason=%s", RANK, reason)
        raise GraphGuardError(f"EXL3XPU_OPEN_CAPTURE_AT_WARMUP rank={RANK}: {reason}; failing the boot")
    runner = getattr(worker, "model_runner", None)
    graphs, detail = count_graphs(runner)
    captures = COUNTS["begin"]
    hooks = ",".join(k for k in HOOK_MODULES if _installed.get(k))
    logger.info("EXL3XPU_WARMUP_DONE rank=%s runner=%s captures=%d graphs=%d detail=%s hooks=%s", RANK,
                type(runner).__name__, captures, graphs,
                ",".join(f"{k}:{v}" for k, v in detail.items()) or "-", hooks or "-")
    if captures != graphs:
        logger.warning("EXL3XPU_CAPTURE_COUNT_MISMATCH rank=%s captures=%d graphs=%d", RANK, captures, graphs)
    try:
        from vllm.compilation.monitor import set_cudagraph_capturing_enabled
        set_cudagraph_capturing_enabled(False)
    except Exception as e:                                          # noqa: BLE001
        logger.warning("EXL3XPU_GRAPH_GUARD could not disable cudagraph capturing: %r", e)
    SERVING = True


def _patch_gpu_worker(mod) -> None:
    worker_cls = getattr(mod, "Worker", None)
    if worker_cls is None or not hasattr(worker_cls, "compile_or_warm_up_model"):
        logger.warning("EXL3XPU_GRAPH_GUARD G1 not installed: %s.Worker.compile_or_warm_up_model missing", mod.__name__)
        return
    orig = worker_cls.compile_or_warm_up_model
    if not _is_patched(orig):
        def compile_or_warm_up_model(self, *args, **kwargs):
            out = orig(self, *args, **kwargs)
            after_warmup(self)
            return out
        worker_cls.compile_or_warm_up_model = _mark(functools.update_wrapper(compile_or_warm_up_model, orig), orig)
    _installed["G1"] = True


# ---------------------------------------------------------------------------------------------------- G3 hooks
def _capturing_enabled() -> bool:
    mon = sys.modules.get("vllm.compilation.monitor")
    return True if mon is None else bool(getattr(mon, "cudagraph_capturing_enabled", True))


def _log_eager_miss(where: str, desc) -> None:
    key = (where, repr(desc))
    COUNTS["eager_miss"] += 1
    if key not in _eager_miss_seen:
        _eager_miss_seen.add(key)
        logger.warning("EXL3XPU_EAGER_MISS rank=%s where=%s desc=%r", RANK, where, desc)


def _patch_cuda_graph(mod) -> None:
    wcls = getattr(mod, "CUDAGraphWrapper", None)
    if wcls is None:
        logger.warning("EXL3XPU_GRAPH_GUARD G3a not installed: %s has no CUDAGraphWrapper", mod.__name__)
        return
    orig = wcls.__call__
    if not _is_patched(orig):
        def __call__(self, *args, **kwargs):
            if not _capturing_enabled() and mod.is_forward_context_available():
                fc = mod.get_forward_context()
                mode = fc.cudagraph_runtime_mode
                if mode != mod.CUDAGraphMode.NONE and mode == self.runtime_mode:
                    desc = fc.batch_descriptor
                    entry = self.concrete_cudagraph_entries.get(desc)
                    if entry is None or entry.cudagraph is None:
                        _log_eager_miss("CUDAGraphWrapper", desc)
                        return self.runnable(*args, **kwargs)
            return orig(self, *args, **kwargs)
        wcls.__call__ = _mark(functools.update_wrapper(__call__, orig), orig)
    _installed["G3a"] = True


def _patch_breakable(mod) -> None:
    wcls = getattr(mod, "BreakableCUDAGraphWrapper", None)
    if wcls is None or not hasattr(wcls, "_capture"):
        logger.warning("EXL3XPU_GRAPH_GUARD G3b not installed: %s has no BreakableCUDAGraphWrapper._capture",
                       mod.__name__)
        return
    orig = wcls._capture
    if not _is_patched(orig):
        def _capture(self, entry, args, kwargs):
            if not _capturing_enabled():
                _log_eager_miss("BreakableCUDAGraphWrapper", getattr(entry, "batch_descriptor", None))
                return self.runnable(*args, **kwargs)
            return orig(self, entry, args, kwargs)
        wcls._capture = _mark(functools.update_wrapper(_capture, orig), orig)
    _installed["G3b"] = True


# ---------------------------------------------------------------------------------------------------- G4 hooks
def _is_real_request(scheduler_output, args, kwargs) -> bool:
    dummy = kwargs.get("dummy_run", args[1] if len(args) > 1 else False)
    if dummy:
        return False
    return int(getattr(scheduler_output, "total_num_scheduled_tokens", 0) or 0) > 0


def run_selftest() -> None:
    """Attempt one capture while serving. G2 must raise inside torch.xpu.graph.__enter__."""
    global _selftest_done
    _selftest_done = True
    import torch
    logger.warning("EXL3XPU_GUARD_SELFTEST rank=%s: attempting one torch.xpu.graph capture while serving", RANK)
    g = torch.xpu.XPUGraph()
    with torch.xpu.graph(g):
        pass
    logger.error("EXL3XPU_GUARD_SELFTEST_FAILED rank=%s: the capture was not blocked", RANK)
    raise GraphGuardError("EXL3XPU_GUARD_SELFTEST_FAILED: a capture after warmup was not blocked")


def _patch_v2_runner(mod) -> None:
    rcls = getattr(mod, "GPUModelRunner", None)
    if rcls is None or not hasattr(rcls, "execute_model"):
        logger.warning("EXL3XPU_GRAPH_GUARD G4 not installed: %s.GPUModelRunner.execute_model missing", mod.__name__)
        return
    orig = rcls.execute_model
    if not _is_patched(orig):
        def execute_model(self, scheduler_output, *args, **kwargs):
            import torch
            if torch.xpu.is_current_stream_capturing():
                COUNTS["open_capture"] += 1
                logger.error("EXL3XPU_OPEN_CAPTURE rank=%s at=execute_model serving=%s", RANK, SERVING)
                raise GraphGuardError(f"EXL3XPU_OPEN_CAPTURE: execute_model entered with the current XPU stream "
                                      f"capturing (rank {RANK})")
            if SELFTEST and SERVING and not _selftest_done and _is_real_request(scheduler_output, args, kwargs):
                run_selftest()
            return orig(self, scheduler_output, *args, **kwargs)
        rcls.execute_model = _mark(functools.update_wrapper(execute_model, orig), orig)
    _installed["G4"] = True


# ------------------------------------------------------------------------------------------ post-import hooks
_PATCHERS: dict[str, Callable] = {
    HOOK_MODULES["G1"]: _patch_gpu_worker,
    HOOK_MODULES["G2"]: _patch_torch_xpu_graphs,
    HOOK_MODULES["G3a"]: _patch_cuda_graph,
    HOOK_MODULES["G3b"]: _patch_breakable,
    HOOK_MODULES["G4"]: _patch_v2_runner,
}
_pending: dict[str, list[Callable]] = {}


class _HookLoader(importlib.abc.Loader):
    def __init__(self, inner, fullname):
        self._inner, self._fullname = inner, fullname

    def create_module(self, spec):
        return self._inner.create_module(spec)

    def exec_module(self, module):
        module.__loader__ = self._inner
        self._inner.exec_module(module)
        _run_hooks(self._fullname, module)

    def __getattr__(self, name):
        return getattr(self._inner, name)


class _HookFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname not in _pending:
            return None
        for finder in sys.meta_path:
            if finder is self or not hasattr(finder, "find_spec"):
                continue
            spec = finder.find_spec(fullname, path, target)
            if spec is not None:
                break
        else:
            return None
        if spec.loader is None or not hasattr(spec.loader, "exec_module"):
            return spec
        spec.loader = _HookLoader(spec.loader, fullname)
        return spec


_finder = _HookFinder()


def _run_hooks(name: str, module) -> None:
    with _lock:
        fns = _pending.pop(name, [])
    for fn in fns:
        try:
            fn(module)
        except Exception:                                           # noqa: BLE001
            logger.exception("EXL3XPU_GRAPH_GUARD hook for %s failed", name)
            raise


def when_imported(name: str, fn: Callable) -> None:
    """Run fn(module) now if `name` is imported, else right after its first import."""
    mod = sys.modules.get(name)
    if mod is not None and getattr(mod, "__spec__", None) is not None and not getattr(
            getattr(mod, "__spec__", None), "_initializing", False):
        fn(mod)
        return
    with _lock:
        _pending.setdefault(name, []).append(fn)
        if _finder not in sys.meta_path:
            sys.meta_path.insert(0, _finder)


def install() -> dict[str, bool]:
    """Install G1-G4 (idempotent). Returns {hook: installed now}; pending hooks apply on import."""
    global SELFTEST
    SELFTEST = os.environ.get("EXL3_GUARD_SELFTEST", "0") == "1"
    for name, patcher in _PATCHERS.items():
        with _lock:
            already = patcher in _pending.get(name, [])
        if not already:
            when_imported(name, patcher)
    logger.info("EXL3XPU_GRAPH_GUARD installed=%s pending=%s selftest=%s",
                ",".join(k for k in HOOK_MODULES if _installed.get(k)) or "-",
                ",".join(sorted(_pending)) or "-", int(SELFTEST))
    return {k: bool(_installed.get(k)) for k in HOOK_MODULES}


def installed() -> dict[str, bool]:
    return {k: bool(_installed.get(k)) for k in HOOK_MODULES}
