"""T6 on CPU: the hook points graph_guard relies on exist in the serving image with the expected source
(torch 2.13.0+xpu, vLLM 0.30 with the V2 runner). If an image update moves any of them, these fail first."""
from __future__ import annotations

import inspect
import re

import pytest
import torch

from results import record


def _src(obj) -> tuple[str, str, int, int]:
    obj = inspect.unwrap(obj)            # decorators (contextmanager, inference_mode) and our own wrappers
    lines, start = inspect.getsourcelines(obj)
    return "".join(lines), inspect.getsourcefile(obj), start, start + len(lines) - 1


def _order(text: str, needles: list[str]) -> list[int]:
    pos = [text.find(n) for n in needles]
    assert all(p >= 0 for p in pos), dict(zip(needles, pos))
    assert pos == sorted(pos), dict(zip(needles, pos))
    return pos


def _rel(path: str) -> str:
    return re.sub(r"^.*?site-packages/", "", path)


def test_torch_xpu_graph_enter_order():
    import torch.xpu.graphs as tg
    assert torch.__version__.startswith("2.13.0") and "xpu" in torch.__version__
    assert torch.xpu.graph is tg.graph and torch.xpu.XPUGraph is tg.XPUGraph
    enter = vars(tg.graph)["__enter__"]
    enter = getattr(enter, "__wrapped__", enter)
    text, path, a, b = _src(enter)
    _order(text, ["torch.xpu.synchronize()", "torch.xpu.empty_cache()", "self.stream_ctx.__enter__()",
                  "self.xpu_graph.capture_begin(*self.pool)"])
    assert "capture_begin" in vars(tg.XPUGraph) and "capture_end" in vars(tg.XPUGraph)   # Python-level, wrappable
    assert "is_current_stream_capturing" in tg.__all__
    record("hook G2", f"{_rel(path)}:{a}-{b} graph.__enter__ = synchronize -> empty_cache -> stream enter -> "
                      f"capture_begin; XPUGraph.capture_begin/_end are Python methods")


def test_vllm_aliases_cuda_graph_to_xpu():
    pytest.importorskip("vllm")
    import vllm.v1.worker.xpu_model_runner as xmr
    text, path, a, b = _src(xmr._torch_cuda_wrapper)
    for s in ("torch.cuda.graph = partial(torch.xpu.graph)", "torch.cuda.CUDAGraph = torch.xpu.XPUGraph",
              "torch.cuda.is_current_stream_capturing = torch.xpu.is_current_stream_capturing"):
        assert s in text, s
    record("alias", f"{_rel(path)}:{a}-{b} torch.cuda.graph/CUDAGraph -> torch.xpu.graph/XPUGraph")


def test_v2_full_capture_goes_through_torch_graph():
    pytest.importorskip("vllm")
    import vllm.v1.worker.gpu.cudagraph_utils as cu
    text, path, a, b = _src(cu.CudaGraphManager.capture)
    _order(text, ["graph = torch.cuda.CUDAGraph()", "with torch.cuda.graph(", "self.graphs[desc] = graph"])
    d_text, _, da, db = _src(cu.CudaGraphManager.dispatch)
    assert "cg_mode=CUDAGraphMode.NONE" in d_text and "self._graphs_captured" in d_text
    record("V2 capture", f"{_rel(path)}:{a}-{b} CudaGraphManager.capture -> torch.cuda.graph; "
                         f"dispatch {da}-{db} returns NONE for uncaptured shapes")


def test_g1_hook_point():
    pytest.importorskip("vllm")
    import vllm.v1.worker.gpu_worker as gw
    import vllm.v1.worker.xpu_worker as xw
    assert "compile_or_warm_up_model" in vars(gw.Worker)
    assert "compile_or_warm_up_model" not in vars(xw.XPUWorker)          # XPUWorker inherits it
    fn = vars(gw.Worker)["compile_or_warm_up_model"]
    fn = getattr(fn, "__wrapped__", fn)
    text, path, a, b = _src(fn)
    _order(text, ["self.model_runner.capture_model()", "enable_gpu_sync_check()", "return CompilationTimes("])
    record("hook G1", f"{_rel(path)}:{a}-{b} Worker.compile_or_warm_up_model (capture_model inside; "
                      f"XPUWorker does not override)")


def test_g4_hook_point():
    pytest.importorskip("vllm")
    import vllm.v1.worker.gpu.model_runner as mr
    import vllm.v1.worker.xpu_model_runner as xmr
    assert issubclass(xmr.XPUModelRunnerV2, mr.GPUModelRunner)
    assert "execute_model" not in vars(xmr.XPUModelRunnerV2) and "capture_model" not in vars(xmr.XPUModelRunnerV2)
    fn = vars(mr.GPUModelRunner)["execute_model"]
    fn = getattr(fn, "__wrapped__", fn)
    text, path, a, b = _src(fn)
    assert "dummy_run" in inspect.signature(fn).parameters
    assert list(inspect.signature(fn).parameters)[:4] == ["self", "scheduler_output", "intermediate_tensors",
                                                          "dummy_run"]
    cap_text = _src(mr.GPUModelRunner.capture_model)[0]
    assert "execute_model" not in cap_text                              # capture never re-enters execute_model
    record("hook G4", f"{_rel(path)}:{a} GPUModelRunner.execute_model(self, scheduler_output, "
                      f"intermediate_tensors, dummy_run, ...) inherited by XPUModelRunnerV2")


def test_g3_hook_points():
    pytest.importorskip("vllm")
    import vllm.compilation.breakable_cudagraph as bcg
    import vllm.compilation.cuda_graph as cg
    import vllm.compilation.monitor as mon
    call = vars(cg.CUDAGraphWrapper)["__call__"]
    text, path, a, b = _src(getattr(call, "__wrapped__", call))
    _order(text, ["is_forward_context_available()", "cudagraph_runtime_mode == CUDAGraphMode.NONE",
                  "entry.cudagraph is None", "validate_cudagraph_capturing_enabled()", "torch.cuda.CUDAGraph()"])
    for name in ("is_forward_context_available", "get_forward_context", "CUDAGraphMode"):
        assert hasattr(cg, name), name                                   # read by the G3a wrapper at call time
    cap = vars(bcg.BreakableCUDAGraphWrapper)["_capture"]
    t2, p2, a2, b2 = _src(getattr(cap, "__wrapped__", cap))
    assert t2.lstrip().startswith("def _capture(") and "validate_cudagraph_capturing_enabled()" in t2
    assert list(inspect.signature(getattr(cap, "__wrapped__", cap)).parameters) == ["self", "entry", "args", "kwargs"]
    assert hasattr(mon, "cudagraph_capturing_enabled") and hasattr(mon, "set_cudagraph_capturing_enabled")
    record("hook G3", f"{_rel(path)}:{a}-{b} CUDAGraphWrapper.__call__; {_rel(p2)}:{a2}-{b2} "
                      f"BreakableCUDAGraphWrapper._capture")


def test_worker_exception_is_fatal():
    """A G2/G4 trip kills the engine: EngineCore treats an execute_model exception as fatal."""
    pytest.importorskip("vllm")
    import vllm.v1.engine.core as core
    text = inspect.getsource(core)
    assert "fatal error" in text.lower()
    m = re.search(r"\n([^\n]*encountered a fatal error[^\n]*)", text, re.I)
    record("fatal path", f"vllm/v1/engine/core.py: {m.group(1).strip() if m else 'fatal error handler present'}")
