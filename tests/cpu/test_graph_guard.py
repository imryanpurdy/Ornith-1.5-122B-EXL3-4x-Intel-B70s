"""T6 on CPU: every branch of graph_guard G1-G4 and the self-test switch, with fake modules / monkeypatched torch.xpu
(no device). The real-class tests at the end patch the actual torch 2.13 xpu and vLLM 0.30 classes of the image
(restored after each test) and exercise the hooks without a GPU."""
from __future__ import annotations

import logging
import subprocess
import sys
import textwrap
import types
from pathlib import Path

import pytest
import torch

from exl3xpu import graph_guard as gg

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def clean_state(monkeypatch, caplog):
    gg.reset_for_tests()
    gg.SELFTEST = False
    saved = dict(gg._installed)
    caplog.set_level(logging.INFO, logger="vllm.exl3xpu.graph_guard")
    monkeypatch.setattr(torch.xpu, "is_current_stream_capturing", lambda: False, raising=False)
    monkeypatch.setattr(torch.xpu, "synchronize", lambda *a, **k: None, raising=False)
    real_mon = sys.modules.get("vllm.compilation.monitor")
    if real_mon is not None:      # G1 flips vLLM's global capture flag; restore it after every test
        monkeypatch.setattr(real_mon, "cudagraph_capturing_enabled", real_mon.cudagraph_capturing_enabled)
    yield
    gg.reset_for_tests()
    gg.SELFTEST = False
    gg._installed.clear()
    gg._installed.update(saved)


def _lines(caplog, marker):
    return [r.getMessage() for r in caplog.records if marker in r.getMessage()]


# ------------------------------------------------------------------------------------------------------- fakes
def fake_graphs_module(log):
    m = types.ModuleType("fake_xpu_graphs")

    class XPUGraph:
        def capture_begin(self, *pool):
            log.append("capture_begin")

        def capture_end(self):
            log.append("capture_end")

    class _Stream:
        def __enter__(self):
            log.append("stream_enter")

        def __exit__(self, *a):
            log.append("stream_exit")

    class graph:                               # same body as torch/xpu/graphs.py graph.__enter__/__exit__
        def __init__(self, xpu_graph, pool=None, stream=None):
            self.xpu_graph, self.pool, self.stream_ctx = xpu_graph, () if pool is None else (pool,), _Stream()

        def __enter__(self):
            log.append("synchronize")
            log.append("empty_cache")
            self.stream_ctx.__enter__()
            self.xpu_graph.capture_begin(*self.pool)

        def __exit__(self, *args):
            self.xpu_graph.capture_end()
            self.stream_ctx.__exit__(*args)

    m.XPUGraph, m.graph = XPUGraph, graph
    return m


class _Mgr:
    def __init__(self, n):
        self.graphs = {i: object() for i in range(n)}


class SpeculatorCudaGraphManager(_Mgr):
    pass


class FakeRunner:
    def __init__(self, n_model=8, n_spec=0):
        self.cudagraph_manager = _Mgr(n_model)
        self.speculator = types.SimpleNamespace(decode_cudagraph_manager=SpeculatorCudaGraphManager(n_spec),
                                                prefill_cudagraph_manager=None)
        enc = types.SimpleNamespace(budget_graphs={"a": {1: 0, 2: 0}})
        self.model_state = types.SimpleNamespace(encoder_runner=types.SimpleNamespace(cudagraph_manager=enc))


def fake_worker_module(runner, raise_in_warmup=False):
    m = types.ModuleType("fake_gpu_worker")

    class Worker:
        def __init__(self):
            self.rank, self.model_runner, self.calls = 2, runner, 0

        def compile_or_warm_up_model(self):
            self.calls += 1
            if raise_in_warmup:
                raise ValueError("warmup failed")
            return "times"

    m.Worker = Worker
    return m


# ---------------------------------------------------------------------------------------------------------- G2
def test_g2_before_serving_captures_and_counts(caplog):
    log = []
    m = fake_graphs_module(log)
    gg._patch_torch_xpu_graphs(m)
    with m.graph(m.XPUGraph()):
        pass
    assert log == ["synchronize", "empty_cache", "stream_enter", "capture_begin", "capture_end", "stream_exit"]
    assert (gg.COUNTS["enter"], gg.COUNTS["begin"], gg.COUNTS["end"], gg.COUNTS["blocked"]) == (1, 1, 1, 0)
    assert gg.capture_count() == 1 and gg.installed()["G2"]


def test_g2_serving_refuses_before_sync_and_stream_switch(caplog):
    log = []
    m = fake_graphs_module(log)
    gg._patch_torch_xpu_graphs(m)
    gg.SERVING, gg.RANK = True, 3
    with pytest.raises(gg.GraphGuardError, match="EXL3XPU_CAPTURE_AFTER_WARMUP"):
        with m.graph(m.XPUGraph()):
            log.append("body")
    assert log == []                                            # no sync, no stream switch, no recording
    assert gg.COUNTS["blocked"] == 1 and gg.COUNTS["begin"] == 0
    (line,) = _lines(caplog, "EXL3XPU_CAPTURE_AFTER_WARMUP")
    assert "rank=3" in line and "via=graph.__enter__" in line and "stack=" in line


def test_g2_direct_capture_begin_while_serving(caplog):
    log = []
    m = fake_graphs_module(log)
    gg._patch_torch_xpu_graphs(m)
    gg.SERVING = True
    with pytest.raises(gg.GraphGuardError):
        m.XPUGraph().capture_begin()
    assert log == [] and "via=XPUGraph.capture_begin" in _lines(caplog, "EXL3XPU_CAPTURE_AFTER_WARMUP")[0]


def test_g2_patch_is_idempotent():
    m = fake_graphs_module([])
    gg._patch_torch_xpu_graphs(m)
    first = m.graph.__enter__
    gg._patch_torch_xpu_graphs(m)
    assert m.graph.__enter__ is first and not gg._is_patched(first.__wrapped__)


# ---------------------------------------------------------------------------------------------------------- G1
def test_g1_ok_logs_counts_and_enters_serving(caplog, monkeypatch):
    mon = types.ModuleType("vllm.compilation.monitor")
    mon.cudagraph_capturing_enabled = True
    mon.set_cudagraph_capturing_enabled = lambda v: setattr(mon, "cudagraph_capturing_enabled", v)
    monkeypatch.setitem(sys.modules, "vllm.compilation.monitor", mon)
    gg.COUNTS.update(begin=10, end=10)
    m = fake_worker_module(FakeRunner(n_model=8, n_spec=0))
    gg._patch_gpu_worker(m)
    w = m.Worker()
    assert w.compile_or_warm_up_model() == "times" and w.calls == 1
    assert gg.SERVING and gg.RANK == 2 and mon.cudagraph_capturing_enabled is False
    (line,) = _lines(caplog, "EXL3XPU_WARMUP_DONE")
    assert "rank=2 runner=FakeRunner captures=10 graphs=10" in line
    assert "model:8" in line and "spec.decode_cudagraph_manager:0" in line and "encoder:2" in line
    assert not _lines(caplog, "EXL3XPU_CAPTURE_COUNT_MISMATCH")


def test_g1_count_mismatch_is_logged(caplog):
    gg.COUNTS.update(begin=9, end=9)
    m = fake_worker_module(FakeRunner(n_model=8))
    gg._patch_gpu_worker(m)
    m.Worker().compile_or_warm_up_model()
    assert _lines(caplog, "EXL3XPU_CAPTURE_COUNT_MISMATCH rank=2 captures=9 graphs=10")


@pytest.mark.parametrize("case", ["capturing", "sync_raises", "unbalanced", "is_capturing_raises"])
def test_g1_open_capture_fails_the_boot(caplog, monkeypatch, case):
    if case == "capturing":
        monkeypatch.setattr(torch.xpu, "is_current_stream_capturing", lambda: True, raising=False)
    elif case == "is_capturing_raises":
        def boom():
            raise RuntimeError("driver")
        monkeypatch.setattr(torch.xpu, "is_current_stream_capturing", boom, raising=False)
    elif case == "sync_raises":
        def sync(*a, **k):
            raise RuntimeError("wait cannot be called for a queue which is recording")
        monkeypatch.setattr(torch.xpu, "synchronize", sync, raising=False)
    else:
        gg.COUNTS.update(begin=3, end=2)
    m = fake_worker_module(FakeRunner())
    gg._patch_gpu_worker(m)
    with pytest.raises(gg.GraphGuardError, match="EXL3XPU_OPEN_CAPTURE_AT_WARMUP"):
        m.Worker().compile_or_warm_up_model()
    assert not gg.SERVING and not _lines(caplog, "EXL3XPU_WARMUP_DONE")
    (line,) = _lines(caplog, "EXL3XPU_OPEN_CAPTURE_AT_WARMUP")
    assert line.startswith("EXL3XPU_OPEN_CAPTURE_AT_WARMUP rank=2 reason=")


def test_g1_warmup_exception_propagates(caplog):
    m = fake_worker_module(FakeRunner(), raise_in_warmup=True)
    gg._patch_gpu_worker(m)
    with pytest.raises(ValueError, match="warmup failed"):
        m.Worker().compile_or_warm_up_model()
    assert not gg.SERVING and not _lines(caplog, "EXL3XPU_WARMUP_DONE")


def test_g1_g2_end_to_end_with_fakes(caplog):
    """warmup captures 3 graphs through the guarded fake torch API, then serving refuses a 4th."""
    log = []
    gm = fake_graphs_module(log)
    gg._patch_torch_xpu_graphs(gm)
    runner = FakeRunner(n_model=0)
    runner.model_state = None

    def warm(self):
        for i in range(3):
            g = gm.XPUGraph()
            with gm.graph(g):
                pass
            self.model_runner.cudagraph_manager.graphs[i] = g
    wm = fake_worker_module(runner)
    wm.Worker.compile_or_warm_up_model = warm
    gg._patch_gpu_worker(wm)
    wm.Worker().compile_or_warm_up_model()
    assert "captures=3 graphs=3" in _lines(caplog, "EXL3XPU_WARMUP_DONE")[0]
    with pytest.raises(gg.GraphGuardError):
        with gm.graph(gm.XPUGraph()):
            pass
    assert gg.COUNTS["begin"] == 3 and gg.COUNTS["blocked"] == 1


# ---------------------------------------------------------------------------------------------------------- G3
def fake_cuda_graph_module(log):
    m = types.ModuleType("fake_cuda_graph")
    m.CUDAGraphMode = types.SimpleNamespace(NONE="NONE", FULL="FULL", PIECEWISE="PIECEWISE")
    m.ctx = types.SimpleNamespace(batch_descriptor="desc8", cudagraph_runtime_mode="PIECEWISE")
    m.available = True
    m.is_forward_context_available = lambda: m.available
    m.get_forward_context = lambda: m.ctx

    class CUDAGraphWrapper:
        def __init__(self, runnable, runtime_mode):
            self.runnable, self.runtime_mode, self.concrete_cudagraph_entries = runnable, runtime_mode, {}

        def __call__(self, *args, **kwargs):
            log.append("orig")
            return "orig"

    class BreakableCUDAGraphWrapper:
        def __init__(self, runnable):
            self.runnable = runnable

        def _capture(self, entry, args, kwargs):
            log.append("orig_capture")
            return "captured"

    m.CUDAGraphWrapper, m.BreakableCUDAGraphWrapper = CUDAGraphWrapper, BreakableCUDAGraphWrapper
    return m


@pytest.fixture
def monitor(monkeypatch):
    mon = types.ModuleType("vllm.compilation.monitor")
    mon.cudagraph_capturing_enabled = True
    monkeypatch.setitem(sys.modules, "vllm.compilation.monitor", mon)
    return mon


def test_g3_wrapper_branches(caplog, monitor):
    log = []
    m = fake_cuda_graph_module(log)
    gg._patch_cuda_graph(m)
    w = m.CUDAGraphWrapper(lambda x: f"eager({x})", "PIECEWISE")
    assert w(1) == "orig" and log == ["orig"]                                  # capturing enabled: vLLM's path
    monitor.cudagraph_capturing_enabled = False
    assert w(2) == "eager(2)" and w(3) == "eager(3)" and log == ["orig"]        # miss: eager, no capture
    assert gg.COUNTS["eager_miss"] == 2 and len(_lines(caplog, "EXL3XPU_EAGER_MISS")) == 1   # logged once
    assert "desc='desc8'" in _lines(caplog, "EXL3XPU_EAGER_MISS")[0]
    w.concrete_cudagraph_entries["desc8"] = types.SimpleNamespace(cudagraph=object())
    assert w(4) == "orig"                                                      # captured entry: replay path
    w.concrete_cudagraph_entries["desc8"] = types.SimpleNamespace(cudagraph=None)
    assert w(5) == "eager(5)"                                                  # entry without graph: miss
    m.ctx.cudagraph_runtime_mode = "NONE"
    assert w(6) == "orig"                                                      # NONE: vLLM runs it eagerly
    m.ctx.cudagraph_runtime_mode = "FULL"
    assert w(7) == "orig"                                                      # other wrapper's mode
    m.ctx.cudagraph_runtime_mode, m.available = "PIECEWISE", False
    assert w(8) == "orig"                                                      # no forward context
    assert gg.installed()["G3a"]


def test_g3_breakable_branches(caplog, monitor):
    log = []
    m = fake_cuda_graph_module(log)
    gg._patch_breakable(m)
    w = m.BreakableCUDAGraphWrapper(lambda *a, **k: ("eager", a, k))
    entry = types.SimpleNamespace(batch_descriptor="d16")
    assert w._capture(entry, (1,), {"y": 2}) == "captured"
    monitor.cudagraph_capturing_enabled = False
    assert w._capture(entry, (1,), {"y": 2}) == ("eager", (1,), {"y": 2})
    assert log == ["orig_capture"] and "where=BreakableCUDAGraphWrapper desc='d16'" in \
        _lines(caplog, "EXL3XPU_EAGER_MISS")[0]


# ---------------------------------------------------------------------------------------------------------- G4
def fake_runner_module(log):
    m = types.ModuleType("fake_model_runner")

    class GPUModelRunner:
        def execute_model(self, scheduler_output, intermediate_tensors=None, dummy_run=False, **kw):
            log.append(("orig", dummy_run))
            return "out"

    m.GPUModelRunner = GPUModelRunner
    return m


def _sched(n):
    return types.SimpleNamespace(total_num_scheduled_tokens=n)


def test_g4_passes_through_and_refuses_open_capture(caplog, monkeypatch):
    log = []
    m = fake_runner_module(log)
    gg._patch_v2_runner(m)
    r = m.GPUModelRunner()
    assert r.execute_model(_sched(4)) == "out" and r.execute_model(_sched(0), None, True) == "out"
    assert log == [("orig", False), ("orig", True)]
    monkeypatch.setattr(torch.xpu, "is_current_stream_capturing", lambda: True, raising=False)
    gg.RANK = 1
    with pytest.raises(gg.GraphGuardError, match="EXL3XPU_OPEN_CAPTURE"):
        r.execute_model(_sched(4))
    assert len(log) == 2 and "EXL3XPU_OPEN_CAPTURE rank=1 at=execute_model" in _lines(caplog, "EXL3XPU_OPEN_CAPTURE")[0]


def _selftest_env(monkeypatch, graphs_mod):
    monkeypatch.setattr(torch.xpu, "XPUGraph", graphs_mod.XPUGraph, raising=False)
    monkeypatch.setattr(torch.xpu, "graph", graphs_mod.graph, raising=False)


def test_selftest_trips_g2_on_first_real_request(caplog, monkeypatch):
    glog, rlog = [], []
    gm = fake_graphs_module(glog)
    gg._patch_torch_xpu_graphs(gm)
    _selftest_env(monkeypatch, gm)
    m = fake_runner_module(rlog)
    gg._patch_v2_runner(m)
    gg.SELFTEST = True
    r = m.GPUModelRunner()
    assert r.execute_model(_sched(5)) == "out"                 # before SERVING: warmup traffic, no self-test
    assert r.execute_model(_sched(0), None, True) == "out"
    gg.SERVING = True
    # dummy runs while serving carry scheduled tokens (_dummy_run sets total_num_scheduled_tokens = num_tokens and
    # passes dummy_run as a keyword): the dummy flag alone must keep the self-test off, positional or keyword
    assert r.execute_model(_sched(6), None, True) == "out"
    assert r.execute_model(_sched(6), intermediate_tensors=None, dummy_run=True) == "out"
    assert r.execute_model(_sched(0)) == "out"                 # no scheduled tokens: no self-test
    with pytest.raises(gg.GraphGuardError, match="EXL3XPU_CAPTURE_AFTER_WARMUP"):
        r.execute_model(_sched(7))
    assert glog == [] and len(rlog) == 5                      # nothing recorded, original not run
    assert _lines(caplog, "EXL3XPU_GUARD_SELFTEST rank=") and _lines(caplog, "EXL3XPU_CAPTURE_AFTER_WARMUP")
    assert not _lines(caplog, "EXL3XPU_GUARD_SELFTEST_FAILED")
    assert r.execute_model(_sched(7)) == "out"                 # only once


def test_selftest_reports_a_broken_guard(caplog, monkeypatch):
    glog, rlog = [], []
    gm = fake_graphs_module(glog)                              # NOT patched: the guard is broken
    _selftest_env(monkeypatch, gm)
    m = fake_runner_module(rlog)
    gg._patch_v2_runner(m)
    gg.SELFTEST, gg.SERVING = True, True
    with pytest.raises(gg.GraphGuardError, match="SELFTEST_FAILED"):
        m.GPUModelRunner().execute_model(_sched(3))
    assert _lines(caplog, "EXL3XPU_GUARD_SELFTEST_FAILED") and "capture_begin" in glog


def test_install_reads_selftest_env(monkeypatch):
    monkeypatch.setattr(gg, "_PATCHERS", {})          # do not patch the real classes in this process
    monkeypatch.setenv("EXL3_GUARD_SELFTEST", "1")
    gg.install()
    assert gg.SELFTEST is True
    monkeypatch.setenv("EXL3_GUARD_SELFTEST", "0")
    gg.install()
    assert gg.SELFTEST is False


# ------------------------------------------------------------------------------------------- post-import hooks
def test_when_imported_runs_after_first_import(tmp_path, monkeypatch):
    pkg = tmp_path / "ggprobe_pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "mod.py").write_text("class Worker:\n    def compile_or_warm_up_model(self):\n        return 1\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    seen = []
    gg.when_imported("ggprobe_pkg.mod", lambda mod: seen.append(mod.__name__))
    assert seen == [] and "ggprobe_pkg.mod" in gg._pending
    import importlib
    import inspect
    mod = importlib.import_module("ggprobe_pkg.mod")
    assert seen == ["ggprobe_pkg.mod"] and "ggprobe_pkg.mod" not in gg._pending
    assert type(mod.__loader__).__name__ == "SourceFileLoader"        # original loader restored on the module
    assert "compile_or_warm_up_model" in inspect.getsource(mod)
    gg.when_imported("ggprobe_pkg.mod", lambda mod: seen.append("again"))   # already imported: runs now
    assert seen == ["ggprobe_pkg.mod", "again"]


# ------------------------------------------------------------------------------- real classes of the image
@pytest.fixture
def restore(monkeypatch):
    """Snapshot class attributes so monkeypatch restores the originals after the test."""
    def snap(cls, name):
        monkeypatch.setattr(cls, name, vars(cls)[name])
    return snap


def test_real_torch_xpu_graph_enter_refuses_before_sync(monkeypatch, restore, caplog):
    import torch.xpu.graphs as tg
    restore(tg.graph, "__enter__")
    restore(tg.XPUGraph, "capture_begin")
    restore(tg.XPUGraph, "capture_end")
    calls = []
    monkeypatch.setattr(torch.xpu, "synchronize", lambda *a, **k: calls.append("synchronize"), raising=False)
    monkeypatch.setattr(torch.xpu, "empty_cache", lambda *a, **k: calls.append("empty_cache"), raising=False)
    gg._patch_torch_xpu_graphs(tg)
    assert torch.xpu.graph is tg.graph and torch.xpu.graph.__enter__.__wrapped__ is not None
    obj = object.__new__(tg.graph)                              # bypass __init__ (it creates an XPU stream)
    obj.pool = ()
    obj.stream_ctx = types.SimpleNamespace(__enter__=lambda: calls.append("stream_enter"))
    obj.xpu_graph = types.SimpleNamespace(capture_begin=lambda *p: calls.append("capture_begin"))
    obj.__enter__()                                             # not serving: the real torch body runs
    assert calls == ["synchronize", "empty_cache", "stream_enter", "capture_begin"] and gg.COUNTS["enter"] == 1
    calls.clear()
    gg.SERVING = True
    with pytest.raises(gg.GraphGuardError):
        obj.__enter__()
    assert calls == []


def test_real_vllm_g1_g3_g4(monkeypatch, restore, caplog):
    pytest.importorskip("vllm")
    import vllm.compilation.breakable_cudagraph as bcg
    import vllm.compilation.cuda_graph as cg
    import vllm.compilation.monitor as mon
    import vllm.v1.worker.gpu.model_runner as mr
    import vllm.v1.worker.gpu_worker as gw
    import vllm.v1.worker.xpu_model_runner as xmr
    import vllm.v1.worker.xpu_worker as xw
    restore(gw.Worker, "compile_or_warm_up_model")
    restore(cg.CUDAGraphWrapper, "__call__")
    restore(bcg.BreakableCUDAGraphWrapper, "_capture")
    restore(mr.GPUModelRunner, "execute_model")
    monkeypatch.setattr(mon, "cudagraph_capturing_enabled", True)
    gg._patch_gpu_worker(gw)
    gg._patch_cuda_graph(cg)
    gg._patch_breakable(bcg)
    gg._patch_v2_runner(mr)
    # inheritance: the XPU classes resolve to the guarded methods
    assert xw.XPUWorker.compile_or_warm_up_model is gw.Worker.compile_or_warm_up_model
    assert gg._is_patched(xw.XPUWorker.compile_or_warm_up_model)
    assert gg._is_patched(xmr.XPUModelRunnerV2.execute_model)
    # G4 on the real class: an open capture is refused before any vLLM code runs
    monkeypatch.setattr(torch.xpu, "is_current_stream_capturing", lambda: True, raising=False)
    with pytest.raises(gg.GraphGuardError):
        mr.GPUModelRunner.execute_model(object.__new__(xmr.XPUModelRunnerV2), None)
    monkeypatch.setattr(torch.xpu, "is_current_stream_capturing", lambda: False, raising=False)
    # G3 on the real CUDAGraphWrapper: a capture miss with capturing disabled runs eagerly; vLLM's own code raises
    from vllm.config import CUDAGraphMode
    w = object.__new__(cg.CUDAGraphWrapper)
    w.__dict__.update(runnable=lambda x: x * 2, runtime_mode=CUDAGraphMode.PIECEWISE, concrete_cudagraph_entries={},
                      cudagraph_options=types.SimpleNamespace(debug_log_enable=False, gc_disable=False))
    ctx = types.SimpleNamespace(batch_descriptor="bd", cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE)
    monkeypatch.setattr(cg, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(cg, "get_forward_context", lambda: ctx)
    mon.set_cudagraph_capturing_enabled(False)
    with pytest.raises(RuntimeError, match="CUDA graph capturing detected"):
        cg.CUDAGraphWrapper.__call__.__wrapped__(w, 21)         # unguarded vLLM 0.30 behaviour
    assert w(21) == 42 and _lines(caplog, "EXL3XPU_EAGER_MISS")
    b = object.__new__(bcg.BreakableCUDAGraphWrapper)
    b.__dict__.update(runnable=lambda *a, **k: "eager")
    with pytest.raises(RuntimeError, match="CUDA graph capturing detected"):
        bcg.BreakableCUDAGraphWrapper._capture.__wrapped__(b, types.SimpleNamespace(batch_descriptor="x"), (), {})
    assert b._capture(types.SimpleNamespace(batch_descriptor="x"), (), {}) == "eager"


def test_install_in_a_fresh_process_before_vllm_imports():
    """The plugin's order: register() -> install() runs before the worker modules are imported. Every hook must
    land on first import, the modules must stay inspectable, and a second install() must not double-wrap."""
    code = textwrap.dedent(f"""
        import sys, inspect
        sys.path[:0] = [{str(ROOT / 'xpu')!r}, {str(ROOT / 'core' / 'src')!r}]
        import torch                     # vLLM imports torch before it loads plugins
        from exl3xpu import graph_guard as gg
        pre = gg.install()
        assert pre["G2"] and not pre["G1"] and not pre["G4"], pre
        import vllm.v1.worker.xpu_worker as xw, vllm.v1.worker.xpu_model_runner as xmr
        import vllm.compilation.cuda_graph as cg, vllm.compilation.breakable_cudagraph as bcg, torch
        gg.install()
        st = gg.installed()
        assert all(st.values()), st
        for fn in (xw.XPUWorker.compile_or_warm_up_model, xmr.XPUModelRunnerV2.execute_model,
                   cg.CUDAGraphWrapper.__call__, bcg.BreakableCUDAGraphWrapper._capture,
                   torch.xpu.graph.__enter__, torch.xpu.XPUGraph.capture_begin, torch.xpu.XPUGraph.capture_end):
            assert gg._is_patched(fn) and not gg._is_patched(fn.__wrapped__), fn
        assert "def compile_or_warm_up_model" in inspect.getsource(sys.modules["vllm.v1.worker.gpu_worker"])
        assert not gg._pending, gg._pending
        print("OK", ",".join(k for k, v in st.items() if v))
    """)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600)
    assert r.returncode == 0 and "OK G1,G2,G3a,G3b,G4" in r.stdout, r.stdout[-2000:] + r.stderr[-4000:]
