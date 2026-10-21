"""T6 on one XPU card (milestone M2c; NOT run on CPU). The caller exposes exactly one card (ZE_AFFINITY_MASK) and
uses the serving image with its driver environment. Run from the repo root:

    python -m pytest -q tests/xpu/test_graph_guard_xpu.py

Checks G2 on the real driver: a capture during warmup records and replays; a capture attempt while SERVING is
refused inside torch.xpu.graph.__enter__ before its synchronize / stream switch, and leaves no recording, no
switched stream and a usable device behind. Also checks G1's open-recording detection on a real recording.
The in-engine self-test (EXL3_GUARD_SELFTEST=1, EngineDeadError, clean exit) is checked from the container log
with tests/xpu/check_guard_log.py selftest."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "xpu")]
from exl3xpu import graph_guard as gg  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.xpu.is_available(), reason="needs one XPU card")


@pytest.fixture(autouse=True)
def guard():
    gg.reset_for_tests()
    gg.install()
    assert gg.installed()["G2"]
    yield
    gg.SERVING = False
    gg.reset_for_tests()


def test_capture_replay_then_refuse_leaves_device_clean():
    x = torch.zeros(4096, device="xpu")
    x.add_(1)                                              # warm the kernel outside any capture
    torch.xpu.synchronize()
    g = torch.xpu.XPUGraph()
    with torch.xpu.graph(g):
        x.add_(1)
    torch.xpu.synchronize()
    assert gg.COUNTS["begin"] == gg.COUNTS["end"] == 1
    v0 = float(x[0])
    g.replay()
    torch.xpu.synchronize()
    assert float(x[0]) == v0 + 1

    default = torch.xpu.current_stream()
    gg.SERVING = True
    with pytest.raises(gg.GraphGuardError, match="EXL3XPU_CAPTURE_AFTER_WARMUP"):
        with torch.xpu.graph(torch.xpu.XPUGraph()):
            x.add_(100)                                    # never runs: __enter__ refused
    assert torch.xpu.current_stream() == default           # no stream context left entered
    assert not torch.xpu.is_current_stream_capturing()
    torch.xpu.synchronize()                                # no queue left recording
    assert gg.COUNTS["begin"] == 1 and gg.COUNTS["blocked"] == 1
    v1 = float(x[0])
    assert v1 == v0 + 1
    g.replay()                                             # warmup graphs still replay
    x.mul_(1)                                              # and eager work still runs
    torch.xpu.synchronize()
    assert float(x[0]) == v1 + 1


def test_g1_sees_a_real_open_recording():
    s = torch.xpu.Stream()
    y = torch.zeros(16, device="xpu")
    y.add_(1)
    torch.xpu.synchronize()
    g = torch.xpu.XPUGraph()
    with torch.xpu.stream(s):
        g.capture_begin()
        try:
            y.add_(1)
            assert gg._open_capture_reason() == "current stream is capturing"
        finally:
            g.capture_end()
    with torch.xpu.stream(s):
        g2 = torch.xpu.XPUGraph()
        g2.capture_begin()
    try:
        # the recording is open on a stream that is no longer current: the begin/end balance catches it
        # before torch.xpu.synchronize() is called
        assert gg._open_capture_reason() == "1 XPUGraph.capture_begin without capture_end"
    finally:
        with torch.xpu.stream(s):
            g2.capture_end()
    assert gg._open_capture_reason() is None
