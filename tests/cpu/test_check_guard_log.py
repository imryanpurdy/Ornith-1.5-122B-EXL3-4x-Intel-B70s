"""The log checker for the capture gate and the M2c guard self-test (tests/xpu/check_guard_log.py), on synthetic
logs shaped like vLLM 0.30's output."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "xpu"))
import check_guard_log as cgl  # noqa: E402


def _warm(rank, cap=8, gr=8, runner="XPUModelRunnerV2"):
    return (f"(Worker_TP{rank} pid=1) INFO 10-21 05:00:0{rank} [graph_guard.py:200] EXL3XPU_WARMUP_DONE rank={rank} "
            f"runner={runner} captures={cap} graphs={gr} detail=model:{gr} hooks=G1,G2,G3a,G3b,G4")


def test_capture_gate_pass_and_failures():
    good = [_warm(r) for r in range(4)] + ["INFO 10-21 05:00:09 [api_server.py:1] Starting"]
    assert cgl.capture_gate(good, 4, 8, "XPUModelRunnerV2") == []
    assert cgl.capture_gate(good[:3], 4, 8, "XPUModelRunnerV2")                      # a rank missing
    assert cgl.capture_gate([_warm(0, 8, 7)] + good[1:], 4, 8, "XPUModelRunnerV2")   # captures != graphs
    assert cgl.capture_gate([_warm(0, 6, 6)] + good[1:], 4, 8, "XPUModelRunnerV2")   # != plan
    assert cgl.capture_gate([_warm(0, runner="XPUModelRunner")] + good[1:], 4, 8, "XPUModelRunnerV2")  # V1
    assert cgl.capture_gate(good + ["ERROR EXL3XPU_CAPTURE_AFTER_WARMUP rank=1 via=graph.__enter__"], 4, 8,
                            "XPUModelRunnerV2")
    assert cgl.capture_gate(good + ["WARNING XPU Graph is disabled by environment variable"], 4, 8,
                            "XPUModelRunnerV2")
    assert cgl.capture_gate([_warm(0, 0, 0)], 1, 0, "XPUModelRunnerV2") == []            # eager run


def _selftest_log(dead_after="05:00:30", failed=False, order_ok=True):
    lines = [_warm(0, 8, 8),
             "INFO 10-21 05:00:20 [graph_guard.py:250] EXL3XPU_GUARD_SELFTEST rank=0: attempting one capture"]
    blocked = "ERROR 10-21 05:00:20 [graph_guard.py:90] EXL3XPU_CAPTURE_AFTER_WARMUP rank=0 via=graph.__enter__"
    dead = f"ERROR 10-21 {dead_after} [async_llm.py:1] vllm.v1.engine.exceptions.EngineDeadError: EngineCore died"
    if failed:
        lines.append("ERROR 10-21 05:00:20 [graph_guard.py:260] EXL3XPU_GUARD_SELFTEST_FAILED rank=0")
    lines += [blocked, dead] if order_ok else [dead, blocked]
    return lines


def test_selftest_checker():
    assert cgl.selftest(_selftest_log(), [], 60) == []
    assert cgl.selftest(_selftest_log(dead_after="05:02:00"), [], 60)                     # too slow
    assert cgl.selftest(_selftest_log(failed=True), [], 60)                               # guard broken
    assert cgl.selftest(_selftest_log(order_ok=False), [], 60)                            # wrong order
    kern = ["Oct 21 05:00:21 host kernel: xe 0000:03:00.0: [drm] GT0: reset done"]
    assert cgl.selftest(_selftest_log(), kern, 60)                                        # an xe reset
    assert cgl.selftest(_selftest_log(), ["Oct 21 05:00:21 host kernel: usb 1-1: new device"], 60) == []


def test_cli(tmp_path):
    log = tmp_path / "run.log"
    log.write_text("\n".join(_warm(r) for r in range(4)) + "\n")
    assert cgl.main(["capture-gate", "--log", str(log), "--ranks", "4", "--expect-graphs", "8"]) == 0
    assert cgl.main(["capture-gate", "--log", str(log), "--ranks", "4", "--expect-graphs", "10"]) == 1
