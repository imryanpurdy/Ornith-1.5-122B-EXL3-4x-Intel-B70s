"""The log checker for the capture gate and the M2c guard self-test (tests/xpu/check_guard_log.py), on synthetic
logs shaped like vLLM 0.30's output."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "xpu"))
import check_guard_log as cgl  # noqa: E402

FDO = "FULL_DECODE_ONLY"


def _warm(rank, cap=8, gr=8, runner="XPUModelRunnerV2", hooks="G1,G2,G3a,G3b,G4"):
    return (f"(Worker_TP{rank} pid=1) INFO 10-21 05:00:0{rank} [graph_guard.py:200] EXL3XPU_WARMUP_DONE rank={rank} "
            f"runner={runner} captures={cap} graphs={gr} detail=model:{gr} hooks={hooks}")


def _engine(mode=FDO):
    num = {"FULL_DECODE_ONLY": "(2, 0)", "NONE": "0", "FULL_AND_PIECEWISE": "(2, 1)"}[mode]
    return ("(EngineCore pid=9) INFO 10-21 04:59:00 [core.py:123] Initializing a V1 LLM engine (v0.30.0) with config: "
            "model='/m', compilation_config={'mode': <CompilationMode.NONE: 0>, 'inductor_passes': {}, "
            f"'cudagraph_mode': <CUDAGraphMode.{mode}: {num}>, 'cudagraph_capture_sizes': [1, 2, 4, 8]}}")


def test_capture_gate_pass_and_failures():
    good = [_engine()] + [_warm(r) for r in range(4)] + ["INFO 10-21 05:00:09 [api_server.py:1] Starting"]
    ok = lambda lines, ranks=4, g=8, mode=FDO: cgl.capture_gate(lines, ranks, g, "XPUModelRunnerV2", mode)  # noqa: E731
    assert ok(good) == []
    assert ok(good[:4])                                                              # a rank missing
    assert ok([good[0], _warm(0, 8, 7)] + good[2:])                                  # captures != graphs
    assert ok([good[0], _warm(0, 6, 6)] + good[2:])                                  # != plan
    assert ok([good[0], _warm(0, runner="XPUModelRunner")] + good[2:])               # V1
    assert ok(good + ["ERROR EXL3XPU_CAPTURE_AFTER_WARMUP rank=1 via=graph.__enter__"])
    assert ok(good + ["WARNING XPU Graph is disabled by environment variable"])
    assert ok([_engine("NONE"), _warm(0, 0, 0)], 1, 0, "NONE") == []                 # eager run
    # hooks that never landed (e.g. G4 pending) fail the gate instead of passing silently
    errs = ok([good[0], _warm(0, hooks="G1,G2,G3a,G3b")] + good[2:])
    assert errs and "G4" in errs[0]
    assert ok([good[0], _warm(0, hooks="-")] + good[2:])
    # the engine config must show the resolved cudagraph mode (DESIGN 4.3)
    assert ok([_engine("FULL_AND_PIECEWISE")] + good[1:])
    assert ok([_engine("NONE")] + good[1:])
    assert ok(good[1:])                                                              # no engine config line
    assert ok([_engine("NONE"), _warm(0, 0, 0)], 1, 0, FDO)                          # eager log vs FDO plan


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
    log.write_text("\n".join([_engine()] + [_warm(r) for r in range(4)]) + "\n")
    base = ["capture-gate", "--log", str(log), "--ranks", "4", "--cudagraph-mode", FDO]
    assert cgl.main(base + ["--expect-graphs", "8"]) == 0
    assert cgl.main(base + ["--expect-graphs", "10"]) == 1
    assert cgl.main(base + ["--expect-graphs", "8", "--require-hooks", "G1,G2,G4"]) == 0
    assert cgl.main(base + ["--expect-graphs", "8", "--require-hooks", "G1,G2,G4,G9"]) == 1
    log.write_text("\n".join([_engine()] + [_warm(r, hooks="G1,G2,G4") for r in range(4)]) + "\n")
    assert cgl.main(base + ["--expect-graphs", "8"]) == 1                     # default: all five hooks
    assert cgl.main(base + ["--expect-graphs", "8", "--require-hooks", "G1,G2,G4"]) == 0
