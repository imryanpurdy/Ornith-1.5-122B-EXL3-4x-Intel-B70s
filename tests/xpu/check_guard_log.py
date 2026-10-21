#!/usr/bin/env python3
"""Check a vLLM container log against the graph-guard gates (DESIGN rev 2, section 4.3). No torch, no vLLM.

  capture-gate  every graph-mode run: N lines EXL3XPU_WARMUP_DONE (one per rank) with runner=XPUModelRunnerV2 and
                captures == graphs == --expect-graphs; zero EXL3XPU_CAPTURE_AFTER_WARMUP, EXL3XPU_OPEN_CAPTURE*,
                EXL3XPU_CAPTURE_COUNT_MISMATCH, EXL3XPU_GUARD_SELFTEST_FAILED, "XPU Graph is disabled" and
                "XPU Graph is not supported" (vLLM forces cudagraph_mode NONE with those, platforms/xpu.py).
  selftest      M2c guard self-test (EXL3_GUARD_SELFTEST=1, one card): after EXL3XPU_WARMUP_DONE, the log shows
                EXL3XPU_GUARD_SELFTEST, then EXL3XPU_CAPTURE_AFTER_WARMUP, then EngineDeadError, within
                --max-seconds; no EXL3XPU_GUARD_SELFTEST_FAILED; and the kernel log excerpt (journalctl -k --since
                <launch>) has no xe reset line. The container exit and the clean relaunch are checked by the caller.

Exit status 0 = PASS. Usage:
  check_guard_log.py capture-gate --log run.log --ranks 4 --expect-graphs 8
  check_guard_log.py selftest --log run.log --kernel-log kernel.log --max-seconds 60
"""
from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime

TS = re.compile(r"\b(\d\d-\d\d \d\d:\d\d:\d\d)\b")
WARM = re.compile(r"EXL3XPU_WARMUP_DONE rank=(\S+) runner=(\S+) captures=(\d+) graphs=(\d+)")
BAD_ALWAYS = ("EXL3XPU_CAPTURE_AFTER_WARMUP", "EXL3XPU_OPEN_CAPTURE", "EXL3XPU_CAPTURE_COUNT_MISMATCH",
              "EXL3XPU_GUARD_SELFTEST_FAILED", "XPU Graph is disabled", "XPU Graph is not supported")
XE_RESET = re.compile(r"\bxe\b.*\b(reset|wedged|GT\d*:? ?hang)\b", re.I)


def _ts(line: str):
    m = TS.search(line)
    return datetime.strptime("2000-" + m.group(1), "%Y-%m-%d %H:%M:%S") if m else None


def capture_gate(lines: list[str], ranks: int, expect_graphs: int, runner: str) -> list[str]:
    errs = []
    warm = [WARM.search(l) for l in lines]
    warm = [m for m in warm if m]
    if len(warm) != ranks:
        errs.append(f"{len(warm)} EXL3XPU_WARMUP_DONE lines, expected {ranks}")
    for m in warm:
        rank, rn, cap, gr = m.group(1), m.group(2), int(m.group(3)), int(m.group(4))
        if rn != runner:
            errs.append(f"rank {rank}: runner={rn}, expected {runner}")
        if not cap == gr == expect_graphs:
            errs.append(f"rank {rank}: captures={cap} graphs={gr}, expected {expect_graphs}")
    if len({m.group(1) for m in warm}) != len(warm):
        errs.append("duplicate rank in EXL3XPU_WARMUP_DONE lines")
    for bad in BAD_ALWAYS:
        n = sum(bad in l for l in lines)
        if n:
            errs.append(f"{n} line(s) with {bad}")
    return errs


def selftest(lines: list[str], kernel: list[str], max_seconds: float) -> list[str]:
    errs = []

    def first(marker, start=0):
        for i in range(start, len(lines)):
            if marker in lines[i]:
                return i
        return None
    w = first("EXL3XPU_WARMUP_DONE")
    s = first("EXL3XPU_GUARD_SELFTEST rank=", 0 if w is None else w)
    c = first("EXL3XPU_CAPTURE_AFTER_WARMUP", 0 if s is None else s)
    d = first("EngineDeadError", 0 if c is None else c)
    for name, idx in (("EXL3XPU_WARMUP_DONE", w), ("EXL3XPU_GUARD_SELFTEST after warmup", s),
                      ("EXL3XPU_CAPTURE_AFTER_WARMUP after the self-test line", c),
                      ("EngineDeadError after EXL3XPU_CAPTURE_AFTER_WARMUP", d)):
        if idx is None:
            errs.append(f"missing {name}")
    if any("EXL3XPU_GUARD_SELFTEST_FAILED" in l for l in lines):
        errs.append("EXL3XPU_GUARD_SELFTEST_FAILED present: the capture was not blocked")
    if s is not None and d is not None:
        t0 = next((_ts(l) for l in lines[s:] if _ts(l)), None)
        t1 = next((_ts(l) for l in reversed(lines[:d + 1]) if _ts(l)), None)
        if t0 is None or t1 is None:
            errs.append("no vLLM timestamps around the self-test")
        elif (t1 - t0).total_seconds() > max_seconds:
            errs.append(f"EngineDeadError {(t1 - t0).total_seconds():.0f}s after the self-test (> {max_seconds:.0f}s)")
    resets = [l for l in kernel if XE_RESET.search(l)]
    if resets:
        errs.append(f"{len(resets)} xe reset line(s) in the kernel log, first: {resets[0].strip()}")
    return errs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("capture-gate")
    a.add_argument("--log", required=True)
    a.add_argument("--ranks", type=int, required=True)
    a.add_argument("--expect-graphs", type=int, required=True)
    a.add_argument("--runner", default="XPUModelRunnerV2")
    b = sub.add_parser("selftest")
    b.add_argument("--log", required=True)
    b.add_argument("--kernel-log", required=True)
    b.add_argument("--max-seconds", type=float, default=60.0)
    args = ap.parse_args(argv)
    with open(args.log, errors="replace") as f:
        lines = f.read().splitlines()
    if args.cmd == "capture-gate":
        errs = capture_gate(lines, args.ranks, args.expect_graphs, args.runner)
    else:
        with open(args.kernel_log, errors="replace") as f:
            kernel = f.read().splitlines()
        errs = selftest(lines, kernel, args.max_seconds)
    print(("FAIL " if errs else "PASS ") + args.cmd)
    for e in errs:
        print("  " + e)
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main())
