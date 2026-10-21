"""Check a vLLM boot log for the plugin's EXL3XPU_LOADED lines (DESIGN rev 2, section 2.3; M2c / M3 / M4 boots).

  python tests/xpu/check_loaded_log.py --log <container log> --ranks 4 --dense 1204 --experts 36864 \
      [--drafter-dense 36 --drafter-experts 768]

Pass = for every rank 0..ranks-1 exactly one target line (and, when --drafter-* is given, exactly one drafter line)
whose dense / experts counts equal both the expected values given here and the expected=<d>/<e> the plugin computed
from the checkpoint index, and no 'differ from the checkpoint index' error anywhere in the log. Exit 0 / 1, one
summary line. CPU only: it reads a text file.
"""
from __future__ import annotations

import argparse
import re
import sys

LINE = re.compile(r"EXL3XPU_LOADED rank=(\d+) dense=(\d+) experts=(\d+) model=(\w+) expected=(\d+)/(\d+)")


def check(text: str, ranks: int, want: dict[str, tuple[int, int]]) -> tuple[bool, list[str]]:
    why: list[str] = []
    seen: dict[tuple[int, str], list[tuple[int, int, int, int]]] = {}
    for m in LINE.finditer(text):
        r, d, e, role, ed, ee = int(m.group(1)), int(m.group(2)), int(m.group(3)), m.group(4), int(m.group(5)), \
            int(m.group(6))
        seen.setdefault((r, role), []).append((d, e, ed, ee))
    if "differ from the checkpoint index" in text:
        why.append("the plugin reported a load-count mismatch")
    for role, (d, e) in want.items():
        for r in range(ranks):
            got = seen.get((r, role), [])
            if len(got) != 1:
                why.append(f"rank {r} {role}: {len(got)} EXL3XPU_LOADED lines, need 1")
                continue
            gd, ge, ed, ee = got[0]
            if (gd, ge) != (d, e) or (ed, ee) != (d, e):
                why.append(f"rank {r} {role}: dense={gd} experts={ge} (index expects {ed}/{ee}), want {d}/{e}")
    extra = sorted({k for k in seen if k[0] >= ranks or k[1] not in want})
    if extra:
        why.append(f"unexpected EXL3XPU_LOADED lines for {extra}")
    return not why, why


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--ranks", type=int, required=True)
    ap.add_argument("--dense", type=int, required=True)
    ap.add_argument("--experts", type=int, required=True)
    ap.add_argument("--drafter-dense", type=int)
    ap.add_argument("--drafter-experts", type=int)
    a = ap.parse_args(argv)
    want = {"target": (a.dense, a.experts)}
    if a.drafter_dense is not None or a.drafter_experts is not None:
        if a.drafter_dense is None or a.drafter_experts is None:
            ap.error("--drafter-dense and --drafter-experts go together")
        want["drafter"] = (a.drafter_dense, a.drafter_experts)
    with open(a.log, errors="replace") as f:
        ok, why = check(f.read(), a.ranks, want)
    print(f"EXL3XPU_LOADED_CHECK {'PASS' if ok else 'FAIL'} ranks={a.ranks} want={want}" + ("" if ok else f" {why}"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
