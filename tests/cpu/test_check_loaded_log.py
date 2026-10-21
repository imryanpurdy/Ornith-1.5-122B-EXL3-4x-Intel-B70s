"""tests/xpu/check_loaded_log.py on synthetic boot logs (the line format is the one loadcheck.close() logs)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "xpu"))

import check_loaded_log as c  # noqa: E402


def _line(r, d, e, role="target", ed=None, ee=None):
    return (f"INFO 10-21 [loadcheck.py:200] EXL3XPU_LOADED rank={r} dense={d} experts={e} model={role} "
            f"expected={ed if ed is not None else d}/{ee if ee is not None else e} moe=ep4 tp=4 layers=193+48 "
            f"dropped=110592\n")


def test_pass_target_and_drafter():
    log = "".join(_line(r, 1204, 36864) + _line(r, 36, 768, "drafter") for r in range(4))
    ok, why = c.check(log, 4, {"target": (1204, 36864), "drafter": (36, 768)})
    assert ok, why


def test_fail_cases():
    want = {"target": (1204, 36864)}
    good = "".join(_line(r, 1204, 36864) for r in range(4))
    assert not c.check("".join(_line(r, 1204, 36864) for r in range(3)), 4, want)[0]           # a rank missing
    assert not c.check(good + _line(0, 1204, 36864), 4, want)[0]                               # duplicate line
    assert not c.check(good.replace("dense=1204", "dense=1200", 1), 4, want)[0]                # wrong count
    assert not c.check(good.replace("expected=1204/36864", "expected=1204/36868", 1), 4, want)[0]
    assert not c.check(good + "LoadCheckError: ... differ from the checkpoint index", 4, want)[0]
    assert not c.check(good + _line(0, 36, 768, "drafter"), 4, want)[0]                        # unexpected drafter
    assert c.check(good, 4, want)[0]


def test_cli(tmp_path):
    p = tmp_path / "boot.log"
    p.write_text("".join(_line(0, 104, 12288) for _ in range(1)))
    assert c.main(["--log", str(p), "--ranks", "1", "--dense", "104", "--experts", "12288"]) == 0
    assert c.main(["--log", str(p), "--ranks", "1", "--dense", "104", "--experts", "12288",
                   "--drafter-dense", "36", "--drafter-experts", "3072"]) == 1
