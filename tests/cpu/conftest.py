"""CPU tests (no device). Run inside the vLLM image with the repo mounted read-only, see tests/README.md.

EXL3_MODEL_DIR: the Ornith-1.5-122B EXL3 checkpoint as published (read-only). Tests that need it are skipped without it.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
for p in (_HERE, _ROOT / "core" / "src", _ROOT / "xpu"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from results import RESULTS as _RESULTS  # noqa: E402  (record() lives in results.py)


@pytest.fixture(scope="session")
def model_dir() -> str:
    d = os.environ.get("EXL3_MODEL_DIR", "")
    if not d or not os.path.isfile(os.path.join(d, "model.safetensors.index.json")):
        pytest.skip("EXL3_MODEL_DIR not set or has no model.safetensors.index.json")
    return d


@pytest.fixture(scope="session")
def ckpt(model_dir):
    from st_range import Checkpoint
    return Checkpoint(model_dir)


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Print results.record() numbers; also append them as JSON lines to $EXL3_TEST_RESULTS if set."""
    if not _RESULTS:
        return
    terminalreporter.section("exl3xpu recorded numbers")
    for k, v in _RESULTS:
        terminalreporter.write_line(f"{k}: {v}")
    out = os.environ.get("EXL3_TEST_RESULTS")
    if out:
        with open(out, "a") as f:
            for k, v in _RESULTS:
                f.write(json.dumps({"key": k, "value": v}) + "\n")
