"""CPU tests (no device). Run inside the vLLM image with the repo mounted read-only, see tests/README.md.

EXL3_MODEL_DIR: the Ornith-1.5-122B EXL3 checkpoint as published (read-only). Tests that need it are skipped without it.
EXL3_MINI_DIR:  Ornith-mini (tools/make_ornith_mini.py), read-only. The plugin's real-tensor tests skip without it.
EXL3_LIB_DIR:   a build of the kernels (_C.so, _moe.so), for the host ops the plugin calls at load time
                (exl3_supported, blob_bytes, moe_plan, moe_info). The plugin's load tests skip without it.
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


@pytest.fixture(scope="session")
def mini_dir() -> str:
    d = os.environ.get("EXL3_MINI_DIR", "")
    if not d or not os.path.isfile(os.path.join(d, "model.safetensors.index.json")):
        pytest.skip("EXL3_MINI_DIR not set or has no model.safetensors.index.json")
    return d


@pytest.fixture(scope="session")
def lib_dir() -> str:
    d = os.environ.get("EXL3_LIB_DIR", "")
    if not d or not all(os.path.isfile(os.path.join(d, f)) for f in ("_C.so", "_moe.so")):
        pytest.skip("EXL3_LIB_DIR not set or lacks _C.so / _moe.so")
    return d


_HARNESS_CACHE: dict = {}


def run_harness(command: str, model: str, cases: list, lib: str | None = None, timeout: int = 3600) -> list:
    """Run tests/cpu/plugin_harness.py in a fresh interpreter (vLLM binds its platform at import) and return its
    JSON result; cached per session. Its stderr (one line per case) is printed on failure."""
    import subprocess
    import tempfile
    key = (command, model, json.dumps(cases, sort_keys=True), lib)
    if key in _HARNESS_CACHE:
        return _HARNESS_CACHE[key]
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    if lib:
        env["EXL3_LIB"] = os.path.join(lib, "_C.so")
        env["EXL3_MOE_LIB"] = os.path.join(lib, "_moe.so")
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "out.json")
        p = subprocess.run([sys.executable, str(_HERE / "plugin_harness.py"), command, "--model", model,
                            "--cases", json.dumps(cases), "--json-out", out],
                           env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout)
        if p.returncode != 0 or not os.path.isfile(out):
            raise AssertionError(f"plugin_harness {command} rc={p.returncode}\n{p.stderr[-8000:]}")
        res = json.loads(Path(out).read_text())
    progress = [ln for ln in p.stderr.splitlines() if ln.startswith(("routing ", "meta ", "real ", "moe ", "units "))]
    _HARNESS_CACHE[key] = res
    for ln in progress:
        _RESULTS.append((f"harness {command}", ln))
    return res
