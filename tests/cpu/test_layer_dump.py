"""EXL3_LAYER_DUMP, the test-only per-layer hidden-state dump (DESIGN M3 row 'per-layer hidden states'):

  unset: install() is a no-op, no global pre-hook and no layer hook is left behind.
  set:   install() registers a discovery pre-hook; the first module forward (the model) attaches one forward
         hook per decoder layer, which captures the layer output hidden_states and the residual stream for the
         first 16 forward passes, then removes every hook and writes layer{L}_rank{R}.pt per layer per rank.
  gating: refused while an XPU graph is capturing or while vLLM's compilation config asks for a cudagraph mode;
         a call is skipped while torch.compile is tracing.

No device: the module tree is fake (no GPU), torch.distributed is faked for the rank tag.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

pytest.importorskip("vllm", reason="needs the vLLM image")

from exl3xpu import layer_dump   # noqa: E402


def _probe(env: dict, drop: tuple = ()) -> dict:
    """Run vllm_plugin.register() in a fresh interpreter (tests/cpu/register_probe.py) under `env`."""
    penv = {k: v for k, v in dict(os.environ, PYTHONDONTWRITEBYTECODE="1", EXL3_VLLM_PATCHES="", **env).items()
            if k not in drop}
    p = subprocess.run([sys.executable, str(Path(__file__).with_name("register_probe.py")), "--xpu-platform"],
                       env=penv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=600)
    lines = [ln for ln in p.stdout.splitlines() if ln.startswith("REGISTER_PROBE ")]
    assert lines, p.stdout[-4000:]
    res = json.loads(lines[-1][len("REGISTER_PROBE "):])
    res["log"] = p.stdout
    return res


class _Leaf(nn.Module):
    def forward(self, x):
        return x


class _Layer(nn.Module):
    """A decoder layer stand-in: returns (hidden_states, residual) so both captures can be told apart."""

    def __init__(self, li: int):
        super().__init__()
        self.self_attn = _Leaf()
        self.delta = float(li + 1)

    def forward(self, x):
        return x + self.delta, x + 2.0 * self.delta


class _Model(nn.Module):
    def __init__(self, n: int = 2):
        super().__init__()
        self.layers = nn.ModuleList([_Layer(i) for i in range(n)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)[0]
        return x


@pytest.fixture(autouse=True)
def _clean():
    layer_dump.uninstall()
    layer_dump.reset()
    yield
    layer_dump.uninstall()
    layer_dump.reset()


def _setup(monkeypatch, tmp_path) -> str:
    monkeypatch.setenv("EXL3_LAYER_DUMP", str(tmp_path))
    assert layer_dump.armed() is True
    assert layer_dump.install() is True
    return str(tmp_path)


def _run(model, x, passes: int) -> None:
    for _ in range(passes):
        model(x)


# ------------------------------------------------------------------------------------------------------ gating
def test_register_skips_without_env():
    r = _probe({}, drop=("EXL3_LAYER_DUMP",))
    assert r["ok"], r
    assert "EXL3XPU_LAYER_DUMP" not in r["log"], r["log"][-2000:]


def test_register_installs_with_env(tmp_path):
    r = _probe({"EXL3_LAYER_DUMP": str(tmp_path)})
    assert r["ok"] and "EXL3XPU_LAYER_DUMP installed" in r["log"], r["log"][-3000:]
    # no model was built, so the shutdown summary reports nothing captured
    assert "EXL3XPU_LAYER_DUMP layers=0" in r["log"], r["log"][-2000:]


def test_install_without_env_is_a_noop(monkeypatch):
    monkeypatch.delenv("EXL3_LAYER_DUMP", raising=False)
    assert layer_dump.armed() is False
    assert layer_dump.install() is False
    assert layer_dump.installed() is False
    assert layer_dump.discovery_installed() is False
    assert layer_dump.discovered() == []
    assert layer_dump.summary_lines()[0].endswith("files=0")


def test_env_must_hold_a_directory(monkeypatch):
    monkeypatch.setenv("EXL3_LAYER_DUMP", "1")
    assert layer_dump.armed() is False
    assert layer_dump.install() is False
    monkeypatch.setenv("EXL3_LAYER_DUMP", "0")
    assert layer_dump.install() is False


def test_install_refuses_while_a_graph_is_capturing(monkeypatch, tmp_path, caplog):
    monkeypatch.setenv("EXL3_LAYER_DUMP", str(tmp_path))
    monkeypatch.setattr(layer_dump, "capturing", lambda: True)
    with caplog.at_level(logging.WARNING):
        assert layer_dump.install() is False
    assert layer_dump.installed() is False and layer_dump.discovery_installed() is False
    assert "XPU graphs are enabled" in caplog.text and "enforce-eager" in caplog.text


def test_install_refuses_under_a_cudagraph_mode(monkeypatch, tmp_path, caplog):
    monkeypatch.setenv("EXL3_LAYER_DUMP", str(tmp_path))
    monkeypatch.setattr(layer_dump, "capturing", lambda: False)
    monkeypatch.setattr(layer_dump, "cudagraph_mode", lambda: "FULL_DECODE_ONLY")
    assert layer_dump.graphs_enabled() is True
    with caplog.at_level(logging.WARNING):
        assert layer_dump.install() is False
    assert "cudagraph_mode=FULL_DECODE_ONLY" in caplog.text
    monkeypatch.setattr(layer_dump, "cudagraph_mode", lambda: "NONE")
    assert layer_dump.graphs_enabled() is False
    assert layer_dump.install() is True


def test_tracing_and_capture_skip_a_call(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path)
    model, x = _Model(), torch.zeros(2, 4)
    monkeypatch.setattr(torch._dynamo, "is_compiling", lambda: True)
    _run(model, x, 1)
    assert layer_dump.discovered() == [0, 1] and layer_dump.STATE.calls == 0
    monkeypatch.setattr(torch._dynamo, "is_compiling", lambda: False)
    _run(model, x, 1)
    assert layer_dump.STATE.calls == 2


# ----------------------------------------------------------------------------------------------------- capture
def test_discovery_attaches_one_hook_per_layer(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path)
    assert layer_dump.discovery_installed() is True
    _Model()(torch.zeros(2, 4))
    assert layer_dump.discovered() == [0, 1]
    assert len(layer_dump.STATE.handles) == 2
    assert layer_dump.discovery_installed() is False        # the global pre-hook is gone after the walk


def test_hook_captures_hidden_states_and_residual(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path)
    x = torch.zeros(2, 4)
    _Model()(x)
    l0, l1 = layer_dump.STATE.layers[0], layer_dump.STATE.layers[1]
    assert torch.equal(l0["hidden"][0], torch.full((2, 4), 1.0))       # layer 0: x + 1
    assert torch.equal(l0["residual"][0], torch.full((2, 4), 2.0))     # layer 0: x + 2
    assert torch.equal(l1["hidden"][0], torch.full((2, 4), 3.0))       # layer 1: 1 + 2
    assert torch.equal(l1["residual"][0], torch.full((2, 4), 5.0))     # layer 1: 1 + 4
    assert l0["hidden"][0].device.type == "cpu" and l0["hidden"][0].dtype == torch.float32


def test_single_tensor_output_gives_the_post_layer_residual_twice():
    t = torch.arange(6.0).reshape(2, 3)
    h, r = layer_dump.split_output(t)
    assert h is t and r is t


def test_16_steps_are_saved_per_layer_per_rank(monkeypatch, tmp_path):
    d = _setup(monkeypatch, tmp_path)
    _run(_Model(), torch.zeros(2, 4), layer_dump.STEPS)
    assert layer_dump.installed() is False                  # the budget's last layer removed every hook
    assert layer_dump.STATE.handles == [] and layer_dump.discovery_installed() is False
    assert layer_dump.STATE.calls == 2 * layer_dump.STEPS
    for li in (0, 1):
        p = Path(d) / f"layer{li}_rank0.pt"
        assert p.is_file()
        blob = torch.load(p, weights_only=False)
        assert blob["meta"]["layer"] == li and blob["meta"]["rank"] == 0
        assert blob["meta"]["dtype"] == "torch.float32" and blob["meta"]["steps"] == layer_dump.STEPS
        assert blob["meta"]["stacked"] is True and blob["meta"]["shape"] == [2, 4]
        assert blob["hidden"].shape == (layer_dump.STEPS, 2, 4)
        assert blob["residual"].shape == (layer_dump.STEPS, 2, 4)
        assert torch.equal(blob["hidden"], torch.full((layer_dump.STEPS, 2, 4), 1.0 + 2.0 * li))
        assert torch.equal(blob["residual"], torch.full((layer_dump.STEPS, 2, 4), 2.0 + 3.0 * li))
    assert "EXL3XPU_LAYER_DUMP layers=2 steps=16" in "\n".join(layer_dump.summary_lines())


def test_the_17th_pass_captures_nothing(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path)
    model, x = _Model(), torch.zeros(2, 4)
    _run(model, x, layer_dump.STEPS)
    calls, saved = layer_dump.STATE.calls, list(layer_dump.STATE.saved)
    _run(model, x, 1)
    assert layer_dump.STATE.calls == calls == 2 * layer_dump.STEPS
    assert layer_dump.STATE.saved == saved


def test_rank_tagging_from_distributed(monkeypatch, tmp_path):
    d = _setup(monkeypatch, tmp_path)
    _run(_Model(), torch.zeros(2, 4), 1)
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 1)
    assert layer_dump.rank() == 1
    paths = layer_dump.save()
    assert sorted(Path(p).name for p in paths) == ["layer0_rank1.pt", "layer1_rank1.pt"]
    blob = torch.load(Path(d) / "layer0_rank1.pt", weights_only=False)
    assert blob["meta"]["rank"] == 1 and blob["meta"]["steps"] == 1


def test_uninstall_writes_what_was_captured(monkeypatch, tmp_path):
    d = _setup(monkeypatch, tmp_path)
    _run(_Model(), torch.zeros(2, 4), 3)
    layer_dump.uninstall()
    assert layer_dump.installed() is False and layer_dump.STATE.done is True
    blob = torch.load(Path(d) / "layer1_rank0.pt", weights_only=False)
    assert blob["meta"]["steps"] == 3
    assert len(layer_dump.flush()) >= 1
