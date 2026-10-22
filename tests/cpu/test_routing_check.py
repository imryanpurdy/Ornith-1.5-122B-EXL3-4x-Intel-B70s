"""EXL3_ROUTING_CHECK, the test-only MoE routing-identity hook (DESIGN M3 row 'routing identical on all ranks'):

  unset: install() is a no-op, Exl3MoEMethod.apply stays stock, observe() does nothing.
  set:   install() wraps Exl3MoEMethod.apply; the wrapper all-gathers each MoE layer's topk_ids over the TP
         group for the first 64 model steps (a step ends when the layer index wraps back), counts identical and
         mismatched steps and rows, logs the first mismatched row with every rank's ids, and reports
         'EXL3XPU_ROUTING_CHECK steps=64 identical=64 pass=<bool>'.
  gating: inactive without the env, at world size 1, and while tracing or capturing a graph.

No device: torch.distributed is faked (world size, rank, all_gather) so the comparison logic runs on CPU.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("vllm", reason="needs the vLLM image")

from exl3xpu import routing_check   # noqa: E402


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


def _layer(li=3, prefix="model.language_model.layers.3.mlp.experts"):
    return SimpleNamespace(exl3_layer_idx=str(li), exl3_prefix=prefix)


def _fake_dist(monkeypatch, world=2, r=0, others=None):
    """torch.distributed as initialized with `world` ranks; `others` maps a rank to the ids it picked."""
    others = {} if others is None else others
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: world)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: r)
    monkeypatch.setattr(routing_check, "_tp_group", lambda: None)

    def all_gather(out, t):
        assert len(out) == world
        for i, o in enumerate(out):
            o.copy_(others[i] if i in others else t)

    monkeypatch.setattr(torch.distributed, "all_gather", all_gather)


def _run_steps(layer_count=2, steps=64):
    ids = torch.arange(8 * 2, dtype=torch.int32).reshape(8, 2)
    for _ in range(steps):
        for li in range(layer_count):
            routing_check.observe(_layer(li), ids)


def test_install_without_env_is_a_noop(monkeypatch):
    routing_check.uninstall()
    routing_check.reset()
    monkeypatch.delenv("EXL3_ROUTING_CHECK", raising=False)
    assert routing_check.install() is False
    assert routing_check.installed() is False
    from exl3xpu import moe_vllm
    assert not getattr(moe_vllm.Exl3MoEMethod.apply, routing_check.MARK, False)
    assert routing_check.observe(_layer(), torch.zeros(4, 2, dtype=torch.int32)) is False
    assert routing_check.report() == []


def test_world_size_one_is_inactive(monkeypatch):
    monkeypatch.setenv("EXL3_ROUTING_CHECK", "1")
    routing_check.reset()
    _fake_dist(monkeypatch, world=1)
    assert routing_check.install() is True
    assert routing_check.active() is False
    ids = torch.zeros(4, 2, dtype=torch.int32)
    assert routing_check.observe(_layer(), ids) is False
    assert routing_check.report() == []
    routing_check.uninstall()


def test_tracing_and_capture_are_skipped(monkeypatch):
    monkeypatch.setenv("EXL3_ROUTING_CHECK", "1")
    routing_check.reset()
    _fake_dist(monkeypatch)
    monkeypatch.setattr(torch._dynamo, "is_compiling", lambda: True)
    assert routing_check.active() is False
    assert routing_check.observe(_layer(), torch.zeros(4, 2, dtype=torch.int32)) is False
    monkeypatch.setattr(torch._dynamo, "is_compiling", lambda: False)
    assert routing_check.active() is True


def test_tp_group_of_one_is_inactive(monkeypatch):
    """A pure-DP run: the world has 2 ranks but the TP group has 1, and DP ranks legitimately pick differently."""
    monkeypatch.setenv("EXL3_ROUTING_CHECK", "1")
    routing_check.reset()
    _fake_dist(monkeypatch, world=2)
    monkeypatch.setattr(routing_check, "_tp_group", lambda: SimpleNamespace(world_size=1))
    assert routing_check.tp_world_size() == 1
    assert routing_check.active() is False
    assert routing_check.observe(_layer(), torch.zeros(4, 2, dtype=torch.int32)) is False
    assert routing_check.report() == []


def test_identical_ids_pass(monkeypatch):
    monkeypatch.setenv("EXL3_ROUTING_CHECK", "1")
    routing_check.reset()
    _fake_dist(monkeypatch, world=2, r=0)
    _run_steps()
    assert routing_check.STATE.step == 64 and routing_check.STATE.done is True
    assert routing_check.STATE.identical_steps == 64
    assert routing_check.STATE.mismatch_steps == 0
    assert routing_check.STATE.rows_total == 64 * 2 * 8 and routing_check.STATE.rows_mismatched == 0
    assert routing_check.report() == ["EXL3XPU_ROUTING_CHECK steps=64 identical=64 pass=True"]
    assert routing_check.verdict() is True
    # once the budget is spent nothing is gathered any more
    assert routing_check.observe(_layer(0), torch.zeros(8, 2, dtype=torch.int32)) is False


def test_mismatch_is_reported_with_rank_detail(monkeypatch):
    monkeypatch.setenv("EXL3_ROUTING_CHECK", "1")
    routing_check.reset()
    local = torch.arange(8 * 2, dtype=torch.int32).reshape(8, 2)
    other = local.clone()
    other[5, 1] = 99                                   # rank 1 routes row 5 to a different expert
    others: dict = {}
    _fake_dist(monkeypatch, world=2, r=0, others=others)

    for li in range(2):                                # step 1, identical
        routing_check.observe(_layer(li), local)
    routing_check.observe(_layer(0), local)            # step 2 starts
    assert routing_check.STATE.step == 2 and routing_check.STATE.identical_steps == 1
    others[1] = other                                  # rank 1 starts picking differently
    routing_check.observe(_layer(1), local)            # step 2, layer 1: rank 1 differs

    lines = routing_check.report()
    assert lines[0].startswith("EXL3XPU_ROUTING_CHECK mismatch step=2 layer=1 rank=0 rows=1/8 first_row=5"), lines
    assert "local=[10, 11]" in lines[0] and "r0=[10, 11]" in lines[0] and "r1=[10, 99]" in lines[0], lines
    assert lines[-1] == "EXL3XPU_ROUTING_CHECK steps=2 identical=1 pass=False", lines
    assert routing_check.verdict() is False
    assert routing_check.STATE.rows_mismatched == 1

    # after the verdict the hook keeps gathering in lockstep but stops comparing and logging
    routing_check.observe(_layer(0), local)
    assert routing_check.STATE.step == 3 and len(routing_check.report()) == 2
    assert routing_check.STATE.done is False            # only the budget or a failed gather ends it


def test_budget_exhausted_after_a_mismatch_is_not_a_pass(monkeypatch):
    monkeypatch.setenv("EXL3_ROUTING_CHECK", "1")
    routing_check.reset()
    local = torch.zeros(8, 2, dtype=torch.int32)
    other = local.clone()
    other[0, 0] = 7
    _fake_dist(monkeypatch, world=2, r=0, others={1: other})
    _run_steps()
    assert routing_check.STATE.done is True
    # the first-mismatch verdict stands: the remaining steps are only gathered to keep the ranks in lockstep
    assert routing_check.report()[-1] == "EXL3XPU_ROUTING_CHECK steps=1 identical=0 pass=False"


def test_wrapper_installs_on_the_moe_method(monkeypatch):
    monkeypatch.setenv("EXL3_ROUTING_CHECK", "1")
    routing_check.uninstall()
    routing_check.reset()
    _fake_dist(monkeypatch)
    assert routing_check.install() is True and routing_check.installed() is True
    from exl3xpu import moe_vllm
    assert getattr(moe_vllm.Exl3MoEMethod.apply, routing_check.MARK, False)

    seen = {}

    def fake_orig(self, layer, x, topk_weights, topk_ids, shared_experts=None, shared_experts_input=None):
        seen["ids"] = topk_ids
        return "out"

    monkeypatch.setattr(moe_vllm.Exl3MoEMethod, "_exl3_routing_check_orig_apply", fake_orig)
    ids = torch.arange(8, dtype=torch.int32).reshape(4, 2)
    out = moe_vllm.Exl3MoEMethod.apply(moe_vllm.Exl3MoEMethod.__new__(moe_vllm.Exl3MoEMethod),
                                       layer=_layer(5), x=torch.zeros(4, 2), topk_weights=torch.zeros(4, 2),
                                       topk_ids=ids)
    assert out == "out" and seen["ids"] is ids
    assert routing_check.STATE.step == 1 and routing_check.STATE.rows_total == 4
    assert not any("pass=" in ln for ln in routing_check.report())
    routing_check.uninstall()
    assert not getattr(moe_vllm.Exl3MoEMethod.apply, routing_check.MARK, False)


def test_atexit_line_for_a_run_that_ended_early(monkeypatch):
    monkeypatch.setenv("EXL3_ROUTING_CHECK", "1")
    routing_check.reset()
    _fake_dist(monkeypatch)
    for li in range(2):
        routing_check.observe(_layer(li), torch.zeros(8, 2, dtype=torch.int32))
    lines = routing_check.flush()
    assert lines[0] == "EXL3XPU_ROUTING_CHECK incomplete steps=1 of 64 identical=1", lines
    assert lines[-1] == "EXL3XPU_ROUTING_CHECK steps=1 identical=1 pass=False", lines

def test_register_skips_without_env():
    r = _probe({}, drop=("EXL3_ROUTING_CHECK",))
    assert r["ok"], r
    assert "EXL3XPU_ROUTING_CHECK" not in r["log"], r["log"][-2000:]


def test_register_installs_with_env():
    r = _probe({"EXL3_ROUTING_CHECK": "1"})
    assert r["ok"] and "EXL3XPU_ROUTING_CHECK installed" in r["log"], r["log"][-3000:]
