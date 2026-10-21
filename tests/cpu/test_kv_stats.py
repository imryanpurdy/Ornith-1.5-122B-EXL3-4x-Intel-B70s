"""EXL3_KV_STATS, the test-only KV magnitude hook (DESIGN 5.3, M2c row 'KV stats'):

  unset: install() is a no-op, Attention.forward stays stock, register() logs no KV_STATS line.
  set:   register() installs the wrapper; it records max|K|/max|V| of the raw tensors Attention.forward is
         handed, for full-attention layers only (decoder, no sliding window), folds repeats into per
         (layer, rank) maxima, and reports one line per layer/rank plus 'pass=<bool> (max < 112)'.
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

from exl3xpu import kv_stats   # noqa: E402


def _layer(name="model.language_model.layers.3.self_attn", attn_type="decoder", sliding_window=None):
    return SimpleNamespace(layer_name=name, attn_type=attn_type, sliding_window=sliding_window)


def _probe(env: dict, drop: tuple = ()) -> dict:
    penv = {k: v for k, v in dict(os.environ, PYTHONDONTWRITEBYTECODE="1", EXL3_VLLM_PATCHES="", **env).items()
            if k not in drop}
    p = subprocess.run([sys.executable, str(Path(__file__).with_name("register_probe.py")), "--xpu-platform"],
                       env=penv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=600)
    lines = [ln for ln in p.stdout.splitlines() if ln.startswith("REGISTER_PROBE ")]
    assert lines, p.stdout[-4000:]
    res = json.loads(lines[-1][len("REGISTER_PROBE "):])
    res["log"] = p.stdout
    return res


def test_register_skips_without_env():
    r = _probe({}, drop=("EXL3_KV_STATS",))
    assert r["ok"], r
    assert "EXL3XPU_KV_STATS" not in r["log"], r["log"][-2000:]


def test_register_installs_with_env():
    r = _probe({"EXL3_KV_STATS": "1"})
    assert r["ok"] and "EXL3XPU_KV_STATS installed" in r["log"], r["log"][-3000:]
    # shutdown flush with nothing measured: no full-attention layer ran, so the verdict must not pass
    assert "EXL3XPU_KV_STATS pass=False (max < 112)" in r["log"], r["log"][-2000:]


def test_install_without_env_is_a_noop(monkeypatch):
    kv_stats.uninstall()
    kv_stats.reset()
    monkeypatch.delenv("EXL3_KV_STATS", raising=False)
    assert kv_stats.install() is False
    assert kv_stats.installed() is False
    from vllm.model_executor.layers.attention import attention as attn
    assert not getattr(attn.Attention.forward, kv_stats.MARK, False)


def test_wrapper_records_raw_kv_and_verdict(monkeypatch):
    monkeypatch.setenv("EXL3_KV_STATS", "1")
    kv_stats.uninstall()
    kv_stats.reset()
    assert kv_stats.install() is True and kv_stats.installed() is True
    from vllm.model_executor.layers.attention import attention as attn
    assert getattr(attn.Attention.forward, kv_stats.MARK, False)

    # the wrapper measures what it is handed, then calls the stock forward
    seen = {}

    def fake_orig(self, query, key, value, *a, **k):
        seen["key"] = key
        return "out"

    monkeypatch.setattr(attn.Attention, "_exl3_kv_stats_orig_forward", fake_orig)
    layer = _layer()
    q = torch.zeros(4, 8)
    key = torch.full((4, 8), 9.0)
    value = torch.full((4, 8), -120.0)
    assert attn.Attention.forward(layer, q, key, value) == "out"
    assert seen["key"] is key
    (li, r), row = next(iter(kv_stats.STATS.items()))
    assert (li, r) == (3, 0) and row == {"max_k": 9.0, "max_v": 120.0, "calls": 1}

    # a smaller second call does not lower the maxima; max_v >= 112 fails the verdict
    attn.Attention.forward(layer, q, torch.full((4, 8), 3.0), torch.full((4, 8), 7.0))
    assert kv_stats.STATS[(3, 0)]["max_k"] == 9.0 and kv_stats.STATS[(3, 0)]["max_v"] == 120.0
    assert kv_stats.STATS[(3, 0)]["calls"] == 2
    lines = kv_stats.report()
    assert lines[0] == "EXL3XPU_KV_STATS layer=3 rank=0 max_k=9 max_v=120", lines
    assert lines[-1] == "EXL3XPU_KV_STATS pass=False (max < 112)", lines

    # bf16 KV, under the threshold: 448/4 = 112 passes
    kv_stats.reset()
    attn.Attention.forward(layer, q, torch.full((4, 8), 100.0, dtype=torch.bfloat16),
                           torch.full((4, 8), 111.0, dtype=torch.bfloat16))
    assert kv_stats.report()[-1] == "EXL3XPU_KV_STATS pass=True (max < 112)"
    kv_stats.uninstall()


def test_only_full_attention_layers_count():
    kv_stats.reset()
    key = torch.full((2, 2), 1.0)
    assert not kv_stats.record(_layer(sliding_window=128), key, key)
    assert not kv_stats.record(_layer(attn_type="encoder"), key, key)
    assert kv_stats.record(_layer(name="model.language_model.layers.7.self_attn"), key, key)
    assert list(kv_stats.STATS) == [(7, 0)]
    kv_stats.reset()
    assert kv_stats.report()[0].startswith("EXL3XPU_KV_STATS measured=0")
    assert kv_stats.report()[-1] == "EXL3XPU_KV_STATS pass=False (max < 112)"
