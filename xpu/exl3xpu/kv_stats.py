"""EXL3_KV_STATS: test-only measurement of the raw K/V that reaches the FP8 KV cache (DESIGN 5.3, M2c row
'KV stats').

Why: the scales row of DESIGN 5.3 runs the FP8 KV cache with k_scale = v_scale = 1.0 because the checkpoint
carries no scales (vLLM's set_default_quant_scales, attention.py:129-150). That is only safe if the magnitudes
fit e4m3fn at that scale: V is un-normalized and e4m3fn's max is 448, so the pass criterion for scale 1.0 is
max|K| < 112 and max|V| < 112 (= 448/4) on every full-attention layer, on every rank, over the mini's M2c run
(layer 3, 16 prompts up to 32K).

What it does: with EXL3_KV_STATS=1, register() installs this module. It wraps
vllm.model_executor.layers.attention.attention.Attention.forward and, for each call of a full-attention layer
(attn_type 'decoder', no sliding window), records max|key| and max|value| of the *incoming* tensors: the raw
bf16/fp16 K/V before the backend scales them to fp8 and writes the KV cache. The cache write happens later in
the same forward (unified_kv_cache_update -> impl.do_kv_cache_update), but that path is bound into
torch.ops.vllm.unified_kv_cache_update at import time, so wrapping the module-level function would miss the
opaque-op path. Attention.forward is the single point that sees the raw tensors on both the direct-call and
the custom-op path, for an fp8 KV cache and a bf16 KV cache alike (a bf16 run measures the same numbers, which
is how the threshold can be checked without a quantized cache).

Eager-safe: a measurement syncs (.item()), so it is skipped while torch.compile is tracing and while an XPU
graph is being captured. Zero overhead when EXL3_KV_STATS is unset: install() returns before importing or
patching anything. Warmup/profile dummy passes go through the same forward and are counted, so the reported
maxima are an upper bound over the whole process.

Output: one line per (layer, rank) at process shutdown, then the verdict:
    EXL3XPU_KV_STATS layer=3 rank=0 max_k=... max_v=...
    EXL3XPU_KV_STATS pass=True (max < 112)
pass is False when nothing was measured or when any max is >= 112.
"""
from __future__ import annotations

import atexit
import logging
import os
import re
import threading

import torch

logger = logging.getLogger("vllm.exl3xpu.kv_stats")

ENV = "EXL3_KV_STATS"
FP8_MAX = 448.0
THRESHOLD = FP8_MAX / 4.0          # 112: with scale 1.0 every |K|, |V| must stay under a quarter of e4m3fn's max
MARK = "_exl3_kv_stats_patched"

STATS: dict[tuple[int, int], dict[str, float]] = {}      # (layer, rank) -> {"max_k", "max_v", "calls"}
_lock = threading.Lock()
_installed = False
_atexit = False


def max_abs(t) -> float | None:
    """max|t| as a float; None for a missing, non-tensor or empty tensor."""
    if t is None or not torch.is_tensor(t) or t.numel() == 0:
        return None
    return float(t.detach().to(torch.float32).abs().max().item())


def layer_index(layer) -> int | None:
    name = getattr(layer, "layer_name", "") or ""
    try:
        from vllm.model_executor.models.utils import extract_layer_index
        return int(extract_layer_index(name))
    except Exception:      # noqa: BLE001
        m = re.search(r"layers\.(\d+)", name)
        return int(m.group(1)) if m else None


def is_full_attention(layer) -> bool:
    """Decoder attention without a sliding window: the layers whose KV cache the scales row covers."""
    attn_type = getattr(layer, "attn_type", None)
    if str(getattr(attn_type, "value", attn_type)) != "decoder":
        return False
    return getattr(layer, "sliding_window", None) is None


def rank() -> int:
    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return int(torch.distributed.get_rank())
    except Exception:      # noqa: BLE001
        pass
    try:
        if torch.xpu.is_available():
            return int(torch.xpu.current_device())
    except Exception:      # noqa: BLE001
        pass
    return 0


def measurable() -> bool:
    """False while tracing (a .item() would break the graph) or while capturing an XPU graph (no sync allowed)."""
    try:
        if torch._dynamo.is_compiling():
            return False
    except Exception:      # noqa: BLE001
        pass
    try:
        if torch.xpu.is_available() and torch.xpu.is_current_stream_capturing():
            return False
    except Exception:      # noqa: BLE001
        pass
    return True


def record(layer, key, value, r: int | None = None) -> bool:
    """Fold one call's max|K|/max|V| into the per (layer, rank) maxima; False if the layer does not count."""
    if not is_full_attention(layer):
        return False
    li = layer_index(layer)
    if li is None:
        return False
    mk, mv = max_abs(key), max_abs(value)
    if mk is None or mv is None:
        return False
    r = rank() if r is None else int(r)
    with _lock:
        cur = STATS.setdefault((li, r), {"max_k": mk, "max_v": mv, "calls": 0})
        cur["max_k"] = max(cur["max_k"], mk)
        cur["max_v"] = max(cur["max_v"], mv)
        cur["calls"] += 1
        periodic = cur["calls"] % 256 == 0
    if periodic:
        flush()                       # periodic line, so a killed process still leaves its maxima behind
    return True


def verdict() -> bool:
    with _lock:
        rows = list(STATS.values())
    return bool(rows) and all(v["max_k"] < THRESHOLD and v["max_v"] < THRESHOLD for v in rows)


def report() -> list[str]:
    """One line per (layer, rank), then the verdict line."""
    with _lock:
        rows = sorted(STATS.items())
    lines = [f"EXL3XPU_KV_STATS layer={li} rank={r} max_k={v['max_k']:.6g} max_v={v['max_v']:.6g}"
             for (li, r), v in rows]
    if not rows:
        lines.append("EXL3XPU_KV_STATS measured=0 (no full-attention layer call was seen)")
    lines.append(f"EXL3XPU_KV_STATS pass={verdict()} (max < {THRESHOLD:g})")
    return lines


def flush() -> list[str]:
    lines = report()
    prev, logging.raiseExceptions = logging.raiseExceptions, False   # at exit the log stream may be closed
    try:
        for ln in lines:
            logger.info("%s", ln)
    finally:
        logging.raiseExceptions = prev
    return lines


def reset() -> None:
    with _lock:
        STATS.clear()


def install() -> bool:
    """Wrap Attention.forward when EXL3_KV_STATS=1 (idempotent). False when the env is unset or vLLM is unusable."""
    global _installed, _atexit
    if os.environ.get(ENV, "0") != "1":
        return False
    if _installed:
        return True
    try:
        from vllm.model_executor.layers.attention import attention as attn
    except Exception as e:     # noqa: BLE001
        logger.warning("EXL3XPU_KV_STATS install failed: attention module not importable (%s)", e)
        return False
    cls = attn.Attention
    cur = cls.forward
    if not getattr(cur, MARK, False):
        def forward(self, query, key, value, *args, **kwargs):
            try:
                if measurable():
                    record(self, key, value)
            except Exception:      # noqa: BLE001 - a measurement must never break serving
                pass
            return cls._exl3_kv_stats_orig_forward(self, query, key, value, *args, **kwargs)

        setattr(forward, MARK, True)
        cls._exl3_kv_stats_orig_forward = cur
        cls.forward = forward
    _installed = True
    if not _atexit:
        atexit.register(flush)
        _atexit = True
    logger.info("EXL3XPU_KV_STATS installed: max|K|/max|V| per full-attention layer and rank, pass if every max "
                "< %g (e4m3fn max %g at scale 1.0)", THRESHOLD, FP8_MAX)
    return True


def uninstall() -> None:
    """Restore the stock Attention.forward (tests, and a re-install after reset())."""
    global _installed
    try:
        from vllm.model_executor.layers.attention import attention as attn
        cls = attn.Attention
        if getattr(cls.forward, MARK, False):
            cls.forward = cls._exl3_kv_stats_orig_forward
            del cls._exl3_kv_stats_orig_forward
    except Exception:      # noqa: BLE001
        pass
    _installed = False


def installed() -> bool:
    return bool(_installed)
