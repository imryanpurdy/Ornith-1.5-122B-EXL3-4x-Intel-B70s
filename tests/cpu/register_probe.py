"""Runs exl3xpu.vllm_plugin.register() in a fresh interpreter (as vLLM's plugin loader does in every process) and
prints one JSON line: {"ok", "error", "registered", "guard", "patches", "fp8kv_patched"}. The environment of the
call decides what register() sees (EXL3_VLLM_PATCHES, EXL3_DRAFT_VOCAB, EXL3_CAPTURE_GUARD).

With --xpu-platform the XPU platform is installed first (as plugin_harness.py does), so modules that consult
vLLM's platform at import (the flash-attention backend the fp8kv patch wraps) import as they would on the box."""
from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
for p in (_ROOT / "core" / "src", _ROOT / "xpu"):
    sys.path.insert(0, str(p))

import torch  # noqa: E402

if "--xpu-platform" in sys.argv:
    import vllm.platforms as _P
    from vllm.platforms.xpu import XPUPlatform
    XPUPlatform.current_device = classmethod(lambda cls: torch.device("cpu"))
    _P._current_platform = XPUPlatform()

res = {"ok": False}
try:
    from exl3xpu import graph_guard, vllm_plugin
    vllm_plugin.register()
    from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS, get_quantization_config
    res["registered"] = "exl3" in QUANTIZATION_METHODS and get_quantization_config("exl3") is vllm_plugin.Exl3Config
    res["guard"] = graph_guard.installed()
    res["patches"] = vllm_plugin.apply_patches()      # second call: idempotent view of what register() applied
    try:
        from vllm.v1.attention.backends import flash_attn as fa
        res["fp8kv_patched"] = bool(getattr(fa.FlashAttentionImpl, "_exl3_fp8kv_patched", False))
    except Exception as e:     # noqa: BLE001
        res["fp8kv_patched"] = f"flash_attn import failed: {type(e).__name__}: {e}"
    res["ok"] = True
except Exception as e:         # noqa: BLE001
    res["error"] = f"{type(e).__name__}: {e}"
print("REGISTER_PROBE " + json.dumps(res))
