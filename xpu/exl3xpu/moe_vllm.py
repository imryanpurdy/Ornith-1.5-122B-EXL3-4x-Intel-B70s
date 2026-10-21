"""EXL3 routed experts as a vLLM 0.30 FusedMoEMethodBase (DESIGN rev 2, section 2.4).

The compute path of the SGLang method (sglang_plugin.Exl3XpuMoEMethod -> ExpertStore.forward ->
exl3xpu_moe::moe_forward), without its host-USM tier, expert cache, staging and prefetch: every local expert of a
layer lives in one device arena [E_local, blob] and the grouped kernel reads them through a static pointer table.

Splitting over the TP group (attention TP stays as vLLM builds it):
  EP (--enable-expert-parallel): rank r owns whole experts (vLLM's expert map, linear placement: 64 of 256 at EP4).
      Global ids are mapped to local ids on the device with the expert map; remote experts and padding become -1,
      which the kernel skips.
  MoE-TP (no EP): every rank owns every expert, narrowed to I/tp: gate/up columns [r*Ip, +Ip), down rows
      [r*Ip, +Ip) (tp.plan_col / tp.plan_row on the checkpoint tensors).
Either way apply() returns this rank's partial routed output (routing weights applied, shared expert not
included), and vLLM's MoE runner adds the shared expert and does the single all-reduce (moe_kernel is None).

Checkpoint names reach the method relative to the experts module, e.g. "17.up_proj.trellis"; RoutedExperts'
stock loader (3-D fused-checkpoint path) is bypassed by an instance-level load_weights override, because
MoERunner.load_weights delegates to routed_experts.load_weights.

Environment:
  EXL3_DUMP_LAYER=<n|mtp> EXL3_DUMP_DIR=<dir> [EXL3_DUMP_MAX=16]
      eager-only debug dump of that layer's MoE inputs and output from apply() (hooks on RoutedExperts never fire:
      the runner calls forward_modular). Files <dir>/L<n>_r<rank>_s<step>.pt. Never while a graph is recorded, and
      never for an all-padding batch (vLLM's profile and capture-warmup forwards).
"""
from __future__ import annotations

import os
import re
import types

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import FUSED_MOE_UNQUANTIZED_CONFIG
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import FusedMoEMethodBase

from . import loadcheck, tp

logger = init_logger("vllm.exl3xpu.moe")

MAX_ROWS = loadcheck.MOE_PLAN_MAX_M          # apply() splits larger batches into chunks of this many rows
MUL1_MARKER = -2082680531                    # int32 view of 0x83DCD12D (exllamav3 mul1 codebook)
ROLES = ("gate_proj", "up_proj", "down_proj")
PARTS = ("trellis", "suh", "svh", "mul1")
_NAME_RE = re.compile(r"(\d+)\.(gate_proj|up_proj|down_proj)\.(trellis|suh|svh|mul1)")
_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")

_DUMP_LAYER = os.environ.get("EXL3_DUMP_LAYER", "")
_DUMP_DIR = os.environ.get("EXL3_DUMP_DIR", "")
_DUMP_MAX = int(os.environ.get("EXL3_DUMP_MAX", "16"))


def blob_bytes(H: int, I: int, K: int) -> int:
    """Expert blob size (exl3_moe.sycl layout): gate|up trellis + down trellis at K bits/weight, then 6 fp16 scale
    vectors (suh_g suh_u svh_g svh_u suh_d svh_d). process_weights_after_loading checks it against the library."""
    return (H * 2 * I * K) // 8 + (I * H * K) // 8 + 2 * (3 * H + 3 * I)


def layer_index(prefix: str) -> str:
    """'language_model.model.layers.17.mlp.experts' -> '17'; MTP experts -> 'mtp'."""
    parts = prefix.split(".")
    if "mtp" in parts:
        return "mtp"
    m = _LAYER_RE.search(prefix + ".")
    return m.group(1) if m else prefix


def cpu_expert_map(layer) -> list[int]:
    """Global expert id -> local slot (-1 = owned by another rank), computed on the CPU from the MoE parallel
    config and checked against the layer's own expert map when that is a real (non-meta) tensor."""
    from vllm.model_executor.layers.fused_moe.expert_map_manager import determine_expert_map
    pc = layer.moe_config.moe_parallel_config
    E = layer.global_num_experts
    if not pc.use_ep:
        return list(range(E))
    with torch.device("cpu"):
        n_local, emap, _ = determine_expert_map(pc.ep_size, pc.ep_rank, E, layer.expert_placement_strategy)
    cpu = emap.tolist()
    have = layer.expert_map
    if have is None:
        raise ValueError(f"exl3: {layer.layer_name}: expert parallel without an expert map")
    if have.device.type != "meta":
        got = have.detach().to("cpu", torch.int64).tolist()
        if got != cpu:
            raise ValueError(f"exl3: {layer.layer_name}: vLLM's expert map differs from the linear placement the "
                             f"loader assumes")
    if n_local != layer.local_num_experts:
        raise ValueError(f"exl3: {layer.layer_name}: {n_local} local experts by placement, layer has "
                         f"{layer.local_num_experts}")
    return cpu


class Exl3MoEMethod(FusedMoEMethodBase):
    """Routed experts of one MoE layer as EXL3 blobs in a device arena (see the module docstring)."""

    def __init__(self, config, moe, prefix: str, K: int):
        super().__init__(moe)
        self.config = config
        self.prefix = prefix
        self.K = int(K)

    # ------------------------------------------------------------------------------------------- weights
    def create_weights(self, layer, num_experts: int, hidden_size: int, intermediate_size_per_partition: int,
                       params_dtype: torch.dtype, **extra_weight_attrs):
        moe = layer.moe_config
        pc = moe.moe_parallel_config
        H, Ip, E = int(hidden_size), int(intermediate_size_per_partition), int(num_experts)
        bad = []
        if H % 128 or Ip % 128:
            bad.append(f"hidden {H} and intermediate/partition {Ip} must be multiples of 128")
        if not moe.is_act_and_mul or moe.activation != MoEActivation.SILU:
            bad.append(f"activation {moe.activation} (only SiLU-gated experts)")
        if getattr(layer, "apply_router_weight_on_input", False):
            bad.append("apply_router_weight_on_input")
        if getattr(layer, "routed_scaling_factor", 1.0) not in (None, 1.0):
            bad.append(f"routed_scaling_factor {layer.routed_scaling_factor}")
        if moe.has_bias:
            bad.append("expert bias")
        if layer.expert_map_manager.num_fused_shared_experts:
            bad.append("fused shared experts")
        if pc.dp_size != 1 or pc.pcp_size != 1 or pc.use_all2all_kernels or pc.is_sequence_parallel:
            bad.append(f"dp {pc.dp_size} / pcp {pc.pcp_size} / all2all {pc.use_all2all_kernels} / sequence parallel "
                       f"{pc.is_sequence_parallel} (only TP with EP or MoE-TP over the TP group)")
        if pc.enable_eplb:
            bad.append("EPLB")
        if H != moe.hidden_dim or E != layer.local_num_experts:
            bad.append(f"sizes: hidden {H} vs {moe.hidden_dim}, local experts {E} vs {layer.local_num_experts}")
        if moe.intermediate_size % pc.tp_size or Ip != moe.intermediate_size // pc.tp_size:
            bad.append(f"intermediate {moe.intermediate_size} / moe tp {pc.tp_size} != {Ip}")
        if params_dtype not in (torch.float16, torch.bfloat16):
            bad.append(f"activation dtype {params_dtype}")
        if bad:
            raise NotImplementedError(f"exl3: {self.prefix}: unsupported MoE configuration: " + "; ".join(bad))

        blob = blob_bytes(H, Ip, self.K)
        layer.exl3_prefix = self.prefix
        layer.exl3_config = self.config
        layer.exl3_K, layer.exl3_H, layer.exl3_I, layer.exl3_E = self.K, H, Ip, E
        layer.exl3_I_full = moe.intermediate_size
        layer.exl3_global_E = layer.global_num_experts
        layer.exl3_topk = moe.experts_per_token
        layer.exl3_blob = blob
        layer.exl3_dtype = params_dtype
        layer.exl3_use_ep = bool(pc.use_ep)
        layer.exl3_ep_size, layer.exl3_ep_rank = pc.ep_size, pc.ep_rank
        layer.exl3_moe_tp_size, layer.exl3_moe_tp_rank = pc.tp_size, pc.tp_rank
        layer.exl3_layer_idx = layer_index(self.prefix)
        layer.exl3_cpu_map = cpu_expert_map(layer)
        if sum(1 for v in layer.exl3_cpu_map if v >= 0) != E:
            raise ValueError(f"exl3: {self.prefix}: expert map holds {sum(v >= 0 for v in layer.exl3_cpu_map)} local "
                             f"experts, layer has {E}")
        # device arena (the default device of model construction), one blob per local expert
        layer.register_buffer("exl3_arena", torch.empty((E, blob), dtype=torch.uint8), persistent=False)
        layer.exl3_stage = {}
        layer.exl3_packed = [False] * E
        layer.exl3_ptrs = None
        layer.exl3_emap = None
        layer.exl3_kernel = None
        layer.exl3_dump_step = 0
        for w in ("w13", "w2"):
            for part in PARTS:
                layer.register_parameter(f"{w}_{part}", torch.nn.Parameter(torch.empty(0, dtype=torch.uint8),
                                                                             requires_grad=False))
        layer.load_weights = types.MethodType(_load_weights, layer)
        loadcheck.add_layer(layer, "moe")

    def get_fused_moe_quant_config(self, layer):
        return FUSED_MOE_UNQUANTIZED_CONFIG

    def process_weights_after_loading(self, layer) -> None:
        E = layer.exl3_E
        missing = [i for i in range(E) if not layer.exl3_packed[i]]
        if missing or layer.exl3_stage:
            partial = {l: {r: sorted(p) for r, p in st.items()} for l, st in list(layer.exl3_stage.items())[:3]}
            raise ValueError(f"exl3: {self.prefix}: {E - len(missing)} of {E} local experts loaded; missing local "
                             f"slots {missing[:8]}{'...' if len(missing) > 8 else ''}; partial {partial}")
        from .moe_offload import ops as moe_ops
        X = moe_ops()
        lib_blob = int(X.blob_bytes(layer.exl3_H, layer.exl3_I, layer.exl3_K))
        if lib_blob != layer.exl3_blob:
            raise ValueError(f"exl3: {self.prefix}: blob layout mismatch: library {lib_blob} B, plugin "
                             f"{layer.exl3_blob} B")
        from .moe_offload import s64
        arena = layer.exl3_arena
        dev = arena.device
        # USM device addresses can lie above 2**63: the int64 table holds their two's-complement (as ExpertStore)
        layer.exl3_ptrs = (s64(arena.data_ptr()) + torch.arange(E, dtype=torch.int64) * layer.exl3_blob).to(dev)
        if layer.exl3_use_ep:
            layer.exl3_emap = layer.expert_map.to(device=dev, dtype=torch.int32).contiguous()
            if layer.exl3_emap.shape != (layer.exl3_global_E,):
                raise ValueError(f"exl3: {self.prefix}: expert map shape {tuple(layer.exl3_emap.shape)}")
        # a planner miss must never reach serving (TORCH_CHECKs in moe_forward would kill the engine)
        loadcheck.moe_info_once(dev)
        plans = loadcheck.moe_plans(layer.exl3_H, layer.exl3_I, layer.exl3_K)
        misses = [m for p, m in plans.items() if not any(p)]
        if misses:
            raise ValueError(f"exl3: {self.prefix}: no MoE kernel for (H={layer.exl3_H}, I={layer.exl3_I}, "
                             f"K={layer.exl3_K}) from M={min(misses)} (moe_plan)")
        layer.exl3_plans = plans
        layer.exl3_kernel = X.moe_forward
        for w in ("w13", "w2"):
            for part in PARTS:
                delattr(layer, f"{w}_{part}")
        del layer.exl3_stage
        layer.__dict__.pop("_exl3_staging", None)          # the pinned staging buffer (load time only)
        loadcheck.layer_done(layer)

    # ------------------------------------------------------------------------------------------- forward
    def apply(self, layer, x: torch.Tensor, topk_weights: torch.Tensor, topk_ids: torch.Tensor,
              shared_experts=None, shared_experts_input=None) -> torch.Tensor:
        return moe_apply(layer, x, topk_weights, topk_ids)


def map_ids(layer, topk_ids: torch.Tensor) -> torch.Tensor:
    """Global expert ids (-1 = padding) -> local ids for the kernel (-1 = padding or remote expert)."""
    ids = topk_ids if topk_ids.dtype == torch.int32 else topk_ids.to(torch.int32)
    emap = layer.exl3_emap
    if emap is not None:
        local = emap.index_select(0, ids.clamp(min=0).reshape(-1)).view(ids.shape)
        ids = torch.where(ids < 0, ids, local)
    return ids.contiguous()


def moe_apply(layer, x: torch.Tensor, topk_weights: torch.Tensor, topk_ids: torch.Tensor) -> torch.Tensor:
    """This rank's partial routed-expert output [M, H] (bf16/fp16 like x): sum over the top-k of w * expert(x),
    local experts only, not reduced over ranks, shared expert not included. No host sync, current stream."""
    if x.dim() != 2 or not x.is_contiguous():
        x = x.reshape(-1, x.shape[-1]).contiguous()
    ids = map_ids(layer, topk_ids)
    w = topk_weights if topk_weights.dtype == torch.float32 and topk_weights.is_contiguous() \
        else topk_weights.float().contiguous()
    fn, ptrs, I, K, E = layer.exl3_kernel, layer.exl3_ptrs, layer.exl3_I, layer.exl3_K, layer.exl3_E
    M = x.shape[0]
    if M <= MAX_ROWS:
        out = fn(x, ids, w, ptrs, I, K, E)
    else:
        # bounds the kernel workspace at the M = 1024 size whatever the batched-token budget; rows are independent
        out = torch.cat([fn(x[i:i + MAX_ROWS], ids[i:i + MAX_ROWS], w[i:i + MAX_ROWS], ptrs, I, K, E)
                         for i in range(0, M, MAX_ROWS)])
    if _DUMP_LAYER and layer.exl3_layer_idx == _DUMP_LAYER:
        _dump(layer, x, topk_ids, topk_weights, ids, out)
    return out


def _capturing(t: torch.Tensor) -> bool:
    if t.device.type != "xpu":
        return False
    return bool(torch.xpu.is_current_stream_capturing())


def _dump(layer, x, topk_ids, topk_weights, ids, out) -> None:
    if _capturing(x) or layer.exl3_dump_step >= _DUMP_MAX:
        return
    if not _DUMP_DIR:
        raise RuntimeError("exl3: EXL3_DUMP_LAYER is set without EXL3_DUMP_DIR")
    # vLLM's profile run and every pre-capture warmup forward are all padding (ids -1): they would fill the
    # EXL3_DUMP_MAX slots with batches that route nowhere. Dump only batches with at least one routed row.
    if not bool((topk_ids >= 0).any()):
        return
    rank = layer.exl3_ep_rank if layer.exl3_use_ep else layer.exl3_moe_tp_rank
    os.makedirs(_DUMP_DIR, exist_ok=True)
    path = os.path.join(_DUMP_DIR, f"L{layer.exl3_layer_idx}_r{rank}_s{layer.exl3_dump_step}.pt")
    layer.exl3_dump_step += 1
    torch.save({"prefix": layer.exl3_prefix, "rank": rank, "use_ep": layer.exl3_use_ep,
                "ep": (layer.exl3_ep_size, layer.exl3_ep_rank), "moe_tp": (layer.exl3_moe_tp_size,
                                                                           layer.exl3_moe_tp_rank),
                "H": layer.exl3_H, "I": layer.exl3_I, "K": layer.exl3_K, "E": layer.exl3_E,
                "x": x.detach().cpu(), "topk_ids": topk_ids.detach().cpu(), "topk_weights": topk_weights.detach().cpu(),
                "local_ids": ids.detach().cpu(), "out": out.detach().cpu()}, path)


# ------------------------------------------------------------------------------------------------- loading
def _staging(layer) -> torch.Tensor:
    buf = getattr(layer, "_exl3_staging", None)
    if buf is None:
        pin = layer.exl3_arena.device.type == "xpu"
        buf = torch.empty(layer.exl3_blob, dtype=torch.uint8, device="cpu", pin_memory=pin)
        layer._exl3_staging = buf
    return buf


def _narrow(layer, eid: int, role: str, part: str, t: torch.Tensor) -> torch.Tensor:
    """MoE-TP slice of one expert tensor (whole under EP / tp 1); checks the full checkpoint shape."""
    H, I = layer.exl3_H, layer.exl3_I_full
    key = f"{layer.exl3_prefix}.{eid}.{role}"
    r, n = layer.exl3_moe_tp_rank, layer.exl3_moe_tp_size
    plan = tp.plan_row(key, I, H, r, n) if role == "down_proj" else tp.plan_col(key, H, I, r, n)
    return plan.narrow(part, t)


def _check_part(layer, eid: int, role: str, part: str, t: torch.Tensor) -> None:
    where = f"exl3: {layer.exl3_prefix}: expert {eid} {role}.{part}"
    if part == "trellis":
        if t.dtype != torch.int16 or t.dim() != 3 or t.shape[-1] != 16 * layer.exl3_K:
            raise ValueError(f"{where}: {tuple(t.shape)} {t.dtype}, expected int16 [.., .., {16 * layer.exl3_K}] "
                             f"(K={layer.exl3_K})")
    elif part == "mul1":
        if t.numel() != 1 or int(t.reshape(-1)[0].item()) != MUL1_MARKER:
            raise ValueError(f"{where}: codebook marker {t.reshape(-1)[:1].tolist()} is not mul1 ({MUL1_MARKER})")
    elif t.dtype != torch.float16 or t.dim() != 1:
        raise ValueError(f"{where}: {tuple(t.shape)} {t.dtype}, expected a float16 vector")


def _pack(layer, slot: int) -> None:
    from .moe_offload import pack_expert
    st = layer.exl3_stage.pop(slot)
    g, u, d = (st[r] for r in ROLES)
    arena = layer.exl3_arena
    if arena.device.type != "xpu":                 # host arena (CPU tests)
        pack_expert(g, u, d, layer.exl3_K, out=arena[slot])
    else:
        buf = _staging(layer)
        pack_expert(g, u, d, layer.exl3_K, out=buf)
        arena[slot].copy_(buf)                      # pinned -> device
        torch.xpu.current_stream(arena.device).synchronize()   # buf is reused for the next expert (load time only)
    layer.exl3_packed[slot] = True


def _load_weights(layer, weights):
    """Instance-level RoutedExperts.load_weights: names relative to the experts module ('17.up_proj.trellis')."""
    cpu_map = layer.exl3_cpu_map
    E_glob = layer.exl3_global_E
    batch = layer.exl3_batch
    for name, w in weights:
        m = _NAME_RE.fullmatch(name)
        if m is None:
            raise ValueError(f"exl3: {layer.exl3_prefix}: unexpected tensor {name!r} under the routed experts")
        eid, role, part = int(m.group(1)), m.group(2), m.group(3)
        if not 0 <= eid < E_glob:
            raise ValueError(f"exl3: {layer.exl3_prefix}: expert id {eid} out of range ({E_glob} experts)")
        slot = cpu_map[eid]
        if slot < 0:
            batch.counts["experts_dropped"] += 1
            continue
        if layer.exl3_packed[slot]:
            raise ValueError(f"exl3: {layer.exl3_prefix}: expert {eid} {role}.{part} delivered after the expert was "
                             f"complete")
        t = _narrow(layer, eid, role, part, w)
        _check_part(layer, eid, role, part, t)
        st = layer.exl3_stage.setdefault(slot, {})
        rs = st.setdefault(role, {})
        if part in rs:
            raise ValueError(f"exl3: {layer.exl3_prefix}: expert {eid} {role}.{part} loaded twice")
        rs[part] = t
        batch.counts["experts"] += 1
        if len(st) == 3 and all(len(st[r]) == 4 for r in ROLES):
            _pack(layer, slot)
        yield f"{'w2' if role == 'down_proj' else 'w13'}_{part}"
