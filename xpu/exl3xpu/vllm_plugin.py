"""
vLLM 0.30 out-of-tree quantization plugin: EXL3 (exllamav3 trellis) weights on Intel XPU, tensor parallel
(DESIGN rev 2, sections 2.2-2.6).

Registered through the `vllm.general_plugins` entry point (exl3xpu = exl3xpu.vllm_plugin:register), so every vLLM
process (API server, engine core, workers) runs register() before the model is built.

Routing (Exl3Config.get_quant_method):
  RoutedExperts (layers.N.mlp.experts, mtp.layers.0.mlp.experts)   -> moe_vllm.Exl3MoEMethod(K)
  LinearBase / ParallelLMHead whose checkpoint matrices are EXL3      -> Exl3LinearMethod(K)
  *.linear_attn.in_proj_ba and anything under visual.*                -> UnquantizedLinearMethod
  any other LinearBase / ParallelLMHead without EXL3 storage          -> raise (never load a matrix silently wrong)
  anything else (embeddings, attention, norms)                        -> None

Exl3LinearMethod keeps vLLM's partitioning: every rank receives the full checkpoint tensor of each component and
its loader cuts the rank's share with tp.plan() (column split: trellis columns + svh, suh whole; row split:
trellis rows + suh, svh whole; 128-aligned). process_weights_after_loading checks that every shard of every
component arrived exactly once and that the dense kernels support the bitrate (no silent Triton fallback;
EXL3_ALLOW_FALLBACK=1 opts in). loadcheck closes each model's batch of layers: EXL3XPU_LOADED counters, scratch
reservation and the kernel variant warmup.

Environment read by register():
  EXL3_VLLM_PATCHES   comma list of vllm_patches to apply; default empty. Only 'fp8kv' (patch_fp8kv_prefill) is
                      accepted on vLLM 0.30; 'align_sync', 'gdn_mask_index', 'xpu_block_size' (and the old '1' =
                      all) are refused with a log line. '0' / 'none' = none. A requested 'fp8kv' that does not
                      apply, or a name the plugin does not recognize, aborts the process (a partial or empty
                      patch set must never boot silently).
  EXL3_DRAFT_VOCAB    the pruned MTP draft head: refused (unvalidated on this model; a later MTP candidate).
  EXL3_CAPTURE_GUARD  'raise' or unset: the graph guard has only that mode (graph_guard.py).
  EXL3_DET_SMALL_LINEAR  '0': test-only opt-out of the 2-page-prefill GEMM chunking (vllm_patches.
                      patch_det_small_linear; installed unconditionally otherwise).
  EXL3_KV_STATS       '1': test-only, per full-attention layer and rank max|K|/max|V| of the raw
                      K/V before the FP8 KV cache write, pass if every max < 112 (kv_stats.py).
  EXL3_ROUTING_CHECK  '1': test-only, the MoE router's topk_ids all-gathered over the TP group for the
                      first 64 eager steps under TP > 1, pass if every rank picks identically
                      (routing_check.py).
  EXL3_LAYER_DUMP     '<dir>': test-only, per-layer hidden_states and residual stream of the first 16 eager
                      forward passes, one layer{L}_rank{R}.pt per layer per rank under <dir> (layer_dump.py).
"""
from __future__ import annotations

import json
import os
import re
from typing import Any

import torch
from torch.nn import Parameter

from vllm.logger import init_logger
from vllm.model_executor.layers.linear import (ColumnParallelLinear, LinearBase, LinearMethodBase,
                                               MergedColumnParallelLinear, QKVParallelLinear, ReplicatedLinear,
                                               RowParallelLinear, UnquantizedLinearMethod)
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead, VocabParallelEmbedding

from . import loadcheck, tp

logger = init_logger("vllm.exl3xpu")

# vLLM fused module -> checkpoint constituents
FUSED = {
    "gate_up_proj": ["gate_proj", "up_proj"],
    "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
    "in_proj_ba": ["in_proj_b", "in_proj_a"],
}
QKV_IDS = {"q": 0, "k": 1, "v": 2}
CB_IDS = {"3inst": 0, "mcg": 1, "mul1": 2}
# int32 view of the codebook multiplier stored in the marker tensor (trellis_core.reference MCG_MULT / MUL1_MULT)
CB_MARKER = {"mcg": 0xCBAC1FED - (1 << 32), "mul1": 0x83DCD12D - (1 << 32)}
ALLOWED_TUPLE_SHARDS = {(0, 1, 2)}       # Qwen3.5 in_proj_qkv -> in_proj_qkvz (0, 1, 2); see tp.py (tp-guard P6)
PATCHES_ACCEPTED = {"fp8kv"}
PATCHES_REFUSED = {"align_sync", "gdn_mask_index", "xpu_block_size", "1", "all"}
PATCH_ALIASES = {"fp8kv_prefill": "fp8kv", "patch_fp8kv_prefill": "fp8kv", "patch_align_sync": "align_sync",
                 "patch_gdn_mask_index": "gdn_mask_index", "patch_xpu_block_size": "xpu_block_size"}


def _norm_key(k: str) -> str:
    """Canonical module name shared by checkpoint keys and vLLM prefixes:
       'model.language_model.layers.3.mlp.gate_proj' / 'language_model.model.layers.3.mlp.gate_proj'
           -> 'layers.3.mlp.gate_proj'
       'mtp.layers.0.mlp.gate_proj' / 'model.mtp.layers.0...' -> 'mtp.layers.0.mlp.gate_proj'
       '...lm_head' -> 'lm_head'"""
    parts = k.split(".")
    if "mtp" in parts:
        return "mtp." + ".".join(parts[parts.index("mtp") + 1:])
    m = re.search(r"(layers\.\d+\..*)$", k)
    if m:
        return m.group(1)
    return parts[-1]


def _is_visual(prefix: str) -> bool:
    parts = prefix.split(".")
    return "visual" in parts or "vision_tower" in parts


class Exl3Config(QuantizationConfig):

    def __init__(self, bits: float, head_bits: int, codebook: str, storage: dict | None = None,
                 mtp_bits: int | None = None):
        super().__init__()
        if codebook not in CB_IDS:
            raise ValueError(f"exl3: unknown codebook {codebook!r}")
        self.bits = bits
        self.mtp_bits = int(mtp_bits or bits)
        self.head_bits = head_bits
        self.codebook = codebook
        self.storage = storage or {}
        self.index_path: str | None = None

    def __repr__(self):
        return f"Exl3Config(bits={self.bits}, head_bits={self.head_bits}, codebook={self.codebook})"

    def get_name(self):
        return "exl3"

    def get_supported_act_dtypes(self):
        return [torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 0

    @staticmethod
    def get_config_filenames() -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "Exl3Config":
        return cls(bits=config.get("bits", 4), head_bits=config.get("head_bits", 6),
                   codebook=config.get("codebook", "mul1"), mtp_bits=config.get("mtp_bits"))

    def _model_file(self, model_name, fname, revision):
        path = os.path.join(model_name, fname)
        if os.path.isfile(path):
            return path
        try:
            from huggingface_hub import hf_hub_download
            return hf_hub_download(model_name, fname, revision=revision)
        except Exception:
            return None

    def maybe_update_config(self, model_name: str, hf_config=None, revision=None):
        # Which modules are EXL3 is decided by the checkpoint itself: a module is EXL3 iff the weight
        # index has '<module>.trellis'. (quantization_config.json's tensor_storage omits the MTP head.)
        idx_path = self._model_file(model_name, "model.safetensors.index.json", revision)
        qc_path = self._model_file(model_name, "quantization_config.json", revision)
        self.index_path = idx_path
        stored = {}
        if qc_path:
            with open(qc_path) as f:
                for key, v in json.load(f).get("tensor_storage", {}).items():
                    if v.get("quant_format") == "exl3":
                        stored[_norm_key(key)] = int(v.get("bits_per_weight", self.bits))
        if idx_path:
            with open(idx_path) as f:
                wmap = json.load(f)["weight_map"]
            for name in wmap:
                if name.endswith(".trellis"):
                    key = _norm_key(name[: -len(".trellis")])
                    default = self.mtp_bits if key.startswith("mtp.") else int(self.bits)
                    self.storage[key] = stored.get(key, default)
        else:
            self.storage.update(stored)
        logger.info("exl3: %d EXL3 modules (%d in MTP head)", len(self.storage),
                    sum(1 for k in self.storage if k.startswith("mtp.")))

    def _bits_for(self, prefix: str) -> int | None:
        """Storage bitrate K of the checkpoint matrices behind a vLLM module prefix, None if none is EXL3."""
        if _is_visual(prefix):
            return None
        key = _norm_key(prefix)
        if not self.storage:
            raise ValueError("exl3: no EXL3 module map (maybe_update_config found neither "
                             "model.safetensors.index.json nor quantization_config.json)")
        base, _, leaf = key.rpartition(".")
        members = FUSED.get(leaf, [leaf])
        names = [f"{base}.{p}" if base else p for p in members]
        bits = [self.storage.get(n) for n in names]
        if all(b is None for b in bits):
            return None
        if any(b is None for b in bits) or len(set(bits)) != 1:
            raise ValueError(f"exl3: {prefix}: fused module mixes EXL3 bitrates / unquantized parts "
                             f"{dict(zip(names, bits))}")
        return int(bits[0])

    def expert_bits(self, prefix: str, num_experts: int) -> int:
        """K shared by all routed experts of an MoE layer (gate/up/down of every expert must be EXL3, same K)."""
        key = _norm_key(prefix)
        bits = {self.storage.get(f"{key}.{e}.{p}") for e in range(num_experts)
                for p in ("gate_proj", "up_proj", "down_proj")}
        if None in bits or len(bits) != 1:
            missing = [f"{key}.{e}.{p}" for e in range(num_experts) for p in ("gate_proj", "up_proj", "down_proj")
                       if f"{key}.{e}.{p}" not in self.storage]
            raise ValueError(f"exl3: {prefix}: routed experts are not all EXL3 with one bitrate: K {sorted(map(str, bits))}"
                             f", e.g. missing {missing[:3]}")
        return int(bits.pop())

    def get_quant_method(self, layer: torch.nn.Module, prefix: str):
        from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
        if isinstance(layer, RoutedExperts):
            if _is_visual(prefix):
                raise ValueError(f"exl3: {prefix}: unexpected MoE layer in the vision tower")
            from .moe_vllm import Exl3MoEMethod
            K = self.expert_bits(prefix, layer.global_num_experts)
            return Exl3MoEMethod(self, layer.moe_config, prefix, K)
        if isinstance(layer, (LinearBase, ParallelLMHead)):
            if _is_visual(prefix):
                if isinstance(layer, ParallelLMHead):
                    raise ValueError(f"exl3: {prefix}: unexpected lm_head in the vision tower")
                return UnquantizedLinearMethod()
            bits = self._bits_for(prefix)
            if bits is not None:
                return Exl3LinearMethod(self, bits, prefix)
            if isinstance(layer, LinearBase) and _norm_key(prefix).endswith("linear_attn.in_proj_ba"):
                return UnquantizedLinearMethod()          # in_proj_a/b are F16 in the checkpoint
            raise ValueError(f"exl3: {prefix}: {type(layer).__name__} has no EXL3 matrices in the checkpoint and is "
                             "not one of the known unquantized linears (in_proj_ba, visual.*)")
        if isinstance(layer, VocabParallelEmbedding) and self._bits_for(prefix) is not None:
            raise ValueError(f"exl3: {prefix}: EXL3 input embeddings are not supported")
        return None


# ---------------------------------------------------------------------------------------------------------------
def _kind(layer) -> str:
    if isinstance(layer, ParallelLMHead):
        return "lmhead"
    if isinstance(layer, QKVParallelLinear):
        return "qkv"
    if isinstance(layer, MergedColumnParallelLinear):
        return "merged"
    if isinstance(layer, RowParallelLinear):
        return "row"
    if isinstance(layer, ColumnParallelLinear):
        return "col"
    if isinstance(layer, ReplicatedLinear):
        if len(getattr(layer, "output_partition_sizes", [0])) != 1:
            raise NotImplementedError(f"exl3: {layer.prefix}: merged replicated linears are not supported")
        return "replicated"
    raise NotImplementedError(f"exl3: {type(layer).__name__} is not a supported EXL3 linear kind")


def _fallback_allowed() -> bool:
    return os.environ.get("EXL3_ALLOW_FALLBACK", "0") == "1"


class Exl3LinearMethod(LinearMethodBase):

    def __init__(self, config: Exl3Config, bits: int, prefix: str = ""):
        self.config = config
        self.K = int(bits)
        self.cb = CB_IDS[config.codebook]
        self.prefix = prefix

    # ----------------------------------------------------------------------------------------- create_weights
    def create_weights(self, layer, input_size_per_partition, output_partition_sizes, input_size,
                       output_size, params_dtype, **extra_weight_attrs):
        k = int(input_size_per_partition)
        parts = [int(s) for s in output_partition_sizes]
        n = sum(parts)
        if k % 128 or any(s % 128 for s in parts):
            raise ValueError(f"exl3: {self.prefix}: local dims must be multiples of 128 (k={k}, n={parts})")
        kind = _kind(layer)
        K = self.K
        layer.exl3_prefix = self.prefix
        layer.exl3_config = self.config
        layer.exl3_kind = kind
        layer.exl3_tp_rank = int(layer.tp_rank)
        layer.exl3_tp_size = int(layer.tp_size)
        layer.exl3_in_full = int(input_size)          # checkpoint k (row splits narrow it)
        layer.exl3_sizes = parts
        layer.exl3_offsets = [sum(parts[:i]) for i in range(len(parts) + 1)]
        layer.exl3_loaded = {}                         # (shard group, part) -> times loaded
        layer.exl3_K = K
        layer.exl3_cb = self.cb
        layer.exl3_dtype = params_dtype
        layer.exl3_cpp = False
        if kind == "qkv":
            layer.exl3_meta = dict(total_num_heads=layer.total_num_heads, total_num_kv_heads=layer.total_num_kv_heads,
                                   head_size=layer.head_size, v_head_size=layer.v_head_size)
            if layer.num_kv_head_replicas != max(1, layer.exl3_tp_size // layer.total_num_kv_heads):
                raise ValueError(f"exl3: {self.prefix}: unexpected KV head replication {layer.num_kv_head_replicas}")
        elif kind == "merged":
            layer.exl3_meta = dict(output_sizes=[int(s) for s in layer.output_sizes])
        elif kind == "lmhead":
            si = layer.shard_indices
            layer.exl3_meta = dict(org_start=si.org_vocab_start_index, org_end=si.org_vocab_end_index,
                                   num_per_partition=layer.num_embeddings_per_partition)
            layer.exl3_n_full = int(layer.org_vocab_size)
        else:
            layer.exl3_meta = {}
            layer.exl3_n_full = int(output_size)

        def reg(name, tensor, part):
            p = Parameter(tensor, requires_grad=False)
            p.weight_loader = self._loader(layer, part)
            layer.register_parameter(name, p)

        reg("trellis", torch.empty((k // 16, n // 16, 16 * K), dtype=torch.int16), "trellis")
        reg("suh", torch.empty((len(parts), k), dtype=torch.float16), "suh")
        reg("svh", torch.empty((n,), dtype=torch.float16), "svh")
        if self.config.codebook != "3inst":
            reg(self.config.codebook, torch.empty((), dtype=torch.int32), self.config.codebook)
        loadcheck.add_layer(layer, "dense")

    # ------------------------------------------------------------------------------------------------ loaders
    def _shard_plan(self, layer, sid) -> tuple[tuple[int, ...], tp.SlicePlan]:
        """(local shard group, slice plan) of one checkpoint matrix arriving with shard id `sid`."""
        kind, meta = layer.exl3_kind, layer.exl3_meta
        r, t = layer.exl3_tp_rank, layer.exl3_tp_size
        k = layer.exl3_in_full
        key = layer.exl3_prefix if sid is None else f"{layer.exl3_prefix}[{sid}]"
        if kind == "merged":
            sizes = meta["output_sizes"]
            if sid is None:
                group = tuple(range(len(sizes)))
            elif isinstance(sid, int):
                group = (sid,)
            elif isinstance(sid, (tuple, list)):
                group = tuple(int(i) for i in sid)
                if group not in ALLOWED_TUPLE_SHARDS:
                    raise ValueError(f"exl3: {layer.exl3_prefix}: tuple shard id {group} is not supported (only "
                                     f"{sorted(ALLOWED_TUPLE_SHARDS)}; vLLM's own tuple loader places others "
                                     "differently)")
            else:
                raise ValueError(f"exl3: {layer.exl3_prefix}: shard id {sid!r}")
            if group[0] < 0 or group[-1] >= len(sizes):
                raise ValueError(f"exl3: {layer.exl3_prefix}: shard id {sid!r} out of range {sizes}")
            n = sum(sizes[i] for i in group)
            plan = tp.plan("merged", key, k, n, tp_rank=r, tp_size=t,
                           shard_id=(group[0] if len(group) == 1 and isinstance(sid, int) else
                                     (None if sid is None else group)), output_sizes=sizes)
        elif kind == "qkv":
            if sid not in QKV_IDS:
                raise ValueError(f"exl3: {layer.exl3_prefix}: QKV shard id must be 'q', 'k' or 'v', got {sid!r} "
                                 "(a fused qkv tensor on disk is not supported)")
            group = (QKV_IDS[sid],)
            hs = meta["head_size"] if sid != "v" else meta["v_head_size"]
            n = (meta["total_num_heads"] if sid == "q" else meta["total_num_kv_heads"]) * hs
            plan = tp.plan("qkv", key, k, n, tp_rank=r, tp_size=t, shard_id=sid, **meta)
        else:
            if sid is not None:
                raise ValueError(f"exl3: {layer.exl3_prefix}: unexpected shard id {sid!r} for a {kind} layer")
            group = (0,)
            plan = tp.plan(kind, key, k, layer.exl3_n_full, tp_rank=r, tp_size=t, **meta)
        # the plan must land exactly on vLLM's local partition
        a, b = layer.exl3_offsets[group[0]], layer.exl3_offsets[group[-1] + 1]
        if (plan.dst_offset, plan.local_n, plan.local_k) != (a, b - a, layer.trellis.shape[0] * 16):
            raise ValueError(f"exl3: {layer.exl3_prefix}: slice plan for shard {sid!r} (dst {plan.dst_offset}, "
                             f"n {plan.local_n}, k {plan.local_k}) does not match vLLM's partition ([{a}, {b}), "
                             f"k {layer.trellis.shape[0] * 16})")
        return group, plan

    def _loader(self, layer, part):
        def load(param, loaded_weight, loaded_shard_id=None):
            sid = loaded_shard_id if loaded_shard_id is not None else getattr(loaded_weight, "shard_id", None)
            group, plan = self._shard_plan(layer, sid)
            if part in ("mcg", "mul1"):
                v = loaded_weight.reshape(-1)
                if v.numel() != 1 or v.dtype != torch.int32 or int(v[0].item()) != CB_MARKER[part]:
                    raise ValueError(f"exl3: {layer.exl3_prefix}: {part} marker {v[:1].tolist()} {v.dtype}")
                param.data.copy_(v[0])
            else:
                w = plan.narrow(part, loaded_weight)
                if part == "trellis":
                    if w.dtype != torch.int16 or w.shape[-1] != param.shape[-1]:
                        raise ValueError(f"exl3: {layer.exl3_prefix}: trellis {tuple(loaded_weight.shape)} "
                                         f"{loaded_weight.dtype} is not K={layer.exl3_K}")
                    a = plan.dst_offset // 16
                    param.data[:, a: a + plan.local_n // 16].copy_(w)
                elif part == "suh":
                    if w.dtype != torch.float16:
                        raise ValueError(f"exl3: {layer.exl3_prefix}: suh dtype {w.dtype}")
                    for i in group:
                        param.data[i].copy_(w)
                elif part == "svh":
                    if w.dtype != torch.float16:
                        raise ValueError(f"exl3: {layer.exl3_prefix}: svh dtype {w.dtype}")
                    param.data[plan.dst_offset: plan.dst_offset + plan.local_n].copy_(w)
                else:
                    raise ValueError(f"exl3: {layer.exl3_prefix}: unexpected component {part}")
            key = (group, part)
            layer.exl3_loaded[key] = layer.exl3_loaded.get(key, 0) + 1
            layer.exl3_batch.counts["dense"] += 1
        return load

    # -------------------------------------------------------------------------------------- post-processing
    def required_parts(self) -> tuple[str, ...]:
        return ("trellis", "suh", "svh") + (() if self.config.codebook == "3inst" else (self.config.codebook,))

    def check_coverage(self, layer) -> list[tuple[int, ...]]:
        loaded = layer.exl3_loaded
        groups = sorted({g for g, _ in loaded})
        if not groups:
            raise ValueError(f"exl3: {layer.exl3_prefix}: no EXL3 tensor was loaded")
        covered = sorted(i for g in groups for i in g)
        if covered != list(range(len(layer.exl3_sizes))):
            raise ValueError(f"exl3: {layer.exl3_prefix}: shards {groups} do not cover the {len(layer.exl3_sizes)} "
                             "output partitions exactly once")
        bad = [(g, p, loaded.get((g, p), 0)) for g in groups for p in self.required_parts()
               if loaded.get((g, p), 0) != 1]
        extra = [k for k in loaded if k[1] not in self.required_parts()]
        if bad or extra:
            raise ValueError(f"exl3: {layer.exl3_prefix}: components not loaded exactly once (group, part, times): "
                             f"{bad + extra}")
        return groups

    def process_weights_after_loading(self, layer):
        groups = self.check_coverage(layer)
        dev = layer.trellis.device
        suh = torch.stack([layer.suh.data[g[0]] for g in groups]).contiguous()
        bounds = [layer.exl3_offsets[g[0]] for g in groups] + [layer.exl3_offsets[-1]]
        n = bounds[-1]
        shard_of_nb = torch.empty(n // 128, dtype=torch.int32)
        for gi in range(len(groups)):
            shard_of_nb[bounds[gi] // 128: bounds[gi + 1] // 128] = gi
        layer.suh = Parameter(suh, requires_grad=False)
        layer.exl3_shard_of_nb = shard_of_nb.to(dev)
        layer.exl3_bounds = bounds
        layer.exl3_groups = groups
        # decide the backend once, outside any traced code: the all-C++ op when the ESIMD library has it
        from . import ops
        E = ops._get_esimd()
        layer.exl3_cpp = bool(E) and hasattr(E, "linear") and bool(E.exl3_supported(layer.exl3_K, layer.exl3_cb))
        if not layer.exl3_cpp:
            if not _fallback_allowed():
                raise RuntimeError(f"exl3: {layer.exl3_prefix}: K={layer.exl3_K} cb={layer.exl3_cb} is not built into "
                                   "_C.so; refusing the Triton fallback (EXL3_ALLOW_FALLBACK=1 opts in)")
            logger.warning("EXL3XPU_FALLBACK %s K=%d cb=%d: Triton path (EXL3_ALLOW_FALLBACK=1)",
                           layer.exl3_prefix, layer.exl3_K, layer.exl3_cb)
        loadcheck.layer_done(layer)

    def apply(self, layer, x, bias=None):
        from . import ops
        if layer.exl3_cpp:
            y = torch.ops.exl3xpu_C.linear(x, layer.trellis, layer.suh, layer.svh, layer.exl3_shard_of_nb,
                                           layer.exl3_bounds, layer.exl3_K, layer.exl3_cb,
                                           ops.SMALL_M_MAX, ops.RECON_SLICE_N)
        else:
            y = ops._exl3_linear_py(x, layer.trellis, layer.suh, layer.svh, layer.exl3_shard_of_nb,
                                    layer.exl3_bounds, layer.exl3_K, layer.exl3_cb)
        if bias is not None:
            y = y + bias
        return y

    # lm_head (ParallelLMHead) path
    def embedding(self, layer, input_):
        raise NotImplementedError("exl3: quantized input embeddings are not supported")


# ---------------------------------------------------------------------------------------------------------------
def _patch_list() -> list[str]:
    raw = os.environ.get("EXL3_VLLM_PATCHES", "").strip()
    if raw.lower() in ("", "0", "none"):
        return []
    names = [PATCH_ALIASES.get(p.strip(), p.strip()) for p in raw.split(",") if p.strip()]
    return list(dict.fromkeys(names))


def apply_patches() -> dict[str, str]:
    """EXL3_VLLM_PATCHES -> {name: applied | already | refused | unknown}; an accepted patch that does not apply,
    or an unrecognized name, raises RuntimeError (never boot with a patch set other than the one requested)."""
    res: dict[str, str] = {}
    for name in _patch_list():
        if name in PATCHES_ACCEPTED:
            from . import vllm_patches
            cls_patched = False
            try:
                from vllm.v1.attention.backends import flash_attn as fa_mod
                cls_patched = bool(getattr(getattr(fa_mod, "FlashAttentionImpl", None), "_exl3_fp8kv_patched", False))
            except Exception:      # noqa: BLE001
                pass
            if cls_patched:
                res[name] = "already"
            else:
                res[name] = "applied" if vllm_patches.patch_fp8kv_prefill() else "skipped"
            if res[name] == "skipped":
                # never serve a configuration other than the one requested (the lane asks for fp8kv on every
                # FP8 run): fail the boot instead of falling back to the stock FP8 prefill path
                logger.error("EXL3XPU_PATCH %s requested but not applied (see the vllm_patches log line above)", name)
                raise RuntimeError(f"exl3: EXL3_VLLM_PATCHES requests {name!r} but the patch did not apply")
        elif name in PATCHES_REFUSED:
            res[name] = "refused"
            logger.warning("EXL3XPU_PATCH %s refused on vLLM 0.30 (only %s is accepted)", name,
                           ",".join(sorted(PATCHES_ACCEPTED)))
        else:
            res[name] = "unknown"
            logger.error("EXL3XPU_PATCH %s unknown (accepted: %s); refusing to boot with a partial patch set",
                         name, ",".join(sorted(PATCHES_ACCEPTED)))
            raise RuntimeError(f"exl3: EXL3_VLLM_PATCHES entry {name!r} is not a known vllm_patch")
    return res


def register():
    """vllm.general_plugins entry point (runs in every vLLM process). An exception here aborts the process."""
    from . import graph_guard
    guard = os.environ.get("EXL3_CAPTURE_GUARD", "raise")
    if guard != "raise":
        raise RuntimeError(f"exl3: EXL3_CAPTURE_GUARD={guard!r}: the graph guard has a single mode, 'raise'")
    graph_guard.install()
    if os.environ.get("EXL3_KV_STATS", "0") == "1":
        from . import kv_stats
        kv_stats.install()
    if os.environ.get("EXL3_ROUTING_CHECK", "0") == "1":
        from . import routing_check
        routing_check.install()
    if os.environ.get("EXL3_LAYER_DUMP", "").strip():
        from . import layer_dump
        layer_dump.install()
    if os.environ.get("EXL3_DRAFT_VOCAB"):
        raise RuntimeError("exl3: EXL3_DRAFT_VOCAB (pruned MTP draft head) is not supported by this plugin build: "
                           "unvalidated on this model and on a vocab-parallel lm_head")
    from . import vllm_patches
    if not vllm_patches.patch_det_small_linear():
        raise RuntimeError("exl3: the det_small_linear patch did not apply (EXL3_DET_SMALL_LINEAR=0 "
                           "is a test-only opt-out; serving without it is not bit-reproducible for "
                           "65..128-token prefills)")
    patches = apply_patches()
    register_quantization_config("exl3")(Exl3Config)
    logger.info("EXL3XPU_PLUGIN registered quantization=exl3 patches=%s",
                ",".join(f"{k}:{v}" for k, v in patches.items()) or "-")
    return None
