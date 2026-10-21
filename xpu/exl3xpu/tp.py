"""Tensor-parallel slice plan for EXL3 linears (DESIGN rev 2, section 2.3).

vLLM hands every rank the FULL checkpoint tensor of each EXL3 component (trellis, suh, svh, mul1) and lets the
parameter's own weight_loader cut the rank's share. This module computes that share. It is pure torch (no vLLM
import) so it can be unit-tested on CPU, and it is built on trellis_core.format.geometry, which owns the slicing
rules:

  column (output) range [a, a+s):  trellis.narrow(1, a/16, s/16), svh[a:a+s], suh whole, mul1 replicated
  row (input) range     [a, a+s):  trellis.narrow(0, a/16, s/16), suh[a:a+s], svh whole, mul1 replicated
  a and s multiples of 128 (geometry raises SplitError otherwise); several column ranges of one matrix may be
  concatenated because they share suh.

One plan function per vLLM layer kind:

  plan_col           ColumnParallelLinear (e.g. mtp.fc)
  plan_merged        MergedColumnParallelLinear, int shard id (in_proj_z -> 3, shared gate/up -> 0/1)
  plan_merged_tuple  MergedColumnParallelLinear, tuple shard id (in_proj_qkv -> (0, 1, 2))
  plan_qkv           QKVParallelLinear 'q' / 'k' / 'v', with KV-head replication when tp_size > num_kv_heads
  plan_row           RowParallelLinear (o_proj, out_proj, down_proj)
  plan_lmhead        ParallelLMHead (vocab-parallel columns)
  plan_replicated    ReplicatedLinear (whole tensors on every rank)

All coordinates in a plan are checkpoint coordinates. `SlicePlan.narrow(part, tensor)` returns the rank's
contiguous slice of one component; `SlicePlan.dst_offset` is where the slice's columns start inside the rank's
fused output (merged / qkv layers), in elements.

The module also carries the per-rank EXL3 tensor counts that the plugin's EXL3XPU_LOADED check asserts
(`expected_loaded_counts`), computed from the checkpoint index.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Sequence

import torch

from trellis_core.format.geometry import SplitError, column_slice, even_shard, kv_shard_index, row_slice
from trellis_core.format.spec import BLOCK, TILE, BitsK, Codebook, MatrixSpec

__all__ = [
    "KINDS", "PARTS", "SlicePlan", "SplitError", "plan", "plan_col", "plan_merged", "plan_merged_tuple", "plan_qkv",
    "plan_row", "plan_lmhead", "plan_replicated", "vocab_shard", "classify", "qwen35_plan", "expected_loaded_counts",
]

KINDS = ("col", "merged", "merged_tuple", "qkv", "row", "lmhead", "replicated")
PARTS = ("trellis", "suh", "svh", "mul1", "mcg", "bias")


def _spec(key: str, k: int, n: int) -> MatrixSpec:
    # geometry only reads key/k/n; bits and codebook are placeholders (the K of the trellis is never changed here).
    return MatrixSpec(key=key, k=int(k), n=int(n), bits=BitsK(8), codebook=Codebook.MUL1)


def _div(a: int, b: int, what: str) -> int:
    if b <= 0 or a % b:
        raise SplitError(f"{what}: {a} is not divisible by {b}")
    return a // b


def _check_rank(tp_rank: int, tp_size: int) -> None:
    if tp_size < 1 or not 0 <= tp_rank < tp_size:
        raise ValueError(f"bad tp_rank {tp_rank} for tp_size {tp_size}")


@dataclass(frozen=True)
class SlicePlan:
    """One rank's share of one stored EXL3 matrix.

    cols: checkpoint output ranges (start, size), concatenated in this order. A single (0, n) range = whole.
    rows: checkpoint input range (start, size). (0, k) = whole.
    At most one of cols / rows is a strict sub-range (EXL3 is split on one side only).
    """
    kind: str
    key: str
    k: int
    n: int
    cols: tuple[tuple[int, int], ...]
    rows: tuple[int, int]
    dst_offset: int
    tp_rank: int
    tp_size: int

    @property
    def local_k(self) -> int:
        return self.rows[1]

    @property
    def local_n(self) -> int:
        return sum(s for _, s in self.cols)

    @property
    def splits_cols(self) -> bool:
        return self.cols != ((0, self.n),)

    @property
    def splits_rows(self) -> bool:
        return self.rows != (0, self.k)

    def col_index(self) -> torch.Tensor:
        """Checkpoint output column of every local output column (int64 [local_n])."""
        return torch.cat([torch.arange(a, a + s, dtype=torch.int64) for a, s in self.cols])

    def row_index(self) -> torch.Tensor:
        """Checkpoint input row of every local input row (int64 [local_k])."""
        a, s = self.rows
        return torch.arange(a, a + s, dtype=torch.int64)

    def local_shapes(self, K: int) -> dict[str, tuple[int, ...]]:
        """Shapes of the rank's narrowed components for bitrate K (integer K)."""
        return {"trellis": (self.local_k // TILE, self.local_n // TILE, 16 * int(K)),
                "suh": (self.local_k,), "svh": (self.local_n,), "mul1": ()}

    # ---------------------------------------------------------------------------------------------- narrowing
    def _check_shape(self, part: str, t: torch.Tensor) -> None:
        if part == "trellis":
            ok = t.dim() == 3 and t.shape[0] * TILE == self.k and t.shape[1] * TILE == self.n
        elif part == "suh":
            ok = t.dim() == 1 and t.shape[0] == self.k
        elif part in ("svh", "bias"):
            ok = t.dim() == 1 and t.shape[0] == self.n
        else:
            ok = t.numel() == 1
        if not ok:
            raise ValueError(f"exl3 tp: {self.key}.{part} has shape {tuple(t.shape)}, plan expects the full "
                             f"checkpoint matrix k={self.k} n={self.n}")

    def narrow(self, part: str, t: torch.Tensor, clone: bool = False) -> torch.Tensor:
        """This rank's contiguous slice of one stored component. `t` must be the full checkpoint tensor.

        The result may share storage with `t` when the slice is already contiguous; pass clone=True to force a copy
        (e.g. when the slice is kept while the checkpoint tensor is released)."""
        if part not in PARTS:
            raise ValueError(f"exl3 tp: unknown component {part!r} of {self.key}")
        self._check_shape(part, t)
        spec = _spec(self.key, self.k, self.n)
        if part in ("mul1", "mcg"):
            out = t
        elif part == "trellis":
            if self.splits_rows:
                sl = row_slice(spec, *self.rows).trellis
                out = t.narrow(sl.dim, sl.start, sl.length)
            else:
                out = t
            if self.splits_cols:
                pieces = []
                for a, s in self.cols:
                    sl = column_slice(spec, a, s).trellis
                    pieces.append(out.narrow(sl.dim, sl.start, sl.length))
                out = pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=1)
        elif part == "suh":
            if self.splits_rows:
                sl = row_slice(spec, *self.rows).suh
                out = t.narrow(sl.dim, sl.start, sl.length)
            else:
                out = t
        else:  # svh, bias: output side
            if self.splits_cols:
                pieces = []
                for a, s in self.cols:
                    sl = column_slice(spec, a, s).svh
                    pieces.append(t.narrow(sl.dim, sl.start, sl.length))
                out = pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=0)
            else:
                out = t   # row split: svh whole; a row-split bias is loaded whole and added on rank 0 by vLLM
        out = out.contiguous()
        if clone and out.untyped_storage().data_ptr() == t.untyped_storage().data_ptr():
            out = out.clone()
        return out

    def narrow_all(self, tensors: dict[str, torch.Tensor], clone: bool = False) -> dict[str, torch.Tensor]:
        return {p: self.narrow(p, t, clone=clone) for p, t in tensors.items()}


def _make(kind: str, key: str, k: int, n: int, cols, rows, dst_offset: int, tp_rank: int, tp_size: int) -> SlicePlan:
    spec = _spec(key, k, n)
    cols = tuple((int(a), int(s)) for a, s in cols)
    rows = (int(rows[0]), int(rows[1]))
    if cols != ((0, n),):
        for a, s in cols:
            column_slice(spec, a, s)          # raises SplitError unless in range and 128-aligned
    if rows != (0, k):
        row_slice(spec, *rows)
    if cols != ((0, n),) and rows != (0, k):
        raise SplitError(f"{key}: a plan splits one side only")
    if k % BLOCK or n % BLOCK:
        raise SplitError(f"{key}: stored dims k={k} n={n} are not multiples of {BLOCK}")
    return SlicePlan(kind, key, int(k), int(n), cols, rows, int(dst_offset), int(tp_rank), int(tp_size))


# ------------------------------------------------------------------------------------------------------ kinds
def plan_col(key: str, k: int, n: int, tp_rank: int, tp_size: int) -> SlicePlan:
    """ColumnParallelLinear: columns [r*n/tp, +n/tp)."""
    _check_rank(tp_rank, tp_size)
    return _make("col", key, k, n, [even_shard(n, tp_size, tp_rank)], (0, k), 0, tp_rank, tp_size)


def plan_merged(key: str, k: int, n: int, output_sizes: Sequence[int], shard_id: int,
                tp_rank: int, tp_size: int) -> SlicePlan:
    """MergedColumnParallelLinear with an int shard id: the checkpoint tensor is that one shard (n ==
    output_sizes[shard_id]); the rank takes [r*n/tp, +n/tp) and writes it at local offset sum(output_sizes[:i])/tp."""
    _check_rank(tp_rank, tp_size)
    if not isinstance(shard_id, int) or not 0 <= shard_id < len(output_sizes):
        raise ValueError(f"{key}: shard id {shard_id!r} out of range for output_sizes {list(output_sizes)}")
    if n != output_sizes[shard_id]:
        raise ValueError(f"{key}: checkpoint n={n} != output_sizes[{shard_id}]={output_sizes[shard_id]}")
    dst = sum(_div(s, tp_size, key) for s in output_sizes[:shard_id])
    return _make("merged", key, k, n, [even_shard(n, tp_size, tp_rank)], (0, k), dst, tp_rank, tp_size)


def plan_merged_tuple(key: str, k: int, n: int, output_sizes: Sequence[int], shard_ids: Sequence[int],
                      tp_rank: int, tp_size: int) -> SlicePlan:
    """MergedColumnParallelLinear with a tuple shard id (consecutive): the checkpoint tensor is those shards fused
    (n == sum of their sizes). Per sub-shard j the rank takes [base_j + r*size_j/tp, +size_j/tp); the pieces are
    concatenated (they share suh) and written at local offset sum(output_sizes[:ids[0]])/tp."""
    _check_rank(tp_rank, tp_size)
    ids = tuple(shard_ids)
    if not ids or any(b - a != 1 for a, b in zip(ids, ids[1:])) or ids[0] < 0 or ids[-1] >= len(output_sizes):
        raise ValueError(f"{key}: shard ids {ids} must be consecutive indices into {list(output_sizes)}")
    sizes = [output_sizes[i] for i in ids]
    if n != sum(sizes):
        raise ValueError(f"{key}: checkpoint n={n} != sum of output_sizes{list(ids)}={sum(sizes)}")
    cols, base = [], 0
    for s in sizes:
        a, sz = even_shard(s, tp_size, tp_rank)
        cols.append((base + a, sz))
        base += s
    dst = sum(_div(s, tp_size, key) for s in output_sizes[:ids[0]])
    return _make("merged_tuple", key, k, n, cols, (0, k), dst, tp_rank, tp_size)


def plan_qkv(key: str, k: int, n: int, shard_id: str, *, total_num_heads: int, total_num_kv_heads: int,
             head_size: int, v_head_size: int | None = None, tp_rank: int, tp_size: int) -> SlicePlan:
    """QKVParallelLinear. total_num_heads is the layer's q head count as vLLM builds it (Qwen3.5 attention:
    num_attention_heads * 2, q and gate interleaved per head). q: [r*q_loc, +q_loc). k/v: KV head shard
    kv_shard_index(tp, r, H) (vLLM: shard rank = r // num_kv_head_replicas, linear.py:1305-1309)."""
    _check_rank(tp_rank, tp_size)
    v_head_size = head_size if v_head_size is None else v_head_size
    q_loc = _div(total_num_heads, tp_size, f"{key} q heads") * head_size
    idx, cnt = kv_shard_index(tp_size, tp_rank, total_num_kv_heads)
    kv_heads_loc = total_num_kv_heads // cnt
    k_loc, v_loc = kv_heads_loc * head_size, kv_heads_loc * v_head_size
    if shard_id == "q":
        if n != total_num_heads * head_size:
            raise ValueError(f"{key}: q checkpoint n={n} != {total_num_heads}*{head_size}")
        return _make("qkv", key, k, n, [(tp_rank * q_loc, q_loc)], (0, k), 0, tp_rank, tp_size)
    if shard_id in ("k", "v"):
        hs, loc = (head_size, k_loc) if shard_id == "k" else (v_head_size, v_loc)
        if n != total_num_kv_heads * hs:
            raise ValueError(f"{key}: {shard_id} checkpoint n={n} != {total_num_kv_heads}*{hs}")
        dst = q_loc if shard_id == "k" else q_loc + k_loc
        return _make("qkv", key, k, n, [(idx * loc, loc)], (0, k), dst, tp_rank, tp_size)
    raise ValueError(f"{key}: qkv shard id must be 'q', 'k' or 'v', got {shard_id!r}")


def plan_row(key: str, k: int, n: int, tp_rank: int, tp_size: int) -> SlicePlan:
    """RowParallelLinear: input rows [r*k/tp, +k/tp); svh whole; partial outputs are all-reduced."""
    _check_rank(tp_rank, tp_size)
    return _make("row", key, k, n, [(0, n)], even_shard(k, tp_size, tp_rank), 0, tp_rank, tp_size)


def plan_lmhead(key: str, k: int, n: int, *, org_start: int, org_end: int, num_per_partition: int,
                tp_rank: int, tp_size: int) -> SlicePlan:
    """ParallelLMHead: columns [org_start, org_end) of the stored vocab (layer.shard_indices.org_vocab_start_index /
    org_vocab_end_index; layer.num_embeddings_per_partition). Vocab padding inside a partition is refused: an EXL3
    matrix cannot grow columns, and Ornith's 248320 = 64 * 3880 needs none at TP 1/2/4."""
    _check_rank(tp_rank, tp_size)
    if org_end - org_start != num_per_partition:
        raise SplitError(f"{key}: rank {tp_rank} holds {org_end - org_start} real vocab rows but the partition is "
                         f"{num_per_partition} (vocab padding / added vocab is not supported for EXL3 lm_head)")
    return _make("lmhead", key, k, n, [(org_start, org_end - org_start)], (0, k), 0, tp_rank, tp_size)


def plan_replicated(key: str, k: int, n: int, tp_rank: int = 0, tp_size: int = 1) -> SlicePlan:
    """ReplicatedLinear (or disable_tp): whole tensors on every rank."""
    _check_rank(tp_rank, tp_size)
    return _make("replicated", key, k, n, [(0, n)], (0, k), 0, tp_rank, tp_size)


def vocab_shard(vocab_size: int, tp_rank: int, tp_size: int, padding: int = 64) -> tuple[int, int, int]:
    """(org_start, org_end, num_per_partition) as vLLM's VocabParallelEmbedding computes them without added
    vocab (pad_vocab_size + vocab_range_from_global_vocab_size, vocab_parallel_embedding.py)."""
    _check_rank(tp_rank, tp_size)
    padded = -(-vocab_size // padding) * padding
    per = _div(padded, tp_size, "padded vocab")
    a = tp_rank * per
    return min(a, vocab_size), min(a + per, vocab_size), per


def plan(kind: str, key: str, k: int, n: int, *, tp_rank: int, tp_size: int, shard_id=None, **meta) -> SlicePlan:
    """Dispatch by kind. meta: output_sizes (merged, merged_tuple); total_num_heads, total_num_kv_heads, head_size,
    v_head_size (qkv); org_start, org_end, num_per_partition (lmhead)."""
    if kind == "col":
        return plan_col(key, k, n, tp_rank, tp_size)
    if kind == "merged" and isinstance(shard_id, int):
        return plan_merged(key, k, n, meta["output_sizes"], shard_id, tp_rank, tp_size)
    if kind in ("merged", "merged_tuple") and isinstance(shard_id, (tuple, list)):
        return plan_merged_tuple(key, k, n, meta["output_sizes"], tuple(shard_id), tp_rank, tp_size)
    if kind == "merged" and shard_id is None:
        # whole fused tensor on disk: same as a tuple of all shards
        return plan_merged_tuple(key, k, n, meta["output_sizes"], tuple(range(len(meta["output_sizes"]))),
                                 tp_rank, tp_size)
    if kind == "qkv":
        return plan_qkv(key, k, n, shard_id, total_num_heads=meta["total_num_heads"],
                        total_num_kv_heads=meta["total_num_kv_heads"], head_size=meta["head_size"],
                        v_head_size=meta.get("v_head_size"), tp_rank=tp_rank, tp_size=tp_size)
    if kind == "row":
        return plan_row(key, k, n, tp_rank, tp_size)
    if kind == "lmhead":
        return plan_lmhead(key, k, n, org_start=meta["org_start"], org_end=meta["org_end"],
                           num_per_partition=meta["num_per_partition"], tp_rank=tp_rank, tp_size=tp_size)
    if kind == "replicated":
        return plan_replicated(key, k, n, tp_rank, tp_size)
    raise ValueError(f"{key}: no slice plan for kind {kind!r} with shard id {shard_id!r}")


# ------------------------------------------------------------------------------------- Qwen3.5-MoE name mapping
# Checkpoint leaf -> (vLLM kind, shard id), as vLLM 0.30 stacks Qwen3.5 / Qwen3.5-MoE weights:
# qwen3_next.py Qwen3NextModel.hf_to_vllm_mapper (q/k/v -> qkv_proj, gate/up -> gate_up_proj) and qwen3_5.py
# Qwen3_5Model.hf_to_vllm_mapper (in_proj_qkv -> in_proj_qkvz (0,1,2), in_proj_z -> 3).
_DENSE_LEAF = [
    (re.compile(r"\.self_attn\.q_proj$"), ("qkv", "q")),
    (re.compile(r"\.self_attn\.k_proj$"), ("qkv", "k")),
    (re.compile(r"\.self_attn\.v_proj$"), ("qkv", "v")),
    (re.compile(r"\.self_attn\.o_proj$"), ("row", None)),
    (re.compile(r"\.linear_attn\.in_proj_qkv$"), ("merged_tuple", (0, 1, 2))),
    (re.compile(r"\.linear_attn\.in_proj_z$"), ("merged", 3)),
    (re.compile(r"\.linear_attn\.out_proj$"), ("row", None)),
    (re.compile(r"\.shared_expert\.gate_proj$"), ("merged", 0)),
    (re.compile(r"\.shared_expert\.up_proj$"), ("merged", 1)),
    (re.compile(r"\.shared_expert\.down_proj$"), ("row", None)),
    (re.compile(r"(^|\.)mlp\.gate_proj$"), ("merged", 0)),      # dense MLP layers (Qwen3.5 dense models)
    (re.compile(r"(^|\.)mlp\.up_proj$"), ("merged", 1)),
    (re.compile(r"(^|\.)mlp\.down_proj$"), ("row", None)),
    (re.compile(r"^lm_head$"), ("lmhead", None)),
    (re.compile(r"^mtp\.fc$"), ("col", None)),
]
_EXPERT = re.compile(r"\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)$")


def classify(module: str) -> tuple[str, object]:
    """Checkpoint module name (without the .trellis suffix) -> (kind, shard id).
    Routed experts return ("expert", (expert_id, proj)). Raises on an unknown EXL3 module (never guess)."""
    m = _EXPERT.search(module)
    if m:
        return "expert", (int(m.group(1)), m.group(2))
    for rx, res in _DENSE_LEAF:
        if rx.search(module):
            return res
    raise ValueError(f"exl3 tp: no slice rule for EXL3 module {module!r}")


def qwen35_plan(module: str, k: int, n: int, cfg: dict, *, tp_rank: int, tp_size: int,
                moe_tp_rank: int | None = None, moe_tp_size: int | None = None) -> SlicePlan:
    """Slice plan of one Qwen3.5 / Qwen3.5-MoE checkpoint matrix as vLLM 0.30 lays it out (cfg = text config).
    Routed experts use the MoE TP split (DESIGN 2.4 step 3): gate/up columns, down rows, at moe_tp_rank/size
    (default: the attention TP rank/size); under EP they are whole (call with moe_tp_size=1)."""
    kind, sid = classify(module)
    if kind == "expert":
        r = tp_rank if moe_tp_rank is None else moe_tp_rank
        t = tp_size if moe_tp_size is None else moe_tp_size
        return (plan_row if sid[1] == "down_proj" else plan_col)(module, k, n, r, t)
    meta: dict = {}
    if kind in ("merged", "merged_tuple"):
        if ".linear_attn." in module:
            kd = cfg["linear_num_key_heads"] * cfg["linear_key_head_dim"]
            vd = cfg["linear_num_value_heads"] * cfg["linear_value_head_dim"]
            meta["output_sizes"] = [kd, kd, vd, vd]
        elif ".shared_expert." in module:
            i = cfg["shared_expert_intermediate_size"]
            meta["output_sizes"] = [i, i]
        else:
            i = cfg["intermediate_size"]
            meta["output_sizes"] = [i, i]
    elif kind == "qkv":
        meta.update(total_num_heads=cfg["num_attention_heads"] * (1 + int(bool(cfg.get("attn_output_gate", False)))),
                    total_num_kv_heads=cfg["num_key_value_heads"], head_size=cfg["head_dim"])
    elif kind == "lmhead":
        a, b, per = vocab_shard(cfg["vocab_size"], tp_rank, tp_size)
        meta.update(org_start=a, org_end=b, num_per_partition=per)
    return plan(kind, module, k, n, tp_rank=tp_rank, tp_size=tp_size, shard_id=sid, **meta)


# ------------------------------------------------------------------------------------- EXL3XPU_LOADED counts
_SUFFIXES = ("trellis", "suh", "svh", "mul1", "mcg", "bias", "su", "sv")


def _exl3_tensor_names(weight_names: Iterable[str]) -> list[str]:
    names = list(weight_names)
    mods = {n[: -len(".trellis")] for n in names if n.endswith(".trellis")}
    out = []
    for n in names:
        mod, _, suf = n.rpartition(".")
        if mod in mods and suf in _SUFFIXES:
            out.append(n)
    return out


def _local_experts(num_experts: int, ep_size: int, ep_rank: int) -> range:
    """vLLM's default "linear" expert placement (expert_map_manager.py): rank r owns [r*E/ep, (r+1)*E/ep)."""
    per = _div(num_experts, ep_size, "num_experts")
    return range(ep_rank * per, (ep_rank + 1) * per)


def expected_loaded_counts(weight_names: Iterable[str], *, num_experts: int, moe_mode: str, tp_size: int,
                           tp_rank: int, with_mtp: bool) -> dict[str, dict[str, int]]:
    """Per-rank counts of EXL3 tensors (trellis/suh/svh/mul1 entries) that reach this rank's EXL3 loaders.

    moe_mode: "ep" (expert parallel over the TP group: whole experts, linear placement, non-local experts are
    dropped before counting) or "tp" (every rank narrows every expert). Dense tensors reach every rank (each rank
    narrows them), so the dense count does not depend on the rank. The vision tower holds no EXL3 tensors.

    Returns {"target": {"dense", "experts"}, "drafter": {"dense", "experts"}} with the drafter zero when
    with_mtp is False. The drafter count includes the lm_head copy: vLLM 0.30's Qwen3_5MTP.load_weights keeps
    every "lm_head" name (qwen3_5_mtp.py:301-310) and the V2 MTP speculator replaces the drafter's lm_head with the
    target's only after get_model() has loaded it (spec_decode/eagle/utils.py:118-132)."""
    if moe_mode not in ("ep", "tp"):
        raise ValueError(f"moe_mode must be 'ep' or 'tp', got {moe_mode!r}")
    local = _local_experts(num_experts, tp_size, tp_rank) if moe_mode == "ep" else range(num_experts)
    res = {"target": {"dense": 0, "experts": 0}, "drafter": {"dense": 0, "experts": 0}}
    for name in _exl3_tensor_names(weight_names):
        mod = name.rpartition(".")[0]
        parts = mod.split(".")
        if "visual" in parts or "vision_tower" in parts:
            raise ValueError(f"exl3 tp: unexpected EXL3 tensor in the vision tower: {name}")
        is_mtp = parts[0] == "mtp"
        kind, sid = classify(mod)
        if kind == "expert":
            if sid[0] not in local:
                continue
            bucket = "experts"
        else:
            bucket = "dense"
        if is_mtp:
            if with_mtp:
                res["drafter"][bucket] += 1
        else:
            res["target"][bucket] += 1
            if with_mtp and kind == "lmhead":
                res["drafter"]["dense"] += 1
    return res
