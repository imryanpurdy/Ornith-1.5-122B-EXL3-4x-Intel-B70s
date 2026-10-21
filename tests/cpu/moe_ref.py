"""Pure-torch CPU stand-ins for the grouped EXL3 MoE kernel (exl3xpu_moe::moe_forward), used to test
moe_vllm.Exl3MoEMethod's id mapping, chunking and EP / MoE-TP partial outputs without a device.

BlobKernel(x, ids, w, ptrs, I, K, E) has the kernel's signature and semantics: ids are LOCAL ids in [-1, E), -1
contributes nothing, ptrs[e] is the address of local expert e's blob. It finds the blob by address in the CPU
arenas registered with it, unpacks it (inverse of moe_offload.pack_expert, K=4 word order) and evaluates
    out[m] = sum_k w[m, k] * down( silu(x[m] @ gate) * (x[m] @ up) )
in float64 with the exact float64 EXL3 weights (exact_ref.weight_exact). It returns float64 and records every call.

reference_moe(x, global_ids, w, weights) is the same sum from weights decoded from the checkpoint tensors.
Because weight_exact is exact, a slice's weights equal the full matrix's block bit for bit, so EP partials and the
TP1 output agree exactly with the reference; MoE-TP partials agree after summation up to float64 reassociation.
"""
from __future__ import annotations

import torch

from exact_ref import weight_exact


def unpack_blob(blob: torch.Tensor, H: int, I: int, K: int) -> dict[str, dict[str, torch.Tensor]]:
    """uint8 blob of one expert -> {gate_proj|up_proj|down_proj: {trellis, suh, svh}} (K=4 only)."""
    if K != 4:
        raise NotImplementedError("unpack_blob: only the K=4 (D=1, checkpoint word order) layout")
    b = blob.contiguous()
    o = 0

    def take(nbytes):
        nonlocal o
        piece = b[o:o + nbytes]
        o += nbytes
        return piece

    gu = take(H // 16 * (2 * I) // 16 * 32 * K).view(torch.int16).view(H // 16, 2 * I // 16, 16 * K)
    dn = take(I // 16 * H // 16 * 32 * K).view(torch.int16).view(I // 16, H // 16, 16 * K)
    f16 = lambda n: take(2 * n).view(torch.float16)   # noqa: E731
    suh_g, suh_u, svh_g, svh_u, suh_d, svh_d = f16(H), f16(H), f16(I), f16(I), f16(I), f16(H)
    if o != b.numel():
        raise ValueError(f"unpack_blob: {o} bytes consumed of {b.numel()}")
    return {"gate_proj": {"trellis": gu[:, :I // 16].contiguous(), "suh": suh_g, "svh": svh_g},
            "up_proj": {"trellis": gu[:, I // 16:].contiguous(), "suh": suh_u, "svh": svh_u},
            "down_proj": {"trellis": dn, "suh": suh_d, "svh": svh_d}}


def expert_weights(parts: dict[str, dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {r: weight_exact(p["trellis"], p["suh"], p["svh"]) for r, p in parts.items()}


def expert_out(x64: torch.Tensor, W: dict[str, torch.Tensor]) -> torch.Tensor:
    g = x64 @ W["gate_proj"]
    u = x64 @ W["up_proj"]
    return (torch.nn.functional.silu(g) * u) @ W["down_proj"]


class BlobKernel:
    def __init__(self):
        self.arenas: list[torch.Tensor] = []
        self.calls: list[dict] = []
        self._cache: dict[tuple, dict] = {}

    def add_arena(self, arena: torch.Tensor) -> None:
        assert arena.device.type == "cpu" and arena.dtype == torch.uint8 and arena.dim() == 2
        self.arenas.append(arena)

    def _blob(self, ptr: int, blob_bytes: int) -> tuple[torch.Tensor, tuple]:
        ptr &= (1 << 64) - 1                      # the table stores addresses as int64 (two's complement)
        for a in self.arenas:
            base = a.data_ptr()
            if base <= ptr < base + a.numel():
                off = ptr - base
                if off % a.shape[1] or a.shape[1] != blob_bytes:
                    raise AssertionError(f"pointer {ptr:#x} is not at a blob boundary of its arena")
                slot = off // a.shape[1]
                return a[slot], (base, slot)
        raise AssertionError(f"pointer {ptr:#x} is in no registered arena")

    def __call__(self, x, ids, w, ptrs, I, K, E):
        M, H = x.shape
        topk = ids.shape[1]
        self.calls.append({"M": M, "ids_dtype": ids.dtype, "ids_contig": ids.is_contiguous(), "w_dtype": w.dtype,
                           "w_contig": w.is_contiguous(), "x_contig": x.is_contiguous(), "I": I, "K": K, "E": E,
                           "ids_min": int(ids.min()) if ids.numel() else None,
                           "ids_max": int(ids.max()) if ids.numel() else None, "ptrs_len": ptrs.numel()})
        if ids.dtype != torch.int32 or w.dtype != torch.float32 or ptrs.dtype != torch.int64:
            raise AssertionError(f"kernel dtypes ids {ids.dtype} w {w.dtype} ptrs {ptrs.dtype}")
        if ids.numel() and (int(ids.min()) < -1 or int(ids.max()) >= E):
            raise AssertionError(f"local ids out of [-1, {E}): {int(ids.min())}..{int(ids.max())}")
        blob_bytes = (H * 2 * I * K) // 8 + (I * H * K) // 8 + 2 * (3 * H + 3 * I)
        out = torch.zeros((M, H), dtype=torch.float64)
        x64 = x.to(torch.float64)
        for e in sorted(set(ids[ids >= 0].tolist())):
            blob, key = self._blob(int(ptrs[e]), blob_bytes)
            if key not in self._cache:
                self._cache[key] = expert_weights(unpack_blob(blob, H, I, K))
            W = self._cache[key]
            rows, ks = (ids == e).nonzero(as_tuple=True)
            y = expert_out(x64[rows], W)
            out.index_add_(0, rows, y * w[rows, ks].to(torch.float64).unsqueeze(1))
        assert topk == w.shape[1]
        return out


def reference_moe(x: torch.Tensor, ids: torch.Tensor, w: torch.Tensor, weights: dict[int, dict]) -> torch.Tensor:
    """float64 sum over the top-k of w * expert(x) with GLOBAL ids (-1 skipped); weights[e] from the checkpoint."""
    out = torch.zeros((x.shape[0], x.shape[1]), dtype=torch.float64)
    x64 = x.to(torch.float64)
    for e in sorted(set(ids[ids >= 0].tolist())):
        rows, ks = (ids == e).nonzero(as_tuple=True)
        out.index_add_(0, rows, expert_out(x64[rows], weights[e]) * w[rows, ks].to(torch.float64).unsqueeze(1))
    return out
