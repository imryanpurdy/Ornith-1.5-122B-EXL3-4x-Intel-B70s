"""Independent CPU reference for EXL3 mul1 linears, transcribed from exllamav3 v1.6.4 (CUDA sources, MIT).

Nothing here comes from trellis-serve. Sources (exllamav3/exllamav3_ext/...):
  * codebook   quant/codebook.cuh  decode_3inst<2>: x *= 0x83DCD12D; sum = dp4a(x, 0x01010101, 0x6400);
               value = __hfma(half(sum & 0xffff), half(0x1eee), half(0xc931))  (one fp16 rounding)
  * bit window quant/exl3_dq.cuh: dq8_aligned_4bits (K=4) and dq4 (K=5,6), selected by dq_dispatch
               (exl3_dq.cuh:288-330); generic dq<bits> (exl3_dq.cuh:15-29) is used as a self-check.
               Funnel/bfe helpers: ptx_portable.cuh exl3_shf_r_wrap / exl3_bfe_u32_16; fshift() in exl3_dq.cuh.
  * tile order quant/reconstruct.cu reconstruct_tile: lane L decodes t = 8L..8L+7, then the shfl_down(4) +
               tile[r][warp][c] half2 stores place values in the row-major 16x16 tile. Simulated literally.
  * rotation   modules/quant/exl3.py LinearEXL3.get_weight_tensor (:239-248): W = H_l(W_inner) * suh[:,None];
               W = H_r(W) * svh[None,:], H = Sylvester 128 / sqrt(128) (util/hadamard.py get_hadamard, data file
               hadamard_1.txt = "+", then sylvester()). y = x @ W  (W is [in, out]; i.e. y = x @ W_torch^T).
Bit operations use numpy uint64 so shifts are well defined (no signed overflow).
"""
from __future__ import annotations
import math

import numpy as np
import torch

MUL1_MULT = 0x83DCD12D
M32 = np.uint64(0xFFFFFFFF)
M16 = np.uint64(0xFFFF)


# ---------------------------------------------------------------- codebook (codebook.cuh decode_3inst<2>)
def _half_from_bits(b: int) -> float:
    return float(np.array([b], dtype=np.uint16).view(np.float16)[0])


_LUT = None


def mul1_lut() -> np.ndarray:
    """fp16 value of every 16-bit state. float64 holds h*k_inv+k_bias exactly (11b x 11b product + 11b
    addend), and numpy's float64->float16 cast rounds once (RNE), matching __hfma's single rounding."""
    global _LUT
    if _LUT is None:
        s = np.arange(65536, dtype=np.uint64)
        x = (s * np.uint64(MUL1_MULT)) & M32
        bsum = (x & np.uint64(0xFF)) + ((x >> np.uint64(8)) & np.uint64(0xFF)) + \
               ((x >> np.uint64(16)) & np.uint64(0xFF)) + ((x >> np.uint64(24)) & np.uint64(0xFF))
        hbits = (np.uint64(0x6400) + bsum).astype(np.uint16)          # dp4a(x, 0x01010101, 0x6400) & 0xffff
        h = hbits.view(np.float16).astype(np.float64)
        k_inv = _half_from_bits(0x1EEE)
        k_bias = _half_from_bits(0xC931)
        _LUT = (h * k_inv + k_bias).astype(np.float16)
    return _LUT


# ---------------------------------------------------------------- packed words
def words_u32(trellis: torch.Tensor) -> np.ndarray:
    """int16 [..., 16K] -> uint64 array of uint32 words [..., 8K]. reconstruct.cu copies the uint16 tile into
    uint32 s_packed with int4 loads, so (little-endian) word i = u16[2i] | u16[2i+1] << 16."""
    u16 = trellis.contiguous().numpy().view(np.uint16).astype(np.uint64)
    return u16[..., 0::2] | (u16[..., 1::2] << np.uint64(16))


def _fshift(b: np.ndarray, a: np.ndarray, shift: int) -> np.ndarray:
    # exl3_dq.cuh fshift(b, a, shift): (uint32)(((uint64)a << 32 | b) >> shift)   (shift may exceed 31)
    return (((a << np.uint64(32)) | b) >> np.uint64(shift)) & M32


def _shf_r_wrap(lo: np.ndarray, hi: np.ndarray, n: int) -> np.ndarray:
    # ptx_portable.cuh exl3_shf_r_wrap: low 32 bits of (hi:lo) >> (n mod 32)
    return (((hi << np.uint64(32)) | lo) >> np.uint64(n & 31)) & M32


def _dq8_aligned_4bits(w: np.ndarray, t_offset: int) -> list[np.ndarray]:
    """exl3_dq.cuh dq8_aligned_4bits: states w0..w7 for t_offset..t_offset+7 (vectorised over tiles)."""
    i1 = t_offset >> 3
    i0 = (i1 + 31) & 31
    a = w[:, i0]
    b = w[:, i1]
    s = _shf_r_wrap(b, a, 20)                                   # FSHF_IMM(s, b, a, 20)
    bfe = lambda src, pos: (src >> np.uint64(pos)) & M16       # BFE16_IMM
    return [bfe(s, 8), bfe(s, 4), s & M16, bfe(b, 16), bfe(b, 12), bfe(b, 8), bfe(b, 4), b & M16]


def _dq4(w: np.ndarray, t_offset: int, bits: int) -> list[np.ndarray]:
    """exl3_dq.cuh dq4<bits>: states for t_offset..t_offset+3."""
    nw = bits * 256 // 32
    b0 = (t_offset + 257) * bits - 16
    b1 = b0 + 3 * bits
    b2 = b1 + 16
    i0 = b0 // 32
    i2 = (b2 - 1) // 32
    s2 = (i2 + 1) * 32 - b2
    a = w[:, i0 % nw]
    b = w[:, i2 % nw]
    w3 = _fshift(b, a, s2) & M16
    w2 = _fshift(b, a, s2 + bits) & M16
    w1 = _fshift(b, a, s2 + bits * 2) & M16
    w0 = _fshift(b, a, s2 + bits * 3) & M16
    return [w0, w1, w2, w3]


def dq_generic(w: np.ndarray, t: int, bits: int) -> np.ndarray:
    """exl3_dq.cuh dq<bits>(ptr, t_offset) (state only). Self-check against the dispatch-specific readers."""
    nw = bits * 256 // 32
    b0 = t * bits + bits - 16 + 256 * bits
    b1 = b0 + 16
    i0 = b0 // 32
    i1 = (b1 - 1) // 32
    s0 = (i1 + 1) * 32 - b1
    a = w[:, i0 % nw]
    b = w[:, i1 % nw]
    return _shf_r_wrap(b, a, s0) & M16      # __funnelshift_r(b, a, s0) & 0xffff


def lane_states(w: np.ndarray, K: int, lane: int) -> list[np.ndarray]:
    """dq_dispatch<K> for t_offset = 8*lane: 8 states (t = 8L .. 8L+7)."""
    t = lane * 8
    if K == 4:
        return _dq8_aligned_4bits(w, t)
    if K in (5, 6):
        return _dq4(w, t, K) + _dq4(w, t + 4, K)
    raise NotImplementedError(f"K={K}")


def decode_tiles(w: np.ndarray, K: int) -> np.ndarray:
    """uint32 words [T, 8K] -> fp16 tiles [T, 16, 16] (row-major, rows = input dim, cols = output dim),
    placing values exactly as reconstruct.cu's warp shuffle does."""
    lut = mul1_lut()
    T = w.shape[0]
    v = [[lut[s.astype(np.int64)] for s in lane_states(w, K, L)] for L in range(32)]   # v[L][j]
    tile = np.zeros((T, 16, 16), dtype=np.float16)
    for L in range(32):
        if L & 4:
            continue                                   # if (!(lane_id & 4)) stores; lanes L+4 feed via shfl
        r0 = (L % 4) * 2
        rows = (r0, r0 + 1, r0 + 8, r0 + 9)
        c0 = L // 8
        c1 = c0 + 4
        for j in range(4):
            # m_j = (frag0 half j of lane L, same of lane L+4) -> tile[rows[j]][warp][c0] (half2 = 2 columns)
            tile[:, rows[j], 2 * c0] = v[L][j]
            tile[:, rows[j], 2 * c0 + 1] = v[L + 4][j]
            # m_{4+j} = frag1 halves -> tile[rows[j]][warp][c1]
            tile[:, rows[j], 2 * c1] = v[L][4 + j]
            tile[:, rows[j], 2 * c1 + 1] = v[L + 4][4 + j]
    return tile


def selfcheck_generic(trellis: torch.Tensor, K: int, ntiles: int = 64) -> int:
    """Number of (tile, t) states where dq_dispatch's reader disagrees with the generic dq<bits>."""
    w = words_u32(trellis.reshape(-1, trellis.shape[-1])[:ntiles])
    bad = 0
    for L in range(32):
        st = lane_states(w, K, L)
        for j in range(8):
            bad += int((st[j] != dq_generic(w, 8 * L + j, K)).sum())
    return bad


def reconstruct_inner(trellis: torch.Tensor, K: int, n0: int = 0, n1: int | None = None,
                      chunk_tiles: int = 1 << 16) -> torch.Tensor:
    """W_inner fp16 [k, n1-n0] (Hadamard domain), columns [n0, n1) (multiples of 16)."""
    tk, tn, _ = trellis.shape
    n1 = tn * 16 if n1 is None else n1
    sub = trellis[:, n0 // 16: n1 // 16].contiguous()
    tns = sub.shape[1]
    flat = sub.reshape(tk * tns, -1)
    out = np.empty((tk * tns, 16, 16), dtype=np.float16)
    for s in range(0, tk * tns, chunk_tiles):
        out[s:s + chunk_tiles] = decode_tiles(words_u32(flat[s:s + chunk_tiles]), K)
    W = out.reshape(tk, tns, 16, 16).transpose(0, 2, 1, 3).reshape(tk * 16, tns * 16)
    return torch.from_numpy(np.ascontiguousarray(W))


# ---------------------------------------------------------------- Hadamard + full weight
def hadamard128(dtype=torch.float64) -> torch.Tensor:
    h = torch.ones((1, 1), dtype=torch.float64)                 # hadamard_1.txt
    while h.shape[0] < 128:                                      # util/hadamard.py sylvester()
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return (h / math.sqrt(128)).to(dtype)


def weight_orig_chunk(W_inner: torch.Tensor, suh: torch.Tensor, svh_chunk: torch.Tensor) -> torch.Tensor:
    """float64 original-basis weight [k, nc] for a column chunk (nc multiple of 128), per get_weight_tensor."""
    k, nc = W_inner.shape
    H = hadamard128()
    w = W_inner.to(torch.float64)
    w = (H @ w.view(k // 128, 128, nc)).view(k, nc)              # preapply_had_l
    w = w * suh.to(torch.float64).unsqueeze(1)
    w = (w.view(k, nc // 128, 128) @ H).view(k, nc)              # preapply_had_r
    w = w * svh_chunk.to(torch.float64).unsqueeze(0)
    return w
