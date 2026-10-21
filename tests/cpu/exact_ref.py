"""Exact float64 reconstruction of an EXL3 mul1 matrix, for slice-equality tests.

W_inner comes from trellis_core.reference.reconstruct_inner (bit-exact fp16 decode). The two 128-block Hadamards
are applied as an unnormalised +-1 fast Walsh-Hadamard transform in float64. Every intermediate is an integer
multiple of the fp16 granularity of W_inner and stays below 2**53 of it (|W_inner| <= 8 is asserted: 128*128*8
* 2**24 = 2**41 < 2**53), so the transforms are exact and independent of summation order. Only the final elementwise
products with suh, svh and 1/128 round, and elementwise products of identical operands are bit-identical. Hence
reconstruct(slice) == reconstruct(full)[slice] bit for bit whenever the slice is a valid EXL3 matrix.

W_orig = diag(suh) . H . W_inner . H . diag(svh), H = Sylvester(128)/sqrt(128) blockwise (as reference.weight_orig).
"""
from __future__ import annotations

import torch

from trellis_core.reference import CB_MUL1, reconstruct_inner


def fwht128_(x: torch.Tensor, dim: int) -> torch.Tensor:
    """In-place unnormalised Sylvester Hadamard over contiguous 128-blocks of `dim` (x float64, contiguous)."""
    assert x.is_contiguous() and x.dtype == torch.float64
    size = x.shape[dim]
    assert size % 128 == 0
    pre = 1
    for d in range(dim):
        pre *= x.shape[d]
    post = x.numel() // (pre * size)
    v = x.view(pre * (size // 128), 128, post)
    h = 1
    while h < 128:
        w = v.view(v.shape[0], 128 // (2 * h), 2, h, post)
        a, b = w[:, :, 0], w[:, :, 1]
        t = a.clone()
        a.add_(b)
        b.neg_().add_(t)            # b <- t - b
        h *= 2
    return x


def weight_exact(trellis: torch.Tensor, suh: torch.Tensor, svh: torch.Tensor) -> torch.Tensor:
    K = trellis.shape[-1] // 16
    w = reconstruct_inner(trellis.contiguous(), K, CB_MUL1).to(torch.float64)
    assert float(w.abs().max()) <= 8.0, "exactness bound assumes |W_inner| <= 8 (mul1 codebook is within 3.46)"
    fwht128_(w, 0)
    fwht128_(w, 1)
    w.mul_(suh.to(torch.float64).unsqueeze(1))
    w.mul_(svh.to(torch.float64).unsqueeze(0))
    w.mul_(1.0 / 128.0)
    return w
