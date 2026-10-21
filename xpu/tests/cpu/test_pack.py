"""CPU test: expert blob layout written by moe_offload.pack_expert.

K=4 (D = 1): the trellis words are the raw checkpoint bytes (gate|up concatenated per k-row, then down), which is
the layout GateUpKernel / DownKernel<4> read (word j of tile (r, c) at ((r * tiles_n + c) * 8K + j)).
K=3 (D = 3): PLANAR4, checked against the index formula, independently of planar4().
Vector tail: suh_g suh_u svh_g svh_u suh_d svh_d (fp16) at the Geom offsets of exl3_moe.sycl.

Usage (no GPU needed):  python test_pack.py [--moe-lib path/to/_moe.so]
With --moe-lib (or EXL3_MOE_LIB), blob sizes are also checked against the library's blob_bytes().
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from exl3xpu.moe_offload import pack_expert, planar4, trellis_words  # noqa: E402


def geom(H, I, K):
    W = 8 * K
    gu = (H // 16) * (2 * I // 16) * W * 4
    d = (I // 16) * (H // 16) * W * 4
    off = {"suh_g": gu + d}
    off["suh_u"] = off["suh_g"] + 2 * H
    off["svh_g"] = off["suh_u"] + 2 * H
    off["svh_u"] = off["svh_g"] + 2 * I
    off["suh_d"] = off["svh_u"] + 2 * I
    off["svh_d"] = off["suh_d"] + 2 * I
    return gu, d, off, off["svh_d"] + 2 * H


def expert(H, I, K, g):
    def mat(k, n):
        return {"trellis": torch.randint(-32768, 32768, (k // 16, n // 16, 16 * K), dtype=torch.int16, generator=g),
                "suh": torch.randn(k, generator=g).half(), "svh": torch.randn(n, generator=g).half()}
    return mat(H, I), mat(H, I), mat(I, H)


def raw(t):
    return t.contiguous().view(torch.uint8).flatten()


def check(H, I, K, lib, g):
    gate, up, down = expert(H, I, K, g)
    blob = pack_expert(gate, up, down, K)
    gu_b, d_b, off, size = geom(H, I, K)
    assert blob.dtype == torch.uint8 and blob.numel() == size, (blob.numel(), size)
    if lib is not None:
        assert int(lib.blob_bytes(H, I, K)) == size, (int(lib.blob_bytes(H, I, K)), size)
    W = 8 * K
    tn_gu, tn_d = 2 * I // 16, H // 16
    words_gu = blob[:gu_b].view(torch.int32).view(H // 16, tn_gu, W)
    words_d = blob[gu_b:gu_b + d_b].view(torch.int32).view(I // 16, tn_d, W)
    gu_src = torch.cat([gate["trellis"], up["trellis"]], dim=1)
    if K == 4:
        # raw bytes, row by row: gate tiles then up tiles of each k-row, then down
        exp = torch.cat([torch.cat([raw(gate["trellis"][r]), raw(up["trellis"][r])]) for r in range(H // 16)])
        assert torch.equal(blob[:gu_b], exp), "K=4 gate|up bytes"
        assert torch.equal(blob[gu_b:gu_b + d_b], raw(down["trellis"])), "K=4 down bytes"
        # word j of tile (r, c) = int16 pair (2j, 2j+1), little-endian
        for r, c, j in ((0, 0, 0), (3, tn_gu - 1, W - 1), (H // 16 - 1, I // 16, 5), (7, I // 16 - 1, 17)):
            lo = int(gu_src[r, c, 2 * j]) & 0xFFFF
            hi = int(gu_src[r, c, 2 * j + 1]) & 0xFFFF
            assert (int(words_gu[r, c, j]) & 0xFFFFFFFF) == (lo | (hi << 16)), ("K=4 word", r, c, j)
    else:
        # PLANAR4: word (row r, tile 4q+t, period p, plane i) of the source lands at [r][q][i][t][p]
        assert torch.equal(words_gu.view(-1), planar4(gu_src, K).view(-1))
        src = gu_src.contiguous().view(torch.int32).view(H // 16, tn_gu, 8, K)
        dst = words_gu.view(H // 16, tn_gu // 4, K, 4, 8)
        gen = torch.Generator().manual_seed(1)
        for _ in range(2000):
            r = int(torch.randint(H // 16, (1,), generator=gen)); c = int(torch.randint(tn_gu, (1,), generator=gen))
            p = int(torch.randint(8, (1,), generator=gen)); i = int(torch.randint(K, (1,), generator=gen))
            assert int(dst[r, c // 4, i, c % 4, p]) == int(src[r, c, p, i]), ("planar4", r, c, p, i)
        assert torch.equal(words_d.view(-1), planar4(down["trellis"], K).view(-1))
    for name, v, n in (("suh_g", gate["suh"], H), ("suh_u", up["suh"], H), ("svh_g", gate["svh"], I),
                       ("svh_u", up["svh"], I), ("suh_d", down["suh"], I), ("svh_d", down["svh"], H)):
        assert torch.equal(blob[off[name]:off[name] + 2 * n], raw(v)), name
    out = torch.full((size,), 0xA5, dtype=torch.uint8)
    pack_expert(gate, up, down, K, out=out)
    assert torch.equal(out, blob), "out= path"
    return size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--moe-lib", default=os.environ.get("EXL3_MOE_LIB"))
    a = ap.parse_args()
    lib = None
    if a.moe_lib:
        torch.ops.load_library(a.moe_lib)
        lib = torch.ops.exl3xpu_moe
    g = torch.Generator().manual_seed(0)
    for H, I, K, what in ((3072, 1024, 4, "Ornith expert (EP)"), (3072, 512, 4, "Ornith MoE-TP2 slice"),
                          (3072, 256, 4, "Ornith MoE-TP4 slice"), (2560, 640, 3, "Flash-Next expert"),
                          (2048, 512, 5, "K=5 planar4 (35B shape)")):
        size = check(H, I, K, lib, g)
        print(f"PASS pack H={H} I={I} K={K} blob={size} B  ({what})")
    try:
        trellis_words(torch.zeros((8, 8, 96), dtype=torch.int16), 6)
        raise AssertionError("K=6 must be refused")
    except ValueError:
        print("PASS pack K=6 refused")
    print("test_pack: all PASS" + ("" if lib is not None else " (blob_bytes not checked: no --moe-lib)"))


if __name__ == "__main__":
    main()
