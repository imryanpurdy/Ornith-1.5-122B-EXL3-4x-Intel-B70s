"""CPU test: the MoE split planner (exl3xpu_moe::moe_plan) and the compiled variant table (moe_info).

No device is opened: moe_plan and moe_info(-1) are host-only ops. Checks:
  1. moe_info(-1): the compiled GEMV kernels are exactly the per-K tables of exl3_moe.sycl (K=4: 11 gate|up (MR, P)
     x {fp16, bf16} + 27 down (MR, P, RB); K=3: the Flash-Next set, down RB = min(40/P, 8)); limits unknown (-1).
  2. Ornith shapes (H=3072, K=4; I = 1024 EP / whole expert, 512 MoE-TP2, 256 MoE-TP4): a valid plan for every
     M = 1..4096 (rule-checked: compiled, H/16 and I/16 splits, RB, T <= 32 at grf_size<256>), and for every chunk
     of the 1024-row M-chunking of M = 1025..4096.
  3. Flash-Next (2560, 640, 3), M = 1..1024: the picks of the pre-planner code (MR 1 / 2 / 4 at M <= 4 / 64 / more,
     gate|up P 5, down P 10 at M = 1 else 5, RB = min(40/P, 8)).
  4. K=4 starting defaults; overrides (moe_set_mr / moe_set_splits): every compiled K=4 variant is reachable, and
     an invalid override gives an empty plan (moe_forward would raise) instead of a wrong launch.
  5. INFO: the 35B shapes (H=2048), which T11 adds kernels for.
Usage:  python test_moe_plan.py --moe-lib path/to/_moe.so [--json out.json]
"""
import argparse
import json
import os
import sys

import torch

ORNITH = [(3072, 1024, 4), (3072, 512, 4), (3072, 256, 4)]
FLASH = (2560, 640, 3)
INFO_35B = [(2048, I, K) for I in (512, 256) for K in (3, 4, 5)]

K4_GU = {(1, 4), (1, 6), (1, 8), (1, 12), (2, 4), (2, 6), (2, 8), (2, 12), (4, 4), (4, 6), (4, 8)}
K4_DN_PAIRS = {(2, 8), (4, 8), (8, 8), (16, 4), (4, 4), (8, 4), (8, 2), (16, 2), (16, 1)}
K4_DN = {(mr, p, rb) for mr in (1, 2, 4) for p, rb in K4_DN_PAIRS}
K3_GU = {(1, 10), (2, 10), (4, 10), (8, 5), (1, 5), (2, 5), (4, 5), (1, 20), (2, 20)}
K3_DN = {(mr, p, min(40 // p, 8)) for mr, p in {(1, 5), (2, 5), (4, 5), (8, 5), (1, 10), (2, 10), (4, 10), (1, 20),
                                                (2, 20), (1, 40)}}


def fail(msg):
    print("FAIL " + msg)
    sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--moe-lib", default=os.environ.get("EXL3_MOE_LIB"), required=os.environ.get("EXL3_MOE_LIB") is None)
    ap.add_argument("--json", default=None, help="write the plan summary (variant sets per shape) here")
    a = ap.parse_args()
    torch.ops.load_library(a.moe_lib)
    X = torch.ops.exl3xpu_moe

    # 1. compiled table
    info = list(X.moe_info(-1))
    F, n = info[0], info[1]
    rows = [info[2 + i * F: 2 + (i + 1) * F] for i in range(n)]
    assert len(info) == 2 + n * F and F == 11, (F, n, len(info))
    gu = {}
    dn = {}
    for kind, K, MR, P, RB, dt, T, grf, slm, wg, priv in rows:
        if wg != -1 or priv != -1:
            fail(f"moe_info(-1) reported a device limit {wg}/{priv} without a device")
        if kind == 0:
            gu.setdefault((K, MR, P), set()).add(dt)
            assert RB == 0 and T == 4 * P and slm == P * MR * 256 * 4
        else:
            dn[(K, MR, P, RB)] = dt
            assert dt == 0 and T == 2 * P and slm == P * MR * 128 * 4
        assert grf == (256 if (K == 4 and MR >= 4) else 128), (K, MR, grf)
        if grf == 256 and T > 32:
            fail(f"grf256 kernel with T={T} > 32: {(kind, K, MR, P, RB)}")
    if any(v != {1, 2} for v in gu.values()):
        fail("gate|up must have fp16 and bf16 kernels for every (K, MR, P)")
    want_gu = {(4,) + v for v in K4_GU} | {(3,) + v for v in K3_GU}
    want_dn = {(4,) + v for v in K4_DN} | {(3,) + v for v in K3_DN}
    if set(gu) != want_gu or set(dn) != want_dn:
        fail(f"compiled table differs: gu extra {set(gu) - want_gu} missing {want_gu - set(gu)}; "
             f"dn extra {set(dn) - want_dn} missing {want_dn - set(dn)}")
    print(f"PASS moe_info(-1): {n} kernels = {len(gu)} gate|up (MR,P) x 2 dtypes + {len(dn)} down (MR,P,RB); "
          f"K=4: {sum(1 for k in gu if k[0] == 4)} + {sum(1 for k in dn if k[0] == 4)}; limits -1 on CPU")

    def plan(H, I, K, M):
        return tuple(int(v) for v in X.moe_plan(H, I, K, M))

    def valid(H, I, K, M, p):
        MR, g, d, RB = p
        if MR <= 0:
            return "no kernel"
        if (K, MR, g) not in gu:
            return f"gate|up {(K, MR, g)} not compiled"
        if (K, MR, d, RB) not in dn:
            return f"down {(K, MR, d, RB)} not compiled"
        tk = H // 16
        if tk % g or (tk // g) % 8:
            return f"gate|up P={g} invalid for H={H}"
        tki = I // 16
        if tki % d:
            return f"down P={d} invalid for I={I}"
        rps = tki // d
        if RB != min(rps, 8) or rps % RB:
            return f"down RB={RB} invalid (rps {rps})"
        if M <= 4 and MR != 1:
            return "pair mode must use MR=1"
        if K == 4 and MR >= 4 and (4 * g > 32 or 2 * d > 32):
            return "grf256 T > 32"
        return None

    summary = {}
    # 2. Ornith shapes, M = 1..4096 and the 1024-row chunks
    for H, I, K in ORNITH:
        seen = {}
        for M in range(1, 4097):
            p = plan(H, I, K, M)
            err = valid(H, I, K, M, p)
            if err:
                fail(f"moe_plan{(H, I, K, M)} = {p}: {err}")
            if M <= 1024:
                seen.setdefault(p, [M, M])[1] = M
        for M in range(1025, 4097):
            for i in range(0, M, 1024):
                c = min(1024, M - i)
                if valid(H, I, K, c, plan(H, I, K, c)):
                    fail(f"chunk {c} of M={M} at {(H, I, K)} has no plan")
        # K=4 starting defaults (DESIGN 3.1 item 2)
        dn0 = min(16, max(4, (I // 16) // 8))
        for M, want in ((1, (1, 12, dn0)), (4, (1, 12, dn0)), (5, (2, 6, dn0)), (64, (2, 6, dn0)), (65, (4, 4, dn0)),
                        (1024, (4, 4, dn0)), (4096, (4, 4, dn0))):
            if plan(H, I, K, M)[:3] != want:
                fail(f"default moe_plan{(H, I, K, M)} = {plan(H, I, K, M)}, want {want}")
        summary[f"{H},{I},{K}"] = [{"plan": list(p), "M": r} for p, r in sorted(seen.items(), key=lambda kv: kv[1][0])]
        print(f"PASS moe_plan {(H, I, K)}: M=1..4096 0 misses, all chunks of M<=4096 planned; plans for M<=1024: "
              + ", ".join(f"{list(p)}@M{r[0]}-{r[1]}" for p, r in sorted(seen.items(), key=lambda kv: kv[1][0])))

    # 3. Flash-Next: the pre-planner picks
    H, I, K = FLASH
    seen = {}
    for M in range(1, 1025):
        mr = 1 if M <= 4 else (2 if M <= 64 else 4)
        d = 10 if M <= 1 else 5
        want = (mr, 5, d, min(40 // d, 8))
        p = plan(H, I, K, M)
        if p != want:
            fail(f"Flash-Next moe_plan{(H, I, K, M)} = {p}, pre-planner code picks {want}")
        seen.setdefault(p, [M, M])[1] = M
    summary[f"{H},{I},{K}"] = [{"plan": list(p), "M": r} for p, r in sorted(seen.items(), key=lambda kv: kv[1][0])]
    print("PASS moe_plan (2560, 640, 3) M=1..1024 == pre-planner picks (M >= 256 runs the K=3 DPAS prefill; the plan "
          "is the GEMV fallback there): " + ", ".join(f"{list(p)}@M{r[0]}-{r[1]}" for p, r in seen.items()))

    # 4. overrides
    try:
        reach = 0
        for H, I, K in ORNITH:
            for (k, mr, g) in sorted(gu):
                if k != 4 or (H // 16) % g or ((H // 16) // g) % 8:
                    continue
                for (k2, mr2, d, rb) in sorted(dn):
                    if k2 != 4 or mr2 != mr or (I // 16) % d or rb != min((I // 16) // d, 8):
                        continue
                    X.moe_set_mr(mr)
                    X.moe_set_splits(g, d)
                    p = plan(H, I, K, 100)                  # grouped mode: the MR override applies there
                    if p != (mr, g, d, rb):
                        fail(f"override MR={mr} gu={g} dn={d} at {(H, I, K)}: plan {p}")
                    reach += 1
        X.moe_set_mr(4); X.moe_set_splits(12, 0)
        if plan(3072, 1024, 4, 100) != (0, 0, 0, 0):
            fail("override K=4 MR=4 P=12 (not compiled) must give an empty plan")
        X.moe_set_mr(0); X.moe_set_splits(5, 0)
        if plan(3072, 1024, 4, 100) != (0, 0, 0, 0):
            fail("override gate|up P=5 at H=3072 (tk 192 % 5 != 0) must give an empty plan")
        X.moe_set_splits(0, 3)
        if plan(3072, 1024, 4, 100) != (0, 0, 0, 0):
            fail("override down P=3 must give an empty plan")
        X.moe_set_mr(8); X.moe_set_splits(0, 0)
        if plan(3072, 1024, 4, 100) != (0, 0, 0, 0):
            fail("override K=4 MR=8 (not compiled) must give an empty plan")
        X.moe_set_mr(8)
        if plan(2560, 640, 3, 100) != (8, 5, 5, 8):
            fail(f"override K=3 MR=8 must give (8, 5, 5, 8), got {plan(2560, 640, 3, 100)}")
        X.moe_set_mr(4)
        if plan(3072, 1024, 4, 2)[0] != 1:
            fail("pair mode ignores the MR override (MR=1)")
        print(f"PASS overrides: {reach} (shape, K=4 variant) combinations reachable exactly; invalid overrides -> empty plan")
    finally:
        X.moe_set_mr(0)
        X.moe_set_splits(0, 0)
    if plan(3072, 1024, 4, 100) != (4, 4, 8, 8):
        fail("overrides not reset")
    for args in ((3072, 1024, 4, 0), (3072, 1000, 4, 8), (3072, 1024, 6, 8), (3000, 1024, 4, 8)):
        if plan(*args) != (0, 0, 0, 0):
            fail(f"moe_plan{args} must be empty")
    print("PASS moe_plan empty for M=0, I%128!=0, K=6, H%128!=0")

    # 5. INFO
    for H, I, K in INFO_35B:
        miss = [M for M in range(1, 1025) if valid(H, I, K, M, plan(H, I, K, M))]
        print(f"INFO 35B shape {(H, I, K)}: {len(miss)} of 1024 M without a plan"
              + (f" (first M={miss[0]}); T11 adds these kernels" if miss else ""))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(summary, f, indent=1)
    print("test_moe_plan: all PASS")


if __name__ == "__main__":
    main()
