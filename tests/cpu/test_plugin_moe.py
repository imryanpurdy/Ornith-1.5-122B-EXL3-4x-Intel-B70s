"""Exl3MoEMethod.apply on real Ornith-mini layers (0 and 3) built and loaded through vLLM 0.30 at TP1, EP2, MoE-TP2,
EP4, MoE-TP4 (every rank), with the grouped kernel replaced by moe_ref.BlobKernel: a pure-torch, float64
interpreter of the kernel contract that finds each expert's blob through the method's pointer table and unpacks it
(plugin_harness.py moe). The reference is computed from the checkpoint tensors (exact float64 EXL3 weights).

Inputs: 48 rows routed over 15 experts that span every EP4 rank, incl. layer 3's dead experts (72, 166, 180, 252);
rows 0-3 are vLLM padding (id -1, weight 0), row 4 of layer 3 routes only to dead experts, row 5 has 4 x -1;
plus a 2,500-row batch (every 7th row padding) that apply() must run in 1,024-row chunks.

Pass, per rank: the kernel always gets int32 contiguous local ids in [-1, E), float32 contiguous weights, I/K/E of
the layer; chunks are exactly [1024, 1024, 452]; padding rows and the dead-expert row are exact zeros; the eager
dump (EXL3_DUMP_LAYER) holds exactly the inputs and output; under EP and at TP1 the partial equals the reference
restricted to the rank's experts bit for bit and the mapped ids are the expert map's. Over ranks: the sum of
partials equals the full reference within 1e-12 (relative to the max |out|), chunked batch included."""
from __future__ import annotations

import pytest

from conftest import run_harness
from results import record

CASES = [{"tp": 1, "ep": False}, {"tp": 2, "ep": True}, {"tp": 2, "ep": False}, {"tp": 4, "ep": True},
         {"tp": 4, "ep": False}]


@pytest.fixture(scope="module")
def moe_results(mini_dir, lib_dir):
    return run_harness("moe", mini_dir, CASES, lib=lib_dir)


@pytest.mark.parametrize("i", range(len(CASES)), ids=[f"tp{c['tp']}{'-ep' if c['ep'] else ''}" for c in CASES])
def test_moe_method_case(moe_results, i):
    res = moe_results[i]
    assert res["case"] == CASES[i]
    assert res["ok"], (res.get("error"), res.get("tb"), {k: v for k, v in res["checks"].items() if not v})
    n_ranks = CASES[i]["tp"]
    per_rank = {"kernel_args", "chunks", "padding_rows_zero", "dump"}
    for r in range(n_ranks):
        for L in (0, 3):
            for chk in per_rank | ({"dead_row_zero"} if L == 3 else set()):
                assert res["checks"][f"L{L}_r{r}_{chk}"] is True, (L, r, chk)
            if CASES[i]["ep"] or n_ranks == 1:
                assert res["checks"][f"L{L}_r{r}_partial_eq_ref_bitwise"] is True
                assert res["checks"][f"L{L}_r{r}_mapped_ids_ok"] is True
    for L in (0, 3):
        assert res["checks"][f"L{L}_sum_vs_ref"] and res["checks"][f"L{L}_sum_vs_ref_chunked"]
        assert res["checks"][f"L{L}_sum_padding_zero"]
    record(f"moe {res['case']}", {k: f"{v:.2e}" for k, v in res["rel"].items() if v is not None and "sum" in k})
