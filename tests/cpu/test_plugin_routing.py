"""Plugin routing over the FULL module tree (DESIGN rev 2, sections 2.2-2.4): vLLM 0.30's own
Qwen3_5MoeForConditionalGeneration (and the Qwen3_5MoeMTP drafter) built on the CPU with the plugin registered,
for the 122B config (vision on, as served) and Ornith-mini (--language-model-only), at TP 1/2/4, every rank
faked, expert parallel on and off (plugin_harness.py routing). Checks, per module and rank:
  - the quant method each module gets (RoutedExperts -> Exl3MoEMethod K4; EXL3 linears and both lm_heads ->
    Exl3LinearMethod with the checkpoint K; in_proj_ba, conv1d, router gates and visual.* -> unquantized; the
    embedding -> UnquantizedEmbeddingMethod) and that nothing else exists and nothing raised;
  - create_weights shapes: local output partitions, trellis [k/16, n/16, 16K], suh [groups, k], svh [n];
  - MoE: local expert count, I per partition, blob size, arena shape, which global experts are local;
  - DESIGN section 2.4 / 6 arena sizes per card (EP4 13.570 GiB + 0.283 MTP, TP4 13.729 + 0.286)."""
from __future__ import annotations

import collections

import pytest

from conftest import run_harness
from results import record

TPS = (1, 2, 4)
CASES_122B = [{"tp": t, "ep": ep, "mtp": 3, "lm_only": False} for t in TPS for ep in (False, True)] + \
             [{"tp": 4, "ep": True, "mtp": 0, "lm_only": True}]
CASES_MINI = [{"tp": t, "ep": ep, "mtp": 3, "lm_only": True} for t in TPS for ep in (False, True)]
H, IE = 3072, 1024
BLOB = {1024: 4743168, 512: 2380800, 256: 1199616}


def blob(I):
    return (H * 2 * I * 4) // 8 + (I * H * 4) // 8 + 2 * (3 * H + 3 * I)


@pytest.fixture(scope="module")
def routing_122b(model_dir):
    return run_harness("routing", model_dir, CASES_122B)


@pytest.fixture(scope="module")
def routing_mini(mini_dir):
    return run_harness("routing", mini_dir, CASES_MINI)


def leaf_of(prefix: str) -> str:
    return prefix.rsplit(".", 1)[-1]


def expected_method(row: dict) -> tuple[str, int | None]:
    cls, prefix = row["cls"], row["prefix"]
    mtp = prefix.startswith("mtp.")
    if cls == "RoutedExperts":
        return "Exl3MoEMethod", 4
    if cls == "VocabParallelEmbedding":
        return "UnquantizedEmbeddingMethod", None
    if cls == "ParallelLMHead":
        return "Exl3LinearMethod", 6
    if prefix.startswith("visual."):
        return "UnquantizedLinearMethod", None
    leaf = leaf_of(prefix)
    if leaf in ("conv1d", "gate", "shared_expert_gate", "in_proj_ba"):
        return "UnquantizedLinearMethod", None
    if leaf in ("qkv_proj", "o_proj", "in_proj_qkvz", "out_proj", "gate_up_proj", "down_proj", "fc"):
        return "Exl3LinearMethod", 4 if mtp else 5
    raise AssertionError(f"unexpected module {row}")


def expected_dense(leaf: str, t: int) -> tuple[list[int], int]:
    """(local output partitions, local k) of the Ornith EXL3 linears at TP t (vLLM 0.30 partitioning)."""
    kv = 256 * max(1, 2 // t)
    return {
        "qkv_proj": ([16384 // t, kv, kv], H),
        "o_proj": ([H], 8192 // t),
        "in_proj_qkvz": ([2048 // t, 2048 // t, 8192 // t, 8192 // t], H),
        "out_proj": ([H], 8192 // t),
        "gate_up_proj": ([1024 // t, 1024 // t], H),
        "down_proj": ([H], 1024 // t),
        "fc": ([H // t], 6144),
        "lm_head": ([248320 // t], H),
    }[leaf]


def check_case(res: dict, n_layers: int) -> dict:
    case, rank = res["case"], res["rank"]
    assert res["ok"], f"{case} rank {rank}: {res.get('error')}\n{res.get('tb')}"
    t, ep = case["tp"], case["ep"] and case["tp"] > 1
    counts = collections.Counter()
    arena_bytes = collections.Counter()
    for part in ("target", "drafter"):
        for row in res.get(part, []):
            meth, K = expected_method(row)
            assert row["method"] == meth, row
            assert row.get("K") == K, row
            counts[(part, row["cls"], meth, K)] += 1
            if meth == "Exl3LinearMethod":
                leaf = "lm_head" if row["cls"] == "ParallelLMHead" else leaf_of(row["prefix"])
                sizes, k = expected_dense(leaf, t)
                n = sum(sizes)
                assert row["sizes"] == sizes and row["tp"] == [rank, t], row
                assert row["trellis"] == [k // 16, n // 16, 16 * K], row
                assert row["suh"] == [len(sizes), k] and row["svh"] == [n], row
            if meth == "Exl3MoEMethod":
                E = 256 // t if ep else 256
                I = IE if ep else IE // t
                assert (row["E"], row["H"], row["I"]) == (E, H, I), row
                assert row["blob"] == blob(I) == BLOB[I] and row["arena"] == [E, BLOB[I]], row
                assert row["use_ep"] == ep and row["ep"] == ([rank, t] if ep else [0, 1]), row
                assert row["moe_tp"] == ([0, 1] if ep else [rank, t]), row
                assert row["local_ids"] == (list(range(rank * E, rank * E + E)) if ep else list(range(256))), row
                arena_bytes[part] += row["arena"][0] * row["arena"][1]
    # exact module inventory: EXL3 dense layers 193 (122B) / 17 (mini); drafter 6; MoE n_layers (+1)
    dense_t = sum(v for (p, c, m, k), v in counts.items() if p == "target" and m == "Exl3LinearMethod")
    moe_t = sum(v for (p, c, m, k), v in counts.items() if p == "target" and m == "Exl3MoEMethod")
    assert moe_t == n_layers and dense_t == {48: 193, 4: 17}[n_layers], counts
    if case.get("mtp"):
        dense_d = sum(v for (p, c, m, k), v in counts.items() if p == "drafter" and m == "Exl3LinearMethod")
        moe_d = sum(v for (p, c, m, k), v in counts.items() if p == "drafter" and m == "Exl3MoEMethod")
        assert (dense_d, moe_d) == (6, 1) and res["drafter_arch"] == "Qwen3_5MoeMTP", counts
    vis = sum(v for (p, c, m, k), v in counts.items() if p == "target" and m == "UnquantizedLinearMethod") - \
        n_layers * 2 - (n_layers * 3 // 4) * 2      # router + shared gate per layer, conv1d + in_proj_ba per GDN layer
    if case.get("lm_only", True):
        assert vis == 0, counts
    else:
        assert vis > 0, counts                       # the vision tower's linears, all unquantized
    return {"counts": counts, "arena_gib": {k: v / 2**30 for k, v in arena_bytes.items()}}


def test_routing_122b_every_rank(routing_122b):
    assert len(routing_122b) == sum(c["tp"] for c in CASES_122B)
    for res in routing_122b:
        check_case(res, 48)


def test_routing_mini_every_rank(routing_mini):
    assert len(routing_mini) == sum(c["tp"] for c in CASES_MINI)
    for res in routing_mini:
        check_case(res, 4)


def test_arena_sizes_match_design(routing_122b):
    """DESIGN 2.4 table: arena per card EP4 13.570 GiB (+0.283 MTP), TP4 13.729 GiB (+0.286)."""
    got = {}
    for res in routing_122b:
        c = res["case"]
        if c["tp"] == 4 and c.get("mtp"):
            got.setdefault("ep4" if c["ep"] else "tp4", set()).add(
                tuple(round(v, 3) for v in check_case(res, 48)["arena_gib"].values()))
    assert got == {"ep4": {(13.570, 0.283)}, "tp4": {(13.729, 0.286)}}, got
    record("arena GiB per card (target, MTP drafter)", got)


def test_tp4_rank_table_literal(routing_122b):
    """DESIGN 2.3 TP4 table, literal local shapes for rank 0..3 (k x n)."""
    want = {"qkv_proj": (3072, 4608), "o_proj": (2048, 3072), "in_proj_qkvz": (3072, 5120),
            "out_proj": (2048, 3072), "gate_up_proj": (3072, 512), "down_proj": (256, 3072), "lm_head": (3072, 62080),
            "fc": (6144, 768)}
    seen = collections.defaultdict(set)
    for res in routing_122b:
        if res["case"]["tp"] != 4 or not res["case"].get("mtp"):
            continue
        for row in res["target"] + res["drafter"]:
            if row["method"] == "Exl3LinearMethod":
                leaf = "lm_head" if row["cls"] == "ParallelLMHead" else leaf_of(row["prefix"])
                seen[leaf].add((row["trellis"][0] * 16, row["trellis"][1] * 16))
    assert {k: v for k, v in seen.items()} == {k: {v} for k, v in want.items()}, seen
