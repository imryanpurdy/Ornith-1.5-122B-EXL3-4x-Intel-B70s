"""Plugin loaders and the EXL3XPU_LOADED counters through vLLM 0.30's real load path (DESIGN rev 2, section 2.3/2.4):
model.load_weights (AutoWeightsLoader, the Qwen3.5 name mappers and their shard ids, MoERunner -> the plugin's
routed-expert loader) then vLLM's process_weights_after_loading, at TP 1/2/4, every rank, expert parallel on and
off, target model then MTP drafter.

  122B (EXL3_MODEL_DIR, vision on): every one of the 152,540 checkpoint tensors with its real dtype and shape
      (routed-expert payloads as meta tensors into meta arenas, codebook markers read from the checkpoint, the rest
      zeros), plugin_harness.py load-meta. Pass = each batch's EXL3XPU_LOADED counts equal tp.expected_loaded_counts
      of the index AND the DESIGN numbers (dense 1,204; drafter 36 = 32 + its lm_head copy; experts per split).
  Ornith-mini (EXL3_MINI_DIR): real tensors through vLLM's DefaultModelLoader (plugin_harness.py load-real). Pass =
      counters as above (104 / 12,288 at TP1) and every rank's dense parameters equal tp.qwen35_plan's slices of the
      checkpoint tensors at the plan's offsets, every local expert blob equals pack_expert of the plan-narrowed
      checkpoint tensors, and the pointer table addresses the arena slots (torch.equal everywhere)."""
from __future__ import annotations

import pytest

from conftest import run_harness
from results import record

SPLITS = [(1, False), (2, True), (2, False), (4, True), (4, False)]
CASES_122B = [{"tp": t, "ep": ep, "mtp": 3, "lm_only": False} for t, ep in SPLITS]
CASES_MINI = [{"tp": t, "ep": ep, "mtp": 3, "lm_only": True} for t, ep in SPLITS]

# per rank: (dense, experts) of the target and of the MTP drafter, DESIGN 2.3 (+36, not +32: tp-guard review D1)
DESIGN_122B = {(1, False): ((1204, 147456), (36, 3072)), (2, True): ((1204, 73728), (36, 1536)),
               (2, False): ((1204, 147456), (36, 3072)), (4, True): ((1204, 36864), (36, 768)),
               (4, False): ((1204, 147456), (36, 3072))}
DESIGN_MINI = {(1, False): ((104, 12288), (36, 3072)), (2, True): ((104, 6144), (36, 1536)),
               (2, False): ((104, 12288), (36, 3072)), (4, True): ((104, 3072), (36, 768)),
               (4, False): ((104, 12288), (36, 3072))}


@pytest.fixture(scope="module")
def load_122b(model_dir, lib_dir):
    return run_harness("load-meta", model_dir, CASES_122B, lib=lib_dir)


@pytest.fixture(scope="module")
def load_mini(mini_dir, lib_dir):
    return run_harness("load-real", mini_dir, CASES_MINI, lib=lib_dir)


def _check_counts(results, design, label):
    seen = set()
    for res in results:
        c = res["case"]
        assert res["ok"], f"{c} rank {res['rank']}: {res.get('error')}\n{res.get('tb')}"
        split = (c["tp"], c["ep"])
        closed = res["closed"]
        assert [b["role"] for b in closed] == ["target", "drafter"], closed
        for b, (d, e) in zip(closed, design[split]):
            assert b["rank"] == res["rank"] and b["tp_size"] == c["tp"], b
            assert b["counts"]["dense"] == b["expected"]["dense"] == d, b["line"]
            assert b["counts"]["experts"] == b["expected"]["experts"] == e, b["line"]
            assert b["line"].startswith(f"EXL3XPU_LOADED rank={res['rank']} dense={d} experts={e} model={b['role']}")
            seen.add((split, res["rank"], b["role"]))
        record(f"{label} tp={c['tp']} ep={c['ep']} rank={res['rank']}", " | ".join(b["line"] for b in closed))
    assert len(seen) == 2 * sum(t for t, _ in SPLITS)


def test_loaded_counts_122b(load_122b):
    _check_counts(load_122b, DESIGN_122B, "122B")


def test_loaded_counts_mini(load_mini):
    _check_counts(load_mini, DESIGN_MINI, "mini")


def test_mini_slices_exact(load_mini):
    for res in load_mini:
        c = res["case"]
        ep = c["ep"] and c["tp"] > 1
        e_local = 256 // c["tp"] if ep else 256
        for part, layers, mats, moe_layers in (("verify_target", 17, 26, 4), ("verify_drafter", 6, 9, 1)):
            v = res[part]
            assert v["mismatch"] == [], (c, res["rank"], part, v["mismatch"][:10])
            assert (v["dense_layers"], v["dense_matrices"]) == (layers, mats), v
            assert v["experts"] == moe_layers * e_local, v
        record(f"mini slices tp={c['tp']} ep={c['ep']} rank={res['rank']}",
               f"target {res['verify_target']['dense_matrices']} matrices + {res['verify_target']['experts']} "
               f"expert blobs, drafter {res['verify_drafter']['dense_matrices']} + "
               f"{res['verify_drafter']['experts']}: all equal")
