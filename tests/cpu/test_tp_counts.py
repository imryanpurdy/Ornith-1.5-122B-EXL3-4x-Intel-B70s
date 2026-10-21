"""M2a, part 3: the per-rank EXL3 tensor counts the plugin's EXL3XPU_LOADED check asserts (DESIGN 2.3), computed
by tp.expected_loaded_counts from the checkpoint index (EXL3_MODEL_DIR) and checked against the DESIGN numbers:
  122B target: dense 301 matrices x 4 = 1,204 per rank; experts EP4 48 x 64 x 12 = 36,864, TP4 147,456;
  MTP drafter: its own 8 matrices x 4 = 32 dense plus the lm_head copy it loads (4), experts EP4 768, TP4 3,072;
  Ornith-mini (layers 0-3 + lm_head + mtp.*, DESIGN section 7): dense 26 x 4 = 104, experts TP1 12,288."""
from __future__ import annotations

import re

import pytest

from exl3xpu import tp
from results import record


@pytest.fixture(scope="module")
def names(ckpt):
    return ckpt.names()


def test_index_inventory(names):
    trellis = [n for n in names if n.endswith(".trellis")]
    mods = {n[: -len(".trellis")] for n in trellis}
    kinds = {}
    for m in mods:
        kinds[tp.classify(m)[0]] = kinds.get(tp.classify(m)[0], 0) + 1   # raises on any unknown EXL3 module
    exl3 = [n for n in names if n.rpartition(".")[0] in mods]
    sufs = {n.rpartition(".")[2] for n in exl3}
    assert sufs == {"trellis", "suh", "svh", "mul1"}, sufs                  # no bias, no legacy su/sv, no mcg
    assert len(mods) == 37941 and len(exl3) == 4 * 37941 and len(names) == 152540
    assert not any(n.split(".")[1] == "visual" for n in exl3)
    record("index", f"{len(names)} tensors, {len(mods)} EXL3 matrices, {len(exl3)} EXL3 tensors, kinds {kinds}")


@pytest.mark.parametrize("moe_mode", ["ep", "tp"])
@pytest.mark.parametrize("with_mtp", [False, True])
def test_counts_122b_tp4(names, moe_mode, with_mtp):
    expect_exp = 36864 if moe_mode == "ep" else 147456
    for r in range(4):
        c = tp.expected_loaded_counts(names, num_experts=256, moe_mode=moe_mode, tp_size=4, tp_rank=r,
                                      with_mtp=with_mtp)
        assert c["target"] == {"dense": 1204, "experts": expect_exp}, (r, c)
        if with_mtp:
            assert c["drafter"] == {"dense": 32 + 4, "experts": 768 if moe_mode == "ep" else 3072}, (r, c)
        else:
            assert c["drafter"] == {"dense": 0, "experts": 0}
    record(f"counts 122B TP4 {moe_mode.upper()}4 mtp={int(with_mtp)}", c)


def test_counts_other_tp(names):
    c1 = tp.expected_loaded_counts(names, num_experts=256, moe_mode="tp", tp_size=1, tp_rank=0, with_mtp=True)
    assert c1 == {"target": {"dense": 1204, "experts": 147456}, "drafter": {"dense": 36, "experts": 3072}}
    for r in range(2):
        c2 = tp.expected_loaded_counts(names, num_experts=256, moe_mode="ep", tp_size=2, tp_rank=r, with_mtp=False)
        assert c2["target"] == {"dense": 1204, "experts": 73728}
    record("counts 122B TP1", c1)


def test_counts_ornith_mini(names):
    """The mini is a byte-exact subset (DESIGN section 7): layers 0-3, embed, norm, lm_head, all mtp.*."""
    keep = re.compile(r"^(model\.language_model\.layers\.[0-3]\.|lm_head\.|mtp\.)")
    mini = [n for n in names if keep.match(n)]
    c = tp.expected_loaded_counts(mini, num_experts=256, moe_mode="tp", tp_size=1, tp_rank=0, with_mtp=False)
    assert c["target"] == {"dense": 104, "experts": 12288}
    c = tp.expected_loaded_counts(mini, num_experts=256, moe_mode="tp", tp_size=1, tp_rank=0, with_mtp=True)
    assert c["drafter"] == {"dense": 36, "experts": 3072}
    for r in range(4):
        e = tp.expected_loaded_counts(mini, num_experts=256, moe_mode="ep", tp_size=4, tp_rank=r, with_mtp=False)
        assert e["target"] == {"dense": 104, "experts": 4 * 64 * 12}
    record("counts mini TP1 / EP4", f"dense 104 experts 12288 / EP4 {4 * 64 * 12} per rank; drafter dense 36 "
                                    f"experts 3072 (TP1)")
