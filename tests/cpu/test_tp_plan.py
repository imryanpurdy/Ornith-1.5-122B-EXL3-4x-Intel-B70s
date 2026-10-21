"""M2a, part 1 (no checkpoint needed): the slice plan of xpu/exl3xpu/tp.py against
  (a) the TP4 table of DESIGN section 2.3, as literal numbers,
  (b) vLLM 0.30's own weight loaders (unquantized layers built on CPU with a fake TP group, fed index-coded
      tensors): the plan must pick exactly the checkpoint columns / rows that vLLM puts at each local position,
  (c) synthetic tensors: narrowing, contiguity, refusals.
Dimensions are Ornith-1.5-122B's (text config): hidden 3072, 32 heads (x2 with the output gate) of 256, 2 KV
heads, GDN key 16x128, value 64x128, shared expert 1024, vocab 248320, MTP fc 6144->3072."""
from __future__ import annotations

import pytest
import torch

from exl3xpu import tp
from trellis_core.format.geometry import SplitError

H = 3072
CFG = dict(hidden_size=H, num_attention_heads=32, attn_output_gate=True, num_key_value_heads=2, head_dim=256,
           linear_num_key_heads=16, linear_key_head_dim=128, linear_num_value_heads=64, linear_value_head_dim=128,
           shared_expert_intermediate_size=1024, moe_intermediate_size=1024, vocab_size=248320)
P = "model.language_model.layers."
# (checkpoint module, k, n)
MATRICES = [
    (P + "3.self_attn.q_proj", H, 16384), (P + "3.self_attn.k_proj", H, 512), (P + "3.self_attn.v_proj", H, 512),
    (P + "3.self_attn.o_proj", 8192, H),
    (P + "0.linear_attn.in_proj_qkv", H, 12288), (P + "0.linear_attn.in_proj_z", H, 8192),
    (P + "0.linear_attn.out_proj", 8192, H),
    (P + "0.mlp.shared_expert.gate_proj", H, 1024), (P + "0.mlp.shared_expert.up_proj", H, 1024),
    (P + "0.mlp.shared_expert.down_proj", 1024, H),
    ("lm_head", H, 248320), ("mtp.fc", 6144, H),
    ("mtp.layers.0.self_attn.q_proj", H, 16384), ("mtp.layers.0.self_attn.k_proj", H, 512),
    ("mtp.layers.0.self_attn.v_proj", H, 512), ("mtp.layers.0.self_attn.o_proj", 8192, H),
    ("mtp.layers.0.mlp.shared_expert.gate_proj", H, 1024), ("mtp.layers.0.mlp.shared_expert.up_proj", H, 1024),
    ("mtp.layers.0.mlp.shared_expert.down_proj", 1024, H),
]


def _plan(mod, k, n, r, t):
    return tp.qwen35_plan(mod, k, n, CFG, tp_rank=r, tp_size=t)


# ------------------------------------------------------------------------------------- (a) DESIGN 2.3 at TP4
def _design_tp4(leaf: str, r: int):
    """DESIGN rev 2 section 2.3 table, rank r of 4: (cols, rows, dst_offset, local k x n)."""
    W = {
        "q_proj": ([(4096 * r, 4096)], (0, H), 0, (H, 4096)),                       # blocks 0-31
        "k_proj": ([(256 * (r // 2), 256)], (0, H), 4096, (H, 256)),                # blocks 32-33
        "v_proj": ([(256 * (r // 2), 256)], (0, H), 4352, (H, 256)),                # blocks 34-35
        "o_proj": ([(0, H)], (2048 * r, 2048), 0, (2048, H)),
        "in_proj_qkv": ([(512 * r, 512), (2048 + 512 * r, 512), (4096 + 2048 * r, 2048)], (0, H), 0, (H, 3072)),
        "in_proj_z": ([(2048 * r, 2048)], (0, H), 3072, (H, 2048)),                 # blocks 24-39
        "out_proj": ([(0, H)], (2048 * r, 2048), 0, (2048, H)),
        "gate_proj": ([(256 * r, 256)], (0, H), 0, (H, 256)),                       # blocks 0-1
        "up_proj": ([(256 * r, 256)], (0, H), 256, (H, 256)),                       # blocks 2-3
        "down_proj": ([(0, H)], (256 * r, 256), 0, (256, H)),
        "lm_head": ([(62080 * r, 62080)], (0, H), 0, (H, 62080)),                   # 485 blocks
        "fc": ([(768 * r, 768)], (0, 6144), 0, (6144, 768)),
    }
    return W[leaf]


@pytest.mark.parametrize("mod,k,n", MATRICES, ids=[m[0].replace(P, "L") for m in MATRICES])
@pytest.mark.parametrize("r", range(4))
def test_design_table_tp4(mod, k, n, r):
    p = _plan(mod, k, n, r, 4)
    cols, rows, dst, (lk, ln) = _design_tp4(mod.rsplit(".", 1)[-1], r)
    assert list(p.cols) == cols and p.rows == rows and p.dst_offset == dst
    assert (p.local_k, p.local_n) == (lk, ln)
    assert all(a % 128 == 0 and s % 128 == 0 for a, s in p.cols) and p.rows[0] % 128 == 0 and p.rows[1] % 128 == 0


def test_lm_head_blocks_and_vocab():
    assert 248320 % 64 == 0 and 248320 // 4 // 128 == 485
    assert [tp.vocab_shard(248320, r, 4) for r in range(4)] == [(62080 * r, 62080 * (r + 1), 62080) for r in range(4)]


# ------------------------------------------------------------------------------------- (b) vLLM loader oracle
class _FakeGroup:
    def __init__(self, rank, size):
        self.rank_in_group, self.world_size = rank, size


@pytest.fixture
def fake_tp():
    pytest.importorskip("vllm", reason="the vLLM loader oracle needs the vLLM image")
    import vllm.distributed.parallel_state as ps
    saved = ps._TP

    def set_(rank, size):
        ps._TP = _FakeGroup(rank, size)
    yield set_
    ps._TP = saved


def _vllm_layer(mod, k, n):
    """The unquantized vLLM 0.30 layer this checkpoint matrix loads into, with the other dim shrunk to 8
    (sharding depends only on the split dim). Returns (layer, shard_id, split side)."""
    import vllm.model_executor.layers.linear as L
    import vllm.model_executor.layers.vocab_parallel_embedding as V
    kind, sid = tp.classify(mod)
    small = 8
    if kind == "qkv":
        return L.QKVParallelLinear(small, 256, 64, 2, bias=False, quant_config=None, prefix="o.qkv_proj"), sid, "col"
    if kind in ("merged", "merged_tuple") and ".linear_attn." in mod:
        return L.MergedColumnParallelLinear(small, [2048, 2048, 8192, 8192], bias=False, quant_config=None,
                                            prefix="o.in_proj_qkvz"), sid, "col"
    if kind == "merged":
        return L.MergedColumnParallelLinear(small, [1024, 1024], bias=False, quant_config=None,
                                            prefix="o.gate_up_proj"), sid, "col"
    if kind == "row":
        return L.RowParallelLinear(k, small, bias=False, quant_config=None, prefix="o.down"), None, "row"
    if kind == "col":
        return L.ColumnParallelLinear(small, n, gather_output=True, bias=False, quant_config=None,
                                      prefix="o.fc"), None, "col"
    if kind == "lmhead":
        return V.ParallelLMHead(n, small, prefix="lm_head"), None, "lmhead"
    raise AssertionError(kind)


@pytest.mark.parametrize("mod,k,n", MATRICES, ids=[m[0].replace(P, "L") for m in MATRICES])
@pytest.mark.parametrize("tp_size", [1, 2, 4])
def test_plan_matches_vllm_loader(fake_tp, mod, k, n, tp_size):
    for r in range(tp_size):
        fake_tp(r, tp_size)
        layer, sid, side = _vllm_layer(mod, k, n)
        p = _plan(mod, k, n, r, tp_size)
        w = layer.weight
        w.data.fill_(-1)
        if side == "row":                     # vLLM layout (out, in): code each input column with its row id
            full = torch.arange(k, dtype=torch.float32).expand(w.shape[0], k).contiguous()
            layer.weight_loader(w, full)
            got = w.data[0].to(torch.int64)
            assert torch.equal(got, p.row_index()), (mod, r)
            assert p.cols == ((0, n),)
            continue
        full = torch.arange(n, dtype=torch.float32)[:, None].expand(n, w.shape[1]).contiguous()
        if side == "lmhead":
            layer.weight_loader(w, full)
            si = layer.shard_indices
            assert tp.vocab_shard(n, r, tp_size) == (si.org_vocab_start_index, si.org_vocab_end_index,
                                                     layer.num_embeddings_per_partition)
        elif sid is None:
            layer.weight_loader(w, full)
        else:
            layer.weight_loader(w, full, sid)
        got = w.data[:, 0].to(torch.int64)
        written = (got >= 0).nonzero().flatten()
        # vLLM wrote exactly the plan's local range, with exactly the plan's checkpoint columns
        assert int(written[0]) == p.dst_offset and len(written) == p.local_n, (mod, r, int(written[0]), len(written))
        assert torch.equal(written, torch.arange(p.dst_offset, p.dst_offset + p.local_n)), (mod, r)
        assert torch.equal(got[p.dst_offset:p.dst_offset + p.local_n], p.col_index()), (mod, r)
        assert p.rows == (0, k)


def test_vllm_kv_replication_tp4(fake_tp):
    import vllm.model_executor.layers.linear as L
    for r in range(4):
        fake_tp(r, 4)
        q = L.QKVParallelLinear(8, 256, 64, 2, bias=False, quant_config=None, prefix="o.qkv_proj")
        assert q.num_kv_head_replicas == 2 and q.num_kv_heads == 1
        p = tp.plan_qkv("k", H, 512, "k", total_num_heads=64, total_num_kv_heads=2, head_size=256,
                        tp_rank=r, tp_size=4)
        assert p.cols == ((256 * (r // q.num_kv_head_replicas), 256),)


# ------------------------------------------------------------------------------------- (c) synthetic tensors
def _coded(k, n, K=4):
    """trellis whose tile (i, j) carries i in word 0 and j in word 1; suh/svh carry their own index."""
    tr = torch.zeros((k // 16, n // 16, 16 * K), dtype=torch.int16)
    tr[:, :, 0] = torch.arange(k // 16, dtype=torch.int16)[:, None]
    tr[:, :, 1] = torch.arange(n // 16, dtype=torch.int16)[None, :]
    return {"trellis": tr, "suh": torch.randn(k).half(), "svh": torch.randn(n).half(),
            "mul1": torch.tensor(-2082680531, dtype=torch.int32)}


@pytest.mark.parametrize("mod,k,n", MATRICES, ids=[m[0].replace(P, "L") for m in MATRICES])
@pytest.mark.parametrize("tp_size", [1, 2, 4])
def test_narrow_selects_planned_tiles(mod, k, n, tp_size):
    t = _coded(k, n, K=1)      # K=1 keeps the lm_head-sized trellis small
    for r in range(tp_size):
        p = _plan(mod, k, n, r, tp_size)
        s = p.narrow_all(t)
        assert all(v.is_contiguous() for v in s.values())
        assert tuple(s["trellis"].shape) == (p.local_k // 16, p.local_n // 16, 16)
        ci, ri = p.col_index(), p.row_index()
        assert torch.equal(s["trellis"][0, :, 1].long(), ci[::16] // 16)
        assert torch.equal(s["trellis"][:, 0, 0].long(), ri[::16] // 16)
        assert torch.equal(s["svh"], t["svh"][ci])
        assert torch.equal(s["suh"], t["suh"][ri])
        assert int(s["mul1"]) == -2082680531


def test_clone_semantics():
    t = _coded(1024, 256)
    p = tp.plan_row("x", 1024, 256, 1, 4)
    view = p.narrow("trellis", t["trellis"])
    assert view.untyped_storage().data_ptr() == t["trellis"].untyped_storage().data_ptr()   # contiguous view
    own = p.narrow("trellis", t["trellis"], clone=True)
    assert own.untyped_storage().data_ptr() != t["trellis"].untyped_storage().data_ptr()
    assert torch.equal(view, own)
    pc = tp.plan_col("x", 1024, 256, 1, 2)
    col = pc.narrow("trellis", t["trellis"])
    assert col.is_contiguous() and col.untyped_storage().data_ptr() != t["trellis"].untyped_storage().data_ptr()


def test_replicated_is_whole():
    t = _coded(256, 384)
    p = tp.plan_replicated("x", 256, 384, 3, 4)
    for part, v in p.narrow_all(t).items():
        assert torch.equal(v, t[part])


def test_refusals():
    with pytest.raises(SplitError):
        tp.plan_col("x", 3072, 1024, 0, 16)
    with pytest.raises(SplitError):
        tp.plan_row("x", 1024, 3072, 1, 16)
    with pytest.raises(SplitError):
        tp.plan_lmhead("lm_head", 3072, 248320, org_start=31040, org_end=62080, num_per_partition=31040,
                       tp_rank=1, tp_size=8)
    with pytest.raises(SplitError):
        tp.plan_lmhead("lm_head", 3072, 1024, org_start=896, org_end=1000, num_per_partition=128,
                       tp_rank=7, tp_size=8)                           # padded partition
    with pytest.raises(SplitError):
        tp.plan_qkv("k", 3072, 512, "k", total_num_heads=64, total_num_kv_heads=2, head_size=256,
                    tp_rank=0, tp_size=3)
    with pytest.raises(ValueError):
        tp.plan_merged("z", 3072, 8192, [2048, 2048, 8192, 8192], 1, 0, 4)     # n does not match shard 1
    with pytest.raises(ValueError):
        tp.plan_merged_tuple("qkv", 3072, 12288, [2048, 2048, 8192, 8192], (0, 2), 0, 4)   # not consecutive
    with pytest.raises(ValueError):
        tp.plan_qkv("q", 3072, 16384, "x", total_num_heads=64, total_num_kv_heads=2, head_size=256,
                    tp_rank=0, tp_size=4)
    with pytest.raises(ValueError):
        tp.plan_col("x", 3072, 1024, 4, 4)                              # rank out of range
    p = tp.plan_col("x", 3072, 1024, 0, 4)
    with pytest.raises(ValueError):
        p.narrow("trellis", torch.zeros((192, 32, 64), dtype=torch.int16))   # already-sliced input
    with pytest.raises(ValueError):
        p.narrow("bogus", torch.zeros(3))
    with pytest.raises(ValueError):
        tp.classify("model.language_model.layers.0.mlp.gate")              # router: not an EXL3 module


def test_tp8_qkv_is_legal():
    p = tp.plan_qkv("q", 3072, 16384, "q", total_num_heads=64, total_num_kv_heads=2, head_size=256,
                    tp_rank=5, tp_size=8)
    assert p.cols == ((5 * 2048, 2048),)
    p = tp.plan_qkv("k", 3072, 512, "k", total_num_heads=64, total_num_kv_heads=2, head_size=256,
                    tp_rank=5, tp_size=8)
    assert p.cols == ((256, 256),) and p.dst_offset == 2048


def test_classify_all_kinds():
    expect = {
        P + "0.self_attn.q_proj": ("qkv", "q"), P + "0.self_attn.k_proj": ("qkv", "k"),
        P + "0.self_attn.v_proj": ("qkv", "v"), P + "0.self_attn.o_proj": ("row", None),
        P + "0.linear_attn.in_proj_qkv": ("merged_tuple", (0, 1, 2)), P + "0.linear_attn.in_proj_z": ("merged", 3),
        P + "0.linear_attn.out_proj": ("row", None), P + "0.mlp.shared_expert.gate_proj": ("merged", 0),
        P + "0.mlp.shared_expert.up_proj": ("merged", 1), P + "0.mlp.shared_expert.down_proj": ("row", None),
        P + "7.mlp.experts.255.up_proj": ("expert", (255, "up_proj")), "lm_head": ("lmhead", None),
        "mtp.fc": ("col", None), "mtp.layers.0.mlp.experts.3.down_proj": ("expert", (3, "down_proj")),
        "mtp.layers.0.self_attn.k_proj": ("qkv", "k"), "model.layers.2.mlp.gate_proj": ("merged", 0),
    }
    for m, e in expect.items():
        assert tp.classify(m) == e, m


def test_dispatch_matches_kind_functions():
    a = tp.plan("merged", "z", 3072, 8192, tp_rank=2, tp_size=4, shard_id=3, output_sizes=[2048, 2048, 8192, 8192])
    b = tp.plan_merged("z", 3072, 8192, [2048, 2048, 8192, 8192], 3, 2, 4)
    assert a == b
    a = tp.plan("merged", "qkv", 3072, 12288, tp_rank=1, tp_size=2, shard_id=(0, 1, 2),
                output_sizes=[2048, 2048, 8192, 8192])
    assert a.cols == ((1024, 1024), (3072, 1024), (8192, 4096)) and a.dst_offset == 0
    whole = tp.plan("merged", "gu", 3072, 2048, tp_rank=1, tp_size=2, shard_id=None, output_sizes=[1024, 1024])
    assert whole.cols == ((512, 512), (1536, 512))


def test_attn_output_gate_default_matches_vllm():
    """vLLM's Qwen3-Next/3.5 attention builds qkv_proj with attn_output_gate=True when the config omits the key."""
    cfg = {k: v for k, v in CFG.items() if k != "attn_output_gate"}
    for r in range(4):
        p = tp.qwen35_plan(P + "3.self_attn.q_proj", H, 16384, cfg, tp_rank=r, tp_size=4)
        assert p.cols == ((4096 * r, 4096),)
        p = tp.qwen35_plan(P + "3.self_attn.v_proj", H, 512, cfg, tp_rank=r, tp_size=4)
        assert p.dst_offset == 4096 + 256
    with pytest.raises(ValueError):          # an explicit False is honoured (q is then 32 heads, not 16384 wide)
        tp.qwen35_plan(P + "3.self_attn.q_proj", H, 16384, dict(CFG, attn_output_gate=False), tp_rank=0, tp_size=4)


@pytest.mark.parametrize("tp_size", [1, 2, 4])
def test_expert_tp_plan(tp_size):
    """MoE TP narrowing (DESIGN 2.4 step 3): gate/up columns [r*I/tp, +I/tp), down rows the same."""
    for r in range(tp_size):
        gu = tp.qwen35_plan(P + "0.mlp.experts.7.gate_proj", H, 1024, CFG, tp_rank=r, tp_size=tp_size)
        dn = tp.qwen35_plan(P + "0.mlp.experts.7.down_proj", 1024, H, CFG, tp_rank=r, tp_size=tp_size)
        I = 1024 // tp_size
        assert gu.cols == ((r * I, I),) and gu.rows == (0, H)
        assert dn.rows == (r * I, I) and dn.cols == ((0, H),)
    ep = tp.qwen35_plan(P + "0.mlp.experts.7.gate_proj", H, 1024, CFG, tp_rank=3, tp_size=4, moe_tp_rank=0,
                        moe_tp_size=1)
    assert ep.cols == ((0, 1024),)
