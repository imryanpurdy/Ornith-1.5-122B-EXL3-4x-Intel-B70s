"""Plugin pieces that need neither a checkpoint nor a vLLM model: the MoE id mapping and row chunking of
moe_vllm (fake layers, a recording kernel), blob sizes, module-name normalisation and storage lookup, the
EXL3_VLLM_PATCHES parser, and loadcheck's batch bookkeeping (when a batch closes, what it refuses). Also the branches
the model-level tests cannot reach on a CPU: Exl3MoEMethod.create_weights refusals and process_weights_after_loading
checks on a stand-in RoutedExperts, the variant-warmup launch list and the device-only steps of a batch close (with a
stand-in for the MoE host ops), routing refusals on bare layer instances, and the eager dump."""
from __future__ import annotations

import types

import pytest
import torch

pytest.importorskip("vllm", reason="the plugin imports vLLM")

from exl3xpu import loadcheck, moe_vllm, vllm_plugin  # noqa: E402


# ------------------------------------------------------------------------------------------------ moe_vllm
def test_blob_bytes():
    assert {I: moe_vllm.blob_bytes(3072, I, 4) for I in (1024, 512, 256)} == {1024: 4743168, 512: 2380800,
                                                                              256: 1199616}
    assert moe_vllm.blob_bytes(2560, 640, 3) == 1862400          # Flash-Next K=3 (moe_offload docstring)


def test_pointer_table_high_addresses():
    """USM device addresses above 2**63 go into the int64 pointer table as two's complement (moe_offload.s64), and
    moe_ref's interpreter reads them back unsigned."""
    from exl3xpu.moe_offload import s64
    a = (1 << 64) - (1 << 40)
    t = s64(a) + torch.arange(4, dtype=torch.int64) * 4743168
    assert t.dtype == torch.int64 and int(t[0]) < 0 and (int(t[3]) & ((1 << 64) - 1)) == a + 3 * 4743168
    assert s64(12345) == 12345


def test_layer_index():
    assert moe_vllm.layer_index("language_model.model.layers.17.mlp.experts") == "17"
    assert moe_vllm.layer_index("mtp.layers.0.mlp.experts") == "mtp"
    assert moe_vllm.layer_index("model.layers.3.mlp.experts") == "3"


def _emap(ep: int, r: int, E: int = 256) -> torch.Tensor:
    m = torch.full((E,), -1, dtype=torch.int32)
    m[r * E // ep:(r + 1) * E // ep] = torch.arange(E // ep, dtype=torch.int32)
    return m


@pytest.mark.parametrize("ep,r", [(2, 0), (2, 1), (4, 0), (4, 3)])
def test_map_ids_ep(ep, r):
    g = torch.Generator().manual_seed(ep * 10 + r)
    ids = torch.stack([torch.randperm(256, generator=g)[:8] for _ in range(32)]).to(torch.int64)
    ids[:3] = -1
    ids[5, 2:] = -1
    lay = types.SimpleNamespace(exl3_emap=_emap(ep, r))
    out = moe_vllm.map_ids(lay, ids)
    E = 256 // ep
    want = torch.where((ids >= r * E) & (ids < (r + 1) * E), ids - r * E, torch.full_like(ids, -1))
    assert out.dtype == torch.int32 and out.is_contiguous() and torch.equal(out, want.to(torch.int32))
    assert int(out.min()) >= -1 and int(out.max()) < E


def test_map_ids_tp_passthrough():
    ids = torch.tensor([[3, -1, 255, 0], [-1, -1, -1, -1]], dtype=torch.int32)
    out = moe_vllm.map_ids(types.SimpleNamespace(exl3_emap=None), ids)
    assert torch.equal(out, ids) and out.dtype == torch.int32


class _RecKernel:
    def __init__(self):
        self.calls = []

    def __call__(self, x, ids, w, ptrs, I, K, E):
        self.calls.append((x.shape[0], ids.dtype, w.dtype, x.is_contiguous(), ids.is_contiguous(), I, K, E))
        return x.float() * 2 + ids.float().sum(1, keepdim=True) + w.sum(1, keepdim=True)


def _fake_moe_layer(kernel, emap=None):
    return types.SimpleNamespace(exl3_kernel=kernel, exl3_ptrs=torch.arange(4, dtype=torch.int64), exl3_I=1024,
                                 exl3_K=4, exl3_E=64, exl3_emap=emap, exl3_layer_idx="0")


@pytest.mark.parametrize("M,chunks", [(0, [0]), (1, [1]), (1024, [1024]), (1025, [1024, 1]),
                                      (4096, [1024] * 4), (2500, [1024, 1024, 452])])
def test_moe_apply_chunks(M, chunks):
    k = _RecKernel()
    lay = _fake_moe_layer(k, _emap(4, 1))
    g = torch.Generator().manual_seed(M)
    x = torch.randn((M, 16), generator=g).to(torch.bfloat16)
    ids = torch.randint(-1, 256, (M, 8), generator=g, dtype=torch.int32)
    w = torch.rand((M, 8), generator=g).to(torch.bfloat16)          # not fp32: apply() converts
    y = moe_vllm.moe_apply(lay, x, w, ids)
    assert [c[0] for c in k.calls] == chunks
    assert all(c[1] == torch.int32 and c[2] == torch.float32 and c[3] and c[4] and c[5:] == (1024, 4, 64)
               for c in k.calls)
    full = _RecKernel()(x, moe_vllm.map_ids(lay, ids), w.float(), lay.exl3_ptrs, 1024, 4, 64)
    assert torch.equal(y, full)


def test_moe_apply_noncontiguous_x():
    k = _RecKernel()
    lay = _fake_moe_layer(k)
    x = torch.randn((8, 32)).to(torch.bfloat16)[:, ::2]
    ids = torch.zeros((8, 8), dtype=torch.int32)
    w = torch.zeros((8, 8))
    y = moe_vllm.moe_apply(lay, x, w, ids)
    assert k.calls[0][3] and torch.equal(y, x.float().contiguous() * 2)


# ----------------------------------------------------------------------------------------------- vllm_plugin
def test_norm_key():
    nk = vllm_plugin._norm_key
    assert nk("model.language_model.layers.3.mlp.gate_proj") == "layers.3.mlp.gate_proj"
    assert nk("language_model.model.layers.3.self_attn.qkv_proj") == "layers.3.self_attn.qkv_proj"
    assert nk("mtp.layers.0.mlp.experts") == "mtp.layers.0.mlp.experts"
    assert nk("model.mtp.fc") == "mtp.fc"
    assert nk("language_model.lm_head") == "lm_head" and nk("lm_head") == "lm_head"


def _cfg(storage):
    c = vllm_plugin.Exl3Config(bits=4, head_bits=6, codebook="mul1", mtp_bits=4)
    c.storage = dict(storage)
    return c


def test_bits_for_and_expert_bits():
    st = {"layers.3.self_attn.q_proj": 5, "layers.3.self_attn.k_proj": 5, "layers.3.self_attn.v_proj": 5,
          "layers.0.linear_attn.in_proj_qkv": 5, "layers.0.linear_attn.in_proj_z": 5, "lm_head": 6,
          "mtp.fc": 4}
    st.update({f"layers.3.mlp.experts.{e}.{p}": 4 for e in range(8) for p in ("gate_proj", "up_proj", "down_proj")})
    c = _cfg(st)
    assert c._bits_for("language_model.model.layers.3.self_attn.qkv_proj") == 5
    assert c._bits_for("language_model.model.layers.0.linear_attn.in_proj_qkvz") == 5
    assert c._bits_for("language_model.model.layers.0.linear_attn.in_proj_ba") is None
    assert c._bits_for("language_model.lm_head") == 6 and c._bits_for("mtp.fc") == 4
    assert c._bits_for("visual.blocks.0.attn.qkv") is None
    assert c.expert_bits("language_model.model.layers.3.mlp.experts", 8) == 4
    with pytest.raises(ValueError, match="not all EXL3"):
        c.expert_bits("language_model.model.layers.3.mlp.experts", 9)
    c.storage["layers.3.self_attn.k_proj"] = 4
    with pytest.raises(ValueError, match="mixes EXL3 bitrates"):
        c._bits_for("language_model.model.layers.3.self_attn.qkv_proj")
    del c.storage["layers.3.self_attn.k_proj"]
    with pytest.raises(ValueError, match="mixes EXL3 bitrates"):
        c._bits_for("language_model.model.layers.3.self_attn.qkv_proj")
    with pytest.raises(ValueError, match="no EXL3 module map"):
        _cfg({})._bits_for("language_model.lm_head")


@pytest.mark.parametrize("raw,want", [("", []), ("0", []), ("none", []), ("fp8kv", ["fp8kv"]),
                                      ("patch_fp8kv_prefill, fp8kv", ["fp8kv"]),
                                      ("1", ["1"]), ("patch_align_sync,foo", ["align_sync", "foo"])])
def test_patch_list(monkeypatch, raw, want):
    monkeypatch.setenv("EXL3_VLLM_PATCHES", raw)
    assert vllm_plugin._patch_list() == want


def test_refused_patches_are_logged_not_applied(monkeypatch):
    monkeypatch.setenv("EXL3_VLLM_PATCHES", "align_sync,gdn_mask_index,xpu_block_size,1")
    called = []
    import exl3xpu.vllm_patches as vp
    for f in ("patch_align_sync", "patch_gdn_mask_index", "patch_xpu_block_size", "patch_fp8kv_prefill", "apply_all"):
        monkeypatch.setattr(vp, f, lambda *a, _f=f: called.append(_f) or True)
    res = vllm_plugin.apply_patches()
    assert res == {"align_sync": "refused", "gdn_mask_index": "refused", "xpu_block_size": "refused",
                   "1": "refused"} and called == []


def test_unknown_patch_aborts(monkeypatch):
    monkeypatch.setenv("EXL3_VLLM_PATCHES", "fp8kv,bogus")
    with pytest.raises(RuntimeError, match="bogus"):
        vllm_plugin.apply_patches()


# ------------------------------------------------------------------------------------------------ loadcheck
class _FakeDense:
    def __init__(self, prefix, rank=0, size=1):
        self.exl3_prefix, self.exl3_kind = prefix, "row"
        self.exl3_tp_rank, self.exl3_tp_size = rank, size
        self.trellis = torch.empty(0)
        self.exl3_config = types.SimpleNamespace(index_path=None)


@pytest.fixture
def lc(monkeypatch):
    loadcheck.reset_for_tests()
    monkeypatch.setattr(loadcheck, "_tp_group", lambda: (0, 1))
    monkeypatch.setattr(loadcheck, "expected_counts", lambda b, role, par: {"dense": b.counts["dense"],
                                                                            "experts": b.counts["experts"]})
    yield loadcheck
    loadcheck.reset_for_tests()


def test_batch_closes_after_last_layer(lc):
    a, b = _FakeDense("language_model.model.layers.0.x"), _FakeDense("language_model.lm_head")
    lc.add_layer(a, "dense")
    lc.add_layer(b, "dense")
    a.exl3_batch.counts["dense"] = 8
    lc.layer_done(a)
    assert lc.CLOSED == []
    lc.layer_done(b)
    assert [c["role"] for c in lc.CLOSED] == ["target"] and "dense=8 experts=0 model=target" in lc.CLOSED[0]["line"]
    assert not hasattr(a, "exl3_batch")                  # references dropped at close
    d = _FakeDense("mtp.fc")
    lc.add_layer(d, "dense")
    lc.layer_done(d)
    assert [c["role"] for c in lc.CLOSED] == ["target", "drafter"]


def test_batch_refusals(lc, monkeypatch):
    a, b = _FakeDense("p.a"), _FakeDense("p.b")
    lc.add_layer(a, "dense")
    lc.add_layer(b, "dense")
    lc.layer_done(a)
    with pytest.raises(lc.LoadCheckError, match="processed twice|ran twice"):
        lc.layer_done(a)
    with pytest.raises(lc.LoadCheckError, match="previous model"):
        lc.add_layer(_FakeDense("p.c"), "dense")
    loadcheck.reset_for_tests()
    c = _FakeDense("p.c", rank=1, size=4)                  # split differently from the TP group (0 of 1)
    lc.add_layer(c, "dense")
    with pytest.raises(lc.LoadCheckError, match="split differently"):
        lc.layer_done(c)
    loadcheck.reset_for_tests()
    monkeypatch.setattr(loadcheck, "expected_counts", lambda b, role, par: {"dense": 1, "experts": 0})
    e = _FakeDense("p.e")
    lc.add_layer(e, "dense")
    with pytest.raises(lc.LoadCheckError, match="differ from the checkpoint index"):
        lc.layer_done(e)
    assert "expected=1/0" in lc.CLOSED[-1]["line"]


def test_drafter_lm_head_sharing_guard(monkeypatch):
    from vllm.model_executor.models import qwen3_5_mtp as m
    loadcheck._check_drafter_shares_lm_head()            # vLLM 0.30: has_own_lm_head is False -> no error
    monkeypatch.setattr(m.Qwen3_5MoeMTP, "has_own_lm_head", True, raising=False)
    with pytest.raises(loadcheck.LoadCheckError, match="has_own_lm_head"):
        loadcheck._check_drafter_shares_lm_head()


def test_scratch_numel():
    lay = types.SimpleNamespace(trellis=torch.empty((192, 1, 1)), exl3_bounds=[0, 62080])   # k=3072, lm_head TP4
    lay2 = types.SimpleNamespace(trellis=torch.empty((128, 1, 1)), exl3_bounds=[0, 3072])   # o_proj TP4
    assert loadcheck.scratch_numel([lay, lay2]) == 3072 * 16384     # DESIGN 2.3: 96 MiB at TP1/2/4


# ----------------------------------------------------------------------------------------------- eager dump
def test_dump_skips_all_padding_batches(tmp_path, monkeypatch):
    """vLLM's profile and capture-warmup forwards are all padding: they must not use up the dump slots."""
    monkeypatch.setattr(moe_vllm, "_DUMP_LAYER", "0")
    monkeypatch.setattr(moe_vllm, "_DUMP_DIR", str(tmp_path))
    lay = _fake_moe_layer(_RecKernel())
    lay.__dict__.update(exl3_dump_step=0, exl3_use_ep=False, exl3_ep_size=1, exl3_ep_rank=0, exl3_moe_tp_size=1,
                        exl3_moe_tp_rank=0, exl3_prefix="language_model.model.layers.0.mlp.experts", exl3_H=16)
    x = torch.randn((6, 16)).to(torch.bfloat16)
    ids = torch.full((6, 8), -1, dtype=torch.int32)
    w = torch.zeros((6, 8))
    for _ in range(3):
        moe_vllm.moe_apply(lay, x, w, ids)
    assert sorted(p.name for p in tmp_path.iterdir()) == [] and lay.exl3_dump_step == 0
    ids[2, 0], w[2, 0] = 3, 1.0
    moe_vllm.moe_apply(lay, x, w, ids)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["L0_r0_s0.pt"]
    d = torch.load(tmp_path / "L0_r0_s0.pt")
    assert torch.equal(d["topk_ids"], ids) and torch.equal(d["x"], x)


# ---------------------------------------------------------------------------------------- EXL3_VLLM_PATCHES
def test_patch_list_default_when_unset(monkeypatch):
    """The lane leaves EXL3_VLLM_PATCHES unset on BF16 runs: unset must mean no patch (DESIGN 2.6 item 2)."""
    monkeypatch.delenv("EXL3_VLLM_PATCHES", raising=False)
    assert vllm_plugin._patch_list() == [] and vllm_plugin.apply_patches() == {}


def test_requested_patch_not_applied_aborts(monkeypatch):
    """A requested fp8kv that does not apply must fail register(), not serve the stock FP8 prefill path."""
    import exl3xpu.vllm_patches as vp
    monkeypatch.setenv("EXL3_VLLM_PATCHES", "fp8kv")
    try:
        from vllm.v1.attention.backends import flash_attn as fa
        monkeypatch.setattr(fa.FlashAttentionImpl, "_exl3_fp8kv_patched", False, raising=False)
    except Exception:          # noqa: BLE001  (backend not importable on this platform: apply_patches sees "not patched")
        pass
    monkeypatch.setattr(vp, "patch_fp8kv_prefill", lambda: False)
    with pytest.raises(RuntimeError, match="did not apply"):
        vllm_plugin.apply_patches()
    monkeypatch.setattr(vp, "patch_fp8kv_prefill", lambda: True)
    assert vllm_plugin.apply_patches() == {"fp8kv": "applied"}


# ---------------------------------------------------------- branches the model-level CPU tests cannot reach
def test_routing_refusals_on_layer_types():
    """get_quant_method on bare instances of vLLM's layer classes (isinstance is all it looks at)."""
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
    from vllm.model_executor.layers.linear import ColumnParallelLinear, UnquantizedLinearMethod
    from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
    c = _cfg({"layers.0.self_attn.q_proj": 5, "layers.0.self_attn.k_proj": 5, "layers.0.self_attn.v_proj": 5})
    head, lin, rexp = (object.__new__(cls) for cls in (ParallelLMHead, ColumnParallelLinear, RoutedExperts))
    with pytest.raises(ValueError, match="no EXL3 matrices"):
        c.get_quant_method(head, "mtp.lm_head")                       # an lm_head without EXL3 storage
    with pytest.raises(ValueError, match="lm_head in the vision tower"):
        c.get_quant_method(head, "visual.head")
    with pytest.raises(ValueError, match="MoE layer in the vision tower"):
        c.get_quant_method(rexp, "visual.blocks.0.mlp.experts")
    assert isinstance(c.get_quant_method(lin, "language_model.model.layers.0.linear_attn.in_proj_ba"),
                      UnquantizedLinearMethod)
    assert isinstance(c.get_quant_method(lin, "visual.blocks.0.mlp.linear_fc1"), UnquantizedLinearMethod)
    with pytest.raises(ValueError, match="no EXL3 matrices"):
        c.get_quant_method(lin, "language_model.model.layers.0.mlp.foo_proj")
    assert isinstance(c.get_quant_method(lin, "language_model.model.layers.0.self_attn.qkv_proj"),
                      vllm_plugin.Exl3LinearMethod)


# Exl3MoEMethod on a stand-in RoutedExperts (create_weights validation, process_weights_after_loading)
from vllm.model_executor.layers.fused_moe.activation import MoEActivation  # noqa: E402

_H, _I, _E = 256, 256, 4


class _FakeRoutedExperts(torch.nn.Module):
    def __init__(self, E=_E, H=_H, I=_I):
        super().__init__()
        pc = types.SimpleNamespace(dp_size=1, pcp_size=1, use_all2all_kernels=False, is_sequence_parallel=False,
                                   enable_eplb=False, tp_size=1, tp_rank=0, use_ep=False, ep_size=1, ep_rank=0)
        self.moe_config = types.SimpleNamespace(moe_parallel_config=pc, is_act_and_mul=True,
                                                activation=MoEActivation.SILU, has_bias=False, hidden_dim=H,
                                                intermediate_size=I, experts_per_token=2)
        self.apply_router_weight_on_input = False
        self.routed_scaling_factor = 1.0
        self.expert_map_manager = types.SimpleNamespace(num_fused_shared_experts=0)
        self.local_num_experts = self.global_num_experts = E
        self.layer_name = "language_model.model.layers.0.mlp.experts"


def _moe_method(layer):
    return moe_vllm.Exl3MoEMethod(_cfg({}), layer.moe_config, layer.layer_name, 4)


def _create(layer, **over):
    kw = dict(num_experts=_E, hidden_size=_H, intermediate_size_per_partition=_I, params_dtype=torch.bfloat16)
    kw.update(over)
    meth = _moe_method(layer)
    meth.create_weights(layer, **kw)
    return meth


def _set(path, value):
    """Mutator: set the attribute at a dotted path below the layer."""
    def f(layer):
        *head, last = path.split(".")
        tgt = layer
        for p in head:
            tgt = getattr(tgt, p)
        setattr(tgt, last, value)
    return f


_BAD_MOE = {
    "gelu": (_set("moe_config.activation", MoEActivation.GELU), {}, "activation"),
    "not_gated": (_set("moe_config.is_act_and_mul", False), {}, "activation"),
    "router_weight_on_input": (_set("apply_router_weight_on_input", True), {}, "apply_router_weight_on_input"),
    "routed_scaling": (_set("routed_scaling_factor", 2.5), {}, "routed_scaling_factor 2.5"),
    "bias": (_set("moe_config.has_bias", True), {}, "expert bias"),
    "fused_shared": (_set("expert_map_manager.num_fused_shared_experts", 1), {}, "fused shared experts"),
    "dp": (_set("moe_config.moe_parallel_config.dp_size", 2), {}, "dp 2"),
    "pcp": (_set("moe_config.moe_parallel_config.pcp_size", 2), {}, "pcp 2"),
    "all2all": (_set("moe_config.moe_parallel_config.use_all2all_kernels", True), {}, "all2all True"),
    "seq_parallel": (_set("moe_config.moe_parallel_config.is_sequence_parallel", True), {}, "sequence parallel True"),
    "eplb": (_set("moe_config.moe_parallel_config.enable_eplb", True), {}, "EPLB"),
    "hidden_not_128": (_set("moe_config.hidden_dim", 200), {"hidden_size": 200}, "multiples of 128"),
    "hidden_mismatch": (_set("moe_config.hidden_dim", 512), {}, "sizes: hidden 256 vs 512"),
    "local_experts_mismatch": (lambda l: None, {"num_experts": 3}, "local experts 3 vs 4"),
    "intermediate_mismatch": (lambda l: None, {"intermediate_size_per_partition": 128}, "intermediate 256"),
    "fp32_activations": (lambda l: None, {"params_dtype": torch.float32}, "activation dtype"),
}


@pytest.mark.parametrize("case", sorted(_BAD_MOE))
def test_moe_create_weights_refusals(lc, case):
    mutate, over, needle = _BAD_MOE[case]
    layer = _FakeRoutedExperts()
    mutate(layer)
    with pytest.raises(NotImplementedError, match="unsupported MoE configuration") as ei:
        _create(layer, **over)
    assert needle in str(ei.value), str(ei.value)
    assert not hasattr(layer, "exl3_arena")


class _FakeMoeOps:
    """moe_offload.ops() stand-in: blob_bytes / moe_info / moe_plan / overrides / moe_forward, recording calls."""

    def __init__(self, blob=None, miss_from=None):
        self.blob, self.miss_from, self.calls = blob, miss_from, []

    def blob_bytes(self, H, I, K):
        return moe_vllm.blob_bytes(H, I, K) if self.blob is None else self.blob

    def moe_info(self, dev):
        self.calls.append(("info", dev))
        return [4, 0]

    def moe_set_mr(self, v):
        self.calls.append(("mr", v))

    def moe_set_splits(self, a, b):
        self.calls.append(("splits", a, b))

    def moe_plan(self, H, I, K, M):
        self.calls.append(("plan", M))
        if self.miss_from is not None and M >= self.miss_from:
            return [0, 0, 0, 0]
        return [1, 12, 8, 8] if M <= 4 else ([2, 6, 8, 8] if M <= 64 else [4, 6, 8, 8])

    def moe_forward(self, x, ids, w, ptrs, I, K, E):
        self.calls.append(("forward", x.shape[0], ids.dtype, tuple(ids.shape), int(ids.min()), int(ids.max()),
                           w.dtype, I, K, E))
        return torch.zeros_like(x)


@pytest.fixture
def fake_moe_ops(monkeypatch):
    from exl3xpu import moe_offload
    holder = {"ops": _FakeMoeOps()}
    monkeypatch.setattr(moe_offload, "ops", lambda: holder["ops"])
    return holder


def _loaded_layer():
    layer = _FakeRoutedExperts()
    meth = _create(layer)
    layer.exl3_packed = [True] * _E            # as if every local expert had been packed by the loader
    return layer, meth


def test_moe_process_weights_success(lc, fake_moe_ops):
    layer, meth = _loaded_layer()
    assert layer.exl3_arena.shape == (_E, moe_vllm.blob_bytes(_H, _I, 4)) and hasattr(layer, "w13_trellis")
    meth.process_weights_after_loading(layer)
    X = fake_moe_ops["ops"]
    want = layer.exl3_arena.data_ptr() + torch.arange(_E, dtype=torch.int64) * layer.exl3_blob
    assert torch.equal(layer.exl3_ptrs, want) and layer.exl3_emap is None
    assert layer.exl3_kernel == X.moe_forward
    assert not any(hasattr(layer, f"{w}_{p}") for w in ("w13", "w2") for p in moe_vllm.PARTS)
    kinds = [c[0] for c in X.calls]
    assert kinds[0] == "info" and kinds.index("info") < kinds.index("plan")          # moe_info before any plan
    assert ("mr", 0) in X.calls[:kinds.index("plan")] and ("splits", 0, 0) in X.calls[:kinds.index("plan")]
    assert sorted(m for c in X.calls if c[0] == "plan" for m in [c[1]]) == list(range(1, 1025))
    assert layer.exl3_plans == {(1, 12, 8, 8): 1, (2, 6, 8, 8): 5, (4, 6, 8, 8): 65}
    assert [c["role"] for c in lc.CLOSED] == ["target"] and "experts=0" in lc.CLOSED[0]["line"]


def test_moe_process_weights_planner_miss_raises(lc, fake_moe_ops):
    layer, meth = _loaded_layer()
    fake_moe_ops["ops"] = _FakeMoeOps(miss_from=600)
    with pytest.raises(ValueError, match=r"no MoE kernel for \(H=256, I=256, K=4\) from M=600"):
        meth.process_weights_after_loading(layer)


def test_moe_process_weights_blob_mismatch_raises(lc, fake_moe_ops):
    layer, meth = _loaded_layer()
    fake_moe_ops["ops"] = _FakeMoeOps(blob=moe_vllm.blob_bytes(_H, _I, 4) + 64)
    with pytest.raises(ValueError, match="blob layout mismatch"):
        meth.process_weights_after_loading(layer)


def test_moe_process_weights_missing_expert_raises(lc, fake_moe_ops):
    layer, meth = _loaded_layer()
    layer.exl3_packed[2] = False
    with pytest.raises(ValueError, match=r"3 of 4 local experts loaded; missing local slots \[2\]"):
        meth.process_weights_after_loading(layer)


class _FakeDenseLayer:
    def __init__(self, k, n, K, bounds=None):
        self.trellis = torch.empty((k // 16, n // 16, 16 * K), dtype=torch.int16)
        self.svh = torch.empty(n, dtype=torch.float16)
        self.exl3_K, self.exl3_cb, self.exl3_dtype, self.exl3_cpp = K, 2, torch.bfloat16, True
        self.exl3_bounds = bounds or [0, n]
        self.exl3_prefix, self.exl3_tp_rank = "language_model.model.layers.0.x", 0
        self.ms: list[int] = []
        layer = self
        self.quant_method = types.SimpleNamespace(apply=lambda l, x: layer.ms.append(x.shape[0]) or x)


def test_warmup_launch_list(fake_moe_ops, monkeypatch):
    """loadcheck.warmup (runs only on a device at boot): every distinct dense shape once at each M of DENSE_WARM_M,
    every distinct MoE plan once at its smallest M, with in-range int32 ids and fp32 weights."""
    loadcheck.reset_for_tests()
    monkeypatch.setattr(torch.xpu, "synchronize", lambda *a, **k: None)
    d1, d2, d3 = _FakeDenseLayer(256, 512, 5), _FakeDenseLayer(256, 512, 5), _FakeDenseLayer(256, 512, 6)
    moe = types.SimpleNamespace(exl3_H=_H, exl3_I=_I, exl3_K=4, exl3_E=_E, exl3_topk=8, exl3_dtype=torch.bfloat16,
                                exl3_ptrs=torch.zeros(_E, dtype=torch.int64), exl3_prefix="m", exl3_tp_rank=0)
    moe2 = types.SimpleNamespace(**vars(moe))
    b = loadcheck.Batch(1)
    b.dense, b.moe = [d1, d2, d3], [moe, moe2]
    res = loadcheck.warmup(b, torch.device("cpu"))
    assert d1.ms == list(loadcheck.DENSE_WARM_M) and d2.ms == [] and d3.ms == list(loadcheck.DENSE_WARM_M)
    fw = [c for c in fake_moe_ops["ops"].calls if c[0] == "forward"]
    assert [c[1] for c in fw] == [1, 5, 65]
    assert all(c[2] == torch.int32 and c[3] == (c[1], 8) and c[4] >= 0 and c[5] < _E and c[6] == torch.float32
               and c[7:] == (_I, 4, _E) for c in fw)
    assert (res["dense_launches"], res["moe_launches"], res["keys"]) == (26, 3, 3)
    loadcheck.reset_for_tests()
    fake_moe_ops["ops"] = _FakeMoeOps(miss_from=900)
    b = loadcheck.Batch(2)
    b.moe = [moe]
    with pytest.raises(loadcheck.LoadCheckError, match="no MoE kernel plan at M=900"):
        loadcheck.warmup(b, torch.device("cpu"))
    loadcheck.reset_for_tests()


def test_close_reserves_scratch_then_warms_on_xpu_only(lc, monkeypatch):
    calls = []
    monkeypatch.setattr(loadcheck, "reserve_scratch", lambda b, dev: calls.append("scratch"))
    monkeypatch.setattr(loadcheck, "warmup", lambda b, dev: calls.append("warmup") or {"ok": 1})
    a = _FakeDense("language_model.model.layers.0.x")
    lc.add_layer(a, "dense")
    lc.layer_done(a)                                          # CPU tensors: neither runs
    assert calls == []
    monkeypatch.setattr(loadcheck, "_device", lambda b: types.SimpleNamespace(type="xpu"))
    c = _FakeDense("language_model.lm_head")
    lc.add_layer(c, "dense")
    lc.layer_done(c)
    assert calls == ["scratch", "warmup"] and lc.CLOSED[-1]["warmup"] == {"ok": 1}
