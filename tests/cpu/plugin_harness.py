"""Builds vLLM 0.30's Qwen3.5-MoE model (and its MTP drafter) on the CPU with this plugin, for the plugin's CPU
tests (tests/cpu/test_plugin_*.py run it in a subprocess and assert on its JSON output).

It runs in its own interpreter because vLLM binds `current_platform` at import time: the serving image has no XPU
device here, so vLLM would pick UnspecifiedPlatform. Before any other vLLM import this script installs the XPU
platform with current_device() answering "cpu" (the only device call made while a model is built), registers the
plugin through its real entry point (exl3xpu.vllm_plugin.register), and fakes vLLM's process groups (TP / EP rank
and size, no collectives are ever issued). Each model build gets a fresh VllmConfig from EngineArgs, exactly as
`vllm serve` would parse the lane's flags, and is constructed through vLLM's initialize_model() under the
loader's dtype / device contexts (meta for the shape-only commands, cpu for the real loads).

  python plugin_harness.py routing   --model DIR --cases JSON [--json-out F]   module -> quant method, shapes
  python plugin_harness.py load-meta --model DIR --cases JSON [--json-out F]   every checkpoint tensor (meta, real
                                      codebook scalars) through model.load_weights + process_weights_after_loading
  python plugin_harness.py load-real --model DIR --cases JSON [--json-out F]   real tensors through vLLM's
                                      DefaultModelLoader; per-rank params compared with tp.py slices of the checkpoint
  python plugin_harness.py moe       --model DIR --cases JSON [--json-out F]   Exl3MoEMethod.apply on real layers
                                      with a pure-torch kernel (see moe_ref.py)

A case is {"tp": 1|2|4, "ep": bool, "ranks": [..] (default all), "mtp": k|0, "lm_only": bool, "mnbt": int}.
Environment: EXL3_LIB / EXL3_MOE_LIB = the built _C.so / _moe.so (host ops: exl3_supported, moe_plan, moe_info).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
for p in (_HERE, _ROOT / "core" / "src", _ROOT / "xpu"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.pop("EXL3_VLLM_PATCHES", None)

import torch  # noqa: E402

import vllm.platforms as _P  # noqa: E402
from vllm.platforms.xpu import XPUPlatform  # noqa: E402

XPUPlatform.current_device = classmethod(lambda cls: torch.device("cpu"))
_P._current_platform = XPUPlatform()

from exl3xpu import loadcheck, vllm_plugin  # noqa: E402

vllm_plugin.register()

import vllm.distributed.parallel_state as ps  # noqa: E402
from vllm.config import set_current_vllm_config  # noqa: E402
from vllm.engine.arg_utils import EngineArgs  # noqa: E402
from vllm.model_executor.model_loader.utils import initialize_model, process_weights_after_loading  # noqa: E402
from vllm.utils.torch_utils import set_default_torch_dtype  # noqa: E402


# ---------------------------------------------------------------------------------------------- fake groups
class FakePG:
    def __init__(self, rank, size):
        self._r, self._s = rank, size

    def size(self):
        return self._s

    def rank(self):
        return self._r


class FakeGroup:
    def __init__(self, rank, size):
        self.rank_in_group, self.world_size = rank, size
        self.ranks = list(range(size))
        self.rank = self.local_rank = rank
        self.device_group = self.cpu_group = FakePG(rank, size)
        self.is_first_rank, self.is_last_rank = rank == 0, rank == size - 1
        self.unique_name = "exl3-fake"


def set_groups(tp: int, rank: int, ep: bool) -> None:
    one = FakeGroup(0, 1)
    for name in ("_PP", "_DP", "_PCP", "_DCP", "_EPLB", "_ETP", "_INNER_DP_WORLD"):
        setattr(ps, name, one)
    ps._TP = FakeGroup(rank, tp)
    ps._WORLD = FakeGroup(rank, tp)
    ps._EP = FakeGroup(rank, tp) if ep else one


def make_config(model: str, case: dict):
    kw = dict(model=model, quantization="exl3", dtype="bfloat16", tensor_parallel_size=case["tp"],
              enable_expert_parallel=bool(case.get("ep")), max_model_len=4096,
              max_num_batched_tokens=case.get("mnbt", 4096), enforce_eager=True, skip_tokenizer_init=True,
              language_model_only=bool(case.get("lm_only", True)), max_num_seqs=16)
    if case.get("mtp"):
        kw["speculative_config"] = {"method": "mtp", "num_speculative_tokens": int(case["mtp"])}
    return EngineArgs(**kw).create_engine_config()


_ARENA_META = False
_orig_moe_create = None


def arena_on_meta(on: bool) -> None:
    """Shape-only runs: replace each MoE layer's device arena by a meta tensor right after create_weights (the
    122B arenas are 58 GiB at TP1). Everything else stays a real (lazily allocated) CPU tensor."""
    global _ARENA_META, _orig_moe_create
    from exl3xpu import moe_vllm
    if _orig_moe_create is None:
        _orig_moe_create = moe_vllm.Exl3MoEMethod.create_weights

        def create_weights(self, layer, *a, **kw):
            _orig_moe_create(self, layer, *a, **kw)
            if _ARENA_META:
                layer.exl3_arena = torch.empty(layer.exl3_arena.shape, dtype=torch.uint8, device="meta")
        moe_vllm.Exl3MoEMethod.create_weights = create_weights
    _ARENA_META = on


def build(vc, drafter: bool = False):
    """initialize_model under the loader's contexts (base_loader.BaseModelLoader.load_model), on the CPU: vLLM's
    expert-map logging cannot run on meta tensors, and torch.empty on the CPU is lazily allocated."""
    mc = vc.speculative_config.draft_model_config if drafter else vc.model_config
    with set_default_torch_dtype(mc.dtype), torch.device("cpu"):
        return initialize_model(vllm_config=vc, model_config=mc), mc


def ranks_of(case):
    return case.get("ranks") or list(range(case["tp"]))


_configs: dict = {}


def config_for(model: str, case: dict, rank: int):
    """One VllmConfig per (model, case) (EngineArgs parsing costs ~20 s per architecture the first time); each
    rank's build starts from an empty static forward context, as in a fresh worker process."""
    key = (model, json.dumps({k: v for k, v in case.items() if k != "ranks"}, sort_keys=True))
    if key not in _configs:
        _configs[key] = make_config(model, case)
    vc = _configs[key]
    vc.compilation_config.static_forward_context.clear()
    vc.compilation_config.static_all_moe_layers.clear()
    loadcheck.reset_for_tests()
    set_groups(case["tp"], rank, bool(case.get("ep")))
    return vc


# --------------------------------------------------------------------------------------------- inspection
def describe(model) -> list[dict]:
    from vllm.model_executor.layers.linear import LinearBase
    from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
    rows = []
    for name, m in model.named_modules():
        qm = getattr(m, "quant_method", None)
        if not isinstance(m, (LinearBase, VocabParallelEmbedding, RoutedExperts)):
            continue
        r = {"name": name, "cls": type(m).__name__, "method": type(qm).__name__ if qm is not None else None,
             "prefix": getattr(m, "prefix", getattr(m, "layer_name", name))}
        if hasattr(m, "exl3_K"):
            r["K"] = m.exl3_K
        if hasattr(m, "exl3_kind"):
            r.update(kind=m.exl3_kind, tp=(m.exl3_tp_rank, m.exl3_tp_size), sizes=m.exl3_sizes,
                     trellis=list(m.trellis.shape), suh=list(m.suh.shape), svh=list(m.svh.shape),
                     k_full=m.exl3_in_full)
        if hasattr(m, "exl3_arena"):
            r.update(E=m.exl3_E, H=m.exl3_H, I=m.exl3_I, blob=m.exl3_blob, arena=list(m.exl3_arena.shape),
                     use_ep=m.exl3_use_ep, ep=(m.exl3_ep_rank, m.exl3_ep_size),
                     moe_tp=(m.exl3_moe_tp_rank, m.exl3_moe_tp_size),
                     local_ids=[g for g, l in enumerate(m.exl3_cpu_map) if l >= 0])
        rows.append(r)
    return rows


def cmd_routing(model: str, cases: list[dict]) -> list[dict]:
    out = []
    for case in cases:
        for rank in ranks_of(case):
            t0 = time.time()
            res = {"case": case, "rank": rank}
            try:
                vc = config_for(model, case, rank)
                arena_on_meta(True)
                with set_current_vllm_config(vc):
                    target, _ = build(vc)
                    res["target"] = describe(target)
                    if case.get("mtp"):
                        draft, dmc = build(vc, drafter=True)
                        res["drafter"] = describe(draft)
                        res["drafter_arch"] = dmc.architecture
                res["ok"] = True
            except Exception as e:     # noqa: BLE001
                res.update(ok=False, error=f"{type(e).__name__}: {e}", tb=traceback.format_exc()[-4000:])
            res["seconds"] = round(time.time() - t0, 2)
            out.append(res)
            print(f"routing tp={case['tp']} ep={case.get('ep')} mtp={case.get('mtp')} rank={rank} "
                  f"ok={res['ok']} {res['seconds']}s", file=sys.stderr, flush=True)
    return out



# --------------------------------------------------------------------------------------------- load helpers
def _is_expert(name: str) -> bool:
    return ".mlp.experts." in name


def meta_weights(ckpt, keep) -> "Iterator[tuple[str, torch.Tensor]]":
    """Every checkpoint tensor, in vLLM's order (shard files sorted, names sorted inside a file), with its real
    dtype and shape: routed-expert components as meta tensors (their arenas are meta), codebook markers read from
    the checkpoint (the loaders check the value), names `keep` rejects as meta, everything else zeros."""
    from st_range import _DT
    by_file: dict = {}
    for n, f in ckpt.weight_map.items():
        by_file.setdefault(f, []).append(n)
    for f in sorted(by_file):
        for n in sorted(by_file[f]):
            m = ckpt.meta(n)
            dt, shape = _DT[m["dtype"]], m["shape"]
            if n.endswith((".mul1", ".mcg")):
                t = ckpt.get(n).clone()
            elif not keep(n) or _is_expert(n):
                t = torch.empty(shape, dtype=dt, device="meta")
            else:
                t = torch.zeros(shape, dtype=dt)
            yield n, t


def _load(model, mc, weights, vc):
    """model.load_weights + vLLM's process_weights_after_loading, as DefaultModelLoader.load_model does."""
    with set_current_vllm_config(vc), set_default_torch_dtype(mc.dtype), torch.device("cpu"):
        loaded = model.load_weights(weights)
        process_weights_after_loading(model, mc, torch.device("cpu"))
    return loaded


def _run_case(model_dir, case, rank, *, real: bool, verify=None):
    from st_range import Checkpoint
    ckpt = Checkpoint(model_dir)
    vc = config_for(model_dir, case, rank)
    arena_on_meta(not real)
    res = {"case": case, "rank": rank}
    target, mc = build(vc)
    if real:
        from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
        loader = DefaultModelLoader(vc.load_config)
        with set_current_vllm_config(vc):
            w = loader.get_all_weights(mc, target)
            _load(target, mc, w, vc)
    else:
        _load(target, mc, meta_weights(ckpt, lambda n: not n.startswith("mtp.")), vc)
    if verify:
        res["verify_target"] = verify(target, ckpt, case, rank)
    del target
    if case.get("mtp"):
        with set_current_vllm_config(vc):
            draft, dmc = build(vc, drafter=True)
        if real:
            with set_current_vllm_config(vc):
                w = loader.get_all_weights(dmc, draft)
                _load(draft, dmc, w, vc)
        else:
            _load(draft, dmc, meta_weights(ckpt, lambda n: n.startswith("mtp.") or "embed_tokens" in n
                                           or "lm_head" in n), vc)
        if verify:
            res["verify_drafter"] = verify(draft, ckpt, case, rank)
        del draft
    res["closed"] = [{k: v for k, v in c.items()} for c in loadcheck.CLOSED]
    return res


def _cases_loop(model_dir, cases, **kw):
    out = []
    for case in cases:
        for rank in ranks_of(case):
            t0 = time.time()
            try:
                res = _run_case(model_dir, case, rank, **kw)
                res["ok"] = True
            except Exception as e:     # noqa: BLE001
                res = {"case": case, "rank": rank, "ok": False, "error": f"{type(e).__name__}: {e}",
                       "tb": traceback.format_exc()[-6000:], "closed": list(loadcheck.CLOSED)}
            res["seconds"] = round(time.time() - t0, 2)
            out.append(res)
            print(f"{'real' if kw.get('real') else 'meta'} tp={case['tp']} ep={case.get('ep')} mtp={case.get('mtp')} "
                  f"rank={rank} ok={res['ok']} {res['seconds']}s {[c['line'] for c in res.get('closed', [])]}",
                  file=sys.stderr, flush=True)
    return out


def cmd_load_meta(model: str, cases: list[dict]) -> list[dict]:
    return _cases_loop(model, cases, real=False)


# ----------------------------------------------------------------------------------- real-load verification
_LEAF_SPLIT = {
    "qkv_proj": [("q_proj", "q"), ("k_proj", "k"), ("v_proj", "v")],
    "gate_up_proj": [("gate_proj", 0), ("up_proj", 1)],
    "in_proj_qkvz": [("in_proj_qkv", (0, 1, 2)), ("in_proj_z", 3)],
}


def ckpt_module(prefix: str) -> str:
    """vLLM module prefix -> checkpoint module (without the fused-leaf split)."""
    if prefix.startswith("mtp."):
        return prefix
    if prefix.endswith("lm_head"):
        return "lm_head"
    if prefix.startswith("language_model.model."):
        return "model.language_model." + prefix[len("language_model.model."):]
    raise ValueError(prefix)


def verify_layers(model, ckpt, case, rank) -> dict:
    """Every EXL3 dense layer: params == tp.qwen35_plan slices of the checkpoint tensors placed at the plan's
    offsets; every local expert blob == pack_expert of the plan-narrowed checkpoint tensors."""
    from exl3xpu import tp as tpm
    from exl3xpu.moe_offload import pack_expert
    from trellis_core.reference import MUL1_MULT
    marker = MUL1_MULT - (1 << 32)
    cfg = ckpt.config()
    stats = {"dense_layers": 0, "dense_matrices": 0, "experts": 0, "mismatch": []}
    for name, m in model.named_modules():
        if hasattr(m, "exl3_kind") and getattr(m, "exl3_groups", None) is not None:
            base = ckpt_module(m.exl3_prefix)
            head, _, leaf = base.rpartition(".")
            members = _LEAF_SPLIT.get(leaf, [(leaf, None)])
            r, t = m.exl3_tp_rank, m.exl3_tp_size
            stats["dense_layers"] += 1
            if int(m.mul1.item()) != marker:
                stats["mismatch"].append((m.exl3_prefix, "mul1"))
            for gi, (cl, sid) in enumerate(members):
                mod = f"{head}.{cl}" if head else cl
                full = ckpt.exl3(mod)
                k, n = full["suh"].shape[0], full["svh"].shape[0]
                plan = tpm.qwen35_plan(mod, k, n, cfg, tp_rank=r, tp_size=t)
                a, ln = plan.dst_offset, plan.local_n
                ok = (torch.equal(m.trellis[:, a // 16:(a + ln) // 16], plan.narrow("trellis", full["trellis"]))
                      and torch.equal(m.svh[a:a + ln], plan.narrow("svh", full["svh"]))
                      and torch.equal(m.suh[gi], plan.narrow("suh", full["suh"])))
                if m.exl3_bounds[gi] != a:            # group gi starts where the plan puts this matrix
                    ok = False
                stats["dense_matrices"] += 1
                if not ok:
                    stats["mismatch"].append((m.exl3_prefix, cl))
        if hasattr(m, "exl3_arena") and m.exl3_ptrs is not None:
            base = ckpt_module(m.exl3_prefix)
            for g, slot in enumerate(m.exl3_cpu_map):
                if slot < 0:
                    continue
                parts = {}
                for role in ("gate_proj", "up_proj", "down_proj"):
                    mod = f"{base}.{g}.{role}"
                    full = ckpt.exl3(mod)
                    k, n = full["suh"].shape[0], full["svh"].shape[0]
                    plan = tpm.qwen35_plan(mod, k, n, cfg, tp_rank=0, tp_size=1,
                                           moe_tp_rank=m.exl3_moe_tp_rank, moe_tp_size=m.exl3_moe_tp_size)
                    parts[role] = plan.narrow_all({p: full[p] for p in ("trellis", "suh", "svh")})
                blob = pack_expert(parts["gate_proj"], parts["up_proj"], parts["down_proj"], m.exl3_K)
                stats["experts"] += 1
                if not torch.equal(blob, m.exl3_arena[slot]):
                    stats["mismatch"].append((m.exl3_prefix, g))
            from exl3xpu.moe_offload import s64
            ptr_ok = torch.equal(m.exl3_ptrs.cpu(), s64(m.exl3_arena.data_ptr())
                                 + torch.arange(m.exl3_E, dtype=torch.int64) * m.exl3_blob)
            if hasattr(m, "_exl3_staging"):
                stats["mismatch"].append((m.exl3_prefix, "staging buffer kept after loading"))
            if not ptr_ok:
                stats["mismatch"].append((m.exl3_prefix, "ptrs"))
    stats["bytes_read"] = ckpt.bytes_read
    return stats


def cmd_load_real(model: str, cases: list[dict]) -> list[dict]:
    return _cases_loop(model, cases, real=True, verify=verify_layers)



# ------------------------------------------------------------------------------------------ MoE method checks
DEAD_L3 = (72, 166, 180, 252)                 # layer-3 experts with all-zero svh (exact-zero probes)
MOE_EXPERTS = (0, 1, 63, 64, 72, 100, 127, 128, 166, 180, 191, 192, 200, 252, 255)   # spans every EP4 rank


def moe_inputs(H: int, topk: int, layer: int) -> dict[str, torch.Tensor]:
    """Fixed routing over MOE_EXPERTS: rows 0-3 padding (id -1, weight 0, as vLLM writes them), row 4 (layer 3)
    only dead experts, row 5 a partial row (4 real experts, 4 x -1), the rest random distinct top-8 choices."""
    g = torch.Generator().manual_seed(1000 + layer)
    M = 48
    x = (torch.randn((M, H), generator=g) * 0.5).to(torch.bfloat16)
    S = torch.tensor(MOE_EXPERTS)
    ids = torch.empty((M, topk), dtype=torch.int32)
    for m in range(M):
        ids[m] = S[torch.randperm(len(S), generator=g)[:topk]]
    w = torch.softmax(torch.randn((M, topk), generator=g), -1).float()
    ids[:4] = -1
    w[:4] = 0
    if layer == 3:
        ids[4] = torch.tensor(list(DEAD_L3) + [-1] * (topk - 4), dtype=torch.int32)
    ids[5, 4:] = -1
    w[5, 4:] = 0
    big = 2500                                  # > 1024: apply() chunks it
    xb = (torch.randn((big, H), generator=g) * 0.5).to(torch.bfloat16)
    idb = torch.stack([S[torch.randperm(len(S), generator=g)[:topk]] for _ in range(big)]).to(torch.int32)
    idb[::7] = -1
    wb = torch.softmax(torch.randn((big, topk), generator=g), -1).float()
    wb[::7] = 0
    return {"x": x, "ids": ids, "w": w, "xb": xb, "idb": idb, "wb": wb}


def _ref_weights(ckpt, layer: int) -> dict[int, dict]:
    from moe_ref import expert_weights
    out = {}
    for e in MOE_EXPERTS:
        base = f"model.language_model.layers.{layer}.mlp.experts.{e}"
        out[e] = expert_weights({r: {p: ckpt.get(f"{base}.{r}.{p}") for p in ("trellis", "suh", "svh")}
                                 for r in ("gate_proj", "up_proj", "down_proj")})
    return out


def _rel(a, b) -> float:
    d = (a - b).abs().max().item()
    s = b.abs().max().item()
    return d / s if s else d


def cmd_moe(model: str, cases: list[dict]) -> list[dict]:
    """For each case (all its ranks) on layers 0 and 3 of the mini: per-rank partials through
    Exl3MoEMethod.apply with moe_ref.BlobKernel; checks per rank and the sum over ranks against the checkpoint
    reference. Returns numbers and pass/fail per check."""
    import tempfile
    from st_range import Checkpoint
    from moe_ref import BlobKernel, reference_moe
    from exl3xpu import moe_vllm
    ckpt = Checkpoint(model)
    layers = (0, 3)
    ref_w = {L: _ref_weights(ckpt, L) for L in layers}
    out = []
    for case in cases:
        t0 = time.time()
        res = {"case": case, "checks": {}, "ok": True}
        try:
            partial = {L: {} for L in layers}
            calls = {}
            for rank in ranks_of(case):
                vc = config_for(model, dict(case, mtp=0), rank)
                arena_on_meta(False)
                target, mc = build(vc)
                from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
                loader = DefaultModelLoader(vc.load_config)
                with set_current_vllm_config(vc):
                    _load(target, mc, loader.get_all_weights(mc, target), vc)
                mods = {int(m.exl3_layer_idx): m for _, m in target.named_modules()
                        if hasattr(m, "exl3_arena") and m.exl3_layer_idx != "mtp"}
                for L in layers:
                    lay = mods[L]
                    kern = BlobKernel()
                    kern.add_arena(lay.exl3_arena)
                    lay.exl3_kernel = kern
                    inp = moe_inputs(lay.exl3_H, lay.exl3_topk, L)
                    meth = lay.quant_method
                    with tempfile.TemporaryDirectory() as td:
                        moe_vllm._DUMP_LAYER, moe_vllm._DUMP_DIR = str(L), td
                        y = meth.apply(lay, inp["x"], inp["w"], inp["ids"].to(torch.int64))
                        moe_vllm._DUMP_LAYER = ""
                        dumps = sorted(os.listdir(td))
                        d = torch.load(os.path.join(td, dumps[0])) if dumps else None
                    yb = meth.apply(lay, inp["xb"], inp["wb"], inp["idb"])
                    partial[L][rank] = (y, yb, inp)
                    calls[(L, rank)] = [c["M"] for c in kern.calls]
                    key = f"L{L}_r{rank}"
                    c = res["checks"]
                    c[f"{key}_kernel_args"] = all(k["ids_dtype"] == torch.int32 and k["ids_contig"]
                                                  and k["w_dtype"] == torch.float32 and k["w_contig"]
                                                  and k["x_contig"] and k["I"] == lay.exl3_I and k["K"] == 4
                                                  and k["E"] == lay.exl3_E for k in kern.calls)
                    c[f"{key}_chunks"] = calls[(L, rank)] == [48, 1024, 1024, 452]
                    c[f"{key}_padding_rows_zero"] = bool((y[:4] == 0).all()) and bool((yb[::7] == 0).all())
                    if L == 3:
                        c[f"{key}_dead_row_zero"] = bool((y[4] == 0).all())
                    c[f"{key}_dump"] = (d is not None and len(dumps) == 1 and dumps[0] == f"L{L}_r{rank}_s0.pt"
                                        and torch.equal(d["x"], inp["x"]) and torch.equal(d["out"], y)
                                        and torch.equal(d["topk_ids"], inp["ids"].to(torch.int64))
                                        and d["local_ids"].dtype == torch.int32)
                    # this rank's partial == reference restricted to its experts (same experts, same rows)
                    own = torch.tensor([lay.exl3_cpu_map[e] >= 0 for e in range(lay.exl3_global_E)])
                    ids_own = torch.where(inp["ids"] >= 0, torch.where(own[inp["ids"].clamp(min=0).long()],
                                                                       inp["ids"], -1), -1)
                    if lay.exl3_use_ep or case["tp"] == 1:
                        r_own = reference_moe(inp["x"], ids_own, inp["w"], ref_w[L])
                        c[f"{key}_partial_eq_ref_bitwise"] = torch.equal(y, r_own)
                        c[f"{key}_mapped_ids_ok"] = torch.equal(d["local_ids"], torch.where(
                            ids_own >= 0, torch.tensor(lay.exl3_cpu_map)[ids_own.clamp(min=0).long()],
                            torch.full_like(ids_own, -1)).to(torch.int32))
                    res.setdefault("rel", {})[f"{key}_partial_vs_own_ref"] = (
                        _rel(y, reference_moe(inp["x"], ids_own, inp["w"], ref_w[L]))
                        if lay.exl3_use_ep or case["tp"] == 1 else None)
                del target, mods
            for L in layers:
                inp = partial[L][ranks_of(case)[0]][2]
                full = reference_moe(inp["x"], inp["ids"], inp["w"], ref_w[L])
                fullb = reference_moe(inp["xb"], inp["idb"], inp["wb"], ref_w[L])
                s = sum(partial[L][r][0] for r in ranks_of(case))
                sb = sum(partial[L][r][1] for r in ranks_of(case))
                res.setdefault("rel", {})[f"L{L}_sum_vs_ref"] = _rel(s, full)
                res["rel"][f"L{L}_sum_vs_ref_chunked"] = _rel(sb, fullb)
                res["checks"][f"L{L}_sum_vs_ref"] = _rel(s, full) <= 1e-12
                res["checks"][f"L{L}_sum_vs_ref_chunked"] = _rel(sb, fullb) <= 1e-12
                res["checks"][f"L{L}_sum_padding_zero"] = bool((s[:4] == 0).all())
            res["ok"] = all(res["checks"].values())
            res["calls"] = {f"L{L}_r{r}": v for (L, r), v in calls.items()}
        except Exception as e:     # noqa: BLE001
            res.update(ok=False, error=f"{type(e).__name__}: {e}", tb=traceback.format_exc()[-6000:])
        res["seconds"] = round(time.time() - t0, 2)
        print(f"moe tp={case['tp']} ep={case.get('ep')} ok={res['ok']} {res['seconds']}s "
              f"{[k for k, v in res['checks'].items() if not v]} {res.get('error', '')}", file=sys.stderr, flush=True)
        out.append(res)
    return out



# ------------------------------------------------------------------------------------------------ refusals
def _expect_raise(fn, *needles):
    try:
        fn()
    except Exception as e:     # noqa: BLE001
        msg = f"{type(e).__name__}: {e}"
        return all(n in msg for n in needles), msg[:400]
    return False, "did not raise"


def _guard(fn):
    """(ok, detail) of a check function; an exception is a failed check, not a harness crash."""
    try:
        return fn()
    except Exception as e:     # noqa: BLE001
        return False, f"raised {type(e).__name__}: {e}"[:600]


def cmd_units(model: str, cases: list[dict]) -> list[dict]:
    """Refusal paths on real vLLM layers of Ornith-mini (TP2, rank 1, fake groups): routing of unknown linears,
    mixed / missing storage, loader shard-id / K / marker / shape / duplicate / coverage refusals, routed-expert
    loader refusals, and the EXL3XPU_LOADED mismatch check. Returns [{name, ok, detail}]."""
    import copy
    import tempfile
    from st_range import Checkpoint
    from vllm.model_executor.layers.linear import ColumnParallelLinear, MergedColumnParallelLinear
    from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
    from exl3xpu.vllm_plugin import Exl3LinearMethod
    ckpt = Checkpoint(model)
    out = []

    def check(name, ok_detail):
        ok, detail = ok_detail
        out.append({"name": name, "ok": bool(ok), "detail": detail})
        print(f"units {name} ok={ok}", file=sys.stderr, flush=True)

    case = {"tp": 2, "ep": True}
    vc = config_for(model, case, 1)
    cfg = vc.quant_config
    arena_on_meta(False)
    P = "language_model.model.layers.0"

    def mk(cls, *a, **kw):
        with set_current_vllm_config(vc), set_default_torch_dtype(torch.bfloat16), torch.device("cpu"):
            return cls(*a, **kw)

    # ---- routing
    check("route_unknown_linear_raises", _expect_raise(
        lambda: mk(ColumnParallelLinear, 3072, 1024, bias=False, quant_config=cfg, prefix=f"{P}.mlp.foo_proj"),
        "no EXL3 matrices"))
    check("route_unknown_mtp_linear_raises", _expect_raise(
        lambda: mk(ColumnParallelLinear, 3072, 1024, bias=False, quant_config=cfg, prefix="mtp.layers.0.mlp.foo"),
        "no EXL3 matrices"))
    c2 = copy.copy(cfg)
    c2.storage = dict(cfg.storage, embed_tokens=4)
    check("route_exl3_embedding_raises", _expect_raise(
        lambda: mk(VocabParallelEmbedding, 248320, 3072, quant_config=c2, prefix="language_model.model.embed_tokens"),
        "embeddings are not supported"))
    c3 = copy.copy(cfg)
    c3.storage = dict(cfg.storage)
    c3.storage["layers.3.self_attn.k_proj"] = 4
    check("route_mixed_fused_raises", _expect_raise(lambda: c3._bits_for(f"language_model.model.layers.3.self_attn."
                                                                       "qkv_proj"), "mixes EXL3 bitrates"))
    c4 = copy.copy(cfg)
    c4.storage = {k: v for k, v in cfg.storage.items() if k != "layers.3.mlp.experts.17.up_proj"}
    check("route_missing_expert_raises", _expect_raise(lambda: c4.expert_bits(
        "language_model.model.layers.3.mlp.experts", 256), "not all EXL3", "experts.17.up_proj"))
    check("route_in_proj_ba_unquantized", _guard(lambda: (type(mk(
        MergedColumnParallelLinear, 3072, [64, 64], bias=False, quant_config=cfg,
        prefix=f"{P}.linear_attn.in_proj_ba").quant_method).__name__ == "UnquantizedLinearMethod", "")))
    check("route_visual_unquantized", _guard(lambda: (type(mk(
        ColumnParallelLinear, 1152, 4304, bias=True, quant_config=cfg,
        prefix="visual.blocks.0.mlp.linear_fc1").quant_method).__name__ == "UnquantizedLinearMethod", "")))

    # ---- dense loader refusals on in_proj_qkvz (TP2 rank 1). A lone layer is its own load batch: its expected
    # EXL3XPU_LOADED counts are its own (the whole-model counts are checked by test_plugin_load.py).
    orig_expected = loadcheck.expected_counts
    loadcheck.expected_counts = lambda b, role, par: {"dense": b.counts["dense"], "experts": b.counts["experts"]}
    qkv = {p: ckpt.get(f"model.language_model.layers.0.linear_attn.in_proj_qkv.{p}") for p in
           ("trellis", "suh", "svh", "mul1")}
    z = {p: ckpt.get(f"model.language_model.layers.0.linear_attn.in_proj_z.{p}") for p in
         ("trellis", "suh", "svh", "mul1")}

    def layer_qkvz():
        loadcheck.reset_for_tests()
        return mk(MergedColumnParallelLinear, 3072, [2048, 2048, 8192, 8192], bias=False, quant_config=cfg,
                  prefix=f"{P}.linear_attn.in_proj_qkvz")

    def feed(lay, parts, sid, only=None):
        for p, t in parts.items():
            if only and p not in only:
                continue
            t = t.clone()
            t.shard_id = sid
            getattr(lay, p).weight_loader(getattr(lay, p), t)

    lay = layer_qkvz()
    check("loader_tuple_shard_23_refused", _expect_raise(lambda: feed(lay, qkv, (2, 3), only=["trellis"]),
                                                         "tuple shard id (2, 3)"))
    lay = layer_qkvz()
    check("loader_wrong_matrix_for_shard_refused", _expect_raise(lambda: feed(lay, z, 0, only=["svh"]),
                                                                 "has shape"))
    lay = layer_qkvz()
    bad = dict(qkv, mul1=torch.tensor(-1, dtype=torch.int32))
    check("loader_bad_marker_refused", _expect_raise(lambda: feed(lay, bad, (0, 1, 2), only=["mul1"]), "marker"))
    lay = layer_qkvz()
    k4 = dict(qkv, trellis=qkv["trellis"][..., :64].contiguous())
    check("loader_wrong_K_refused", _expect_raise(lambda: feed(lay, k4, (0, 1, 2), only=["trellis"]), "not K=5"))
    def complete():
        lay = layer_qkvz()
        feed(lay, qkv, (0, 1, 2))
        feed(lay, z, 3)
        lay.quant_method.process_weights_after_loading(lay)
        return (lay.exl3_groups == [(0, 1, 2), (3,)] and lay.exl3_bounds == [0, 6144, 10240] and lay.exl3_cpp
                and loadcheck.CLOSED[-1]["counts"]["dense"] == 8, f"{lay.exl3_bounds} {loadcheck.CLOSED[-1]['line']}")
    check("loader_complete_layer_processes", _guard(complete))
    lay = layer_qkvz()
    feed(lay, qkv, (0, 1, 2))
    check("process_missing_shard_raises", _expect_raise(
        lambda: lay.quant_method.process_weights_after_loading(lay), "do not cover"))
    lay = layer_qkvz()
    feed(lay, qkv, (0, 1, 2))
    feed(lay, z, 3)
    feed(lay, z, 3, only=["svh"])
    check("process_duplicate_raises", _expect_raise(
        lambda: lay.quant_method.process_weights_after_loading(lay), "not loaded exactly once"))
    lay = layer_qkvz()
    feed(lay, qkv, (0, 1, 2))
    feed(lay, z, 3, only=["trellis", "suh", "svh"])
    check("process_missing_marker_raises", _expect_raise(
        lambda: lay.quant_method.process_weights_after_loading(lay), "not loaded exactly once", "mul1"))
    lay = layer_qkvz()
    feed(lay, qkv, (0, 1, 2))
    feed(lay, z, 3)
    os.environ["EXL3_ALLOW_FALLBACK"] = "0"
    lay.exl3_K = 7
    check("process_unsupported_K_refuses_fallback", _expect_raise(
        lambda: lay.quant_method.process_weights_after_loading(lay), "refusing the Triton fallback"))

    loadcheck.expected_counts = orig_expected

    # ---- routed-expert loader refusals (TP2 + EP, rank 1 owns experts 128..255)
    loadcheck.reset_for_tests()
    vc.compilation_config.static_forward_context.clear()
    vc.compilation_config.static_all_moe_layers.clear()
    with set_current_vllm_config(vc):
        target, mc = build(vc)
    moe = {int(m.exl3_layer_idx): m for _, m in target.named_modules() if hasattr(m, "exl3_arena")
           and m.exl3_layer_idx != "mtp"}[3]

    def expert(e):
        base = f"model.language_model.layers.3.mlp.experts.{e}"
        return [(f"{e}.{r}.{p}", ckpt.get(f"{base}.{r}.{p}")) for r in ("down_proj", "gate_proj", "up_proj")
                for p in ("mul1", "suh", "svh", "trellis")]

    def run(pairs):
        return list(moe.load_weights(iter(pairs)))

    check("moe_unexpected_name_refused", _expect_raise(lambda: run([("17.gate_proj.weight", torch.zeros(1))]),
                                                       "unexpected tensor"))
    check("moe_expert_out_of_range_refused", _expect_raise(lambda: run([("256.gate_proj.trellis", torch.zeros(1))]),
                                                           "out of range"))
    before = moe.exl3_batch.counts["experts_dropped"]
    def remote():
        names = run(expert(3))        # expert 3 belongs to rank 0: dropped, nothing staged
        return (names == [] and moe.exl3_batch.counts["experts_dropped"] - before == 12 and not moe.exl3_stage,
                str(names))
    check("moe_remote_expert_dropped", _guard(remote))

    def local():
        names = run(expert(200))
        return len(names) == 12 and moe.exl3_packed[200 - 128] and not moe.exl3_stage, str(names[:3])
    check("moe_local_expert_packed", _guard(local))
    check("moe_after_complete_refused", _expect_raise(lambda: run(expert(200)[:1]), "after the expert was complete"))
    check("moe_duplicate_refused", _expect_raise(lambda: run(expert(201)[:1] + expert(201)[:1]), "loaded twice"))
    bad = [(n, torch.tensor(-1, dtype=torch.int32) if n.endswith("mul1") else t) for n, t in expert(202)]
    check("moe_bad_marker_refused", _expect_raise(lambda: run(bad), "not mul1"))
    check("moe_process_missing_raises", _expect_raise(
        lambda: moe.quant_method.process_weights_after_loading(moe), "of 128 local experts loaded"))
    del target, moe

    # ---- EXL3XPU_LOADED mismatch: expected counts from an index with 4 extra expert tensors local to rank 0
    with tempfile.TemporaryDirectory() as td:
        idx = json.load(open(os.path.join(model, "model.safetensors.index.json")))
        for p in ("trellis", "suh", "svh", "mul1"):
            idx["weight_map"][f"model.language_model.layers.4.mlp.experts.0.gate_proj.{p}"] = "x"
        path = os.path.join(td, "index.json")
        json.dump(idx, open(path, "w"))
        case4 = {"tp": 4, "ep": True}
        vc4 = config_for(model, case4, 0)
        old = vc4.quant_config.index_path
        vc4.quant_config.index_path = path
        try:
            from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
            arena_on_meta(False)
            t4, mc4 = build(vc4)

            def go():
                with set_current_vllm_config(vc4):
                    _load(t4, mc4, DefaultModelLoader(vc4.load_config).get_all_weights(mc4, t4), vc4)
            check("loaded_count_mismatch_raises", _expect_raise(
                go, "EXL3XPU_LOADED rank=0 dense=104 experts=3072 model=target expected=104/3076",
                "differ from the checkpoint index"))
        finally:
            vc4.quant_config.index_path = old
    return out


COMMANDS = {"routing": cmd_routing, "load-meta": cmd_load_meta, "load-real": cmd_load_real, "moe": cmd_moe,
            "units": cmd_units}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=sorted(COMMANDS))
    ap.add_argument("--model", required=True)
    ap.add_argument("--cases", required=True, help="JSON list of cases")
    ap.add_argument("--json-out", default="")
    a = ap.parse_args(argv)
    res = COMMANDS[a.command](a.model, json.loads(a.cases))
    text = json.dumps(res)
    if a.json_out:
        Path(a.json_out).write_text(text)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
