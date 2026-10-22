"""
Targeted fixes to vLLM internals that matter on XPU. Applied from the plugin entry point, each one
checks the exact source it expects and logs + skips if vLLM changed (never patches blindly).

On vLLM 0.30 only patch_fp8kv_prefill is applied through EXL3_VLLM_PATCHES=fp8kv (vllm_plugin.apply_patches);
the others are kept for reference and refused by the plugin. patch_det_small_linear is not an
EXL3_VLLM_PATCHES entry: it is a correctness fix (EXL3_DET_SMALL_LINEAR=0 turns it off) and the plugin
installs it unconditionally.

gdn_mask_index: GDNAttentionMetadataBuilder.build indexes device tensors with a CPU boolean mask
  (e.g. block_table_tensor[spec_sequence_masks_cpu, :k]). That forces a pageable host->device copy of
  the mask, which on XPU's in-order queue waits for all prior GPU work: ~20 ms of stall every speculative
  decode step (measured: 1.0 s over 50 steps), serialising host prep with the previous step. The patch
  routes those through _rows(): a slice when the mask is all-true (pure spec decode, the hot path), else
  CPU-computed indices copied asynchronously from pinned memory + index_select (no sync).
"""
from __future__ import annotations
import inspect
import os
import re
import textwrap

import torch

from vllm.logger import init_logger

logger = init_logger("vllm.exl3xpu")


def _rows(t: torch.Tensor, mask_cpu: torch.Tensor) -> torch.Tensor:
    """t[mask_cpu] without a host<->device sync (mask is a CPU bool tensor over t's first dim)."""
    if t.device.type == "cpu":
        return t[mask_cpu]
    n = mask_cpu.shape[0]
    if bool(mask_cpu.all()):
        return t[:n]
    idx = mask_cpu.nonzero().squeeze(1).pin_memory().to(t.device, non_blocking=True)
    return t.index_select(0, idx)


_PATTERNS = [
    # (regex on source, replacement)
    (r"(\w+)\[\s*spec_sequence_masks_cpu\s*,\s*: self\.num_spec \+ 1\s*\]", r"_rows(\1, spec_sequence_masks_cpu)[:, : self.num_spec + 1]"),
    (r"(\w+)\[\s*~spec_sequence_masks_cpu\s*,\s*0\s*\]", r"_rows(\1, ~spec_sequence_masks_cpu)[:, 0]"),
    (r"(\w+)\[\s*~spec_sequence_masks_cpu\s*\]", r"_rows(\1, ~spec_sequence_masks_cpu)"),
    (r"(\w+)\[\s*spec_sequence_masks_cpu\s*\]", r"_rows(\1, spec_sequence_masks_cpu)"),
]


def patch_gdn_mask_index() -> bool:
    try:
        from vllm.v1.attention.backends import gdn_attn
    except Exception as e:  # noqa
        logger.warning("exl3xpu: gdn_attn not importable (%s); GDN sync patch skipped", e)
        return False
    cls = getattr(gdn_attn, "GDNAttentionMetadataBuilder", None)
    if cls is None or getattr(cls, "_exl3_sync_patched", False):
        return False
    src = textwrap.dedent(inspect.getsource(cls.build))
    n_total = 0
    for pat, rep in _PATTERNS:
        src, n = re.subn(pat, rep, src)
        n_total += n
    if n_total < 2 or "_rows(block_table_tensor, spec_sequence_masks_cpu)" not in src:
        logger.warning("exl3xpu: GDNAttentionMetadataBuilder.build source changed (%d matches); sync patch skipped",
                       n_total)
        return False
    ns = dict(vars(gdn_attn))
    ns["_rows"] = _rows
    exec(compile(src, f"<exl3xpu patched {gdn_attn.__file__}>", "exec"), ns)
    cls.build = ns["build"]
    cls._exl3_sync_patched = True
    logger.info("exl3xpu: patched GDN metadata build (%d mask-index sites made sync-free)", n_total)
    return True


FP8KV_PREFILL_MIN_SEQ = 4096
FP8KV_PREFILL_MIN_QUERY = 64      # genuine prefill chunks only (not MTP verify batches of k+1 tokens)

# vLLM 0.30: source lines of the stock FlashAttentionImpl whose semantics the routed path relies on (cache layout,
# fp8 view, metadata fields, scalar k/v scales). patch_fp8kv_prefill refuses to apply when one is missing, and the
# plugin then aborts the boot (vllm_plugin.apply_patches).
_FP8KV_FORWARD_MARKERS = (
    "if attn_metadata is None:",
    "key_cache, value_cache = kv_cache.transpose(1, 2).split(self.head_size, dim=-1)",
    "key_cache = key_cache.view(current_platform.fp8_dtype())",
    "value_cache = value_cache.view(current_platform.fp8_dtype())",
    "if not attn_metadata.use_cascade:",
    "cu_seqlens_q = attn_metadata.query_start_loc",
    "seqused_k = attn_metadata.seq_lens",
    "max_seqlen_k = attn_metadata.max_seq_len",
    "block_table = attn_metadata.block_table",
    "k_descale = layer._k_scale.expand(descale_shape)",
    "v_descale = layer._v_scale.expand(descale_shape)",
)
_FP8KV_UPDATE_MARKERS = (
    "key_cache, value_cache = kv_cache.transpose(1, 2).split(self.head_size, dim=-1)",
    "reshape_and_cache_flash(",
)
_FP8KV_FORWARD_PARAMS = ("self", "layer", "query", "key", "value", "kv_cache", "attn_metadata", "output",
                         "output_scale", "output_block_scale")
_FP8KV_MD_FIELDS = ("num_actual_tokens", "max_query_len", "query_start_loc", "max_seq_len", "seq_lens",
                    "block_table", "use_cascade", "causal", "mm_prefix_query_range_tensor", "rswa_prefix_lens")

# routed prefill chunks in this process (tests and the first-use log line read it)
FP8KV_STATS = {"routed": 0}


def _fp8kv_source_problems(fa_mod) -> list[str]:
    """What in the installed flash_attn backend differs from what the routed path was written against."""
    cls = fa_mod.FlashAttentionImpl
    problems = []
    try:
        fwd = inspect.getsource(cls.forward)
        upd = inspect.getsource(cls.do_kv_cache_update)
    except (OSError, TypeError, AttributeError) as e:
        return [f"source not readable: {type(e).__name__}: {e}"]
    problems += [f"forward lacks {m!r}" for m in _FP8KV_FORWARD_MARKERS if m not in fwd]
    problems += [f"do_kv_cache_update lacks {m!r}" for m in _FP8KV_UPDATE_MARKERS if m not in upd]
    params = tuple(inspect.signature(cls.forward).parameters)
    if params != _FP8KV_FORWARD_PARAMS:
        problems.append(f"forward parameters {params}")
    md_fields = set(getattr(fa_mod.FlashAttentionMetadata, "__dataclass_fields__", {}))
    problems += [f"FlashAttentionMetadata lacks {f}" for f in _FP8KV_MD_FIELDS if f not in md_fields]
    for name in ("is_quantized_kv_cache", "current_platform", "AttentionType"):
        if not hasattr(fa_mod, name):
            problems.append(f"flash_attn module lacks {name}")
    return problems


def patch_fp8kv_prefill() -> bool:
    """Route single-sequence prefill chunks over an fp8 KV cache through 16-bit attention on the dequantized
    sequence (exl3xpu.fp8kv_prefill.prefill_attention_paged): vllm_xpu_kernels' FA2 runs about 2x slower on fp8 K/V
    than on bf16/fp16 at head_dim 256 (B70, vllm_xpu_kernels 0.1.14.1: 26-34 vs 56-75 TFLOPS, notes T15a).

    Routed: one sequence in the batch, >= FP8KV_PREFILL_MIN_QUERY queries, >= FP8KV_PREFILL_MIN_SEQ keys, not
    capturing a graph, a plain causal decoder layer (no ALiBi, sinks, soft cap, sliding window, DCP/PCP, cascade,
    multimodal prefix ranges, R-SWA), fp16/bf16 queries. Everything else runs the stock kernel unchanged.
    vLLM 0.30 port: the routed call is one paged causal FA call in the query's dtype over a gathered, dequantized
    copy of the sequence, not trellis' KB-block merge (prefill_attention): the merge needs non-paged head_dim-256
    kernels with LSE output that vllm_xpu_kernels 0.1.14.1 does not ship, so every merge part fell back to PyTorch
    attention. No mixed decode+prefill route (0.30 fills num_decode_reqs / num_prefill_reqs only under DCP, and the
    route needed a .item() sync), no oneDNN route; EXL3_FP8KV_MIXED=1 or EXL3_ONEDNN_ATTN=1 make it refuse.
    EXL3_FP8KV_CHECK=1 (tests): each routed call syncs and checks that the sequence's real length is max_seq_len
    (the V2 runner passes a CPU upper bound there, exact for a prefill chunk)."""
    try:
        from vllm.v1.attention.backends import flash_attn as fa_mod
    except Exception as e:  # noqa
        logger.warning("exl3xpu: flash_attn backend not importable (%s); fp8-KV prefill patch skipped", e)
        return False
    cls = getattr(fa_mod, "FlashAttentionImpl", None)
    if cls is None or getattr(cls, "_exl3_fp8kv_patched", False):
        return False
    if os.environ.get("EXL3_FP8KV_MIXED", "0") == "1":
        logger.warning("exl3xpu: EXL3_FP8KV_MIXED=1 is not supported on vLLM 0.30; fp8-KV prefill patch not applied")
        return False
    from . import fp8kv_prefill
    if fp8kv_prefill.ONEDNN_ATTN:
        logger.warning("exl3xpu: EXL3_ONEDNN_ATTN=1 is not validated on vLLM 0.30; fp8-KV prefill patch not applied")
        return False
    problems = _fp8kv_source_problems(fa_mod)
    if problems:
        logger.warning("exl3xpu: FlashAttentionImpl differs from vLLM 0.30 (%s); fp8-KV prefill patch skipped",
                       "; ".join(problems))
        return False
    orig = cls.forward
    is_q = fa_mod.is_quantized_kv_cache
    fp8_dtype = fa_mod.current_platform.fp8_dtype()
    decoder = fa_mod.AttentionType.DECODER
    check = os.environ.get("EXL3_FP8KV_CHECK", "0") == "1"

    def eligible(self, md, query, output_scale, output_block_scale) -> bool:
        if md is None or output_scale is not None or output_block_scale is not None:
            return False
        if not is_q(self.kv_cache_dtype) or md.use_cascade:
            return False
        if md.query_start_loc.shape[0] != 2 or md.max_query_len < FP8KV_PREFILL_MIN_QUERY \
                or md.max_seq_len < FP8KV_PREFILL_MIN_SEQ:
            return False
        # one sequence: every token of the batch is its query, and its keys end at max_seq_len
        if md.num_actual_tokens != md.max_query_len or md.max_seq_len < md.max_query_len:
            return False
        if query.dtype not in (torch.float16, torch.bfloat16):
            return False
        if torch.xpu.is_available() and torch.xpu.is_current_stream_capturing():
            return False
        if self.alibi_slopes is not None or getattr(self, "sinks", None) is not None:
            return False
        if self.logits_soft_cap not in (None, 0, 0.0):
            return False
        if getattr(self, "dcp_world_size", 1) != 1 or getattr(self, "pcp_world_size", 1) != 1:
            return False
        if getattr(self, "attn_type", decoder) != decoder or getattr(self, "fa4_hd256", False):
            return False
        # 0.30: causal is a bool or a per-sequence Tensor; only the plain causal bool qualifies
        if getattr(md, "causal", True) is not True:
            return False
        sw = self.sliding_window
        if sw is not None and tuple(sw) != (-1, -1):
            return False
        # multimodal prefix ranges (0.30 name, and the older one) and R-SWA change the mask
        if getattr(md, "mm_prefix_query_range_tensor", None) is not None \
                or getattr(md, "mm_prefix_range_tensor", None) is not None \
                or getattr(md, "rswa_prefix_lens", None) is not None:
            return False
        return True

    def forward(self, layer, query, key, value, kv_cache, attn_metadata, output,
                output_scale=None, output_block_scale=None):
        md = attn_metadata
        if not eligible(self, md, query, output_scale, output_block_scale):
            return orig(self, layer, query, key, value, kv_cache, attn_metadata, output,
                        output_scale, output_block_scale)
        n = md.num_actual_tokens
        seq_len = int(md.max_seq_len)
        if check:
            real_len, real_q = int(md.seq_lens[0].item()), int(md.query_start_loc[1].item())
            if real_len != seq_len or real_q != n:
                raise RuntimeError(f"fp8kv prefill: max_seq_len {seq_len} / num_actual_tokens {n} differ from the "
                                   f"sequence's seq_len {real_len} / query length {real_q}")
        key_cache, value_cache = kv_cache.transpose(1, 2).split(self.head_size, dim=-1)
        key_cache = key_cache.view(fp8_dtype)
        value_cache = value_cache.view(fp8_dtype)
        q = query[:n]
        out = output[:n].view(n, q.shape[1], self.head_size)
        # host-side scale copies (no device sync); fall back to the tensors only if vLLM lacks them
        ks = getattr(layer, "_k_scale_float", None)
        vs = getattr(layer, "_v_scale_float", None)
        ks = float(layer._k_scale) if ks is None else float(ks)
        vs = float(layer._v_scale) if vs is None else float(vs)
        fp8kv_prefill.prefill_attention_paged(q, key_cache, value_cache, md.block_table[0], seq_len,
                                              ks, vs, float(self.scale), out)
        FP8KV_STATS["routed"] += 1
        if FP8KV_STATS["routed"] == 1:
            logger.info("EXL3XPU_FP8KV_ROUTE first routed prefill chunk: %d queries over %d keys, %s compute, "
                        "k_scale %g v_scale %g", n, seq_len, str(q.dtype).replace("torch.", ""), ks, vs)
        return output

    cls.forward = forward
    cls._exl3_fp8kv_patched = True
    cls._exl3_fp8kv_orig_forward = orig
    logger.info("exl3xpu: fp8-KV prefill attention routed through dequantized 16-bit paged FA (one sequence, "
                ">= %d queries, >= %d keys, outside graph capture)", FP8KV_PREFILL_MIN_QUERY, FP8KV_PREFILL_MIN_SEQ)
    return True


def patch_xpu_block_size() -> bool:
    """Opt-in (EXL3_KV_BLOCK_EXACT=1): keep the hybrid-negotiated attention block size (a multiple of 64 for the
    GDN kernel) instead of rounding it up to a power of two. On Qwen3.8 with fp8 KV the negotiation gives 1600
    tokens (= the 3.1 MiB GDN state page); the XPU platform rounds that to 2048 and pads every page to 4 MiB,
    wasting ~22% of the KV pool. The power-of-two rule exists for the ESIMD paged-attention decode fast path;
    this model's decode/verify attention runs through FA2 varlen instead."""
    import os
    if os.environ.get("EXL3_KV_BLOCK_EXACT") != "1":
        return False
    try:
        from vllm.platforms import xpu as xpu_mod
        from vllm.platforms.interface import Platform
    except Exception as e:  # noqa
        logger.warning("exl3xpu: XPU platform not importable (%s); block-size patch skipped", e)
        return False
    cls = getattr(xpu_mod, "XPUPlatform", None)
    if cls is None or getattr(cls, "_exl3_block_patched", False):
        return False

    def update_block_size_for_backend(klass, vllm_config) -> None:
        Platform.update_block_size_for_backend.__func__(klass, vllm_config)
        from vllm.config.vllm import get_layers_from_vllm_config
        from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
        cc = vllm_config.cache_config
        layers = get_layers_from_vllm_config(vllm_config, AttentionLayerBase)
        if not any(l.get_attn_backend().get_name() == "GDN_ATTN" for l in layers.values()):
            return
        bs = max(cc.block_size, 64)
        new = (bs + 63) // 64 * 64
        if new == cc.block_size:
            logger.info("exl3xpu: keeping attention block size %d (multiple of 64, not rounded to a power of 2)", new)
            return
        if cc.mamba_cache_mode == "align":
            cc.mamba_block_size = new
        if cc.mamba_page_size_padded is not None:
            cc.mamba_page_size_padded = new * (cc.mamba_page_size_padded // cc.block_size)
        cc.block_size = new
        logger.info("exl3xpu: attention block size %d (multiple of 64)", new)

    cls.update_block_size_for_backend = classmethod(update_block_size_for_backend)
    cls._exl3_block_patched = True
    logger.info("exl3xpu: XPU block size keeps the hybrid negotiation (EXL3_KV_BLOCK_EXACT=1)")
    return True


# ------------------------------------------------------------------------------------------------
# align_sync: with prefix caching on, hybrid models run mamba_cache_mode "align", and GPUModelRunner._prepare_inputs
# then calls num_accepted_tokens_event.synchronize() EVERY step (the non-align async path skips it). The host
# cannot prepare step N+1 while step N runs: py-spy put 82% of EngineCore time on that line and prose decode
# lost 20-30%. The CPU counts are only needed when preprocess_mamba copies a request's GDN state into a new
# block (a block boundary crossing, ~1 in 400 steps at block 1600) or when the batch rows changed; otherwise the
# GPU copy (written by postprocess_mamba_align_gpu) is already correct. The patch skips the sync and the two
# CPU->GPU re-copies of the (then stale) counts on exactly those steps.

ALIGN_SKIP_MARGIN = 16    # tokens of slack on num_computed_tokens (async corrections, draft rejections)


def _align_can_skip(runner, scheduler_output, num_reqs) -> bool:
    if not getattr(runner, "use_async_scheduling", False) or os.environ.get("EXL3_ALIGN_SYNC_SKIP", "0") != "1":
        return False
    req_ids = list(runner.input_batch.req_ids[:num_reqs])
    prev = getattr(runner, "_exl3_prev_req_ids", None)
    runner._exl3_prev_req_ids = req_ids
    if prev != req_ids:
        return False
    try:
        bs = runner._get_mamba_bufs().preprocess.mamba_spec.block_size
    except Exception:
        return False
    nst = scheduler_output.num_scheduled_tokens
    for rid in req_ids:
        prev_idx = runner.mamba_state_idx.get(rid)
        st = runner.requests.get(rid)
        if prev_idx is None or st is None:
            return False
        n = nst.get(rid, 0)
        lo = max(st.num_computed_tokens - ALIGN_SKIP_MARGIN, 0) + n
        hi = st.num_computed_tokens + ALIGN_SKIP_MARGIN + n
        if -(-lo // bs) - 1 != prev_idx or -(-hi // bs) - 1 != prev_idx:
            return False
    return True


_ALIGN_SYNC_OLD = """        if needs_cpu_accepted_counts:
            assert self.num_accepted_tokens_event is not None
            self.num_accepted_tokens_event.synchronize()"""
_ALIGN_SYNC_NEW = """        self._exl3_skipped_sync = bool(needs_cpu_accepted_counts) and _align_can_skip(
            self, scheduler_output, num_reqs)
        if self._exl3_skipped_sync:
            self._exl3_skips = getattr(self, "_exl3_skips", 0) + 1
        elif needs_cpu_accepted_counts:
            assert self.num_accepted_tokens_event is not None
            self.num_accepted_tokens_event.synchronize()"""
_ALIGN_RESYNC_OLD = """                self.num_accepted_tokens.np[:num_reqs] = (
                    self.input_batch.num_accepted_tokens_cpu[:num_reqs]
                )
                self.num_accepted_tokens.copy_to_gpu(num_reqs)"""
_ALIGN_RESYNC_NEW = """                if not getattr(self, "_exl3_skipped_sync", False):
                    self.num_accepted_tokens.np[:num_reqs] = (
                        self.input_batch.num_accepted_tokens_cpu[:num_reqs]
                    )
                    self.num_accepted_tokens.copy_to_gpu(num_reqs)"""


def patch_align_sync() -> bool:
    try:
        from vllm.v1.worker import gpu_model_runner as gmr
    except Exception as e:  # noqa
        logger.warning("exl3xpu: gpu_model_runner not importable (%s); align sync patch skipped", e)
        return False
    cls = gmr.GPUModelRunner
    if getattr(cls, "_exl3_align_patched", False):
        return False
    fixes = []
    for name, old, new in (("_prepare_inputs", _ALIGN_SYNC_OLD, _ALIGN_SYNC_NEW),
                           ("execute_model", _ALIGN_RESYNC_OLD, _ALIGN_RESYNC_NEW)):
        src = textwrap.dedent(inspect.getsource(getattr(cls, name)))
        old_d, new_d = textwrap.dedent(old), textwrap.dedent(new)
        # sources are dedented by 4 (method level); match the snippets at that indentation
        old_d = "\n".join(l[4:] for l in old.split("\n"))
        new_d = "\n".join(l[4:] for l in new.split("\n"))
        if src.count(old_d) != 1 or "super()" in src:
            logger.warning("exl3xpu: GPUModelRunner.%s source changed; align sync patch skipped", name)
            return False
        fixes.append((name, src.replace(old_d, new_d)))
    ns = dict(vars(gmr))
    ns["_align_can_skip"] = _align_can_skip
    for name, src in fixes:
        exec(compile(src, f"<exl3xpu patched {gmr.__file__}:{name}>", "exec"), ns)
        setattr(cls, name, ns[name])
    cls._exl3_align_patched = True
    logger.info("exl3xpu: align-mode accepted-token sync skipped except on block crossings / batch changes")
    return True


# ------------------------------------------------------------------------------------------------
# det_small_linear: oneDNN's XPU bf16/fp16 GEMM - the path F.linear takes for every unquantized vLLM
# linear (GDN in_proj_ba, the MoE router gate, the shared-expert gate) - is not run-to-run bitwise
# reproducible when the row count lands in 65..128 AND the output width is narrow. Few column tiles
# means oneDNN splits K across workgroups and reduces the fp32 partials with atomics, so the sum order
# varies. Measured on one B70, 100 repeats per cell, K=3072 bf16: M<=64 and M>=129 are always
# bit-identical; M=65..128 with N in 24..192 differs on 3-100% of the repeats (N=32 at M=76: 93/100,
# N=128 at M=90: 99/100), while N<=16 and N>=256 stay exact. 65..128 rows is exactly a 2-page
# (2 x 64-token) prefill, which is why only that prompt-length band diverges: the perturbed rows are
# ~1 ulp, then bf16 rounding plus MoE top-k turn them into 1e-3-level logit differences and answer
# flips. Fix: run those calls as <=64-row GEMMs, the one regime oneDNN never splits; it costs one
# extra tiny GEMM per affected linear (N=32/K=3072: ~5 us) and nothing outside the band.
DET_LINEAR_ROWS = 64     # rows per chunk: <= 64 is the reproducible regime
DET_LINEAR_MAX_M = 128   # only 64 < M <= 128 is affected
DET_LINEAR_N_MAX = 192   # only narrow-N GEMMs split K this way (N >= 256 measured exact)
_DET_LINEAR_DTYPES = (torch.bfloat16, torch.float16)


def patch_det_small_linear() -> bool:
    """Chunk the unquantized-linear GEMM to <= DET_LINEAR_ROWS rows for the 2-page prefill band."""
    if os.environ.get("EXL3_DET_SMALL_LINEAR", "1") == "0":
        return False
    try:
        from vllm.model_executor.layers.linear import UnquantizedLinearMethod
    except Exception as e:  # noqa: BLE001
        logger.warning("exl3xpu: UnquantizedLinearMethod not importable (%s); det_small_linear skipped", e)
        return False
    if getattr(UnquantizedLinearMethod, "_exl3_det_small_linear", False):
        return True
    orig = UnquantizedLinearMethod.apply

    def apply(self, layer, x, bias=None):
        w = getattr(layer, "weight", None)
        if (w is not None and w.dim() == 2 and w.shape[0] <= DET_LINEAR_N_MAX
                and x.dim() >= 2 and x.dtype in _DET_LINEAR_DTYPES
                and DET_LINEAR_ROWS < x.shape[0] <= DET_LINEAR_MAX_M):
            x2 = x.reshape(-1, x.shape[-1])
            out = torch.cat([orig(self, layer, x2[i:i + DET_LINEAR_ROWS], bias)
                             for i in range(0, x2.shape[0], DET_LINEAR_ROWS)], dim=0)
            return out.reshape(*x.shape[:-1], out.shape[-1])
        return orig(self, layer, x, bias)

    UnquantizedLinearMethod.apply = apply
    UnquantizedLinearMethod._exl3_det_small_linear = True
    logger.info("exl3xpu: unquantized linear GEMMs chunked to <= %d rows for %d..%d rows with N <= %d "
                "(oneDNN XPU split-K is not bit-reproducible there)", DET_LINEAR_ROWS,
                DET_LINEAR_ROWS + 1, DET_LINEAR_MAX_M, DET_LINEAR_N_MAX)
    return True

def apply_all():
    """Not used on vLLM 0.30: the plugin applies only the patches EXL3_VLLM_PATCHES names, through
    vllm_plugin.apply_patches (fp8kv only; the others are refused there)."""
    raise RuntimeError("exl3xpu: vllm_patches.apply_all() is not supported on vLLM 0.30; use EXL3_VLLM_PATCHES")
