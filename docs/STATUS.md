# Status

## Milestone 1: build and manifest (done)

- **Build:** `_C.so` and `_moe.so` build with icpx 2026.1.1 (`intel/omix:0.4.0-devel-ubuntu24.04`) against the runtime image's torch 2.13.0+xpu.
  - That runtime's SYCL runtime is `libsycl.so.9` 2026.0, the same ABI.
  - Both libraries load into the runtime's torch, and all kernels register.
  - The runtime's own ocloc/IGC compiles every kernel image for the B70 (device id `0xe223`).
  - Rebuilds are byte-identical.
- **Manifest** (read from the safetensors headers): 37,941 EXL3 linears, all `mul1`, and every width is a multiple of 128.
  - K=4: 37,640 (the 256 routed experts per layer, plus the whole MTP layer).
  - K=5: 300 (attention, Gated DeltaNet, shared experts).
  - K=6: 1 (lm_head).
  - The vision tower is BF16; GDN `in_proj_a/b`, the router gate and `shared_expert_gate` are unquantized F16.
- **Kernel coverage:** the stock build already decodes `mul1` at K=3..6. `-DEXL3_ALL_CODEBOOKS` only adds `mul1` K=2 and the 3inst/mcg codebooks at K=4/6.
- **Gap:** the grouped MoE kernel is K=3 only, with split tables tuned for a smaller MoE (hidden 2560, expert width 640). Ornith needs K=4 at hidden 3072 and expert width 1024 (256 per card at TP4).

## Milestone 1b: dense kernels on a B70 (done)

**Correctness**, on real tensors of the checkpoint at K=4/5/6, including the full lm_head:
- `exl3_reconstruct` is **bit-exact** against an independent float64 reference transcribed from exllamav3 (964M unique weights).
- A second reference, decoded from exllamav3's writer side, agrees bit-exactly.
- Every GEMM path (GEMV, DPAS, reconstruct + GEMM) sits at the fp16 rounding floor.
- Decoded lm_head rows match their own token's embedding row, which two deliberately broken decoders do not.

**Speed** (one B70, XPU-graph replay; measured VRAM read roof 583 GB/s):
- Large tensors at batch 1: 318–566 GB/s; lm_head reaches 94–97 % of the roof.
- Prefill at 1,024 rows: 75–89 TFLOPS.

**5-bit vs 4- and 6-bit:**
- Single-row decode (batch 1–2): K=5 costs about what its bytes predict.
- Batched decode (3–32 rows): K=5 is 1.40–1.91x K=4 and slower than K=6. The K=5 DPAS path decodes 8 lanes x 32 steps, without K=4's 32-lane builder.
- Prefill: K=5 is within 3–11 %.

The 5-bit tensors are ~54 % of the weight bytes read per decoded token, so this was the first kernel fix, together with routing 5-bit batches of 3–8 rows to the single-row kernel and shape-aware split-K for narrow tensors. All three landed in milestone 1c below.

## Milestone 1c: dense kernels for Ornith's shapes (done)

Two changes, each reviewed independently and re-measured on a second card. Outputs are bit-identical to before except where a route moved to a different kernel, and every bf16 row passes the reference check.

- **Wide 5-bit DPAS build:** the 5-bit B-operand builder now decodes 32 lanes like the 4-bit one. Batched 5-bit decode went from 1.63-1.93x the 4-bit time to 1.02-1.21x at 3-8 rows; 190 changed cells are 1.02-1.83x faster (median 1.28x). The 6-bit build also widened at 16+ rows (median 1.08x).
- **Dispatch:** a per-shape GEMV/DPAS cut-over for 5-bit at 3-8 rows, and split-K sized for narrow tensors (the tensor-parallel slices are mostly narrow).

Dense EXL3 time per card at 4-way tensor parallelism (attention, Gated DeltaNet, shared experts, lm_head slice, MTP; routed experts excluded), one B70, XPU-graph replay:

| rows | before | after | ratio |
|---|---:|---:|---:|
| 1 | 3.99 ms | 3.65 ms | 0.91 |
| 4 | 5.06 ms | 3.67 ms | 0.73 |
| 16 | 5.36 ms | 4.09 ms | 0.76 |
| 64 | 8.80 ms | 5.50 ms | 0.63 |

At 16 streams with 3 MTP draft tokens (64 rows), a whole decode step's dense work goes from 10.19 ms to 6.70 ms per card.

## Milestones 2-5 (2-4 done; 5 not done: the gate soak fails two rows)

Milestones 2-4 are complete: the EXL3 MoE method in vLLM matches the reference on one card (2), two cards match
one (3), and Ornith boots and answers on four (4). For milestone 5, long-context recall, tool calls and throughput
pass on the live engine, but the 60-minute gate soak fails two rows (wrong table lookups at ~22K tokens and
fill-probe flips, both under load; see README Status). The evidence is the measured tables in [README.md](../README.md#measured), the status table in
[README.md](../README.md#status), and the honesty note there on the final gate bundle's verdict.

Open: the manual KV-bytes variant, the source fix for the capture leak behind the first-request guard, replacing
the fill-12 probe with one whose fp8 answer margin has been measured, and the other items in README's Open list.
