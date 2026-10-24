# Ornith-1.5-122B EXL3 on 4x Intel Arc Pro B70

Serving recipe and vLLM XPU plugin for
**[Mia-AiLab/Ornith-1.5-122B-EXL3-EAQv1-4-bpw](https://huggingface.co/Mia-AiLab/Ornith-1.5-122B-EXL3-EAQv1-4-bpw)**:
the 122B `qwen3_5_moe` checkpoint (EXL3 `mul1` codebook, EAQv1 4 bpw, 61.0 GiB) served as published — no
re-quantizing, no llama.cpp.

The EXL3 kernels and the vLLM plugin come from [trellis-serve](https://github.com/sybil-solutions/trellis-serve)
(branch `ft-xpu_bmg`, MIT); this repository is a fork that adds what a 122B MoE across four cards needs. The
upstream README is kept as [TRELLIS-SERVE-README.md](TRELLIS-SERVE-README.md); [ORIGIN.md](ORIGIN.md) records the
exact source commit.


## What this is

The shipping configuration, on four Arc Pro B70 cards, one rank per card:

| | |
|---|---|
| parallelism | tensor parallel 4 + expert parallel (whole experts: 64 of 256 per card) |
| context | `--max-model-len 262144` (256K), paged KV in 2,112-token blocks |
| speculation | MTP draft layer, `num_speculative_tokens 2` |
| KV cache | `--kv-cache-dtype fp8_e4m3` for the attention layers (the GDN/SSM state stays bf16), `--gpu-memory-utilization 0.85` |
| graphs | `cudagraph_mode: FULL_DECODE_ONLY` inside `--compilation-config` (not a standalone flag), capture sizes 1,2,4,8,12,16,24,32,40,48,56,64,96 → 24 graphs per rank (9 target + 15 drafter). No graph is captured while serving; the plugin's graph guard raises if one is. |
| batching | `--max-num-seqs 16`, `--max-num-batched-tokens 4096` |
| API | auto tool choice with the `qwen3_xml` parser; `qwen3_effort` reasoning parser plus a chat template that turns reasoning off with an empty think block |
| per card | 15.73 GiB of weights, 7.8 GiB of KV |

## Requirements

- 4x Intel Arc Pro B70 (32 GB each, Xe2 / Battlemage) on the `xe` kernel driver, all four idle: no other container
  may hold `/dev/dri` (`start.sh` refuses to start while one does).
- Docker with `/dev/dri` passed through.
- Host-side `python3` and `curl`: `start.sh` needs both (curl for the first-request guard and the READY gates,
  python3 for the gates).
- ~61 GiB of disk for the checkpoint (61.04 GiB at the validated revision). Fetch it with the `hf` CLI —
  `pip install -U "huggingface_hub[cli]"`, then
  `hf download Mia-AiLab/Ornith-1.5-122B-EXL3-EAQv1-4-bpw --revision 9c5c913b3de1eef901520d6fedaf450044ea0423 --local-dir <MODEL_DIR>`.
- The pinned runtime image — vLLM 0.30 XPU, torch 2.13.0+xpu: see [docs/runtime-image.md](docs/runtime-image.md).
  That image is **not publicly obtainable today**; §1 of that page records which of its pieces are reachable and
  which are not.
- A oneAPI DPC++ toolchain image for the kernel build: `intel/omix:0.4.0-devel-ubuntu24.04` (icpx 2026.1). The
  build runs on the CPU in throwaway containers; no GPU is needed for it.

## Quick start

```bash
git clone https://github.com/imryanpurdy/Ornith-1.5-122B-EXL3-4x-Intel-B70s.git && cd Ornith-1.5-122B-EXL3-4x-Intel-B70s
cp .env.example .env       # set MODEL_DIR to the weights and RUNTIME_IMAGE to the pinned image
./scripts/build-vxk.sh     # once: the vllm_xpu_kernels overlay (~2-2.5 h CPU-only at -j2, needs network)
./start.sh                 # builds _C.so/_moe.so on the first run (cached afterwards), launches, waits for READY
```

`scripts/build-vxk.sh` is a separate, once-only step: it builds the `vllm_xpu_kernels` overlay the engine cannot
start without, needs network for its source and dependency clones, and caches its output in `$VXK_DIR`.

Serves an OpenAI-compatible API on `$PORT` under the model id `$SERVED_MODEL_NAME`. There are two builds. The
overlay above is the CPU-only one, run once. `start.sh` builds the EXL3 op libs `_C.so` and `_moe.so` against the
runtime image's own torch on the first run (cached afterwards on source mtimes), then captures the decode graphs.
Later boots reuse `$BUILD_DIR` and `$VXK_DIR`.

## Measured

Shipping config above, greedy, thinking off, 4x B70.

| workload | how it was measured | tok/s |
|---|---|---:|
| 1 stream | 20 requests, first discarded, median of 19, 200-token cap | 101.9 |
| 1 stream | gate run, median of 19, 600-token cap | 112.6 |
| 16 concurrent | sustained aggregate = mean of rounds 2–4, 600-token cap per stream | 457.7 |
| 16 concurrent | gate run, median of 15 rounds | 454.8 |
| 32 concurrent | gate run, median of 15 rounds (informational, not a gate) | 448.8 |
| 1 stream, no MTP | as row 1 | 74.0 |
| 16 concurrent, no MTP | as row 3 | 430.6 |

KV pool, and the same pool as a multiple of the 407,455 tokens the previous production engine held:

| config | pool tokens | × |
|---|---:|---:|
| fp8 @ 0.85, MTP k=2 (shipping) | 1,140,422 | 2.80 |
| fp8 @ 0.85, no MTP | 1,420,780 | 3.49 |
| fp8 @ manual 11.61 GiB, MTP k=2 (measured, not shipped) | 1,695,325 | 4.16 |

60-minute soak: 9 streams × ~116K-token prompts (measured mean 116,410), 9/9 completed, 0 errors, mean KV 94.2% over the steady window
(min 81.3%, max 99.8%), 157 preemptions all at 97–100% KV, 0 OOM, VRAM drift ≤ 0.01 MiB per card.

Long context: needle-in-a-haystack correct at 131,457 and 205,322 prompt tokens (TTFT 101.7 s and 174.4 s), an
118K-token prefix-cache scan with no drift beyond the limit, and image input all pass.

Greedy decode here is not bit-deterministic across batch sizes: the same prompt at a different concurrency can
produce a different, equally valid answer. That is why the numbers above are medians and sustained means, not
single runs.

## How to test it

`./start.sh` runs the READY checks before it reports success:

- a 14-fill greedy arithmetic sweep (the `17*23 = 391` family) from 33 to 33,634 prompt tokens, with the four
  documented fp8 flip fills (3, 12, 75, 150) passed as known answers, so a clean engine reads 14/14;
- a tool-call probe: 5 single-tool calls, one multi-tool pick, one nested-argument call.

Then, from any client:

```bash
set -a; . ./.env; set +a    # $PORT, $SERVED_MODEL_NAME
curl -s "http://localhost:$PORT/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$SERVED_MODEL_NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"What is 17*23?\"}],\"max_tokens\":32,\"temperature\":0}" \
  | jq -r '.choices[0].message.content'
```

Send two or three short requests after a boot before the first long prompt (see Known limits 3).

## How to stop it

```bash
./start.sh stop
```

`start.sh` takes the name from `CONTAINER_NAME` in `.env` and prints it at launch; `stop` is just
`docker rm -f "$CONTAINER_NAME"` with the name resolved from `.env`. The other subcommands are `status` (container
and API state), `logs` (`docker logs -f`) and `--print` (print the full `docker run` line and the gate plan, run
nothing). Boots poll `/v1/models` for up to `READY_WAIT_SECONDS` (default 1800 s) — raise it on a slower host.

## Known limits

1. **fp8 KV near-tie flips.** Greedy answers on knife-edge prompts can differ from bf16 KV. The documented family
   is the `17*23 = 391` fill task, where fills 3, 12, 75 and 150 answer `401` — deterministic, well-formed, finite
   logprobs. The known-flip list is frozen and handed to the READY sweep; the flips are accepted and documented,
   not fixed. A paired A/B (recall at 60K/118K/200K, 220 short checkable prompts, tool calls) put fp8 within noise
   of bf16 on every set; on the short A/B set fp8 scored 219/220 against bf16's 220/220 — one fp8-only miss, judged
   within noise by the frozen rule.
2. **MTP draft-KV cost.** The 2-token draft layer reserves its own KV: 1,140,422 tokens against 1,420,780 on the
   identical config without MTP — 280,358 tokens, 19.7% of the no-MTP pool. It buys ~1.4× single-stream and ~1.06×
   at 16 streams.
3. **The first-request guard.** A cold FULL_DECODE_ONLY engine dies with `EXL3XPU_OPEN_CAPTURE` if its first
   request is a long prefill (≥29K tokens; 29K and 60K both reproduce): a SYCL/oneMKL command-graph recording is
   left open at the first unwarmed GEMM shape. Three short requests first leave the queue clean and everything
   after that is fine; `start.sh` does this as part of READY. The source fix is open.
4. **Long-context residency.** The pool holds about 9 concurrent ~116K-token sessions, not 16: on this hybrid
   model the GDN/SSM state cache, not the attention pool, caps residency at 9. 16 streams is reachable with short
   prompts.

## Status

Serving.

Two gate rows are recorded here honestly, from two different runs. The final full gate bundle (18:21Z boot)
holds verdict **NOT PASSED (full)**: its `nan_sweep_concurrency` row ran 13/14, with the one miss at concurrency 3 —
the documented KNOWN fill-12 `401` flip counted as a failure by a one-line counting bug in that gate harness's round
loop (the flip itself is documented; the counting bug was left as-is and never hidden); `nan_sweep_strict` passed
14/14 in the same bundle. Separately, a partial re-run (20:17Z boot) of the concurrency row showed a different
signature: the knife-edge fill probe truncated to `17<eos>` at concurrency 14. That near-tie truncation was accepted
at cutover after 800 real fill prompts showed zero truncations at concurrency 14.

| Milestone | State |
|---|---|
| 1. Kernels build for the runtime's torch; the checkpoint's EXL3 manifest checks out | done |
| 1b. Dense kernels run correctly on a B70 against an independent reference | done |
| 1c. Dense kernels faster for Ornith's shapes (5-bit DPAS, small-batch routing, split-K) | done |
| 2. EXL3 MoE method in vLLM matches the reference on one card | done |
| 3. Two cards match one card | done |
| 4. Ornith on four cards boots and answers | done |
| 5. Long-context recall, tool calls, throughput and a 60-minute soak | done |

Open:

- the manual KV-bytes variant (1,695,325 tokens): measured and probe-checked, ships only after its own 60-minute
  soak;
- the source fix for the capture leak behind the first-request guard;
- replacing the near-tie fill probe with a non-near-tie one — its concurrency variant is a canary for fp8 batch
  numerics, not a correctness gate.

Details: [docs/STATUS.md](docs/STATUS.md).

## Credits

- [trellis-serve](https://github.com/sybil-solutions/trellis-serve) (MIT): the B70 EXL3 kernels, the vLLM and
  SGLang plugins and the split rules this builds on. Its `LICENSE` and `NOTICE` are kept unchanged.
- [Mia-AiLab](https://huggingface.co/Mia-AiLab): the EXL3 quantization recipe and the weights served here.
- [exllamav3](https://github.com/turboderp-org/exllamav3): the EXL3 format, the reference reconstruction and the
  tensor-parallel layouts.
- [vLLM](https://github.com/vllm-project/vllm) and Intel's XPU kernels and oneAPI toolchain.

## License

MIT, inherited from trellis-serve; see [LICENSE](LICENSE) and [NOTICE](NOTICE).
