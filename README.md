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

**Not yet run end to end:** `start.sh`'s cold launch path (kernel build, dist, `docker run`, first-request guard,
READY gates) has never launched an engine. The validated deployment was started by a separate launch script with
the same image. Its 43 serve arguments and its container environment match `./start.sh --print` when `.env` sets
the same model id and port; host paths, the container name and the container labels differ. Of `start.sh` itself
only `--print` and dry runs against a stand-in `docker` have been exercised.

`scripts/build-vxk.sh` is a separate, once-only step: it builds the `vllm_xpu_kernels` overlay the engine cannot
start without, needs network for its source and dependency clones, and caches its output in `$VXK_DIR`.

Serves an OpenAI-compatible API on `$PORT` under the model id `$SERVED_MODEL_NAME`. There are two builds. The
overlay above is the CPU-only one, run once. `start.sh` builds the EXL3 op libs `_C.so` and `_moe.so` against the
runtime image's own torch on the first run (cached afterwards on source mtimes), then captures the decode graphs.
Later boots reuse `$BUILD_DIR` and `$VXK_DIR`.

## Measured

Greedy, 4x B70. Every row runs the shipping config above (`--max-num-seqs 16`, MTP k=2) unless the row says
otherwise. The 1-stream rows are chat requests; the 16- and 32-way rows send a raw prompt to `/v1/completions`,
so the chat template's thinking switch does not apply to them.

| workload | how it was measured | tok/s |
|---|---|---:|
| 1 stream, thinking off | 20 requests, first discarded, median of 19, 200-token cap | 101.9 |
| 1 stream, thinking on (the template's default) | gate run, 20 requests, first discarded, median of 19, 600-token cap | 112.6 |
| 16 concurrent | sustained aggregate = mean of rounds 2–4, 600-token cap per stream | 457.7 |
| 16 concurrent | gate run, 15 rounds, median of rounds 2–15 | 454.8 |
| 32 concurrent | gate run, 15 rounds, median of rounds 2–15 (informational, not a gate) | 448.8 |
| 1 stream, no MTP, thinking off | as row 1 (74.0 at `--max-num-seqs` 16 and at 32) | 74.0 |
| 16 concurrent, no MTP, `--max-num-seqs 32` | as row 3 | 430.6 |
| 16 concurrent, MTP k=2, `--max-num-seqs 32` | as row 3: the same-`--max-num-seqs` reference for the row above | 447.9 |

KV pool, and the same pool as a multiple of the 407,455 tokens the previous production engine held:

| config | pool tokens | × |
|---|---:|---:|
| fp8 @ 0.85, MTP k=2, `--max-num-seqs 16` (shipping) | 1,140,422 | 2.80 |
| fp8 @ 0.85, MTP k=2, `--max-num-seqs 32` | 1,138,508 | 2.79 |
| fp8 @ 0.85, no MTP, `--max-num-seqs 32` | 1,420,780 | 3.49 |
| fp8 @ manual 11.61 GiB, MTP k=2, `--max-num-seqs 16` (measured, not shipped) | 1,695,325 | 4.16 |

60-minute soak: 9 streams × ~116K-token prompts (measured mean 116,410), 9/9 completed, 0 errors, mean KV 94.2% over the steady window
(min 81.3%, max 99.8%), 157 preemptions, 0 OOM, VRAM drift ≤ 0.01 MiB per card. The preemptions fell in six
20-second sampler intervals, 124 of them in one; the KV samples on either side of those intervals read 88.9–99.5%.
Nine such streams sit right at the pool's limit (Known limits 4). At that load each stream decoded at 1.7–3.7 tok/s
(completion tokens over decode time), time to first token was 93–1,042 s, and the longest stall inside one stream
was 536 s (a preemption recompute).

Long context: needle-in-a-haystack correct at 131,457 and 205,322 prompt tokens (TTFT 101.7 s and 174.4 s), an
118K-token prefix-cache scan with no drift beyond the limit, and image input all pass.

Greedy decode here is not bit-deterministic across batch sizes: the same prompt at a different concurrency can
produce a different answer, usually an equally valid one (Known limits 1 lists the known exceptions). That is why
the numbers above are medians and sustained means, not single runs.

## How to test it

`./start.sh` runs the READY checks before it reports success:

- a 14-fill greedy arithmetic sweep (the `17*23 = 391` family) from 33 to 33,634 prompt tokens, with `401` at the
  four documented flip fills (3, 12, 75, 150; see Known limits 1) passed as a known answer, so a clean engine reads
  14/14;
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

`start.sh` takes the name from `CONTAINER_NAME` in `.env` and prints it at launch. It only ever stops, removes or
replaces a container it created itself: every container it starts carries the labels `ornith.created-by=start.sh`,
`ornith.checkout=<an id of this checkout>` and `ornith.port=<port>`. `stop`:

1. refuses, and leaves the container exactly as it is, when a container of that name lacks the first two labels
   (another deployment, another checkout, or one started by an older `start.sh`), or is paused or restarting;
2. removes its own stopped container with a plain `docker rm`;
3. for its own running container, waits until the engine's `/metrics` reports no requests running or waiting
   (`vllm:num_requests_running` plus `vllm:num_requests_waiting`, read on the port in its `ornith.port` label, or
   on `PORT` when that label is missing; up to `STOP_DRAIN_SECONDS`, default 300), sends SIGTERM, waits up to
   `STOP_WAIT_SECONDS` (default 180) for the engine to exit, then removes the container. If `/metrics` does not
   answer or lacks either gauge, requests are still running or waiting, or the engine does not exit in time, it
   refuses and exits non-zero. Stop the clients first: a request that arrives between the last drain check and the
   SIGTERM is not covered.

It never sends SIGKILL and never runs `docker rm -f` or `docker stop`: on the `xe` driver, killing an engine with
GPU kernels in flight has reset the GPU engine. A plain `./start.sh` refuses the same way when a container of that
name exists and is not its own, or when its own is still running; `restart` is `stop`, then a fresh start.

The other subcommands are `status` (container, owner and API state), `logs` (`docker logs -f`) and `--print` (print
the full `docker run` line and the gate plan, run nothing). The printed line carries the three ownership labels
next to `ornith.config`; labels are bookkeeping, not serve configuration, and the serve arguments and container
environment are unchanged by them. Boots poll `/v1/models` for up to `READY_WAIT_SECONDS` (default 1800 s) — raise
it on a slower host.

## Known limits

1. **Answer flips on the fill probe.** Greedy answers on some prompts differ between fp8 and bf16 KV, and between
   batch sizes. The documented family is the `17*23 = 391` fill task. Fill 3 (with fp8 KV) and fill 75 (with bf16
   KV) have flipped every time so far; fills 12 and 150 flip intermittently:
   - fill 3 answers `401` with fp8 KV in every sweep on the shipping config so far (15 of 15), and `391` with bf16
     on the current build (4 of 4 bf16 sweeps; the one bf16 `401`, in 9 bf16 sweeps, came from an earlier build
     with a since-fixed oneDNN GEMM nondeterminism at 65–128-token prompts);
   - fill 12 answers `391` when it runs alone (34 of 34), but in 84 batched rounds it answered `401` four times and
     stopped after `17` twice (see Status);
   - fill 75 answers `401` with bf16 KV only (9 of 9 bf16 sweeps); with fp8 it has answered `391` every time (16 of 16);
   - fill 150 answered `401` once in 15 sweeps.

   The `401` answers are well-formed, with finite logprobs. The READY sweep accepts `401` at these four fills as a
   known answer; the flips are documented, not fixed. Each of those four entries was added after the failure it
   exempts, and the sweep matches the known answer as a substring rather than exactly; both are under review, and
   neither turns a recorded gate failure into a pass (see Status). A paired A/B (recall at 60K/118K/200K, 220 short
   checkable prompts, tool calls) put fp8 within noise of bf16 on every set; on the short A/B set fp8 scored
   219/220 against bf16's 220/220 — one fp8-only miss, judged within noise by the frozen rule. That A/B ran on the
   no-MTP build at `--max-num-seqs 32`, greedy, one request at a time, with thinking on, and it did not include the
   thinking-off fill-3 case above. It has not yet been repeated on the shipping config (MTP k=2,
   `--max-num-seqs 16`); that re-run is scheduled.
2. **MTP draft-KV cost.** The 2-token draft layer reserves its own KV: about 282K tokens, ~19.9% of the no-MTP
   pool, compared at the same `--max-num-seqs`:
   - at 16: 282,359 tokens (19.85%), the shipping 1,140,422 against a no-MTP reading whose boot log was not kept;
   - at 32: 282,272 tokens (19.87%), 1,138,508 against 1,420,780, both from boot logs.

   It buys ~1.4× single-stream (101.9 against 74.0 tok/s) and ~1.04× at 16 streams (447.9 against 430.6, both at
   `--max-num-seqs 32`; no-MTP 16-way at `--max-num-seqs 16` was not measured).
3. **The first-request guard.** A cold FULL_DECODE_ONLY engine dies with `EXL3XPU_OPEN_CAPTURE` if its first
   request is a long prefill: an XPU command-graph recording is left open on a worker's queue after the warmup
   captures, and the plugin's graph guard then stops the engine. First requests of 29,008 and 60,147 tokens both
   crashed (measured before MTP was enabled). Nothing between a 25-token first request and 29,008 tokens was
   tested, so the real threshold is unknown: treat any long first request as unsafe. What opens the recording is
   not identified. An earlier oneMKL explanation was ruled out (torch-xpu's fp16 matmul runs on oneDNN, not
   oneMKL), and the source of the leak is unproven. Three short requests first leave the queue clean, and every
   long request after that has been fine; `start.sh` sends them before its READY gates. The API is already
   listening while they run, so keep clients away until `start.sh` reports READY. The source fix is open.
4. **Long-context residency.** The pool in the boot log (1,140,422 tokens on the shipping config) is vLLM's
   estimate for maximum-length requests: 596 blocks of 2,112 tokens divided by the 137 blocks it counts for one
   262,144-token request, times 262,144. Underneath, those blocks (595 usable) are shared by the attention layers
   and the GDN/SSM state: every request holds its attention blocks plus 9–12 state blocks. Nine ~116K-token
   sessions (prompt plus 4K generated) need 585–603 blocks, so about 8 fit at once, not 16; the 60-minute soak ran
   9 and took its 157 preemptions there. 16 streams fit when each stays under about 59K tokens (steady decode), or
   53K while all 16 prefill.

## Status

Serving.

Two gate rows are recorded here honestly, from two different runs. The final full gate bundle (18:21Z boot)
holds verdict **NOT PASSED (full)**. Its `nan_sweep_concurrency` row ran 13/14: at concurrency 3 the fill-12 probe
answered `401`. The rule in force failed any round whose probe did not answer `391`. The KNOWN `12:401` entry and
the round-loop change that would pass that round were both added after failures; under this project's rule that
no gate rule is relaxed after a failure, neither is accepted, so the row stands as a failure. `nan_sweep_strict`
passed 14/14 in the same bundle. Separately, a partial re-run (20:17Z boot) of the concurrency row failed on a
different signature: at concurrency 14 the fill probe stopped after `17` (`17<eos>`), a wrong, truncated answer.
That was its second sighting on the shipping config; the first was at 16:22Z the same day. It was accepted as a
known risk at cutover. An 800-prompt early-stop test run after cutover reported no truncations at concurrency 14,
but that test sent bare `a*b = ` prompts rather than the fill probe and did not count 3-token replies, so it could
not have seen this case. It is open.

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
- replacing the fill-12 probe with one whose fp8 answer margin has been measured (fill 12's fp8 margin never was,
  so calling it a near-tie is unproven);
- the `17<eos>` truncation above;
- the fp8/bf16 A/B on the shipping config (Known limits 1);
- a first end-to-end launch by `start.sh` itself (see Quick start).

Details: [docs/STATUS.md](docs/STATUS.md).

## Credits

- [trellis-serve](https://github.com/sybil-solutions/trellis-serve) (MIT): the B70 EXL3 kernels, the vLLM and
  SGLang plugins and the split rules this builds on. Its `LICENSE` and `NOTICE` are kept unchanged.
- [ornith-ai](https://huggingface.co/ornith-ai): the Ornith-1.5-122B base model, and the chat template that
  `templates/ornith-chat-template.jinja` is adapted from (MIT, as declared on the model card).
- [Mia-AiLab](https://huggingface.co/Mia-AiLab): the EXL3 quantization recipe and the weights served here.
- [exllamav3](https://github.com/turboderp-org/exllamav3): the EXL3 format, the reference reconstruction and the
  tensor-parallel layouts.
- [vLLM](https://github.com/vllm-project/vllm) and Intel's XPU kernels and oneAPI toolchain.

## License

MIT, inherited from trellis-serve; see [LICENSE](LICENSE) and [NOTICE](NOTICE).
