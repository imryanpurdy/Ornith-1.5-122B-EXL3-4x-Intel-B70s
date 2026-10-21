# Tests

## `tests/cpu`: CPU only (milestone M2a, the graph guard, the vLLM plugin)

Runs inside the vLLM 0.30 XPU serving image (torch 2.13.0+xpu) with no device, no network, and the repo, models
and kernel build mounted read-only. One command:

```
IMAGE=<vllm-0.30-xpu-image> MODEL_DIR=<Ornith-1.5-122B-EXL3 checkpoint dir> MINI_DIR=<Ornith-mini dir> \
  LIB_DIR=<dir with the built _C.so and _moe.so> tests/run_cpu.sh
```

which is

```
docker run --rm --network none --cpus 16 --memory 48g -v "$PWD:/src:ro" -e PYTHONDONTWRITEBYTECODE=1 \
  -v <model-dir>:/model:ro -e EXL3_MODEL_DIR=/model -v <mini-dir>:/mini:ro -e EXL3_MINI_DIR=/mini \
  -v <lib-dir>:/exl3lib:ro -e EXL3_LIB_DIR=/exl3lib --entrypoint /opt/venv/bin/python <image> \
  -m pytest -p no:cacheprovider -q -W ignore /src/tests/cpu
```

Tests that need a missing directory skip (`MODEL_DIR`: the 122B checkpoint; `MINI_DIR`: Ornith-mini made by
`tools/make_ornith_mini.py`; `LIB_DIR`: the kernel build, for the host ops the plugin calls at load time:
`exl3_supported`, `blob_bytes`, `moe_plan`, `moe_info`).

The plugin tests build vLLM's real model in `plugin_harness.py`, run in a fresh interpreter per command: vLLM binds
its platform at import, and the image has no XPU device here, so the harness installs the XPU platform first
(its `current_device()` answers "cpu"), registers the plugin through `register()`, and fakes vLLM's TP / EP process
groups for each rank (no collective is ever issued while a model is built or loaded). The checkpoint tests read only the tensors they need, by byte
range. Peak RAM is about 20 GiB (the float64 reconstruction of `lm_head`). Measured numbers are printed in the
"exl3xpu recorded numbers" section of the summary. If `EXL3_TEST_RESULTS=<file>` is set, they are also appended
to that file as JSON lines.

| file | what |
|---|---|
| `test_tp_plan.py` | slice plan (`xpu/exl3xpu/tp.py`) vs the TP4 table of the design, vs vLLM 0.30's own weight loaders at TP 1/2/4 (index-coded tensors through unquantized layers), refusals, synthetic narrowing |
| `test_tp_real.py` | real checkpoint tensors, every dense kind + MTP + routed experts, TP 1/2/4, every rank: slices reassemble exactly, per-rank weight == TP1 block bit for bit, row-split partial sums within the bf16 bound |
| `test_tp_counts.py` | per-rank EXL3 tensor counts for the plugin's `EXL3XPU_LOADED` check, from the checkpoint index |
| `test_graph_guard.py` | every branch of the graph guard G1-G4 and the self-test switch, with fakes and on the image's real torch / vLLM classes |
| `test_graph_guard_sources.py` | the hook points exist in the image with the expected source |
| `test_check_guard_log.py` | the log checker below |
| `test_plugin_routing.py` | quant method of every module of vLLM's Qwen3.5-MoE model and MTP drafter (122B with vision, mini), TP 1/2/4, every rank, EP on/off; create_weights shapes; arena sizes per card |
| `test_plugin_load.py` | the real load path (`model.load_weights` + `process_weights_after_loading`): EXL3XPU_LOADED counts on the 122B (every tensor, meta payloads) and the mini (real tensors), every rank; mini params and expert blobs equal the slice plan's |
| `test_plugin_moe.py` | `Exl3MoEMethod.apply` on loaded mini layers 0 and 3 with a float64 kernel stand-in (`moe_ref.py`): id mapping, -1 padding, dead experts, 1,024-row chunks, the eager dump, EP / MoE-TP partials and their sums |
| `test_plugin_refusals.py` | everything the plugin refuses (routing, dense loaders, expert loader, count mismatch) and `register()` per environment (`register_probe.py`) |
| `test_plugin_units.py` | id mapping, chunking, blob sizes, name normalisation, `EXL3_VLLM_PATCHES`, loadcheck's batch bookkeeping |
| `test_check_loaded_log.py` | the EXL3XPU_LOADED log checker below |
| `plugin_harness.py`, `moe_ref.py`, `register_probe.py` | model builder / loader driver; float64 grouped-MoE stand-in and blob unpacker; register() in a fresh process |
| `st_range.py`, `exact_ref.py` | byte-range safetensors reader; exact float64 EXL3 reconstruction |

## `tests/xpu`: needs a card (not part of the CPU run)

- `test_graph_guard_xpu.py`: G2 and G1 on one real card (capture/replay during warmup, refusal while serving
  leaves no recording or stream behind, open-recording detection).
- `check_guard_log.py capture-gate|selftest`: checks a container log against the capture gate (one
  `EXL3XPU_WARMUP_DONE` per rank, `runner=XPUModelRunnerV2`, `captures == graphs == plan`, all five guard hooks
  installed on every rank, the engine config's resolved `cudagraph_mode` equal to `--cudagraph-mode`, no guard
  trips; e.g. `capture-gate --log L --ranks 4 --expect-graphs 8 --cudagraph-mode FULL_DECODE_ONLY`) and the
  one-card guard self-test (`EXL3_GUARD_SELFTEST=1`: `EXL3XPU_CAPTURE_AFTER_WARMUP` then `EngineDeadError`
  within 60 s, no xe reset in the kernel log).
- `m2_moe_parity.py`: milestone M2b on one card (DESIGN section 7: checks i-xi and the dense determinism probe),
  Ornith-mini layers 0 and 3; writes a JSON result and `splits-L<n>.json` (microbench). `--cpu-dryrun` runs the
  harness without a device (the kernel replaced by `tests/cpu/moe_ref.BlobKernel`), to check its plumbing.
- `check_loaded_log.py --log L --ranks N --dense D --experts E [--drafter-dense D2 --drafter-experts E2]`: every rank
  logged exactly one EXL3XPU_LOADED line per model with the expected counts (M2c: 104 / 12,288 at TP1; M4: 1,204 /
  36,864 x 4 at EP4, drafter 36 / 768).
