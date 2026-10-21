# Tests

## `tests/cpu`: CPU only (milestone M2a and the graph guard)

Runs inside the vLLM 0.30 XPU serving image (torch 2.13.0+xpu) with no device, no network, and the repo and model
mounted read-only. One command:

```
IMAGE=<vllm-0.30-xpu-image> MODEL_DIR=<Ornith-1.5-122B-EXL3 checkpoint dir> tests/run_cpu.sh
```

which is

```
docker run --rm --network none --cpus 16 --memory 48g -v "$PWD:/src:ro" -v <model-dir>:/model:ro \
  -e EXL3_MODEL_DIR=/model -e PYTHONDONTWRITEBYTECODE=1 --entrypoint /opt/venv/bin/python <image> \
  -m pytest -p no:cacheprovider -q -W ignore /src/tests/cpu
```

Without `MODEL_DIR` the checkpoint tests skip. The checkpoint tests read only the tensors they need, by byte
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
