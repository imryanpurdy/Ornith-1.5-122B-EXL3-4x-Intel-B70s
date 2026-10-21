#!/usr/bin/env bash
# CPU test suite (tests/cpu) inside the vLLM XPU serving image: no device, no network, repo and models read-only.
#   IMAGE=<vllm-0.30-xpu-image> MODEL_DIR=<Ornith-1.5-122B-EXL3 checkpoint dir> MINI_DIR=<Ornith-mini dir> \
#   LIB_DIR=<dir with the built _C.so and _moe.so> tests/run_cpu.sh [pytest args]
# Tests that need a missing directory are skipped. Peak RAM ~20 GiB (the lm_head reconstruction).
set -euo pipefail
: "${IMAGE:?set IMAGE to the vLLM 0.30 XPU serving image}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
ARGS=(--rm --network none --cpus "${CPUS:-16}" --memory "${MEMORY:-48g}" -v "$REPO:/src:ro" -e PYTHONDONTWRITEBYTECODE=1)
if [[ -n "${NAME:-}" ]]; then ARGS+=(--name "$NAME"); fi
if [[ -n "${MODEL_DIR:-}" ]]; then ARGS+=(-v "$MODEL_DIR:/model:ro" -e EXL3_MODEL_DIR=/model); fi
if [[ -n "${MINI_DIR:-}" ]]; then ARGS+=(-v "$MINI_DIR:/mini:ro" -e EXL3_MINI_DIR=/mini); fi
if [[ -n "${LIB_DIR:-}" ]]; then ARGS+=(-v "$LIB_DIR:/exl3lib:ro" -e EXL3_LIB_DIR=/exl3lib); fi
exec docker run "${ARGS[@]}" --entrypoint "${PYTHON:-/opt/venv/bin/python}" "$IMAGE" \
    -m pytest -p no:cacheprovider -q -W ignore /src/tests/cpu "$@"
