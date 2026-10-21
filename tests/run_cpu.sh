#!/usr/bin/env bash
# CPU test suite (tests/cpu) inside the vLLM XPU serving image: no device, no network, repo and model read-only.
#   IMAGE=<vllm-0.30-xpu-image> MODEL_DIR=<Ornith-1.5-122B-EXL3 checkpoint dir> tests/run_cpu.sh [pytest args]
# Without MODEL_DIR the checkpoint tests are skipped. Peak RAM ~20 GiB (the lm_head reconstruction).
set -euo pipefail
: "${IMAGE:?set IMAGE to the vLLM 0.30 XPU serving image}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
ARGS=(--rm --network none --cpus "${CPUS:-16}" --memory "${MEMORY:-48g}" -v "$REPO:/src:ro" -e PYTHONDONTWRITEBYTECODE=1)
if [[ -n "${NAME:-}" ]]; then ARGS+=(--name "$NAME"); fi
if [[ -n "${MODEL_DIR:-}" ]]; then ARGS+=(-v "$MODEL_DIR:/model:ro" -e EXL3_MODEL_DIR=/model); fi
exec docker run "${ARGS[@]}" --entrypoint "${PYTHON:-/opt/venv/bin/python}" "$IMAGE" \
    -m pytest -p no:cacheprovider -q -W ignore /src/tests/cpu "$@"
