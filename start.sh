#!/usr/bin/env bash
# start.sh: build the EXL3 XPU kernels for the runtime image in .env, then launch the 4-card vLLM lane.
# The lane itself is still being built (see README "Status"); until it lands this script stops after the kernels.
set -euo pipefail
cd "$(dirname "$0")"
[ -f .env ] || { echo "No .env: copy .env.example to .env and fill it in." >&2; exit 1; }
set -a; . ./.env; set +a
for v in MODEL_DIR RUNTIME_IMAGE TOOLCHAIN_IMAGE BUILD_DIR; do
  [ -n "${!v:-}" ] || { echo ".env: $v is empty" >&2; exit 1; }
done
[ -f "$MODEL_DIR/config.json" ] || { echo "MODEL_DIR has no config.json: $MODEL_DIR" >&2; exit 1; }

./scripts/build-kernels.sh

echo "Kernels built into $BUILD_DIR/lib. The 4-card serving lane is not in this repo yet (README: Status)."
exit 3
