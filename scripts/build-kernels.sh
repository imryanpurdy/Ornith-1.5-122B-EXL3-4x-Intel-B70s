#!/usr/bin/env bash
# build-kernels.sh: build the EXL3 XPU op libraries against the runtime image's torch, on the CPU.
#   _C.so   from xpu/csrc/exl3_ops.sycl  (torch library exl3xpu_C: EXL3 linear, GEMV/DPAS, reconstruct)
#   _moe.so from xpu/csrc/exl3_moe.sycl  (torch library exl3xpu_moe: grouped EXL3 MoE)
#
# Steps (each in a throwaway container with no GPU, no network):
#   torch: copy the runtime image's torch headers + the 5 libraries the build links against into $BUILD_DIR/torch,
#          and record where torch lives inside that image (the .so files get that RUNPATH).
#   ext / moe: compile with icpx from $TOOLCHAIN_IMAGE (the runtime images ship no compiler).
#   verify: load both libraries into the runtime image's torch and list the registered ops.
# Usage: scripts/build-kernels.sh [all|torch|ext|moe|verify]   (reads .env)
set -euo pipefail
cd "$(dirname "$0")/.."
REPO=$(pwd)
[ -f .env ] && { set -a; . ./.env; set +a; }
: "${RUNTIME_IMAGE:?set RUNTIME_IMAGE in .env}" "${TOOLCHAIN_IMAGE:?set TOOLCHAIN_IMAGE in .env}"
BUILD_DIR=${BUILD_DIR:-./build}
mkdir -p "$BUILD_DIR" && B=$(cd "$BUILD_DIR" && pwd)
UIDGID="$(id -u):$(id -g)"
RUN=(docker run --rm --network none --cpus "$(nproc)" --entrypoint bash -e "OWNER=$UIDGID")

step_torch() {
  "${RUN[@]}" --name ornith-exl3-build-torch -v "$B":/build -v "$REPO/scripts/inner":/inner:ro "$RUNTIME_IMAGE" \
    -lc 'bash /inner/extract_torch.sh'
}
step_lib() {   # ext|moe
  [ -f "$B/torch/RUNTIME_INFO" ] || step_torch
  local flags=""; [ "$1" = ext ] && flags="-DEXL3_ALL_CODEBOOKS"
  "${RUN[@]}" --name "ornith-exl3-build-$1" -v "$B":/build -v "$REPO/xpu":/src:ro -v "$REPO/scripts/inner":/inner:ro \
    -e "FLAGS=$flags" "$TOOLCHAIN_IMAGE" -lc "bash /inner/build_lib.sh $1" | tee "$B/build-$1.txt"
}
step_verify() {
  "${RUN[@]}" --name ornith-exl3-build-verify -v "$B":/build:ro "$RUNTIME_IMAGE" -lc '
    python3 -c "
import torch
for lib in (\"/build/lib/_C.so\", \"/build/lib/_moe.so\"):
    torch.ops.load_library(lib)
ops = sorted(n for n in torch._C._dispatch_get_all_op_names() if n.startswith((\"exl3xpu_C::\", \"exl3xpu_moe::\")))
print(len(ops), \"ops registered:\", \" \".join(ops))
print(\"exl3_supported mul1 K=4,5,6:\", [bool(torch.ops.exl3xpu_C.exl3_supported(k, 2)) for k in (4, 5, 6)])
"'
}

case "${1:-all}" in
  all) step_torch; step_lib ext; step_lib moe; step_verify ;;
  torch) step_torch ;; ext) step_lib ext ;; moe) step_lib moe ;; verify) step_verify ;;
  *) echo "usage: $0 [all|torch|ext|moe|verify]" >&2; exit 2 ;;
esac
