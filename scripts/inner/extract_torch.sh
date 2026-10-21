#!/usr/bin/env bash
# Runs INSIDE the runtime image (CPU only, no network, no devices).
# Copies its torch headers + the 5 libraries the EXL3 builds link against to /build/torch, verifies the copy,
# and records where torch lives in this image (the built .so files get that RUNPATH).
set -euo pipefail
trap 'chown -R "$OWNER" /build/torch 2>/dev/null || true' EXIT
T=$(python3 -c "import torch,os;print(os.path.dirname(torch.__file__))")
LIBS="libc10.so libtorch.so libtorch_cpu.so libc10_xpu.so libtorch_xpu.so"
rm -rf /build/torch && mkdir -p /build/torch/lib
cp -a "$T/include" /build/torch/
for L in $LIBS; do cp -a "$T/lib/$L" /build/torch/lib/; done
(cd "$T/lib" && sha256sum $LIBS) > /build/torch/SHA256.orig
(cd /build/torch/lib && sha256sum -c --quiet /build/torch/SHA256.orig)
{
  echo "torch_lib_dir_in_image=$T/lib"
  python3 -c "import torch;print('torch', torch.__version__, '| xpu', torch.version.xpu, '| cxx11abi', torch.compiled_with_cxx11_abi())"
} | tee /build/torch/RUNTIME_INFO
