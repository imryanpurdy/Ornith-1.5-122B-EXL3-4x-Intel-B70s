#!/usr/bin/env bash
# Runs INSIDE the toolchain image (oneAPI icpx; CPU only, no network, no devices).
# Configure for the vllm_xpu_kernels rebuild: torch/python come from /ts (extracted from the runtime image),
# oneDNN and sycl-tla are mounted at /deps so FetchContent never touches the network, the build tree is /cb.
set -o pipefail
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
# configure imports torch through the interpreter it finds (/usr/bin/python3): torch lives in /ts, not in this image
export PYTHONPATH=/ts${PYTHONPATH:+:$PYTHONPATH}
export LD_LIBRARY_PATH=/ts/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
# FetchContent clones oneDNN/sycl-tla from the bind-mounted /deps trees, which the container runs as root:
# git refuses those clones as dubious ownership unless every directory is marked safe.
git config --global --add safe.directory '*'
trap 'chown -R "$OWNER" /cb 2>/dev/null || true' EXIT
cd /cb || exit 1
cmake /src -G "Unix Makefiles" \
  -DCMAKE_BUILD_TYPE=Release -DVLLM_TARGET_DEVICE=xpu \
  -DCMAKE_TOOLCHAIN_FILE=/src/cmake/toolchain.cmake \
  -DVLLM_PYTHON_EXECUTABLE=/usr/bin/python3 -DVLLM_PYTHON_PATH=/ts \
  -DPython_INCLUDE_DIR=/ts/python3.12 -DPython_LIBRARY=/ts/lib/libpython3.12.so.1.0 \
  -DCMAKE_INSTALL_PREFIX=/out \
  -DVLLM_PAGED_DECODE_CONFIG=paged_decode_default.conf \
  -DVLLM_CHUNK_PREFILL_CONFIG=chunk_prefill_default.conf \
  -DVLLM_CUTLASS_SRC_DIR=/deps/sycl-tla \
  -DONEDNN_GIT_REPO=/deps/oneDNN -DONEDNN_GIT_TAG="${ONEDNN_REV:?}" \
  -DFETCHCONTENT_BASE_DIR=/cb/_deps
rc=$?
echo "cfg_rc=$rc CMPLR=$CMPLR_ROOT"
grep -E "Generated (paged_decode|chunk_prefill)_kernel_template" CMakeFiles/CMakeOutput.log 2>/dev/null || true
exit $rc
