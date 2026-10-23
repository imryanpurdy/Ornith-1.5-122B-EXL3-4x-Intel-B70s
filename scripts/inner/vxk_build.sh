#!/usr/bin/env bash
# Runs INSIDE the toolchain image (oneAPI icpx; CPU only, no network, no devices).
# Build + install the configured tree. -j2 is deliberate: higher job counts were OOM-killed inside a 48 GiB cap
# (chunk_prefill head128/192, gdn chunk_gated_delta_rule, grouped_gemm xe_2, one paged_decode TU).
set -o pipefail
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
export PYTHONPATH=/ts${PYTHONPATH:+:$PYTHONPATH}
export LD_LIBRARY_PATH=/ts/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
trap 'chown -R "$OWNER" /cb /out 2>/dev/null || true' EXIT
cd /cb || exit 1
t0=$(date +%s)
make -j"${JOBS:-2}"
rc=$?
t1=$(date +%s)
echo "build_rc=$rc wall_s=$((t1 - t0))"
if [ "$rc" = 0 ]; then
  make install
  echo "install_rc=$?"
  find /out -type f -name '*.so' | sort
fi
exit $rc
