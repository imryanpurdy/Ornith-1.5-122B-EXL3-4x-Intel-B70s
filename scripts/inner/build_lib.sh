#!/usr/bin/env bash
# Runs INSIDE the toolchain image (oneAPI icpx; CPU only, no network, no devices).
# Usage: build_lib.sh ext|moe     (extra compiler flags in $FLAGS)
# Same icpx line as trellis-serve's xpu/scripts/build_ext.sh / build_moe.sh, except:
#   - torch comes from the runtime image (/build/torch), and the RUNPATH is torch's lib dir INSIDE that image;
#   - no oneDNN for _C.so (the runtime images ship no libdnnl), and the compiler output is not filtered.
set -o pipefail
which=${1:?ext|moe}
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
set -u
trap 'chown -R "$OWNER" /build/lib 2>/dev/null || true' EXIT
T=/build/torch
RT=$(sed -n 's/^torch_lib_dir_in_image=//p' $T/RUNTIME_INFO)
case "$which" in
  ext) SRC=csrc/exl3_ops.sycl; OUT=/build/lib/_C.so ;;
  moe) SRC=csrc/exl3_moe.sycl; OUT=/build/lib/_moe.so ;;
  *) echo "unknown target $which"; exit 2 ;;
esac
mkdir -p /build/lib && rm -f "$OUT"
cd /src
cmd=(icpx -fsycl -fsycl-targets=spir64 -O3 -ffast-math -fPIC -std=c++17 -shared
     -fsycl-device-code-split=per_kernel -D_GLIBCXX_USE_CXX11_ABI=1 ${FLAGS:-}
     -I csrc -I$T/include -I$T/include/torch/csrc/api/include
     -x c++ $SRC -x none -o "$OUT"
     -L$T/lib -Wl,-rpath,$RT -lc10 -ltorch -ltorch_cpu -lc10_xpu -ltorch_xpu)
icpx --version | head -1
printf '%q ' "${cmd[@]}"; echo
t0=$(date +%s)
"${cmd[@]}" 2>&1
rc=$?
echo "icpx exit=$rc wall_s=$(( $(date +%s) - t0 ))"
[ $rc -eq 0 ] && sha256sum "$OUT"
exit $rc
