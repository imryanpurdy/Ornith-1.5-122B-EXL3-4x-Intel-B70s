#!/usr/bin/env bash
# build-vxk.sh: rebuild vllm_xpu_kernels 0.1.14.1 with the paged-decode instantiation Ornith needs at TP4, laid
# out so the container can bind-mount it over the image's package (docs/runtime-image.md §3).
#
#   src     clone vllm-project/vllm-xpu-kernels at VXK_REV into $VXK/src, add the one config line
#   deps    clone oneDNN + sycl-tla at the pinned revisions into $VXK/deps (build containers run --network none)
#   torch   copy the runtime image's torch + python headers/libs into $VXK/torch-site
#   cfg     cmake configure in a throwaway container (no GPU, no network, no devices)
#   build   make -j$JOBS && make install into $VXK/vllm_xpu_kernels
#   rpath   patchelf --set-rpath '$ORIGIN' on every .so (without it `import vllm_xpu_kernels` cannot find
#           libgrouped_gemm_xe_default.so)
#   layout  copy the package's python files from the source tree and libgdn_index64.so from the image, so the
#           overlay has the same file set as the image's package
#   verify  assert the qgroup16/head256/page64 template is in the attn library and that the package imports
#
# Usage: scripts/build-vxk.sh [all|src|deps|torch|cfg|build|rpath|layout|verify]   (reads .env)
#   all runs every step, cached: a step whose outputs are newer than its inputs is skipped.
# Env: RUNTIME_IMAGE, TOOLCHAIN_IMAGE (from .env), BUILD_DIR, VXK_DIR (default $BUILD_DIR/vxk), JOBS (default 2).
#
# CPU only. Every container is --rm, --network none, no devices. Expect ~70 min for build on 2 jobs: -j8 and -j16
# were OOM-killed inside a 48 GiB cap on the rig this was developed on.
set -euo pipefail
cd "$(dirname "$0")/.."
REPO=$(pwd)
[ -f .env ] && { set -a; . ./.env; set +a; }
: "${RUNTIME_IMAGE:?set RUNTIME_IMAGE in .env}" "${TOOLCHAIN_IMAGE:?set TOOLCHAIN_IMAGE in .env}"
BUILD_DIR=${BUILD_DIR:-./build}
VXK=${VXK_DIR:-$BUILD_DIR/vxk}
mkdir -p "$VXK" && V=$(cd "$VXK" && pwd)
JOBS=${JOBS:-2}

# Pinned inputs. The kernel source is the same upstream release the image ships (0.1.14.1); the config patch is
# one added line; the dependencies are the revisions the image's own build used.
VXK_REV=6d92b1bfbf32767ecda8e819613eb151e70030ad
ONEDNN_REV=0e2a5bfeef1bfbffc3137464606540233086ce9b
SYCLTLA_REV=87f6850680a580654b9ea2c80dbc01aeb36ad231
PATCHELF_VER=0.18.0
PATCHELF_SHA256=b699cb82300dfd08ea6c895d8829b46bea970cfa737fa34188d8256ca89404e5
CONF_REL=csrc/xpu/attn/kernel_configs/paged_decode_default.conf
CONF_LINE='16,256,64,false,false,false'
UIDGID="$(id -u):$(id -g)"
RUN=(docker run --rm --network none --cpus "$(nproc)" --memory 48g --entrypoint bash -e "OWNER=$UIDGID")

say() { printf '\033[1;34m[INFO]\033[0m  %s\n' "$*"; }
ok()  { printf '\033[1;32m[ OK ]\033[0m  %s\n' "$*"; }
die() { printf '\033[1;31m[ERR ]\033[0m  %s\n' "$*" >&2; exit 1; }

newer_than() {   # newer_than OUT SRC... : true when OUT exists and is newer than every SRC
  local out=$1; shift
  [ -e "$out" ] || return 1
  local f
  for f in "$@"; do
    [ -e "$f" ] || continue
    [ "$f" -nt "$out" ] && return 1
  done
  return 0
}

clone_at() {   # clone_at DIR URL REV  (shallow: fetch the one commit, no history)
  local dir=$1 url=$2 rev=$3
  if [ -d "$dir/.git" ] && [ "$(git -C "$dir" rev-parse HEAD 2>/dev/null || true)" = "$rev" ]; then
    ok "cached $dir at $rev"
    return 0
  fi
  rm -rf "$dir"
  mkdir -p "$dir"
  git -C "$dir" init --quiet
  git -C "$dir" remote add origin "$url"
  git -C "$dir" fetch --quiet --depth 1 origin "$rev"
  git -C "$dir" checkout --quiet FETCH_HEAD
  [ "$(git -C "$dir" rev-parse HEAD)" = "$rev" ] || die "$dir is not at $rev"
  ok "fetched $url at $rev"
}

step_src() {
  say "=== src: vllm-xpu-kernels @ $VXK_REV + the one-line paged-decode config ==="
  clone_at "$V/src" https://github.com/vllm-project/vllm-xpu-kernels.git "$VXK_REV"
  local conf="$V/src/$CONF_REL"
  [ -f "$conf" ] || die "$CONF_REL missing in the source tree"
  if ! grep -qx "$CONF_LINE" "$conf"; then
    # insert after the Qwen3-Next head-256/page-64 entry, which is unique in the file
    [ "$(grep -c '^8,256,64,false,false,false$' "$conf")" = 1 ] \
      || die "anchor line 8,256,64,false,false,false is not unique in $CONF_REL; refusing to guess where to patch"
    sed -i "/^8,256,64,false,false,false\$/a $CONF_LINE" "$conf"
  fi
  [ "$(grep -c "^$CONF_LINE\$" "$conf")" = 1 ] || die "$CONF_REL does not carry exactly one $CONF_LINE"
  ok "$CONF_REL carries $CONF_LINE (one added line, nothing else)"
}

step_deps() {
  say "=== deps: oneDNN @ $ONEDNN_REV, sycl-tla @ $SYCLTLA_REV ==="
  mkdir -p "$V/deps"
  clone_at "$V/deps/oneDNN" https://github.com/oneapi-src/oneDNN.git "$ONEDNN_REV"
  clone_at "$V/deps/sycl-tla" https://github.com/NVIDIA/cutlass.git "$SYCLTLA_REV"
}

step_torch() {
  say "=== torch: headers + libs + python headers from $RUNTIME_IMAGE ==="
  if newer_than "$V/torch-site/RUNTIME_INFO" "$REPO/.env"; then ok "cached $V/torch-site"; return 0; fi
  "${RUN[@]}" --name ornith-vxk-torch -v "$V":/v -v "$REPO/scripts/inner":/inner:ro "$RUNTIME_IMAGE" \
    -lc 'bash /inner/vxk_torch.sh'
  [ -f "$V/torch-site/RUNTIME_INFO" ] || die "torch extraction produced no RUNTIME_INFO"
  ok "torch-site: $(head -2 "$V/torch-site/RUNTIME_INFO" | tr '\n' ' ')"
}

step_cfg() {
  say "=== cfg: cmake configure (icpx 2026.1 from $TOOLCHAIN_IMAGE) ==="
  [ -d "$V/src" ] || die "no source tree; run: $0 src"
  [ -d "$V/deps/oneDNN" ] && [ -d "$V/deps/sycl-tla" ] || die "no deps; run: $0 deps"
  [ -f "$V/torch-site/RUNTIME_INFO" ] || die "no torch-site; run: $0 torch"
  mkdir -p "$V/cmake-build" "$V/vllm_xpu_kernels"
  "${RUN[@]}" --name ornith-vxk-cfg -e "ONEDNN_REV=$ONEDNN_REV" \
    -v "$V/src":/src:ro -v "$V/deps":/deps:ro -v "$V/torch-site":/ts:ro \
    -v "$V/cmake-build":/cb -v "$REPO/scripts/inner":/inner:ro "$TOOLCHAIN_IMAGE" \
    -lc 'bash /inner/vxk_cfg.sh' 2>&1 | tee "$V/cfg.log"
  grep -q 'paged_decode_kernel_template sources' "$V/cfg.log" \
    || die "configure did not report the paged-decode templates (see $V/cfg.log)"
  ok "$(grep -o 'Generated paged_decode_kernel_template sources: [0-9]* files' "$V/cfg.log" | tail -1)"
}

step_build() {
  say "=== build: make -j$JOBS && make install (this is the long step, ~70 min at -j2) ==="
  [ -f "$V/cmake-build/CMakeCache.txt" ] || die "not configured; run: $0 cfg"
  "${RUN[@]}" --name ornith-vxk-build -e "JOBS=$JOBS" \
    -v "$V/cmake-build":/cb -v "$V":/out -v "$V/torch-site":/ts:ro \
    -v "$V/src":/src:ro -v "$V/deps":/deps:ro -v "$REPO/scripts/inner":/inner:ro "$TOOLCHAIN_IMAGE" \
    -lc 'bash /inner/vxk_build.sh' 2>&1 | tee "$V/build.log"
  grep -q 'build_rc=0' "$V/build.log" || die "build failed (see $V/build.log)"
  # `make install` only installs the five python extension modules; the kernel libraries stay in the build tree.
  # The image's package carries them, and the overlay replaces the whole package directory, so copy them across.
  local f
  for f in libattn_kernels_xe_2.so libgdn_attn_kernels_xe_2.so libgrouped_gemm_xe_2.so \
           libgrouped_gemm_xe_default.so libmhc_kernels_xe_2.so libmqa_logits_kernels_xe_2.so; do
    [ -f "$V/cmake-build/$f" ] || die "$f was not built"
    cp -a "$V/cmake-build/$f" "$V/vllm_xpu_kernels/$f"
  done
  for f in _C.abi3.so _moe_C.abi3.so _vllm_fa2_C.abi3.so _xpu_C.abi3.so xpumem_allocator.abi3.so \
           libattn_kernels_xe_2.so libgdn_attn_kernels_xe_2.so libgrouped_gemm_xe_2.so \
           libgrouped_gemm_xe_default.so libmhc_kernels_xe_2.so libmqa_logits_kernels_xe_2.so; do
    [ -f "$V/vllm_xpu_kernels/$f" ] || die "$f was not installed"
  done
  ok "installed into $V/vllm_xpu_kernels"
}

step_rpath() {
  say "=== rpath: patchelf --set-rpath '\$ORIGIN' on every .so ==="
  if [ ! -x "$V/tools/bin/patchelf" ]; then
    say "fetching patchelf $PATCHELF_VER (build-time tool, not shipped)"
    mkdir -p "$V/tools"
    curl -sSL -o "$V/tools/patchelf.tar.gz" \
      "https://github.com/NixOS/patchelf/releases/download/$PATCHELF_VER/patchelf-$PATCHELF_VER-x86_64.tar.gz"
    tar xzf "$V/tools/patchelf.tar.gz" -C "$V/tools"
    echo "$PATCHELF_SHA256  $V/tools/bin/patchelf" | sha256sum -c --quiet - \
      || die "patchelf binary does not match the pinned sha256"
  fi
  local f
  for f in "$V"/vllm_xpu_kernels/*.so; do
    "$V/tools/bin/patchelf" --set-rpath '$ORIGIN' "$f" || die "patchelf failed on $f"
  done
  ok "rpath \$ORIGIN on $(ls "$V"/vllm_xpu_kernels/*.so | wc -l) libraries"
}

step_layout() {
  say "=== layout: python package files + libgdn_index64.so (no public source, taken from the image) ==="
  local sp="$V/src/vllm_xpu_kernels"
  [ -d "$sp" ] || die "no $sp; run: $0 src"
  cp -a "$sp"/__init__.py "$sp"/flash_attn_interface.py "$sp"/fused_moe_interface.py "$sp"/moe_utils.py \
        "$sp"/rotary.py "$sp"/.gitkeep "$V/vllm_xpu_kernels/"
  mkdir -p "$V/vllm_xpu_kernels/quantization"
  cp -a "$sp"/quantization/__init__.py "$sp"/quantization/_quantize_convert.py "$V/vllm_xpu_kernels/quantization/"
  # the image's package carries libgdn_index64.so, which this source tree does not build; the overlay replaces the
  # whole package directory, so it has to keep it.
  "${RUN[@]}" --name ornith-vxk-gdn -v "$V/vllm_xpu_kernels":/out "$RUNTIME_IMAGE" -lc '
    set -e
    cp -a /opt/venv/lib/python3.12/site-packages/vllm_xpu_kernels/libgdn_index64.so /out/
    chown "$OWNER" /out/libgdn_index64.so
    echo "libgdn_index64.so $(sha256sum /out/libgdn_index64.so | cut -d" " -f1)"
  '
  ok "$(ls "$V/vllm_xpu_kernels" | wc -l) entries in the overlay"
}

step_verify() {
  say "=== verify: the qgroup16 instantiation is present and the package imports ==="
  local attn="$V/vllm_xpu_kernels/libattn_kernels_xe_2.so"
  [ -f "$attn" ] || die "no $attn"
  local n_tpl n_q16
  n_tpl=$(strings -a "$attn" | grep -o 'paged_decode_kernel_template_[a-z0-9_]*\.cpp' | sort -u | wc -l)
  n_q16=$(strings -a "$attn" | grep -c 'paged_decode_kernel_template_q16_h256_p64' || true)
  [ "$n_q16" -ge 1 ] || die "the q16/head256/page64 decode template is not in $attn"
  ok "paged-decode templates: $n_tpl (the image ships 32; 33 with the added line), q16_h256_p64 present"
  "${RUN[@]}" --name ornith-vxk-import -v "$V/vllm_xpu_kernels":/overlay:ro "$RUNTIME_IMAGE" -lc '
    set -e
    P=/opt/venv/lib/python3.12/site-packages/vllm_xpu_kernels
    mv "$P" "$P.image" && cp -a /overlay "$P"
    python3 -c "
import vllm_xpu_kernels, vllm_xpu_kernels.flash_attn_interface, vllm_xpu_kernels.fused_moe_interface
print(\"IMPORT OK\", vllm_xpu_kernels.__file__)
"
  '
  ok "import check passed (loading only: no device is touched)"
}

case "${1:-all}" in
  all)
    step_src; step_deps; step_torch; step_cfg
    if newer_than "$V/vllm_xpu_kernels/libattn_kernels_xe_2.so" \
         "$V/src/$CONF_REL" "$V/src/csrc/xpu/attn" "$V/src/csrc/xpu" "$V/src/CMakeLists.txt"; then
      ok "cached build outputs are newer than the sources; skipping build/rpath/layout"
    else
      step_build; step_rpath; step_layout
    fi
    step_verify ;;
  src) step_src ;; deps) step_deps ;; torch) step_torch ;; cfg) step_cfg ;;
  build) step_build ;; rpath) step_rpath ;; layout) step_layout ;; verify) step_verify ;;
  *) die "usage: $0 [all|src|deps|torch|cfg|build|rpath|layout|verify]" ;;
esac
