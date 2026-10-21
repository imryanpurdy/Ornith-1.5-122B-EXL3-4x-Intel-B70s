#!/usr/bin/env bash
# make_dist.sh (DESIGN 2.1, T7): assemble an immutable plugin dist from a clean checkout.
#
#   dist/<commit12>-<sha12>/exl3xpu/{*.py,_C.so,_moe.so}
#   dist/<commit12>-<sha12>/trellis_core/{__init__.py,reference.py,format/{__init__,spec,geometry}.py}
#   dist/<commit12>-<sha12>/exl3xpu-0.2.0.dev0.dist-info/{METADATA,entry_points.txt}
#   dist/<commit12>-<sha12>/templates/{ornith-chat-template.jinja,qwen3_effort_parser.py}
#   dist/<commit12>-<sha12>/BUILD_INFO    # commit, branch, per-file sha256 ("<sha256>  <relpath>"), build provenance
#
# <sha12> = first 12 hex of sha256 over the sorted "<sha256>  <relpath>" listing of every file except BUILD_INFO
# (the same digest tools/ornith/planlib.py dist_manifest() recomputes, so the lane's --go verifies the dist bit
# for bit). A dist dir is never overwritten: if dist/<id> exists the script refuses.
#
# Usage: make_dist.sh --src DIR --libs DIR [--templates DIR] [--out DIR]
#   --src        clean git checkout/worktree of this repo; supplies the Python, trellis_core and the identity
#   --libs       build output dir (build.sh): must hold _C.so, _moe.so and SOURCE_INFO for the SAME commit,
#                clean tree (the build snapshot digest must match <src> HEAD)
#   --templates  dir with ornith-chat-template.jinja + qwen3_effort_parser.py; default <src>/templates
#   --out        dist root; default <src>/../dist via git toplevel (i.e. <repo>/dist)
#
# CPU only, no containers, no devices: it assembles files and hashes them. The .so build itself is build.sh's job.
set -euo pipefail

die() { echo "make_dist.sh: $*" >&2; exit 1; }
SRC=; LIBS=; TPL=; OUT=
while [ $# -gt 0 ]; do
  case "$1" in
    --src) SRC=$2; shift 2 ;;
    --libs) LIBS=$2; shift 2 ;;
    --templates) TPL=$2; shift 2 ;;
    --out) OUT=$2; shift 2 ;;
    *) die "unknown arg $1 (usage: make_dist.sh --src DIR --libs DIR [--templates DIR] [--out DIR])" ;;
  esac
done
[ -n "$SRC" ] || die "--src is required (a clean checkout of this repo)"
[ -n "$LIBS" ] || die "--libs is required (the build.sh output with _C.so/_moe.so/SOURCE_INFO)"
SRC=$(cd "$SRC" && pwd) || die "--src $SRC: not a dir"
LIBS=$(cd "$LIBS" && pwd) || die "--libs $LIBS: not a dir"
TPL=${TPL:-$SRC/templates}
OUT=${OUT:-$(git -C "$SRC" rev-parse --show-toplevel)/dist}

# --- source tree: clean, on a commit -------------------------------------------------------------
[ -z "$(git -C "$SRC" status --porcelain)" ] || die "$SRC is dirty: dists are built from clean checkouts only"
COMMIT=$(git -C "$SRC" rev-parse HEAD)
C12=${COMMIT:0:12}
BRANCH=$(git -C "$SRC" rev-parse --abbrev-ref HEAD)

# --- build provenance: the .so must come from this exact commit -----------------------------------
[ -f "$LIBS/_C.so" ] || die "$LIBS/_C.so missing (run build.sh first)"
[ -f "$LIBS/_moe.so" ] || die "$LIBS/_moe.so missing (run build.sh first)"
[ -f "$LIBS/SOURCE_INFO" ] || die "$LIBS/SOURCE_INFO missing (build.sh records the snapshot it built)"
b_commit=$(sed -n 's/^commit=//p' "$LIBS/SOURCE_INFO")
b_dirty=$(sed -n 's/^dirty=//p' "$LIBS/SOURCE_INFO")
b_digest=$(sed -n 's/^digest=//p' "$LIBS/SOURCE_INFO")
[ "$b_commit" = "$COMMIT" ] || die "libs commit $b_commit != src HEAD $COMMIT: rebuild with build.sh --src"
[ "$b_dirty" = "0" ] || die "libs were built from a dirty tree: rebuild"

# --- templates --------------------------------------------------------------------------------------
[ -f "$TPL/ornith-chat-template.jinja" ] || die "$TPL/ornith-chat-template.jinja missing"
[ -f "$TPL/qwen3_effort_parser.py" ] || die "$TPL/qwen3_effort_parser.py missing"

# --- stage ------------------------------------------------------------------------------------------
STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT
mkdir -p "$STAGE/exl3xpu" "$STAGE/trellis_core/format" "$STAGE/exl3xpu-0.2.0.dev0.dist-info" "$STAGE/templates"

cp "$SRC"/xpu/exl3xpu/*.py "$STAGE/exl3xpu/"
cp "$LIBS/_C.so" "$LIBS/_moe.so" "$STAGE/exl3xpu/"
cp "$SRC"/core/src/trellis_core/__init__.py "$SRC"/core/src/trellis_core/reference.py "$STAGE/trellis_core/"
cp "$SRC"/core/src/trellis_core/format/__init__.py "$SRC"/core/src/trellis_core/format/spec.py \
   "$SRC"/core/src/trellis_core/format/geometry.py "$STAGE/trellis_core/format/"
cat > "$STAGE/exl3xpu-0.2.0.dev0.dist-info/METADATA" <<EOF
Metadata-Version: 2.1
Name: exl3xpu
Version: 0.2.0.dev0
Summary: EXL3 XPU kernels and vLLM plugin (Ornith)
EOF
cat > "$STAGE/exl3xpu-0.2.0.dev0.dist-info/entry_points.txt" <<'EOF'
[vllm.general_plugins]
exl3xpu = exl3xpu.vllm_plugin:register
EOF
cp "$TPL/ornith-chat-template.jinja" "$TPL/qwen3_effort_parser.py" "$STAGE/templates/"

# --- content id (planlib.dist_manifest: sorted "<sha256>  <relpath>" listing, sha256[:12]) ----------
LISTING=$(cd "$STAGE" && find . -type f ! -name BUILD_INFO -printf '%P\n' | sort |
  while IFS= read -r r; do printf '%s  %s\n' "$(sha256sum "$r" | cut -d' ' -f1)" "$r"; done)
SHA12=$(printf '%s\n' "$LISTING" | sha256sum | cut -d' ' -f1 | cut -c1-12)
ID="$C12-$SHA12"
[ ! -e "$OUT/$ID" ] || die "$OUT/$ID already exists: dists are immutable, never rebuilt in place"

# --- BUILD_INFO + publish ---------------------------------------------------------------------------
mkdir -p "$OUT"
{
  echo "id=$ID"
  echo "commit=$COMMIT"
  echo "branch=$BRANCH"
  echo "built=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "libs=$LIBS (build.sh: commit=$b_commit digest=$b_digest)"
  echo "provenance: per-file sha256 of every file in this dist (BUILD_INFO itself excluded from the id)"
  printf '%s\n' "$LISTING"
} > "$STAGE/BUILD_INFO"
mv "$STAGE" "$OUT/$ID"
trap - EXIT
echo "dist $ID"
echo "$OUT/$ID"
