# Runtime image and the kernel overlay

`start.sh` serves from one container image plus one bind-mounted overlay. This page records exactly what those
are, where every piece comes from, and what a newcomer can and cannot obtain. Nothing here is built or pushed by
this repository: the recipe was validated against the image already on the rig, and `start.sh` never builds an
image.

## 1. The engine image

| Field | Value |
|---|---|
| Local tag | `b70-lumnus-trial:v1` |
| Local image ID (what `.env` pins) | `sha256:6021b4b8d99d6fa7e139dff53028356135201eb4e6479f091ce55f0f78c4307d` |
| Engine | vLLM 0.30.0 XPU, torch 2.13.0+xpu, `vllm_xpu_kernels` 0.1.14.1, Python 3.12, Ubuntu 24.04 |
| OCI labels | `org.opencontainers.image.version=0.30.0-b70.1-trial`, `…revision=27239a65755d7853e52bf197030ffb7638fa141d`, `…source=https://github.com/Lumnus/b70-flash-next` |

The ID is a *local* image ID — the identity of the bytes on disk, so `docker run sha256:6021b4b8…` is unambiguous.
It is not a registry digest of a published image (see §1.3).

### 1.1 The chain its own labels describe

1. **Base** — `docker.io/vllm/vllm-openai-xpu:v0.30.0@sha256:fc0e112afb64e3a06fe8daff34652435822a629412f38efce8f0f67a46636b8d`
   (labels `org.opencontainers.image.base.name` and `base.digest`).
   **Reachable: yes, publicly.** `docker manifest inspect` on that digest resolves from a host with plain
   registry access (checked 2026-10-23).
2. **Pack overlay** — `ghcr.io/wu1ff/qwen38-flashnext-b70:1.0.0@sha256:85512b52c09fa660a2e7fe441417129e7c47fac727fd85f66ccea6b65e0a9122`
   (label `net.lumnus.wu1ff.image`). Two binaries with no public source come from this image by digest:
   `/opt/venv/lib/python3.12/site-packages/vllm_xpu_kernels/libgdn_index64.so` (a GDN op library with 64-bit
   conv-state offsets) and `/opt/b70-residency-shim/libl0_peer_residency_shim.so` (a Level Zero peer-residency
   shim, the file `env.list`'s `LD_PRELOAD` points at).
   **Reachable: NOT verified — currently unresolvable.** An anonymous pull-token request for both the tag and the
   digest returns `404 MANIFEST_UNKNOWN` from `ghcr.io` (checked 2026-10-23), so the pack is private, withdrawn or
   moved. If you do have access, pull by digest: the digest is the identity, not the tag.
3. **Patch series** — `github.com/Lumnus/b70-flash-next` at revision
   `27239a65755d7853e52bf197030ffb7638fa141d` (label `org.opencontainers.image.revision`).
   **Reachable: yes, publicly.** The repository is public (`private: false`) and that commit resolves through the
   GitHub API (checked 2026-10-23). At that revision it carries:
   - `image/Dockerfile` — the build of record;
   - `patches/vllm/` — 21 patches (`0001`-`0019`, including `0013b` and `0014a`-`0014f`) plus `patches/series.txt`
     mapping each to its fork commit;
   - `image/verify-overlay.sh` — the byte-identity gate the build runs;
   - `PROVENANCE.md` — per-patch file hashes, layer digests and provenance grades.
   The patches are written against vLLM `v0.30.0` = `ced6857afa0ea7b2e3f0846a62e1394e90f15607` (label
   `net.lumnus.vllm.source`) and the image applies them with `patch --fuzz=0` to both
   `/opt/venv/lib/python3.12/site-packages` and `/workspace/vllm`.
   The revision is a *trial* revision, not a release: `main` and the newest tag in that repository have moved on
   since, so pin the revision, never a branch name.

### 1.2 Obtaining an equivalent image

```bash
git clone https://github.com/Lumnus/b70-flash-next.git && cd b70-flash-next
git checkout 27239a65755d7853e52bf197030ffb7638fa141d
docker buildx build -f image/Dockerfile \
  --build-arg GIT_SHA=0.30.0-b70.1-trial \
  --build-arg SOURCE_SHA=$(git rev-parse HEAD) \
  -t b70-lumnus-trial:v1 .
```

That build pulls the base by digest and takes the two binaries from the pack image by digest, so it stops at the
`COPY --from=wu1ff` steps unless you can reach the pack image (§1.1 item 2). **We have not re-run this build**;
every number in this repository was measured on the image already present on the rig.

### 1.3 What this repository does *not* ship

This repository does not ship or publish any image. The *older* plugin-image layout on
`intel/llm-scaler-vllm:0.26.0-b2` — an unrelated base, from before the engine image existed — was removed with the
upstream-project cleanup; `start.sh` never used it.

## 2. The `vllm_xpu_kernels` overlay (required, not optional)

The container bind-mounts a rebuilt `vllm_xpu_kernels` over the image's own package at
`/opt/venv/lib/python3.12/site-packages/vllm_xpu_kernels` (`VXK_DIR` in `.env`).

### 2.1 Why it exists

The image ships `vllm_xpu_kernels` 0.1.14.1 built from `csrc/xpu/attn/kernel_configs/paged_decode_default.conf`,
which has qgroup-8 entries at head size 256 but **none at qgroup 16**. Ornith's full attention is 32 query heads
and 2 KV heads at head_dim 256 with page size 64, so at TP4 each card runs 16 query heads against 1 KV head — a
qgroup-16 decode. Without the overlay the engine aborts at the post-KV warmup:

```
RuntimeError: Paged decode kernel not compiled for this configuration   (wants 16,256,64,false,false,false)
```

There is no fallback for that shape in this package, so the mount is mandatory: `start.sh` refuses to start without
it.

### 2.2 What it is

Upstream `github.com/vllm-project/vllm-xpu-kernels` at `6d92b1bfbf32767ecda8e819613eb151e70030ad` (the same
0.1.14.1 the image ships) with exactly **one added line** in that config file, after the Qwen3-Next head-256 entry:

```
16,256,64,false,false,false
```

Nothing else changed: the source diff is `89a90`, a single added line. The rebuild produces 33 paged-decode
instantiations instead of 32 (`libattn_kernels_xe_2.so` 209,500,920 B vs 194,978,032 B), and the extra template is
`paged_decode_kernel_template_q16_h256_p64_fff.cpp`. Every other `.so` is byte-different from the image's copy
because it comes from a different build environment (sizes differ by well under 1%), not because of a source
change.

### 2.3 How we know the overlay is what it claims to be

We extracted the image's baked package with a throwaway container (`docker create` + `docker cp`, never `run`,
no device) and compared sha256 per file:

- **identical**: `__init__.py`, `flash_attn_interface.py`, `fused_moe_interface.py`, `moe_utils.py`, `rotary.py`,
  `quantization/*`, `libgdn_index64.so`;
- **different**: all compiled extensions and kernel libraries — and the paged-decode template set is 33 files in
  the overlay against 32 in the image, with `q16_h256_p64` present only in the overlay.

The overlay also carries the package's python files and a copy of the image's `libgdn_index64.so` (that source tree
does not build it, and the mount replaces the whole directory), so its file set equals the image's.

### 2.4 Building it

`scripts/build-vxk.sh` reproduces the overlay on the CPU. No GPU, every build container `--rm --network none`, no
devices:

| Piece | Pin |
|---|---|
| kernel source | `vllm-project/vllm-xpu-kernels` @ `6d92b1bfbf32767ecda8e819613eb151e70030ad` + the one config line |
| compiler | `icpx` 2026.1.1 from `TOOLCHAIN_IMAGE` (`intel/omix:0.4.0-devel-ubuntu24.04`) |
| torch | headers + libraries taken read-only from `RUNTIME_IMAGE` (torch 2.13.0+xpu) |
| oneDNN | `0e2a5bfeef1bfbffc3137464606540233086ce9b` (mounted, so FetchContent never uses the network) |
| sycl-tla | `87f6850680a580654b9ea2c80dbc01aeb36ad231` |
| patchelf | 0.18.0, fetched from its release tarball and sha256-checked (`b699cb82…`) |

`make -j2` (higher `-j` values were OOM-killed inside a 48 GiB cap), then `patchelf --set-rpath '$ORIGIN'` on every
`.so` — without it `import vllm_xpu_kernels` cannot find `libgrouped_gemm_xe_default.so`. `make install` only
installs the five python extension modules, so the script copies the six kernel libraries across from the build
tree. Budget ~70 minutes and ~4 GiB of disk for `VXK_DIR` (the torch copy dominates).

```bash
./scripts/build-vxk.sh            # src, deps, torch, cfg, build, rpath, layout, verify — cached step by step
./scripts/build-vxk.sh verify     # just the assertions
```

### 2.5 Reproducibility status (honest)

The whole pipeline was re-run from this script on the rig on 2026-10-23/24 — `src`, `deps`, `torch`, `cfg`,
`build` (8131 s at `-j2` inside a 48 GiB cap), `rpath`, `layout`, `verify` — and the result compares like this
against the overlay the serving container actually mounts:

- **byte-identical** (sha256): all six kernel libraries, including
  `libattn_kernels_xe_2.so` = `78a1abd2039b84ca1a50693f199cccac4d6a246758fa789d7ddb9c9e44e39c56`, and the copied
  `libgdn_index64.so`;
- **same size, different bytes**: the five python extension modules (`_C.abi3.so`, `_moe_C.abi3.so`,
  `_vllm_fa2_C.abi3.so`, `_xpu_C.abi3.so`, `xpumem_allocator.abi3.so`) — 92 differing bytes in `_C.abi3.so`,
  in the build-id/padding region; `readelf -d` shows the same `$ORIGIN` RUNPATH and the same `NEEDED` list;
- against the **image's baked** package, `libattn_kernels_xe_2.so` is 209,500,920 B versus 194,978,032 B, and
  `strings` counts 33 paged-decode templates versus 32 — the one added config line, nothing else;
- `verify` reports `IMPORT OK` for `vllm_xpu_kernels`, `flash_attn_interface` and `fused_moe_interface` inside the
  runtime image (loading only; no device is touched).

Not reproduced here: the *image itself* (§1.2), and `libgdn_index64.so`, which has no public source and is copied
out of the image rather than built. Neither is needed to rebuild the overlay — the overlay takes
`libgdn_index64.so` from whatever image you run.
