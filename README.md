# Ornith-1.5-122B EXL3 on 4x Intel Arc Pro B70

Serving recipe and engine plugin for **[Mia-AiLab/Ornith-1.5-122B-EXL3-EAQv1-4-bpw](https://huggingface.co/Mia-AiLab/Ornith-1.5-122B-EXL3-EAQv1-4-bpw)**
(the 122B `qwen3_5_moe` model, EXL3 `mul1` codebook, 61.1 GiB) on **4x Intel Arc Pro B70 (32 GB, Xe2 / Battlemage)** with **vLLM**,
using the published weights as they are: no re-quantizing, no llama.cpp.

The EXL3 kernels and the vLLM plugin come from [trellis-serve](https://github.com/sybil-solutions/trellis-serve)
(branch `ft-xpu_bmg`, MIT); this repository is a fork of it that adds what a 122B MoE across four cards needs.
The upstream README is kept as [TRELLIS-SERVE-README.md](TRELLIS-SERVE-README.md); [ORIGIN.md](ORIGIN.md) records the exact source commit.

> **Status: work in progress, not servable yet.** See [Status](#status). `./start.sh` builds the kernels today and stops
> before launching a server until the 4-card lane lands.

## Quick start (target shape)

```bash
git clone https://github.com/imryanpurdy/Ornith-1.5-122B-EXL3-4x-Intel-B70s.git && cd Ornith-1.5-122B-EXL3-4x-Intel-B70s
cp .env.example .env    # MODEL_DIR, RUNTIME_IMAGE, TOOLCHAIN_IMAGE, BUILD_DIR, PORT, SERVED_MODEL_NAME, GPUS
./start.sh
```

Requirements: 4x Arc Pro B70 on the `xe` kernel driver, Docker, ~62 GiB of disk for the weights, a vLLM 0.30 XPU image
(torch 2.13.0+xpu) as the runtime, and `intel/omix:0.4.0-devel-ubuntu24.04` (oneAPI icpx 2026.1) to build the kernels.
The kernel build runs on the CPU in throwaway containers; no GPU is needed for it.

## How it works

| Piece | Where | Notes |
|---|---|---|
| EXL3 dense kernels (GEMV, DPAS, reconstruct + GEMM) | `xpu/csrc/exl3_esimd.h`, `xpu/csrc/exl3_ops.sycl` | SYCL/ESIMD, JIT `spir64`; `mul1` at K=3..6; wide 5-bit DPAS build, per-shape GEMV/DPAS cut-over, shape-aware split-K |
| EXL3 grouped MoE kernels | `xpu/csrc/exl3_moe.sycl` | upstream is K=3 only; Ornith's experts are K=4 (in progress) |
| vLLM plugin | `xpu/exl3xpu/` | quantization config + linear method; MoE method and tensor parallelism in progress |
| EXL3 tensor-parallel split rules | `core/src/trellis_core/format/geometry.py` | 128-aligned input/output splits, KV-head replication |
| Kernel build | `scripts/build-kernels.sh` | builds `_C.so` and `_moe.so` against the runtime image's own torch |

Plan: attention and Gated DeltaNet layers split 4 ways (tensor parallel), the 256 routed experts split by whole experts
(64 per card) or by width (1024 -> 256 per card), whichever measures faster; FP8 KV cache; XPU graphs for decode only
(`FULL_DECODE_ONLY`; no graph is ever captured while serving).
Every Ornith linear width is a multiple of 128, so every split is an exact slice of the EXL3 tensors.

## Status

| Milestone | State |
|---|---|
| 1. Kernels build for the runtime's torch; the checkpoint's EXL3 manifest checks out | done |
| 1b. Dense kernels run correctly on a B70 against an independent reference | done |
| 1c. Dense kernels faster for Ornith's shapes (5-bit DPAS, small-batch routing, split-K) | done |
| 2. EXL3 MoE method in vLLM matches the reference on one card | in progress |
| 3. Two cards match one card | planned |
| 4. Ornith on four cards boots and answers | planned |
| 5. Long-context recall, tool calls, throughput and a 60-minute soak | planned |

Details: [docs/STATUS.md](docs/STATUS.md).

## Acknowledgements

- [trellis-serve](https://github.com/sybil-solutions/trellis-serve) (MIT): the B70 EXL3 kernels, the vLLM and SGLang plugins and the split rules this builds on. Its `LICENSE` and `NOTICE` are kept unchanged.
- [exllamav3](https://github.com/turboderp-org/exllamav3): the EXL3 format, the reference reconstruction and the tensor-parallel layouts.
- [vLLM](https://github.com/vllm-project/vllm) and Intel's `vllm-xpu-kernels`, `llm-scaler` and oneAPI toolchain.
- [Mia-AiLab](https://huggingface.co/Mia-AiLab) for the EXL3 quantization, and [ornith-ai](https://huggingface.co/ornith-ai) for the Ornith-1.5 models.

## License

MIT, inherited from trellis-serve; see [LICENSE](LICENSE) and [NOTICE](NOTICE).
