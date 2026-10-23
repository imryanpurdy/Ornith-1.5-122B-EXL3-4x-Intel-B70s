#!/usr/bin/env bash
# Runs INSIDE the runtime image (CPU only, no network, no devices).
# Copies what the vllm_xpu_kernels build needs from the runtime image into /v/torch-site:
#   torch/                 the package (headers come with it)
#   python3.12/            CPython headers, both the generic and the multiarch (pyconfig.h) directories merged
#   lib/libpython3.12.so   the shared library cmake links the extensions against
#   <pure-python deps>     the modules torch needs to be importable during configure/codegen
set -euo pipefail
trap 'chown -R "$OWNER" /v/torch-site 2>/dev/null || true' EXIT
rm -rf /v/torch-site
mkdir -p /v/torch-site/lib /v/torch-site/python3.12

T=$(python3 -c 'import torch,os;print(os.path.dirname(torch.__file__))')
cp -a "$T" /v/torch-site/torch
cp -a /usr/include/python3.12/. /v/torch-site/python3.12/
cp -a /usr/include/x86_64-linux-gnu/python3.12/. /v/torch-site/python3.12/
cp -a /usr/lib/x86_64-linux-gnu/libpython3.12.so.1.0 /v/torch-site/lib/
# torch also needs the PTI profiling library, which the image keeps in the venv lib dir, not in torch/lib
cp -a /opt/venv/lib/libpti_view.so* /v/torch-site/lib/

SP=$(python3 -c 'import sysconfig;print(sysconfig.get_paths()["purelib"])')
for m in numpy numpy.libs setuptools typing_extensions.py sympy networkx filelock jinja2 markupsafe fsspec \
         mpmath torchgen packaging wheel _distutils_hack pkg_resources distutils-precedence.pth; do
  cp -a "$SP/$m" /v/torch-site/ 2>/dev/null || echo "note: $m not in this image (not needed by the build)"
done

{
  echo "torch_lib_dir_in_image=$T/lib"
  python3 -c 'import torch;print("torch", torch.__version__, "| xpu", torch.version.xpu, "| cxx11abi", torch.compiled_with_cxx11_abi())'
} | tee /v/torch-site/RUNTIME_INFO
