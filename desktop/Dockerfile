# STATUS, FIRST, BECAUSE THIS FILE DOES NOT BUILD TODAY.
#
# `weftspun-pixal3d:cdbb2bb` is a local image tag. Nothing tracked ever described how it was
# built, it was never pushed to a registry, and this desk's Docker store is now empty -- both
# contexts report zero images. So the base cannot be resolved by anyone, including the desk
# that made it.
#
# What survives is its digest, from the build log of the layer below:
#
#   sha256:b95888832720856ed840cc94ef8baf0a94185d16456ced7f0a8cd77ef21425b5
#
# That names the thing exactly and fetches nothing. It is the whole argument for tracking a
# recipe at the time it works rather than when it is needed: this file's measurements are all
# still true, and the environment they were measured in is gone.
#
# The tag is upstream's commit `cdbb2bb`, the base carried the weights at /weights/pixal3d,
# and python 3.10. Reconstructing it is the work this file is waiting on. It is kept as the
# record of nineteen measured fixes, not as a build that runs.

# The base environment RFD 1028's image never installed, taken from upstream's own Space.
#
# WHAT THIS FILE USED TO BE, and why it is worth saying. It compiled four CUDA extensions
# from git, one of them for thirty minutes, and it accumulated nineteen fixes discovered by
# running the thing and reading the error: python headers, Eigen, cmake, a pip too old to
# read PEP 621 metadata, a torch pin that brought the wrong triton, a safetensors that could
# not open a checkpoint, a natten with no kernels.
#
# Every one of those was a rediscovery. `TencentARC/Pixal3D`'s Hugging Face Space publishes
# `requirements_th26_cu124.txt`: the exact version matrix for torch 2.6 and CUDA 12.4, with
# PREBUILT WHEELS for natten, cumesh, flex_gemm, o_voxel, nvdiffrast, nvdiffrec_render and
# flash-attn 3. All cp310, which is what this base image runs.
#
# So the compiles are gone, and so is the version archaeology. The lesson is cheaper than
# the nineteen fixes: look for the environment the authors actually run before reconstructing
# one from their README.
FROM weftspun-pixal3d:cdbb2bb

ENV PIP_NO_CACHE_DIR=1

# gcc AND THE PYTHON HEADERS ARE A RUNTIME DEPENDENCY HERE, WHICH IS NOT OBVIOUS.
#
# Wheels removed every compile from the build, and then `flex_gemm` and `o_voxel` failed on
# the GPU with `fatal error: Python.h: No such file or directory`. Triton JIT-compiles a
# small C shim -- `cuda_utils.cpython-310-x86_64-linux-gnu.so` -- the first time a kernel
# runs, so it needs a compiler and Python's headers inside the container at run time, every
# time, not while the image is being built.
#
# Upstream's Space never meets this: Hugging Face's base image ships a full toolchain. A lean
# container does not, and the failure arrives inside a diffusion step rather than at install.
# cmake and ninja are here for natten, which is the one package still built from source
# below. When this file first moved to the Space's wheels the toolchain was trimmed to
# `build-essential python3-dev`, on the reasoning that wheels do not compile -- correct for
# five of the six, and natten then failed with `Cannot find CMake executable`.
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential python3-dev cmake ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Modern pip and setuptools first, and this one still matters even with wheels. The base
# carries pip 22.0.2, which cannot read a PEP 621 `pyproject.toml`: with
# `--no-build-isolation` every such project installs as `UNKNOWN-0.0.0`, and since they all
# share that name each install uninstalls the previous one. Nothing errors, the layers look
# right, and the image ends up missing packages it reported installing.
RUN pip install --no-cache-dir --upgrade "pip>=24.0" "setuptools>=70.0" wheel

# Upstream's matrix, fetched at the commit this image pins rather than transcribed. A copy
# here would drift from the Space the moment they bump a wheel, and the whole point of this
# file is to stop maintaining a second opinion about their dependencies.
ADD https://huggingface.co/spaces/TencentARC/Pixal3D/raw/main/requirements_th26_cu124.txt \
    /tmp/requirements_th26_cu124.txt

# MOGE IS FILTERED OUT, AND UPSTREAM'S OWN FILE DOES NOT INSTALL AS WRITTEN WITHOUT THAT.
#
# Line 21 is `git+https://github.com/microsoft/MoGe.git`, and pip answers:
#
#   ERROR: Cannot install -r requirements_th26_cu124.txt (line 21) and trimesh==4.10.1
#   because these package versions have conflicting dependencies.
#
# MoGe wants a trimesh the same file pins away. On a Space that resolves differently or the
# demo tolerates it; here it stops the build.
#
# Removing it costs nothing on this path and is the same conclusion reached earlier for the
# pixi environment: MoGe estimates the camera from the image, `inference.py` imports it
# lazily inside `load_moge_model`, and that runs only when `--fov` is absent. Every image
# here is rendered by us, so the camera is a number we hold rather than one to infer. The
# consequence, stated so it is not discovered: `--fov` is mandatory in this image, in
# RADIANS, and 40 degrees is 0.698.
#
# The filter is a grep rather than an edited copy of the file, so the rest stays whatever
# upstream says today.
RUN grep -v "github.com/microsoft/MoGe" /tmp/requirements_th26_cu124.txt > /tmp/requirements.filtered.txt \
    && pip install -r /tmp/requirements.filtered.txt

# NATTEN IS THE ONE WHEEL THAT CANNOT BE USED, AND THE REASON IS THE GPU.
#
# The Space's `natten-0.21.0+torch2.6cu124` installs cleanly, imports, and carries
# `libnatten.cpython-310-x86_64-linux-gnu.so`. It then fails on this card:
#
#   NATTEN failure: CUDA runtime error: no kernel image is available for execution
#
# The kernels are real and compiled for the wrong architecture. Hugging Face Spaces run
# H-series cards, sm_90; this desk has an RTX 3090, sm_86. Upstream hints at it -- their
# README says `requirements-hfdemo.txt` targets "H-series GPU architecture" -- and the wheel
# filename says nothing about it, so the only way to learn this is to run a kernel.
#
# The other five wheels are architecture-independent enough to work: the failing run got
# through sparse-structure sampling on flex_gemm at 1.9 it/s and died only here.
#
# So natten is built from source for the local architecture, and it is the one thirty-minute
# compile that survives. `--no-deps` keeps it from moving torch.
ARG NATTEN_CUDA_ARCH=8.6
RUN NATTEN_CUDA_ARCH="${NATTEN_CUDA_ARCH}" NATTEN_N_WORKERS=8     pip install --no-cache-dir --force-reinstall --no-deps natten==0.21.0 --no-build-isolation  && python3 -c "import natten, pathlib; so = list(pathlib.Path(natten.__file__).parent.glob('libnatten*.so')); assert so, 'natten built without kernels'; print(f'natten {natten.__version__} with {so[0].name}')"

# utils3d, which the Space's file does not list and Pixal3D's code imports.
#
# It is absent there because MoGe pulled it in, and MoGe is the line filtered out above. This
# is Pixal3D's own pin, from README step 4, rather than TRELLIS.2's git ref: where two
# upstreams disagree, the one whose code runs decides.
RUN pip install --no-cache-dir --no-deps \
      https://github.com/LDYang694/Storages/releases/download/20260430/utils3d-0.0.2-py3-none-any.whl

# safetensors is not in upstream's file, and the base pins 0.4.5, which cannot open
# `ss_flow_img_dit_1_3B_64_bf16.safetensors` -- `InvalidHeaderDeserialization` on a file whose
# sha256 matches what Hugging Face publishes. 0.8.0 reads it. The failure surfaces as a 401
# from Hugging Face, because `pipelines/base.py` swallows the real exception and retries the
# path as a repo id.
RUN pip install --no-cache-dir --upgrade safetensors

# THE BACKGROUND REMOVER IS SWAPPED FOR THE MIT ONE, IN CONFIG RATHER THAN IN CODE.
#
# `pipeline.json` names `BiRefNet` with `model_name: briaai/RMBG-2.0`, and the pipeline builds
# it in `from_pretrained` -- before it looks at the input -- so an RGBA image with an exact
# matte does not avoid it. RMBG-2.0 is gated and non-commercial, and blocklisted here for
# both. `ZhengPeng7/BiRefNet` is the original: MIT, ungated, and `BiRefNet.__init__`'s own
# default, which upstream overrode in config.
#
# The assert matters. If upstream changes that entry, this fails at build rather than quietly
# leaving the gated model in place.
RUN python3 - <<'EOF'
import json, pathlib
p = pathlib.Path("/weights/pixal3d/pipeline.json")
d = json.loads(p.read_text())
entry = d["args"]["rembg_model"]
assert entry["args"]["model_name"] == "briaai/RMBG-2.0", entry
entry["args"]["model_name"] = "ZhengPeng7/BiRefNet"
p.write_text(json.dumps(d, indent=4))
print("rembg checkpoint ->", entry["args"]["model_name"])
EOF

# THE IMPORT CHECK CANNOT LIVE IN THE BUILD, and the first version of this file put it there.
#
# `docker build` has no device, and triton initialises its driver at import:
#
#   RuntimeError: 0 active drivers ([]). There should only be one.
#
# so `RUN python3 -c "import flex_gemm"` fails on a working image and passes on none. The
# check runs where the answer means something:
#
#   docker run --rm --gpus all <image> python3 /opt/weftspun/smoke.py
COPY smoke.py /opt/weftspun/smoke.py
