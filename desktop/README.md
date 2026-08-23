# desktop: the local 3090 path

The root `Dockerfile` builds the vast.ai worker RFD 0036 describes. This directory is the
other thing that was actually run: upstream's cascade on the desktop 4090/3090, driven by
`inference.py` rather than through `server.py`, which is how the two sample meshes in
`6-datasource/anny-render-corpus`'s corpus were produced.

It is here because it was living in a session temp directory, which is the failure the
workspace already has a rule about. Two of its four files describe an environment that no
longer exists, and that is stated in each file rather than left for the next reader.

## What it measured that the worker image does not know

**`requirements-hfdemo.txt` does not give a working environment on this card.** Its natten
wheel installs, imports, and carries `libnatten.cpython-310-x86_64-linux-gnu.so`, and then the
GPU answers `no kernel image is available for execution`. The kernels are real and compiled
for sm_90, because Hugging Face Spaces run H-series cards. This desk is sm_86.

The root README says that file pins prebuilt wheels for every heavy dependency and that there
are no source compiles. That is true of the Space and false of anywhere else: natten has to be
built for the local architecture, which is a thirty-minute compile, and it is the only one of
the six that does.

This reaches the worker image directly. vast.ai rents whatever is available, so an instance is
as likely to be sm_86 or sm_89 as sm_90, and the failure arrives inside a diffusion step rather
than at install.

**Triton needs a compiler and Python's headers at run time.** It JIT-compiles a shim the first
time a kernel runs, so `build-essential python3-dev` is a runtime dependency of a wheels-only
image. The Space never meets this because its base image ships a toolchain.

**`briaai/RMBG-2.0` is named in `pipeline.json` and is blocklisted.** The pipeline builds it in
`from_pretrained`, before it looks at the input, so supplying an RGBA image with an exact matte
does not avoid it. `ZhengPeng7/BiRefNet` is the original, MIT, ungated, and already the class
default that upstream overrode. `desktop/Dockerfile` swaps it in config and asserts on the
entry it expects, so an upstream change fails the build rather than quietly restoring the gated
model.

## The gate

`smoke.py` launches `na2d` on the device. Three weaker checks were tried first and each passed
on a broken image: importing natten passes with no kernels at all, the `.so` existing passes
with sm_90 kernels, and a `+` in the version string fails on a correct local build. A file is
not a capability.

Run it where the answer means something, which is not `docker build`:

    docker run --rm --gpus all <image> python3 /opt/weftspun/smoke.py
