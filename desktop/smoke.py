"""Does this image actually work? Run it with --gpus all; it is meaningless without one.

Every import here needs a CUDA device at module load, which is why this is not a build step.
`docker build` has no device, so the same check inside a `RUN` fails on a working image.
"""

import sys

import torch

print(f"torch   {torch.__version__}")
print(f"cuda    {torch.cuda.is_available()} "
      f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else '(no device)'}")
if not torch.cuda.is_available():
    print("FAIL no CUDA device. Run with --gpus all, or this proves nothing either way.")
    sys.exit(1)

import triton

print(f"triton  {triton.__version__}")

failed = []
for name in ("cumesh", "flex_gemm", "o_voxel", "utils3d", "nvdiffrast.torch", "natten"):
    try:
        __import__(name)
        print(f"  ok   {name}")
    except Exception as exc:
        print(f"  FAIL {name}: {type(exc).__name__}: {exc}")
        failed.append(name)

# THE ONLY CHECK THAT MEANS ANYTHING IS RUNNING A KERNEL.
#
# Three weaker checks were tried here and each passed on a broken image:
#
#   `import natten`            passes on a source build with no kernels at all
#   `libnatten*.so` exists     passes on the Space's wheel, whose kernels are sm_90
#   the version carries `+`    fails on a correct local build, which reports plain 0.21.0
#
# A file is not a capability. The wheel installs, imports, carries its `.so`, and then says
# `no kernel image is available for execution on the device` inside a diffusion step. So this
# launches the smallest neighbourhood attention it can and sees whether the GPU runs it.
try:
    import torch
    from natten.functional import na2d

    q = torch.randn(1, 8, 8, 1, 16, device="cuda", dtype=torch.float16)
    out = na2d(q, q, q, kernel_size=3)
    torch.cuda.synchronize()
    print(f"  ok   natten kernels run: na2d on {tuple(q.shape)} -> {tuple(out.shape)}")
except Exception as exc:
    print(f"  FAIL natten kernels: {type(exc).__name__}: {str(exc).splitlines()[0][:120]}")
    failed.append("natten kernels")

sys.exit(1 if failed else 0)
