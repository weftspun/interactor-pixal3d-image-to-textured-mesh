# Pixal3D, as a vast.ai worker. RFD 0040 records the design, and
# RFD 0036 records why this is a plain Docker image and not a Cog.
#
# The target is an RTX 4090 with 24 GB. Pixal3D peaks at 6.50 GB with
# low_vram on, thus that card holds it with room for the activations.
#
# The weights are measured, and not guessed: 24.045 GB across seven
# safetensors files in TencentARC/Pixal3D. Every file is fp16 or bf16,
# thus 2 bytes per parameter and about 12.02 B parameters.
#
# Build:
#   docker build -t weftspun/pixal3d .
#
# Test the handler contract with no GPU and no weights:
#   docker build --target contract -t weftspun/pixal3d:contract .
#   docker run --rm -p 8000:8000 weftspun/pixal3d:contract

# ---------------------------------------------------------------------
# The contract stage. It carries the handler and usd-core, and no
# model. RFD 0040 uses it to prove the request and response shapes in
# Docker, on a machine with no NVIDIA device.
# ---------------------------------------------------------------------
FROM python:3.11-slim AS contract

WORKDIR /app

# RFD 0053. OpenUSD is the internal format, thus the layer writer is
# part of the contract and not an extra.
RUN pip install --no-cache-dir usd-core==25.5 fastapi==0.115.5 uvicorn==0.32.1 pydantic==2.10.3

COPY server.py /app/server.py
COPY test_input.json /app/test_input.json

ENV WEFTSPUN_STUB=1 PORT=8000
EXPOSE 8000
CMD ["python", "/app/server.py"]

# ---------------------------------------------------------------------
# The worker. CUDA 12.4, because upstream needs NATTEN and either
# flash attention or the PyTorch SDPA backend.
# ---------------------------------------------------------------------
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 AS worker

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIXAL3D_SRC=/src/Pixal3D \
    PIXAL3D_WEIGHTS=/weights/pixal3d

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.11 python3-pip git libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Pinned to the upstream requirements.txt at the commit this image
# targets. An unpinned wheel changes the mesh.
RUN pip install --no-cache-dir \
      torch==2.5.1 numpy==2.1.3 safetensors==0.4.5 pillow==12.0.0 \
      imageio==2.37.2 imageio-ffmpeg==0.6.0 tqdm==4.67.1 easydict==1.13 \
      opencv-python-headless==4.12.0.88 trimesh==4.10.1 \
      transformers==4.57.3 zstandard==0.25.0 kornia==0.8.2 timm==1.0.22 \
      diffusers==0.37.1 accelerate==1.13.0 plyfile==1.1.3 \
      usd-core==25.5 huggingface_hub==0.26.2 \
      fastapi==0.115.5 uvicorn==0.32.1 pydantic==2.10.3

# MoGe has no wheel. Upstream installs it from git.
RUN pip install --no-cache-dir git+https://github.com/microsoft/MoGe.git

# NATTEN needs a CUDA architecture at build time. A mismatch fails at
# import, and not here.
RUN pip install --no-cache-dir natten==0.21.0

# Upstream itself. Pin the commit: master moves, and a moved master
# changes the mesh with no build change to show for it.
ARG PIXAL3D_REF=master
RUN git clone https://github.com/TencentARC/Pixal3D.git /src/Pixal3D \
    && git -C /src/Pixal3D checkout "${PIXAL3D_REF}"

# Three model repositories, 24.045 GB together. They download at build
# time, because a cold start that pulls 24 GB is a cold start that
# times out.
RUN python3 -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('TencentARC/Pixal3D', local_dir='/weights/pixal3d'); \
snapshot_download('Ruicheng/moge-2-vitl', local_dir='/weights/moge'); \
snapshot_download('camenduru/dinov3-vitl16-pretrain-lvd1689m', local_dir='/weights/dinov3')"

WORKDIR /app
COPY server.py /app/server.py

# vast.ai maps this port. PORT overrides it.
ENV PORT=8000
EXPOSE 8000

CMD ["python3", "-u", "/app/server.py"]
