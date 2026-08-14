# interactor-pixal3d-image-to-textured-mesh -- vast.ai worker, RFD 0036/0040.
#
# Worker env contract: python 3.10 + torch 2.6 cu124 + upstream's
# requirements-hfdemo.txt, which pins PREBUILT wheels for every heavy dep
# (flash_attn_3, nvdiffrast, o_voxel, cumesh, flex_gemm, natten) -- zero
# source compiles. The wheels are cp310: the base image MUST be python 3.10.

FROM python:3.11-slim AS contract
WORKDIR /app
RUN pip install --no-cache-dir usd-core==25.5 fastapi==0.115.5 uvicorn==0.32.1 pydantic==2.10.3
COPY server.py worker_entry.py synth_views.py /app/
COPY test_input.json /app/test_input.json
ENV WEFTSPUN_STUB=1 PORT=8000
EXPOSE 8000
CMD ["python", "/app/server.py"]

FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 AS worker

# python 3.10 exactly -- the hfdemo wheels are cp310.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3-pip git curl \
    && rm -rf /var/lib/apt/lists/*
RUN python3.10 -m venv /venv
ENV PATH="/venv/bin:$PATH"

ARG PIXAL3D_REF=main
RUN git clone https://github.com/TencentARC/Pixal3D.git /src/Pixal3D \
    && git -C /src/Pixal3D checkout "${PIXAL3D_REF}"

# Upstream's own pinned env: torch 2.6 cu124 + prebuilt wheels. No compiles.
RUN pip install --no-cache-dir -r /src/Pixal3D/requirements-hfdemo.txt \
    && pip install --no-cache-dir usd-core==25.5 fastapi==0.115.5 uvicorn==0.32.1 pydantic==2.10.3

WORKDIR /app
COPY server.py worker_entry.py synth_views.py /app/
ENV PYTHONPATH=/src/Pixal3D PIXAL3D_WEIGHTS=TencentARC/Pixal3D LOW_VRAM=1 PORT=8000
EXPOSE 8000
CMD ["python", "-u", "/app/server.py"]
