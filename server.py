"""Pixal3D image to textured mesh, PBR. RFD 0040, extended with the
generate/extract latent flow upstream's own app.py ships.

Thin HTTP shim over worker_entry.py, which vendors upstream app.py's
init_models/generate/extract verbatim (RFD 0040: "call the upstream entry
point, do not reimplement the cascade" -- app.py IS upstream's entry point
for the latent-state + preview-render flow).

  POST /predict  -- image -> latent state + nvdiffrast preview renders
                    (+ cameras). No GLB decode happens here.
  POST /extract  -- latent state -> GLB + USD layer. The one decode.

The split is the decode-only latent contract: a latent consumer (e.g. a
VoxHammer edit over the same TRELLIS.2-backbone slat family) takes `state`
directly -- no decode/re-encode round trip. The preview renders come from
upstream's nvdiffrast-backed differentiable renderer, the same path a later
adaptation pass backpropagates through.

Environment: python 3.10 + torch 2.6 cu124 + upstream requirements-hfdemo.txt
(prebuilt wheels: flash_attn_3, nvdiffrast, o_voxel, cumesh, flex_gemm).

RFD 0053 makes OpenUSD the internal format, thus a result carries a USD
layer beside the GLB that ships.
"""

import base64
import os
import tempfile
import urllib.request
from pathlib import Path

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("ATTN_BACKEND", "flash_attn")

# src/library/aiModelsCatalog.js API_MAX_MESH_VERTICES. The API rejects
# a mesh above this, thus it fails every later stage.
MAX_MESH_VERTICES = 210000

# WEFTSPUN_STUB=1 skips the model and answers with the same shape, so the
# server contract is testable in Docker with no GPU and no 24 GB download.
STUB = os.environ.get("WEFTSPUN_STUB") == "1"

_READY = {"loaded": False}


class InputError(ValueError):
    """The request is wrong. This is the caller's fault, and not ours."""


def _fetch(image: str, work: Path) -> Path:
    """Takes a URL, a data URI, or raw base64, and writes a file."""
    target = work / "input.png"
    if image.startswith(("http://", "https://")):
        urllib.request.urlretrieve(image, target)
        return target
    if image.startswith("data:"):
        image = image.split(",", 1)[1]
    target.write_bytes(base64.b64decode(image))
    return target


def _validate_predict(job_input: dict) -> dict:
    if not job_input.get("image"):
        raise InputError("image is required: a URL, a data URI, or base64")
    resolution = job_input.get("resolution", 1024)
    if resolution not in (512, 1024):
        raise InputError(f"resolution must be 512 or 1024, got {resolution!r}")
    nviews = int(job_input.get("nviews", 8))
    if not 0 <= nviews <= 60:
        raise InputError("nviews must be between 0 and 60")
    return {
        "image": job_input["image"],
        "seed": int(job_input.get("seed", 42)),
        "fov": float(job_input.get("fov", -1.0)),
        "resolution": resolution,
        "nviews": nviews,
        "view_resolution": int(job_input.get("view_resolution", 512)),
    }


def _validate_extract(job_input: dict) -> dict:
    if not job_input.get("state"):
        raise InputError("state is required: the base64 latent state a /predict returned")
    decimation = int(job_input.get("decimation_target", MAX_MESH_VERTICES))
    if not 1000 <= decimation <= MAX_MESH_VERTICES:
        raise InputError(f"decimation_target must be between 1000 and {MAX_MESH_VERTICES}")
    return {
        "state": job_input["state"],
        "decimation_target": decimation,
        "texture_size": int(job_input.get("texture_size", 2048)),
    }


def _to_usd(glb: Path, work: Path) -> Path:
    """Writes the base USD layer for this asset (RFD 0053). The GLB is an
    asset-path attribute, not a `references` arc -- plain usd-core reads no
    glTF, and a reference would fail to resolve."""
    from pxr import Sdf, Usd, UsdGeom

    layer = work / "layer.usda"
    stage = Usd.Stage.CreateNew(str(layer))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Asset")
    stage.SetDefaultPrim(root.GetPrim())
    geometry = stage.DefinePrim("/Asset/Geometry")
    geometry.CreateAttribute("weftspun:sourceAsset", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath(glb.name)
    )
    geometry.CreateAttribute("weftspun:sourceFormat", Sdf.ValueTypeNames.Token).Set("gltf")
    geometry.CreateAttribute("weftspun:stage", Sdf.ValueTypeNames.Token).Set("image_to_mesh")
    stage.GetRootLayer().Save()
    return layer


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def predict(job_input: dict) -> dict:
    """image -> latent state + preview views. No GLB decode."""
    args = _validate_predict(job_input)
    work = Path(tempfile.mkdtemp())
    image_path = _fetch(args["image"], work)

    if STUB:
        stub_png = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8"
            "z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
        )
        return {
            "state": _b64(b"stubstate"),
            "views": [_b64(stub_png)] * args["nviews"],
            "cameras": [],
            "seed": args["seed"],
            "stub": True,
        }

    import worker_entry

    result = worker_entry.generate(
        str(image_path),
        seed=args["seed"],
        resolution=args["resolution"],
        manual_fov=args["fov"],
        nviews=args["nviews"],
        view_resolution=args["view_resolution"],
    )
    return {
        "state": _b64(result["state"]),
        "views": [_b64(v) for v in result["views"]],
        "cameras": result["cameras"],
        "camera_params": result["camera_params"],
        "seed": args["seed"],
        "stub": False,
    }


def extract(job_input: dict) -> dict:
    """latent state -> GLB + USD layer. The one decode."""
    args = _validate_extract(job_input)
    work = Path(tempfile.mkdtemp())
    glb = work / "output.glb"

    if STUB:
        glb.write_bytes(bytes([0x67, 0x6C, 0x54, 0x46, 0x02]) + b"stub")
    else:
        import worker_entry

        worker_entry.extract(
            base64.b64decode(args["state"]),
            args["decimation_target"],
            args["texture_size"],
            str(glb),
        )

    layer = _to_usd(glb, work)
    return {"glb": _b64(glb.read_bytes()), "layer": _b64(layer.read_bytes()), "stub": STUB}


def load() -> None:
    """In STUB mode nothing loads. Otherwise worker_entry.init_models()
    loads upstream's pipeline + image-cond models + MoGe, once."""
    if STUB:
        _READY["loaded"] = True
        return

    import worker_entry

    worker_entry.init_models()
    _READY["loaded"] = True


def build_app():
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel

    app = FastAPI(title="pixal3d", version="0.3.0")

    class PredictRequest(BaseModel):
        image: str
        seed: int = 42
        fov: float = -1.0
        resolution: int = 1024
        nviews: int = 8
        view_resolution: int = 512

    class ExtractRequest(BaseModel):
        state: str
        decimation_target: int = MAX_MESH_VERTICES
        texture_size: int = 2048

    @app.get("/health")
    def health():
        return {"status": "ok", "ready": _READY["loaded"], "stub": STUB}

    @app.post("/predict")
    def run_predict(request: PredictRequest):
        try:
            return predict(request.model_dump())
        except InputError as error:
            # 400 and not 500: the caller can fix this.
            return JSONResponse(status_code=400, content={"error": str(error)})

    @app.post("/extract")
    def run_extract(request: ExtractRequest):
        try:
            return extract(request.model_dump())
        except InputError as error:
            return JSONResponse(status_code=400, content={"error": str(error)})

    return app


if __name__ == "__main__":
    import uvicorn

    load()
    uvicorn.run(build_app(), host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
