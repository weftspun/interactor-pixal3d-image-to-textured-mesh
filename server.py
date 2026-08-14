"""Pixal3D image to textured mesh, PBR. RFD 0040.

An HTTP server in a plain Docker image. RFD 0036 records why this is
not a Cog.

vast.ai rents a whole instance and runs a Docker image on it, thus
there is no serverless handler contract to satisfy. The image serves a
port, and the instance lives as long as it is rented.

The target is an RTX 4090 with 24 GB. Pixal3D peaks at 6.50 GB with
`low_vram` on, thus that card holds it with room for the activations.

Upstream is TencentARC/Pixal3D, MIT. Three cascading stages, each a
diffusion transformer:

  1. sparse structure   32 -> 64
  2. shape              256 -> 512 -> 1024
  3. texture            256 -> 512 -> 1024

RFD 0053 makes OpenUSD the internal format, thus a result carries a
USD layer beside the GLB that ships.
"""

import base64
import os
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

# Upstream reads these before it imports torch.
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("ATTN_BACKEND", "flash_attn")

SRC = os.environ.get("PIXAL3D_SRC", "/src/Pixal3D")
WEIGHTS = os.environ.get("PIXAL3D_WEIGHTS", "/weights/pixal3d")

# src/library/aiModelsCatalog.js API_MAX_MESH_VERTICES. The API rejects
# a mesh above this, thus it fails every later stage.
MAX_MESH_VERTICES = 210000

# WEFTSPUN_STUB=1 skips the model and answers with the same shape. It
# exists so the server contract can be tested in Docker with no GPU
# and no 24 GB download. RFD 0040 records that test.
STUB = os.environ.get("WEFTSPUN_STUB") == "1"

# The model loads once, at start, and not per request. The weights are
# 24.045 GB across seven files. A load per request would dominate every
# response, and an instance is rented by the hour either way.
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


def _validate(job_input: dict) -> dict:
    """Reads the input, and names what is wrong before any GPU work."""
    if not job_input.get("image"):
        raise InputError("image is required: a URL, a data URI, or base64")

    resolution = job_input.get("resolution", -1)
    if resolution not in (-1, 512, 1024):
        raise InputError(f"resolution must be -1, 512, or 1024, got {resolution!r}")

    decimation = int(job_input.get("decimation_target", MAX_MESH_VERTICES))
    if not 1000 <= decimation <= MAX_MESH_VERTICES:
        raise InputError(f"decimation_target must be between 1000 and {MAX_MESH_VERTICES}")

    return {
        "image": job_input["image"],
        "seed": int(job_input.get("seed", 42)),
        "fov": float(job_input.get("fov", -1.0)),
        "resolution": resolution,
        "low_vram": bool(job_input.get("low_vram", True)),
        "decimation_target": decimation,
    }


def _run_upstream(image_path: Path, glb: Path, args: dict) -> None:
    """Calls upstream's own entry point.

    A copy of the three-stage cascade here would drift from the commit
    this image pins.
    """
    command = [
        sys.executable,
        "inference.py",
        "--image",
        str(image_path),
        "--output",
        str(glb),
        "--seed",
        str(args["seed"]),
        "--fov",
        str(args["fov"]),
        "--model_path",
        WEIGHTS,
        "--resolution",
        str(args["resolution"]),
    ]

    if args["low_vram"]:
        command.append("--low_vram")

    subprocess.run(command, cwd=SRC, check=True)


def _to_usd(glb: Path, work: Path) -> Path:
    """Writes the base USD layer for this asset.

    RFD 0053 makes USD the internal format. This model authors the base
    layer, thus every later stage adds a sublayer over it and none
    rewrites this one.

    The GLB is recorded as an asset path, and not as a `references`
    arc. A reference makes USD resolve and open the target, and plain
    `usd-core` reads no glTF. That resolution fails with:

        Cannot determine file format for @output.glb@

    A glTF file format plugin would make the arc work. Until this image
    carries one, the attribute states where the geometry is without
    claiming USD can open it.
    """
    from pxr import Sdf, Usd, UsdGeom

    layer = work / "layer.usda"
    stage = Usd.Stage.CreateNew(str(layer))

    # Y up, and one unit is one metre. Every later stage reads these,
    # and a stage that guesses them puts the asset at the wrong scale.
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


def _encode(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def predict(job_input: dict) -> dict:
    """Runs one request. `job_input` is the JSON body."""
    args = _validate(job_input)

    work = Path(tempfile.mkdtemp())
    image_path = _fetch(args["image"], work)
    glb = work / "output.glb"

    if STUB:
        # The contract, with no model. These bytes are not a mesh, and
        # the shape is the one a real run returns.
        glb.write_bytes(bytes([0x67, 0x6C, 0x54, 0x46, 0x02]) + b"stub")
    else:
        _run_upstream(image_path, glb, args)

    layer = _to_usd(glb, work)

    return {
        "glb": _encode(glb),
        "layer": _encode(layer),
        "seed": args["seed"],
        "stub": STUB,
    }


def load() -> None:
    """Checks the image holds what a run needs, before it serves.

    A server that answers `ready` and then fails every request is worse
    than one that never starts.
    """
    if STUB:
        _READY["loaded"] = True
        return

    if not Path(SRC).is_dir():
        raise RuntimeError("the upstream source is absent: " + SRC)

    if not Path(WEIGHTS).is_dir():
        raise RuntimeError("the weights are absent: " + WEIGHTS)

    _READY["loaded"] = True


def build_app():
    """The HTTP surface. Two routes, and nothing else.

    This image serves one model, thus a route tree would be one entry
    long.
    """
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel

    app = FastAPI(title="pixal3d", version="0.1.0")

    class PredictRequest(BaseModel):
        image: str
        seed: int = 42
        fov: float = -1.0
        resolution: int = -1
        low_vram: bool = True
        decimation_target: int = MAX_MESH_VERTICES

    @app.get("/health")
    def health():
        """vast.ai runs no health probe of its own, thus a caller polls
        this until the instance is ready."""
        return {"status": "ok", "ready": _READY["loaded"], "stub": STUB}

    @app.post("/predict")
    def run(request: PredictRequest):
        try:
            return predict(request.model_dump())
        except InputError as error:
            # 400 and not 500. The caller can fix this, and a 500 would
            # send them to retry a request that can never work.
            return JSONResponse(status_code=400, content={"error": str(error)})

    return app


if __name__ == "__main__":
    import uvicorn

    load()
    uvicorn.run(build_app(), host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
