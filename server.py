"""Pixal3D image to textured mesh, PBR. RFD 0040, extended per the
generate/extract latent flow upstream's own app.py ships.

An HTTP server in a plain Docker image. RFD 0036 records why this is
not a Cog.

Two routes now mirror upstream `app.py`'s split exactly:

  POST /predict  -- image -> latent state + nvdiffrast preview renders
                    (+ cameras). No GLB decode happens here.
  POST /extract  -- latent state -> GLB + USD layer. Decode happens
                    once, here, at the edge.

That split IS the decode-only latent contract: a downstream latent
consumer (e.g. a VoxHammer edit over the same TRELLIS.2-backbone slat
family) takes `state` and never pays for -- or loses quality to -- a
decode/re-encode round trip. The preview renders come from upstream's
nvdiffrast-backed MeshRenderer/PbrMeshRenderer, i.e. the same
differentiable-rendering path a later adaptation pass backpropagates
through.

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
import io
import os
import tempfile
import urllib.request
from pathlib import Path

# Upstream reads these before it imports torch.
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("ATTN_BACKEND", "flash_attn")

SRC = os.environ.get("PIXAL3D_SRC", "/src/Pixal3D")
WEIGHTS = os.environ.get("PIXAL3D_WEIGHTS", "TencentARC/Pixal3D")

# src/library/aiModelsCatalog.js API_MAX_MESH_VERTICES. The API rejects
# a mesh above this, thus it fails every later stage.
MAX_MESH_VERTICES = 210000

# WEFTSPUN_STUB=1 skips the model and answers with the same shape. It
# exists so the server contract can be tested in Docker with no GPU
# and no 24 GB download. RFD 0040 records that test.
STUB = os.environ.get("WEFTSPUN_STUB") == "1"

# The model loads once, at start, and not per request (24.045 GB of
# weights across seven files; a load per request would dominate every
# response, and an instance is rented by the hour either way).
_READY = {"loaded": False}
_MODELS = {"pipeline": None, "moge": None}


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

    resolution = job_input.get("resolution", -1)
    if resolution not in (-1, 512, 1024):
        raise InputError(f"resolution must be -1, 512, or 1024, got {resolution!r}")

    nviews = int(job_input.get("nviews", 8))
    if not 0 <= nviews <= 60:
        raise InputError("nviews must be between 0 and 60")

    return {
        "image": job_input["image"],
        "seed": int(job_input.get("seed", 42)),
        "fov": float(job_input.get("fov", -1.0)),
        "resolution": resolution,
        "low_vram": bool(job_input.get("low_vram", True)),
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


# --- Latent state, ported from upstream app.py's pack_state/unpack_state.
# One difference: the state travels IN the HTTP response as base64 npz
# bytes instead of a server-side temp path -- the server stays stateless,
# and the latent is the caller's to hold, pass to /extract, or hand to a
# latent-space editor (VoxHammer) without this instance staying alive.


def _pack_state(shape_slat, tex_slat, res: int) -> str:
    import numpy as np

    buffer = io.BytesIO()
    np.savez_compressed(
        buffer,
        shape_slat_feats=shape_slat.feats.cpu().numpy(),
        tex_slat_feats=tex_slat.feats.cpu().numpy(),
        coords=shape_slat.coords.cpu().numpy(),
        res=res,
    )
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _unpack_state(state_b64: str):
    import numpy as np
    import torch
    from trellis2.modules.sparse import SparseTensor  # upstream's tensor type

    data = np.load(io.BytesIO(base64.b64decode(state_b64)))
    shape_slat = SparseTensor(
        feats=torch.from_numpy(data["shape_slat_feats"]).cuda(),
        coords=torch.from_numpy(data["coords"]).cuda(),
    )
    tex_slat = shape_slat.replace(torch.from_numpy(data["tex_slat_feats"]).cuda())
    return shape_slat, tex_slat, int(data["res"])


def _run_generate(image_path: Path, args: dict) -> dict:
    """image -> latent state + preview renders. Ported from app.py's
    generate flow: pipeline.run(..., return_latent=True), then
    render_utils on the (not-yet-extracted) mesh sample. Faithful to
    upstream; not re-derived."""
    import numpy as np
    from PIL import Image

    pipeline = _MODELS["pipeline"]

    image = Image.open(image_path).convert("RGBA")

    mesh_list, (shape_slat, tex_slat, res) = pipeline.run(
        image,
        seed=args["seed"],
        return_latent=True,
    )
    mesh = mesh_list[0]
    state = _pack_state(shape_slat, tex_slat, res)

    views, cameras = [], []
    if args["nviews"] > 0:
        import sys

        sys.path.insert(0, SRC)
        from pixal3d.utils import render_utils

        result = render_utils.render_multiview(
            mesh, resolution=args["view_resolution"], nviews=args["nviews"]
        )
        # render_multiview returns frames + the camera parameters used;
        # callers need the cameras for any downstream fitting.
        frames, extrinsics, intrinsics = (
            result if isinstance(result, tuple) and len(result) == 3
            else (result, None, None)
        )
        for i, frame in enumerate(frames):
            buffer = io.BytesIO()
            Image.fromarray(np.asarray(frame)).save(buffer, format="PNG")
            views.append(base64.b64encode(buffer.getvalue()).decode("ascii"))
        if extrinsics is not None:
            cameras = [
                {
                    "extrinsics": np.asarray(e.cpu() if hasattr(e, "cpu") else e).tolist(),
                    "intrinsics": np.asarray(k.cpu() if hasattr(k, "cpu") else k).tolist(),
                }
                for e, k in zip(extrinsics, intrinsics)
            ]

    return {"state": state, "views": views, "cameras": cameras}


def _run_extract(args: dict, work: Path) -> Path:
    """latent state -> GLB. Ported from app.py's extract_glb_api:
    decode_latent then o_voxel.postprocess.to_glb."""
    import numpy as np

    pipeline = _MODELS["pipeline"]
    shape_slat, tex_slat, res = _unpack_state(args["state"])
    mesh = pipeline.decode_latent(shape_slat, tex_slat, res)[0]

    import o_voxel

    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices,
        faces=mesh.faces,
        attr_volume=mesh.attrs,
        coords=mesh.coords,
        attr_layout=pipeline.pbr_attr_layout,
        grid_size=res,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=args["decimation_target"],
        texture_size=args["texture_size"],
        remesh=True,
        remesh_band=1,
        remesh_project=0,
        use_tqdm=True,
    )
    # Upstream's axis correction, verbatim.
    rot = np.array(
        [[-1, 0, 0, 0], [0, 0, -1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=np.float64
    )
    glb.apply_transform(rot)

    target = work / "output.glb"
    glb.export(str(target))
    return target


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
    """image -> latent state + preview views. No GLB decode."""
    args = _validate_predict(job_input)

    work = Path(tempfile.mkdtemp())
    image_path = _fetch(args["image"], work)

    if STUB:
        # The contract, with no model. The state is not a latent, and
        # the views are not renders; the shape is what a real run returns.
        stub_png = base64.b64encode(
            base64.b64decode(
                "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8"
                "z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
            )
        ).decode("ascii")
        result = {
            "state": base64.b64encode(b"stubstate").decode("ascii"),
            "views": [stub_png] * args["nviews"],
            "cameras": [],
        }
    else:
        result = _run_generate(image_path, args)

    result.update({"seed": args["seed"], "stub": STUB})
    return result


def extract(job_input: dict) -> dict:
    """latent state -> GLB + USD layer. The one decode."""
    args = _validate_extract(job_input)
    work = Path(tempfile.mkdtemp())

    if STUB:
        glb = work / "output.glb"
        glb.write_bytes(bytes([0x67, 0x6C, 0x54, 0x46, 0x02]) + b"stub")
    else:
        glb = _run_extract(args, work)

    layer = _to_usd(glb, work)

    return {
        "glb": _encode(glb),
        "layer": _encode(layer),
        "stub": STUB,
    }


def load() -> None:
    """Loads the pipeline in-process, once, per upstream app.py's
    init_models (low-VRAM branch). In STUB mode nothing loads."""
    if STUB:
        _READY["loaded"] = True
        return

    if not Path(SRC).is_dir():
        raise RuntimeError("the upstream source is absent: " + SRC)

    import sys

    sys.path.insert(0, SRC)
    from pixal3d.pipelines import Pixal3DImageTo3DPipeline

    pipeline = Pixal3DImageTo3DPipeline.from_pretrained(WEIGHTS)
    pipeline._device = __import__("torch").device("cuda")
    pipeline.low_vram = True
    _MODELS["pipeline"] = pipeline

    _READY["loaded"] = True


def build_app():
    """The HTTP surface: health, predict (latent+views), extract (GLB)."""
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel

    app = FastAPI(title="pixal3d", version="0.2.0")

    class PredictRequest(BaseModel):
        image: str
        seed: int = 42
        fov: float = -1.0
        resolution: int = -1
        low_vram: bool = True
        nviews: int = 8
        view_resolution: int = 512

    class ExtractRequest(BaseModel):
        state: str
        decimation_target: int = MAX_MESH_VERTICES
        texture_size: int = 2048

    @app.get("/health")
    def health():
        """vast.ai runs no health probe of its own, thus a caller polls
        this until the instance is ready."""
        return {"status": "ok", "ready": _READY["loaded"], "stub": STUB}

    @app.post("/predict")
    def run_predict(request: PredictRequest):
        try:
            return predict(request.model_dump())
        except InputError as error:
            # 400 and not 500. The caller can fix this, and a 500 would
            # send them to retry a request that can never work.
            return JSONResponse(status_code=400, content={"error": str(error)})

    @app.post("/extract")
    def run_extract_route(request: ExtractRequest):
        try:
            return extract(request.model_dump())
        except InputError as error:
            return JSONResponse(status_code=400, content={"error": str(error)})

    return app


if __name__ == "__main__":
    import uvicorn

    load()
    uvicorn.run(build_app(), host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
