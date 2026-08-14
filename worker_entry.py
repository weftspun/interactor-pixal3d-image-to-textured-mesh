"""Vendored worker core: upstream TencentARC/Pixal3D `app.py`'s own
init_models / generate / extract flow, verbatim minus gradio/spaces/progress
plumbing. server.py is a thin HTTP shim over this module.

Vendoring rationale (RFD 0040: "call the upstream entry point, do not
reimplement the cascade"): app.py IS upstream's entry point for the
latent-state + preview-render flow; a hand-rolled partial port (this repo's
v0.2.0) missed the image-cond model init, MoGe camera estimation, and
preprocessing, and would not run. Function bodies below are copied from
app.py; deviations are marked # WEFTSPUN.

Environment contract (see Dockerfile / requirements-hfdemo.txt upstream):
python 3.10 + torch 2.6 cu124 + the prebuilt wheels (flash_attn_3,
nvdiffrast, o_voxel, cumesh, flex_gemm, natten).
"""

import io
import math
import os
import time

import numpy as np
import torch
from PIL import Image

from pixal3d.modules.sparse import SparseTensor
from pixal3d.pipelines import Pixal3DImageTo3DPipeline
from pixal3d.utils import render_utils

import o_voxel

# --- Constants, verbatim from app.py ---
CASCADE_MAX_NUM_TOKENS = 49152
MOGE_MODEL_NAME = "Ruicheng/moge-2-vitl"
WILD_MESH_SCALE = 1.0
WILD_EXTEND_PIXEL = 0
WILD_IMAGE_RESOLUTION = 512

IMAGE_COND_CONFIGS = {
    "ss": {
        "model_name": "camenduru/dinov3-vitl16-pretrain-lvd1689m",
        "image_size": 512,
        "grid_resolution": 16,
    },
    "shape_512": {
        "model_name": "camenduru/dinov3-vitl16-pretrain-lvd1689m",
        "image_size": 512,
        "grid_resolution": 32,
        "use_naf_upsample": True,
        "naf_target_size": 512,
    },
    "shape_1024": {
        "model_name": "camenduru/dinov3-vitl16-pretrain-lvd1689m",
        "image_size": 1024,
        "grid_resolution": 64,
        "use_naf_upsample": True,
        "naf_target_size": 512,
    },
    "tex_1024": {
        "model_name": "camenduru/dinov3-vitl16-pretrain-lvd1689m",
        "image_size": 1024,
        "grid_resolution": 64,
        "use_naf_upsample": True,
        "naf_target_size": 1024,
    },
}

pipeline = None
moge_model = None
LOW_VRAM = os.environ.get("LOW_VRAM", "1") == "1"  # WEFTSPUN: default on (RFD 0027, 24GB tier)
MODEL_PATH = os.environ.get("PIXAL3D_WEIGHTS", "TencentARC/Pixal3D")


def build_image_cond_model(config: dict):
    from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import (
        DinoV3ProjFeatureExtractor,
    )

    model = DinoV3ProjFeatureExtractor(**config)
    model.eval()
    return model


def load_moge_model(device="cuda", model_name=MOGE_MODEL_NAME):
    from moge.model.v2 import MoGeModel

    moge_model = MoGeModel.from_pretrained(model_name).to(device)
    moge_model.eval()
    return moge_model


def init_models():
    """app.py's init_models, minus diagnostics/envmap (envmap only feeds the
    PBR-lit demo video; render_multiview does not need it). # WEFTSPUN"""
    global pipeline, moge_model
    if pipeline is not None:
        return

    pipeline = Pixal3DImageTo3DPipeline.from_pretrained(MODEL_PATH)
    pipeline.image_cond_model_ss = build_image_cond_model(IMAGE_COND_CONFIGS["ss"])
    pipeline.image_cond_model_shape_512 = build_image_cond_model(IMAGE_COND_CONFIGS["shape_512"])
    pipeline.image_cond_model_shape_1024 = build_image_cond_model(IMAGE_COND_CONFIGS["shape_1024"])
    pipeline.image_cond_model_tex_1024 = build_image_cond_model(IMAGE_COND_CONFIGS["tex_1024"])

    if LOW_VRAM:
        for attr in [
            "image_cond_model_ss",
            "image_cond_model_shape_512",
            "image_cond_model_shape_1024",
            "image_cond_model_tex_1024",
        ]:
            m = getattr(pipeline, attr, None)
            if m is not None and getattr(m, "use_naf_upsample", False):
                m._load_naf()
        pipeline._device = torch.device("cuda")
        pipeline.low_vram = True
        moge_model = load_moge_model(device="cpu")
    else:
        pipeline.low_vram = False
        pipeline.cuda()
        pipeline.image_cond_model_ss.cuda()
        pipeline.image_cond_model_shape_512.cuda()
        pipeline.image_cond_model_shape_1024.cuda()
        pipeline.image_cond_model_tex_1024.cuda()
        for attr in [
            "image_cond_model_ss",
            "image_cond_model_shape_512",
            "image_cond_model_shape_1024",
            "image_cond_model_tex_1024",
        ]:
            m = getattr(pipeline, attr, None)
            if m is not None and getattr(m, "use_naf_upsample", False):
                m._load_naf()
        moge_model = load_moge_model(device="cuda")


# --- Camera estimation, verbatim from app.py ---

def compute_f_pixels(camera_angle_x: float, resolution: int) -> float:
    focal_length = 16.0 / torch.tan(torch.tensor(camera_angle_x / 2.0))
    f_pixels = focal_length * resolution / 32.0
    return float(f_pixels.item())


def distance_from_fov(camera_angle_x, grid_point, target_point, mesh_scale, image_resolution):
    rotation_matrix = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    gp = grid_point.to(torch.float32) @ rotation_matrix.T
    gp = gp / mesh_scale / 2
    xw, yw, zw = gp[0].item(), gp[1].item(), gp[2].item()
    xt, yt = float(target_point[0].item()), float(target_point[1].item())
    f_pixels = compute_f_pixels(camera_angle_x, image_resolution)
    x_ndc = xt - image_resolution / 2.0
    y_ndc = -(yt - image_resolution / 2.0)
    distance_x = f_pixels * xw / x_ndc - yw
    return {"distance_from_x": float(distance_x), "f_pixels": float(f_pixels)}


def get_camera_params_wild_moge(
    image_path, device="cuda", mesh_scale=1.0, extend_pixel=0, image_resolution=512
):
    pil_image = Image.open(image_path).convert("RGB")
    width, height = pil_image.size
    image_np = np.array(pil_image).astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).to(device)
    if LOW_VRAM:
        moge_model.to(device)
    with torch.no_grad():
        output = moge_model.infer(image_tensor)
    if LOW_VRAM:
        moge_model.cpu()
        torch.cuda.empty_cache()
    intrinsics = output["intrinsics"].squeeze().cpu().numpy()
    fx_normalized = intrinsics[0, 0]
    fx = fx_normalized * width
    camera_angle_x = 2 * math.atan(width / (2 * fx))

    grid_point = torch.tensor([-1.0, 0.0, 0.0])
    distance = distance_from_fov(
        camera_angle_x,
        grid_point,
        torch.tensor([0 - extend_pixel, image_resolution - 1 + extend_pixel]),
        mesh_scale,
        image_resolution,
    )["distance_from_x"]
    return {"camera_angle_x": camera_angle_x, "distance": distance, "mesh_scale": mesh_scale}


# --- Latent state. app.py packs to a temp file; here the npz bytes go over
# HTTP so the server stays stateless. # WEFTSPUN ---

def pack_state_bytes(shape_slat, tex_slat, res: int) -> bytes:
    buffer = io.BytesIO()
    np.savez_compressed(
        buffer,
        shape_slat_feats=shape_slat.feats.cpu().numpy(),
        tex_slat_feats=tex_slat.feats.cpu().numpy(),
        coords=shape_slat.coords.cpu().numpy(),
        res=res,
    )
    return buffer.getvalue()


def unpack_state_bytes(blob: bytes):
    data = np.load(io.BytesIO(blob))
    shape_slat = SparseTensor(
        feats=torch.from_numpy(data["shape_slat_feats"]).cuda(),
        coords=torch.from_numpy(data["coords"]).cuda(),
    )
    tex_slat = shape_slat.replace(torch.from_numpy(data["tex_slat_feats"]).cuda())
    return shape_slat, tex_slat, int(data["res"])


# --- Generate / extract, following app.py's generate_3d / extract_glb_api ---

def generate(
    image_path: str,
    seed: int = 42,
    resolution: int = 1024,
    manual_fov: float = -1.0,
    nviews: int = 8,
    view_resolution: int = 512,
) -> dict:
    init_models()
    torch.manual_seed(seed)

    img = Image.open(image_path)
    image_preprocessed = pipeline.preprocess_image(img)
    processed_path = image_path + ".processed.png"
    image_preprocessed.save(processed_path)

    if manual_fov > 0:
        camera_angle_x = float(manual_fov)  # radians, matching inference.py's --fov
        grid_point = torch.tensor([-1.0, 0.0, 0.0])
        distance = distance_from_fov(
            camera_angle_x,
            grid_point,
            torch.tensor([0 - WILD_EXTEND_PIXEL, WILD_IMAGE_RESOLUTION - 1 + WILD_EXTEND_PIXEL]),
            WILD_MESH_SCALE,
            WILD_IMAGE_RESOLUTION,
        )["distance_from_x"]
        camera_params = {
            "camera_angle_x": camera_angle_x,
            "distance": distance,
            "mesh_scale": WILD_MESH_SCALE,
        }
    else:
        camera_params = get_camera_params_wild_moge(
            processed_path,
            device="cuda",
            mesh_scale=WILD_MESH_SCALE,
            extend_pixel=WILD_EXTEND_PIXEL,
            image_resolution=WILD_IMAGE_RESOLUTION,
        )

    pipeline_type = f"{int(resolution)}_cascade"
    mesh_list, (shape_slat, tex_slat, res) = pipeline.run(
        image_preprocessed,
        camera_params=camera_params,
        seed=seed,
        preprocess_image=False,
        return_latent=True,
        pipeline_type=pipeline_type,
        max_num_tokens=CASCADE_MAX_NUM_TOKENS,
    )
    mesh = mesh_list[0]
    state = pack_state_bytes(shape_slat, tex_slat, res)

    views, cameras = [], []
    if nviews > 0:
        frames, extrinsics, intrinsics = render_utils.render_multiview(
            mesh, resolution=view_resolution, nviews=nviews
        )
        for frame in frames:
            buffer = io.BytesIO()
            Image.fromarray(np.asarray(frame)).save(buffer, format="PNG")
            views.append(buffer.getvalue())
        cameras = [
            {
                "extrinsics": np.asarray(e.cpu() if hasattr(e, "cpu") else e).tolist(),
                "intrinsics": np.asarray(k.cpu() if hasattr(k, "cpu") else k).tolist(),
            }
            for e, k in zip(extrinsics, intrinsics)
        ]

    return {"state": state, "views": views, "cameras": cameras, "camera_params": camera_params}


def extract(state: bytes, decimation_target: int, texture_size: int, output_path: str) -> str:
    init_models()
    shape_slat, tex_slat, res = unpack_state_bytes(state)
    mesh = pipeline.decode_latent(shape_slat, tex_slat, res)[0]

    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices,
        faces=mesh.faces,
        attr_volume=mesh.attrs,
        coords=mesh.coords,
        attr_layout=pipeline.pbr_attr_layout,
        grid_size=res,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=decimation_target,
        texture_size=texture_size,
        remesh=True,
        remesh_band=1,
        remesh_project=0,
        use_tqdm=True,
    )
    rot = np.array(
        [[-1, 0, 0, 0], [0, 0, -1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=np.float64
    )
    glb.apply_transform(rot)
    glb.export(output_path)
    return output_path
