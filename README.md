# interactor-pixal3d-image-to-textured-mesh

Standalone repo for `pixal3d_image_to_textured_mesh`, ported verbatim from
[weftspun's RFD 0040](https://github.com/weftspun/request-for-discussion/tree/main/0040-pixal3d-image-to-textured-mesh)
— RFD 0036's own worked example, already built and design-reviewed. Nothing changed
except the wrapper: file content below is copied as-is, not re-derived.

---

# RFD 0040: Model image for pixal3d_image_to_textured_mesh

**State:** discussion
**Feature:** model packaging

## Problem

Pixal3D is the image to 3D path in daily use. It writes a metal map
and a roughness map, which TRELLIS.2 does not.

This RFD first recorded its parameter count and its license as
unknown, and it asked for a measurement on the DGX. There is no DGX.
The measurement came from the published checkpoints instead.

## Decision

Package Pixal3D as the primary image to 3D worker. Upstream is
TencentARC/Pixal3D, and it uses the MIT license.

It is a plain Docker image that serves HTTP, and not a Cog. RFD 0036
records why: vast.ai rents an instance and runs a container on it.

Call the upstream `inference.py`, and do not reimplement the cascade.
A copy here would drift from the commit this image pins.

See `DETAILS.md` for the Docker contract-stage test, the checkpoint
measurement, and the format choice. It also gives the `predict()`
interface, the build-time downloads, and what this RFD corrects.

## Related

RFD 0026 gives the memory per model. RFD 0027 gives the GPU tier.
RFD 0038 packages the backbone. RFD 0053 gives the asset format.

---

# Extension: multiview loop (views in, latent + differentiable renders out)

Extends the verbatim RFD 0040 port above. Three pieces:

## 1. Generate/extract split (`server.py` v0.2.0) — the decode-only latent contract

`/predict` no longer decodes a GLB. It returns:

| Field | What |
|---|---|
| `state` | The packed latent (`shape_slat` + `tex_slat` + coords + res as base64 npz) — the inter-stage contract. A latent-space consumer (e.g. a VoxHammer edit; Pixal3D `main` is built on the TRELLIS.2 backbone, the same slat family VoxHammer inverts) takes this directly, no decode/re-encode round trip. |
| `views` | `nviews` preview renders (base64 PNG) from upstream's **nvdiffrast-backed** renderer — the same differentiable path a later adaptation pass backpropagates through. |
| `cameras` | Per-view extrinsics/intrinsics (callers need the cameras for any downstream fitting). |

`/extract` takes `state` (+ `decimation_target`, `texture_size`) and performs the one decode:
`decode_latent` → `to_glb` → GLB + USD layer. Ported verbatim from upstream `app.py`'s
`generate`/`extract_glb_api` split — nothing invented.

## 2. Views-in (`synth_views.py`)

photo + camera angles → N synthesized views, recipe verified against multimodalart's
`qwen-image-multiple-angles-3d-camera` Space code: `Qwen/Qwen-Image-Edit-2511` +
`lightx2v/…-Lightning` (4-step) + `fal/…-Multiple-Angles-LoRA`, prompt
`<sks> {azimuth} {elevation} {distance}`, cfg 1.0. Upstream Pixal3D conditions on a **single**
image, so the frontal/best view is meshed and the other views are held as targets for the
adaptation pass.

## 3. Adaptation (queued, not in this repo yet)

Image- or prompt-driven mesh adaptation goes through VoxHammer
(`interactor-voxhammer-*-mesh-editing`): invert the slat latent (flow sampler run backwards,
caching the per-timestep trajectory + attention KV), denoise under the target condition with
the cached source latents pasted back outside the mask each step, hard-splice at the end.
The `state` + `cameras` this server returns are exactly its inputs; the nvdiffrast renders are
the verification signal.

## Status

**Scaffolded from upstream's own app.py flow, not yet executed against the real weights.** The
port is faithful (pack/unpack_state, `pipeline.run(..., return_latent=True)`, `decode_latent`,
`to_glb`, `render_multiview`) but the import paths (`trellis2.modules.sparse.SparseTensor`,
`o_voxel`) and the in-process `load()` were not run on a GPU in this pass — confirm against a
real deployment before trusting the worker stage. STUB mode fully exercises both routes' contracts.
