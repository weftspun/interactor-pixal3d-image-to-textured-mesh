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

| Field     | What                                                                                                                                                                                                                                                                                                   |
| --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `state`   | The packed latent (`shape_slat` + `tex_slat` + coords + res as base64 npz) — the inter-stage contract. A latent-space consumer (e.g. a VoxHammer edit; Pixal3D `main` is built on the TRELLIS.2 backbone, the same slat family VoxHammer inverts) takes this directly, no decode/re-encode round trip. |
| `views`   | `nviews` preview renders (base64 PNG) from upstream's **nvdiffrast-backed** renderer — the same differentiable path a later adaptation pass backpropagates through.                                                                                                                                    |
| `cameras` | Per-view extrinsics/intrinsics (callers need the cameras for any downstream fitting).                                                                                                                                                                                                                  |

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

**Contract stage verified; worker stage faithful-but-unproven.** v0.3.0 vendors upstream
app.py's own init/generate/extract into `worker_entry.py` (image-cond models, MoGe camera
estimation, preprocessing — all the parts a shallow port would miss), and the Dockerfile builds
the py3.10 + prebuilt-wheels env upstream's `requirements-hfdemo.txt` pins (no source compiles).
Not yet executed on a GPU — one real `/predict`→`/extract` round trip is the remaining gate
before the worker stage can be trusted.

**Corrected: "no source compiles" holds on the Space and nowhere else.** `requirements-hfdemo.txt`
pins a natten wheel whose kernels are sm_90, and this desk's 3090 is sm_86 — it installs,
imports, carries its `.so`, and then fails inside a diffusion step with `no kernel image is
available for execution`. natten has to be built for the target architecture, a thirty-minute
compile, and it is the only one of the six wheels that does. vast.ai rents whatever is free, so
the worker image inherits this. `desktop/` records the measurement, and `desktop/smoke.py` is
the check that answers it by launching a kernel rather than importing a module.

The cascade itself has now run on this desk and produced meshes, through upstream's
`inference.py` in the image `desktop/Dockerfile` describes. That is not the same as the worker
stage: no `/predict`→`/extract` round trip has been served, so the sentence above still stands
for `server.py`.

Known upstream-recipe caveat for `synth_views.py`: on the 24GB tier it quantizes with bnb 4-bit,
and this session's runs produced camera-correct but noise-corrupted images across every
software combination tried (diffusers 0.35/0.36, model 2508/2511, cfg/Lightning) — the one
constant was bnb 4-bit on torch 2.4.1 (bnb even warns of a misaligned inner dimension, 3420 %
64 != 0). Treat 4-bit Qwen-Image on that stack as suspect; retest on torch>=2.6 or with 8-bit
before relying on it.
