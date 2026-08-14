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
