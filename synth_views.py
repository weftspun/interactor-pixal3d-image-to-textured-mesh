"""Novel-view synthesis for the views-in half of the multiview loop.

photo + camera angles -> N synthesized views, which feed /predict (the
frontal/best view is meshed; the others are held for the VoxHammer /
differentiable-render adaptation pass -- upstream Pixal3D conditions on a
single image, so views are not batched into one generation).

Recipe verified against multimodalart's qwen-image-multiple-angles-3d-camera
Space (the exact code it runs), not guessed:
  base:      Qwen/Qwen-Image-Edit-2511 (QwenImageEditPlusPipeline)
  LoRA 1:    lightx2v/Qwen-Image-Edit-2511-Lightning (4-step distill)
  LoRA 2:    fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA (camera control)
  prompt:    "<sks> {azimuth name} {elevation name} {distance name}"
  sampling:  4 steps, guidance 1.0

Torch 2.4 note: if running on the pinned torch==2.4.1 stack, apply the same
SDPA enable_gqa shim used across this repo family (see README).

Usage:
  python synth_views.py --image fullbody.jpg --out views/ \
      --azimuths 0 45 90 315 --elevation 0 --distance 1.0 --seed 42
"""

import argparse
import time
from pathlib import Path

AZIMUTH_MAP = {
    0: "front view",
    45: "front-right quarter view",
    90: "right side view",
    135: "back-right quarter view",
    180: "back view",
    225: "back-left quarter view",
    270: "left side view",
    315: "front-left quarter view",
}
ELEVATION_MAP = {
    -30: "low-angle shot",
    0: "eye-level shot",
    30: "elevated shot",
    60: "high-angle shot",
}
DISTANCE_MAP = {0.6: "close-up", 1.0: "medium shot", 1.8: "wide shot"}


def snap(value, options):
    return min(options, key=lambda x: abs(x - value))


def build_camera_prompt(azimuth: float, elevation: float, distance: float) -> str:
    return (
        f"<sks> {AZIMUTH_MAP[snap(azimuth, list(AZIMUTH_MAP))]} "
        f"{ELEVATION_MAP[snap(elevation, list(ELEVATION_MAP))]} "
        f"{DISTANCE_MAP[snap(distance, list(DISTANCE_MAP))]}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--out", default="views")
    parser.add_argument("--azimuths", type=float, nargs="+", default=[0, 45, 90, 315])
    parser.add_argument("--elevation", type=float, default=0.0)
    parser.add_argument("--distance", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--size", type=int, default=1024)
    args = parser.parse_args()

    import torch

    # torch 2.4 compat shim (enable_gqa kwarg landed in torch 2.5; it is a
    # GQA fast-path hint, not a correctness flag).
    _sdpa = torch.nn.functional.scaled_dot_product_attention

    def _sdpa_compat(*a, enable_gqa=False, **kw):
        return _sdpa(*a, **kw)

    torch.nn.functional.scaled_dot_product_attention = _sdpa_compat

    from PIL import Image
    from diffusers import QwenImageEditPlusPipeline

    t0 = time.time()
    # bnb 4-bit on the big components: the Space's recipe runs bf16 on
    # 80GB cards; on the RTX 4090 tier (RFD 0027) the transformer alone
    # overflows 24GB, so quantization is our one documented divergence.
    from diffusers.quantizers import PipelineQuantizationConfig

    pipe = QwenImageEditPlusPipeline.from_pretrained(
        "Qwen/Qwen-Image-Edit-2511",
        quantization_config=PipelineQuantizationConfig(
            quant_backend="bitsandbytes_4bit",
            quant_kwargs={"load_in_4bit": True, "bnb_4bit_compute_dtype": torch.bfloat16},
            components_to_quantize=["transformer", "text_encoder"],
        ),
        torch_dtype=torch.bfloat16,
    )
    pipe.load_lora_weights(
        "lightx2v/Qwen-Image-Edit-2511-Lightning",
        weight_name="Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
        adapter_name="lightning",
    )
    pipe.load_lora_weights(
        "fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA",
        weight_name="qwen-image-edit-2511-multiple-angles-lora.safetensors",
        adapter_name="angles",
    )
    pipe.set_adapters(["lightning", "angles"], adapter_weights=[1.0, 1.0])
    pipe.to("cuda")
    pipe.vae.enable_tiling()
    print(f"[timing] pipeline+loras load: {time.time()-t0:.1f}s", flush=True)

    image = Image.open(args.image).convert("RGB")
    image.thumbnail((args.size, args.size))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for azimuth in args.azimuths:
        prompt = build_camera_prompt(azimuth, args.elevation, args.distance)
        t1 = time.time()
        with torch.inference_mode():
            result = pipe(
                image=[image],
                prompt=prompt,
                num_inference_steps=args.steps,
                true_cfg_scale=1.0,
                generator=torch.manual_seed(args.seed),
            )
        target = out / f"view_az{int(azimuth):03d}_el{int(args.elevation):+03d}.png"
        result.images[0].save(target)
        print(
            f"[timing] az={azimuth}: {time.time()-t1:.1f}s -> {target} ({prompt!r})",
            flush=True,
        )


if __name__ == "__main__":
    main()
