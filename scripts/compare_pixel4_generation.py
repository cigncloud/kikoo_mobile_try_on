"""Run a fixed-input FP32 CUDA control and analyze the Pixel 4 generation."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from prepare_pixel4_generation import ROOT, blend_tiles, posterior_sample, write_json


def reference(output: Path, checkpoint: Path) -> None:
    from Mobile_VTON.models.autoencoders.vae import Decoder
    from Mobile_VTON.models.unets.unet_2d_condition_tryon import UNet2DConditionModel
    from mobile_vton_vae import EncoderOnlyVAE

    if not torch.cuda.is_available():
        raise RuntimeError("Reference requires CUDA in the pinned vLLM container")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    folder = output / "reference"
    folder.mkdir(exist_ok=True)
    config = json.loads((output / "generation.json").read_text())
    data = output / "data"
    image = np.fromfile(data / "person.f32", np.float32).reshape(1, 3, 1024, 768)
    vae = (
        EncoderOnlyVAE.from_checkpoint(checkpoint / "vae")
        .eval()
        .requires_grad_(False)
        .float()
        .cuda()
    )
    vae.tile_sample_min_size = 512
    vae.tile_latent_min_size = 64
    vae.tile_overlap_factor = 0.25
    vae.enable_tiling()
    start = time.perf_counter()
    with torch.inference_mode():
        moments = (
            vae.encode(torch.from_numpy(image).cuda())
            .latent_dist.parameters.float()
            .cpu()
            .numpy()
        )
    moments.tofile(folder / "person_moments.f32")
    epsilon = np.fromfile(data / "person_epsilon.f32", np.float32).reshape(
        1, 16, 128, 96
    )
    person = (posterior_sample(moments, epsilon) - 0.0609) * 1.5305
    person.tofile(folder / "person_latent.f32")
    del vae
    gc.collect()
    torch.cuda.empty_cache()
    print("Reference person VAE complete", flush=True)
    model = UNet2DConditionModel.from_pretrained(
        checkpoint / "denoiser", local_files_only=True
    )
    model.set_sample_size_rsd(256, 96)
    model = model.eval().requires_grad_(False).float().cuda()
    cloth = np.fromfile(data / "garment_latent.f32", np.float32).reshape(1, 16, 128, 96)
    condition = torch.from_numpy(np.concatenate([person, cloth], axis=2)).cuda()
    noise = np.fromfile(data / "initial_noise.f32", np.float32).reshape(1, 16, 128, 96)
    latents = torch.from_numpy(np.concatenate([noise, noise], axis=2)).clone()
    branches = ("unconditional", "conditional")
    text = {
        b: torch.from_numpy(
            np.fromfile(data / f"text_{b}.f32", np.float32).reshape(1, 333, 4096)
        ).cuda()
        for b in branches
    }
    images = {
        b: torch.from_numpy(
            np.fromfile(data / f"image_{b}.f32", np.float32).reshape(1, 16, 4096)
        ).cuda()
        for b in branches
    }
    shapes = json.loads((output / "garment_precompute.json").read_text())["steps"][0][
        "shapes"
    ]
    with torch.inference_mode():
        for step, timestep in enumerate(config["timesteps"]):
            features = []
            for index, shape in enumerate(shapes):
                doubled = [shape[0], shape[1] * 2, shape[2]]
                values = (
                    np.fromfile(
                        data / f"features_{step:02d}/feature_{index}.f16", np.float16
                    )
                    .astype(np.float32)
                    .reshape(doubled)
                )
                features.append(torch.from_numpy(values).cuda())
            sample = torch.cat([latents.cuda(), condition], dim=1)
            predictions = []
            for branch in branches:
                prediction = (
                    model(
                        sample=sample,
                        timestep=torch.tensor([timestep], device="cuda"),
                        encoder_hidden_states=text[branch],
                        added_cond_kwargs={"image_embeds": images[branch]},
                        garment_features=features
                        if branch == "conditional"
                        else [torch.zeros_like(x) for x in features],
                        return_dict=False,
                    )[0]
                    .float()
                    .cpu()
                )
                if not torch.isfinite(prediction).all():
                    raise ValueError("Reference produced nonfinite values")
                predictions.append(prediction)
                if step == 0:
                    prediction.numpy().tofile(folder / f"first_{branch}.f32")
            delta = np.float32(config["sigmas"][step + 1]) - np.float32(
                config["sigmas"][step]
            )
            guided = predictions[0] + config["guidance_scale"] * (
                predictions[1] - predictions[0]
            )
            latents = latents + float(delta) * guided
            latents.numpy().tofile(folder / f"latents_step_{step + 1:02d}.f32")
            print(f"Reference CFG step {step + 1}/28", flush=True)
    latent = latents[:, :, :128] / 1.5305 + 0.0609
    latent.numpy().tofile(folder / "decoder_latents.f32")
    del model, condition, text, images, features, sample, predictions, prediction
    gc.collect()
    torch.cuda.empty_cache()
    decoder = Decoder(
        **json.loads((checkpoint / "vae_decoder/decoder.json").read_text())
    )
    decoder.load_state_dict(
        torch.load(
            checkpoint / "vae_decoder/decoder.pt", map_location="cpu", weights_only=True
        )
    )
    decoder = decoder.eval().requires_grad_(False).float().cuda()
    with torch.inference_mode():
        pixels = decoder(latent.cuda()).float().cpu().numpy()
    pixels.tofile(folder / "image.f32")
    save_image(pixels, folder / "reference.png")
    write_json(
        folder / "metrics.json",
        {
            "status": "completed",
            "seconds": time.perf_counter() - start,
            "dtype": "FP32",
            "tf32": False,
            "garment_features": "same FP16 precomputed features as phone",
            "steps": 28,
            "guidance_scale": 2.5,
            "rng": config["source"]["rng"],
        },
    )


def save_image(pixels: np.ndarray, path: Path) -> None:
    pixels = pixels.reshape(1, 3, 1024, 768)
    if not np.isfinite(pixels).all():
        raise ValueError("Nonfinite pixels")
    array = (
        ((pixels[0].transpose(1, 2, 0) / 2 + 0.5).clip(0, 1) * 255)
        .round()
        .astype(np.uint8)
    )
    Image.fromarray(array).save(path)


def compare(actual: np.ndarray, expected: np.ndarray) -> dict:
    if (
        actual.shape != expected.shape
        or not np.isfinite(actual).all()
        or not np.isfinite(expected).all()
    ):
        raise ValueError("Invalid comparison tensors")
    difference = actual.astype(np.float64) - expected.astype(np.float64)
    rmse = float(np.sqrt(np.mean(difference**2)))
    return {
        "mae": float(np.abs(difference).mean()),
        "rmse": rmse,
        "relative_rmse": rmse
        / max(float(np.sqrt(np.mean(expected.astype(np.float64) ** 2))), 1e-12),
        "max_absolute_error": float(np.abs(difference).max()),
    }


def analyze(output: Path) -> None:
    metrics = {}
    for phase in ("vae", "denoise", "decode"):
        metrics[phase] = json.loads((output / f"{phase}_metrics.json").read_text())
        if metrics[phase]["status"] != "completed":
            raise ValueError("Incomplete generation")
    comparisons = {}
    config = json.loads((output / "generation.json").read_text())
    noise = np.fromfile(output / "data/initial_noise.f32", np.float32).reshape(
        1, 16, 128, 96
    )
    initial = np.concatenate([noise, noise], axis=2)
    negative = np.fromfile(output / "first_unconditional.f32", np.float32).reshape(
        initial.shape
    )
    positive = np.fromfile(output / "first_conditional.f32", np.float32).reshape(
        initial.shape
    )
    delta = np.float32(config["sigmas"][1]) - np.float32(config["sigmas"][0])
    expected_step = initial + delta * (
        negative + np.float32(config["guidance_scale"]) * (positive - negative)
    )
    actual_step = np.fromfile(output / "latents_step_01.f32", np.float32).reshape(
        initial.shape
    )
    scheduler_check = compare(actual_step, expected_step)
    if scheduler_check["max_absolute_error"] > 1e-6:
        raise ValueError("Native CFG/Euler integration does not match the fixed inputs")
    tiles = []
    for i, height in enumerate((64, 64, 32)):
        tiles.append(
            [
                np.fromfile(output / f"moments_tile_{i}_{j}.f32", np.float32).reshape(
                    1, 32, height, width
                )
                for j, width in enumerate((64, 48))
            ]
        )
    native_moments = np.fromfile(output / "person_moments.f32", np.float32).reshape(
        1, 32, 128, 96
    )
    blend_check = compare(native_moments, blend_tiles(tiles))
    if blend_check["max_absolute_error"] > 1e-5:
        raise ValueError(
            "Native VAE tile blending differs from the reference algorithm"
        )
    for name in (
        "person_moments",
        "person_latent",
        "first_unconditional",
        "first_conditional",
        "decoder_latents",
        "image",
    ):
        actual = np.fromfile(output / f"{name}.f32", np.float32)
        expected = np.fromfile(output / f"reference/{name}.f32", np.float32)
        comparisons[name] = compare(actual, expected)
    steps = []
    for step in range(1, 29):
        name = f"latents_step_{step:02d}.f32"
        steps.append(
            {
                "step": step,
                **compare(
                    np.fromfile(output / name, np.float32),
                    np.fromfile(output / "reference" / name, np.float32),
                ),
            }
        )
    pixels = np.fromfile(output / "image.f32", np.float32).reshape(1, 3, 1024, 768)
    save_image(pixels, output / "pixel4_result.png")
    pairs = [
        ("Person", ROOT / "experiments/2026-10-23-mobile_vton/inputs/person.jpg"),
        ("Garment", ROOT / "experiments/2026-10-23-mobile_vton/inputs/garment.jpg"),
        ("FP32 control", output / "reference/reference.png"),
        ("Pixel 4 mixed", output / "pixel4_result.png"),
    ]
    sheet = Image.new("RGB", (384 * 4, 548), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (label, path) in enumerate(pairs):
        image = (
            Image.open(path).convert("RGB").resize((384, 512), Image.Resampling.LANCZOS)
        )
        sheet.paste(image, (384 * index, 36))
        draw.text((384 * index + 12, 12), label, fill="black")
    sheet.save(output / "comparison.png")
    import re

    log = (output / "generation.monitor.log").read_text()
    wall = re.findall(r"online_wall_seconds=(\d+)", log)
    if not wall or "exit_code=0" not in log:
        raise ValueError("Missing end-to-end completion marker")
    available = [int(x) for x in re.findall(r"mem_available_kb=(\d+)", log)]
    result = {
        "status": "completed",
        "full_generation_completed_on_phone": True,
        "steps": 28,
        "cfg_forwards": 56,
        "resolution": [768, 1024],
        "online_cold_wall_seconds": int(wall[-1]),
        "process_peak_rss_bytes": max(
            x["process_peak_rss_bytes"] for x in metrics.values()
        ),
        "mem_available_min_kib": min(available),
        "phases": metrics,
        "comparisons": comparisons,
        "native_math_checks": {
            "first_cfg_euler_step": scheduler_check,
            "vae_tile_blending": blend_check,
        },
        "per_step_comparisons": steps,
        "image": str(output / "pixel4_result.png"),
        "comparison_image": str(output / "comparison.png"),
        "scope": "One fixed person/garment pair and seed; cached garment features and image/text conditioning are precomputed. Neural models request OpenCL, while CPU handles CFG, Euler integration, tile stitching and posterior arithmetic. RSS is not exclusive GPU memory and may omit driver allocations. FP32 control shares FP16 garment feature cache.",
    }
    write_json(output / "comparison.json", result)
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in ("phases", "per_step_comparisons")
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["reference", "analyze"], required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "experiments/2026-10-24-pixel4/generation_hybrid_low",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("/tmp/mobile_tryon/weights/mobile_vton/checkpoint"),
    )
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.stage == "reference":
        reference(args.output, args.checkpoint)
    else:
        analyze(args.output)


if __name__ == "__main__":
    main()
