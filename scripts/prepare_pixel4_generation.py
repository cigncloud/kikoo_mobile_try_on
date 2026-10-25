"""Prepare pinned inputs and timestep-specific garment features for Pixel 4."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "models/mobile_vton/upstream"))


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n")


def posterior_sample(parameters: np.ndarray, noise: np.ndarray) -> np.ndarray:
    mean, logvar = np.split(parameters, 2, axis=1)
    return mean + np.exp(0.5 * np.clip(logvar, -30, 20)) * noise


def blend_tiles(tiles: list[list[np.ndarray]], extent: int = 16) -> np.ndarray:
    """Mirror Diffusers' vertical-then-horizontal, in-place tile blending."""
    result = []
    limit = 64 - extent
    for i, row in enumerate(tiles):
        pieces = []
        for j, tile in enumerate(row):
            if i:
                above = tiles[i - 1][j]
                count = min(above.shape[2], tile.shape[2], extent)
                for y in range(count):
                    tile[:, :, y, :] = above[:, :, -count + y, :] * (
                        1 - y / count
                    ) + tile[:, :, y, :] * (y / count)
            if j:
                left = row[j - 1]
                count = min(left.shape[3], tile.shape[3], extent)
                for x in range(count):
                    tile[:, :, :, x] = left[:, :, :, -count + x] * (
                        1 - x / count
                    ) + tile[:, :, :, x] * (x / count)
            pieces.append(tile[:, :, :limit, :limit])
        result.append(np.concatenate(pieces, axis=3))
    return np.concatenate(result, axis=2)


def prepare(output: Path, checkpoint: Path) -> None:
    data = output / "data"
    data.mkdir(parents=True, exist_ok=True)
    image_path = ROOT / "experiments/2026-10-23-mobile_vton/inputs/person.jpg"
    image = (
        Image.open(image_path)
        .convert("RGB")
        .resize((768, 1024), Image.Resampling.LANCZOS)
    )
    person = np.array(image, dtype=np.float32).transpose(2, 0, 1)[None] / 127.5 - 1
    person.tofile(data / "person.f32")
    generator = torch.Generator().manual_seed(42)
    for name in ("initial_noise", "person_epsilon", "garment_epsilon"):
        torch.randn(1, 16, 128, 96, generator=generator).numpy().tofile(
            data / f"{name}.f32"
        )
    cache_path = (
        ROOT
        / "experiments/2026-10-23-mobile_vton/outputs_garment_precompute/garment_cache.pt"
    )
    cache = torch.load(cache_path, map_location="cpu", weights_only=True)
    params = cache["posterior_parameters"].float().numpy()
    epsilon = np.fromfile(data / "garment_epsilon.f32", np.float32).reshape(
        1, 16, 128, 96
    )
    cloth = (posterior_sample(params, epsilon) - 0.0609) * 1.5305
    cloth.astype(np.float32).tofile(data / "garment_latent.f32")
    embeddings = torch.load(
        ROOT / "experiments/2026-10-23-mobile_vton/outputs_bf16_cached/embeddings.pt",
        map_location="cpu",
        weights_only=True,
    )
    for branch, index in (("conditional", 0), ("unconditional", 1)):
        embeddings["prompt"][index].float().numpy().tofile(data / f"text_{branch}.f32")
        cache["projected_image_embeds"][1 if index == 0 else 0].float().numpy().tofile(
            data / f"image_{branch}.f32"
        )
    scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=3.0)
    scheduler.set_timesteps(28)
    models = {
        "tryon": "../tryon_structural_int4_fp16_block32.mnn",
        "decoder": "../decoder_fp16.mnn",
    }
    for h, w in ((512, 512), (512, 384), (256, 512), (256, 384)):
        models[f"vae_{h}_{w}"] = (
            "../vae_encoder_fp32.mnn" if h == w == 512 else f"models/vae_{h}_{w}.mnn"
        )
    config = {
        "schema": 1,
        "height": 1024,
        "width": 768,
        "steps": 28,
        "guidance_scale": 2.5,
        "scheduler_shift": 3.0,
        "seed": 42,
        "backend": 3,
        "gpu_mode": 80,
        "tryon_precision": 2,
        "models": models,
        "timesteps": scheduler.timesteps.tolist(),
        "sigmas": scheduler.sigmas.tolist(),
        "garment_feature_storage": "float16",
        "feature_count": 24,
        "data": "data",
        "output": ".",
        "source": {
            "person_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
            "garment_cache": str(cache_path.relative_to(ROOT)),
            "garment_cache_sha256": hashlib.sha256(cache_path.read_bytes()).hexdigest(),
            "garment_cache_identity": cache["identity"],
            "conditioning": "Fixed cached BF16 posterior/projected embeddings promoted to FP32; original FP32 GarmentNet features computed for all 28 timesteps, then stored as FP16.",
            "rng": "Explicit CPU FP32 torch seed-42 noise artifacts, shared across phone and reference; not bitwise RNG-equivalent to prior BF16/CUDA runs.",
        },
    }
    write_json(output / "generation.json", config)
    print(
        "Prepared fixed noise, person image, cached conditioning and 28-step scheduler.",
        flush=True,
    )


def export_vae(output: Path, checkpoint: Path) -> None:
    from mobile_vton_vae import EncoderOnlyVAE

    class Raw(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = EncoderOnlyVAE.from_checkpoint(checkpoint / "vae")

        def forward(self, image):
            return self.model._encode(image)

    model = Raw().eval().requires_grad_(False).float()
    folder = output / "models"
    folder.mkdir(exist_ok=True)
    image = np.fromfile(output / "data/person.f32", np.float32).reshape(1, 3, 1024, 768)
    references = {}
    for h, w in ((512, 512), (512, 384), (256, 512), (256, 384)):
        y, x = (768 if h == 256 else 0), (384 if w == 384 else 0)
        tile = torch.from_numpy(image[:, :, y : y + h, x : x + w].copy())
        with torch.inference_mode():
            reference = model(tile)
            reference.numpy().tofile(output / f"data/vae_reference_{h}_{w}.f32")
            references[f"{h}_{w}"] = list(reference.shape)
            if h == w == 512:
                continue
            graph = folder / f"vae_{h}_{w}.onnx"
            torch.onnx.export(
                model,
                (tile,),
                str(graph),
                input_names=["image"],
                output_names=["output_0"],
                opset_version=17,
                dynamo=False,
            )
        subprocess.run(
            [
                "/tmp/mobile_tryon_mnn_host/MNNConvert",
                "-f",
                "ONNX",
                "--modelFile",
                str(graph),
                "--MNNModel",
                str(folder / f"vae_{h}_{w}.mnn"),
                "--bizCode",
                "pixel4_generation_vae",
            ],
            check=True,
        )
        print(f"Exported exact boundary tile {h}x{w}", flush=True)
    write_json(output / "vae_export_shapes.json", references)


def precompute_garment(output: Path, checkpoint: Path) -> None:
    from Mobile_VTON.models.unets.unet_2d_condition_garment import UNet2DConditionModel

    if not torch.cuda.is_available():
        raise RuntimeError(
            "Garment precomputation requires CUDA; use the pinned vLLM container."
        )
    model = UNet2DConditionModel.from_pretrained(
        checkpoint / "denoiser_garment", local_files_only=True
    )
    model.set_sample_size_rsd(128, 96)
    model = model.eval().requires_grad_(False).float().cuda()
    text = (
        torch.load(
            ROOT
            / "experiments/2026-10-23-mobile_vton/outputs_bf16_cached/embeddings.pt",
            map_location="cpu",
            weights_only=True,
        )["garment_prompt"][0]
        .float()
        .cuda()
    )
    cloth = torch.from_numpy(
        np.fromfile(output / "data/garment_latent.f32", np.float32).reshape(
            1, 16, 128, 96
        )
    ).cuda()
    config = json.loads((output / "generation.json").read_text())
    records = []
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        for step, timestep in enumerate(config["timesteps"]):
            folder = output / f"data/features_{step:02d}"
            folder.mkdir(exist_ok=True)
            _, features = model(
                sample=cloth,
                timestep=torch.tensor([timestep], device="cuda"),
                encoder_hidden_states=text,
                return_dict=False,
            )
            for index, feature in enumerate(features[: config["feature_count"]]):
                duplicated = torch.cat([feature, feature], dim=1)
                if not torch.isfinite(duplicated).all():
                    raise ValueError("Nonfinite garment feature")
                duplicated.half().cpu().numpy().tofile(folder / f"feature_{index}.f16")
            records.append(
                {
                    "step": step,
                    "timestep": timestep,
                    "shapes": [list(x.shape) for x in features[:24]],
                }
            )
            print(f"Garment cache {step + 1}/28", flush=True)
    torch.cuda.synchronize()
    write_json(
        output / "garment_precompute.json",
        {
            "status": "completed",
            "device": torch.cuda.get_device_name(),
            "seconds": time.perf_counter() - start,
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "bytes": sum(
                p.stat().st_size for p in (output / "data").glob("features_*/*.f16")
            ),
            "steps": records,
            "scope": "Offline fixed-garment preprocessing; excluded from phone online latency.",
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["prepare", "vae", "garment"], required=True)
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
    args.output.mkdir(parents=True, exist_ok=True)
    {"prepare": prepare, "vae": export_vae, "garment": precompute_garment}[args.stage](
        args.output, args.checkpoint
    )


if __name__ == "__main__":
    main()
