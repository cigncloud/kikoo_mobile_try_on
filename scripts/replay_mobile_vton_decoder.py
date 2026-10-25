"""Isolate FP16 decoder error using the exact saved native-resolution latents."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
ROOT = Path(__file__).resolve().parents[1]


def replay(experiment: Path, output: Path) -> dict:
    import numpy as np
    import torch
    from PIL import Image

    report = json.loads((experiment / "outputs_decoder_fp16/metrics.json").read_text())
    config = report["config"]
    torch.set_num_threads(config["threads"])
    sys.path.insert(0, str(ROOT / config["upstream"]))
    from Mobile_VTON.models.autoencoders.vae import Decoder

    checkpoint = Path(config["checkpoint"]) / "vae_decoder"
    decoder = Decoder(**json.loads((checkpoint / "decoder.json").read_text()))
    original_state = torch.load(
        checkpoint / "decoder.pt", weights_only=True, map_location="cpu"
    )
    decoder.load_state_dict(original_state, strict=True)
    decoder = decoder.eval().requires_grad_(False).cuda()
    output.mkdir(parents=True, exist_ok=True)
    results = {}
    for name in ("fp16", "fp16_vae_tiled512"):
        folder = experiment / f"outputs_decoder_{name}"
        recorded = json.loads((folder / "metrics.json").read_text())
        latents_cpu = torch.load(folder / "decoder_latents.pt", weights_only=True)
        latent_hash = hashlib.sha256(latents_cpu.numpy().tobytes()).hexdigest()
        if (
            latent_hash
            != recorded["decoder_optimization"]["events"][-1]["latent_sha256_fp32"]
        ):
            raise ValueError("Saved latent hash does not match generation record")
        images, arrays = {}, {}
        for label, dtype in (("fp32", torch.float32), ("fp16", torch.float16)):
            decoder.to(dtype=dtype)
            # Recover original FP32 weights after a previous half-precision pass.
            decoder.load_state_dict(original_state, strict=True)
            with torch.inference_mode():
                decoded = decoder(latents_cpu.cuda().to(dtype=dtype)).float().cpu()
            if not torch.isfinite(decoded).all():
                raise ValueError(f"Nonfinite decoder output: {name}/{label}")
            arrays[label] = decoded.numpy()
            pixels = (decoded / 2 + 0.5).clamp(0, 1).permute(0, 2, 3, 1)[0].numpy()
            images[label] = (pixels * 255).round().astype("uint8")
            Image.fromarray(images[label]).save(output / f"{name}_{label}.png")
        expected_fp32 = (
            experiment / "outputs_decoder_fp32/result_01.png"
            if name == "fp16"
            else experiment / "outputs_vae_encoder_tiled512/result_01.png"
        )
        expected_fp16 = folder / "result_01.png"
        matches = {
            "fp32_replay_matches_pipeline": bool(
                np.array_equal(images["fp32"], np.asarray(Image.open(expected_fp32)))
            ),
            "fp16_replay_matches_pipeline": bool(
                np.array_equal(images["fp16"], np.asarray(Image.open(expected_fp16)))
            ),
        }
        if not all(matches.values()):
            raise ValueError(f"Replayed decoder output differs from pipeline: {name}")
        raw_delta = arrays["fp16"] - arrays["fp32"]
        pixel_delta = (
            images["fp16"].astype(np.float32) - images["fp32"].astype(np.float32)
        ) / 255
        mse = float(np.mean(pixel_delta**2))
        results[name] = {
            "latent_sha256_fp32": latent_hash,
            **matches,
            "raw_decoder_mae": float(np.mean(np.abs(raw_delta))),
            "raw_decoder_max_absolute_difference": float(np.max(np.abs(raw_delta))),
            "image_mae_0_to_1": float(np.mean(np.abs(pixel_delta))),
            "image_psnr_db": -10 * math.log10(mse) if mse else None,
            "image_max_absolute_difference_0_to_1": float(np.max(np.abs(pixel_delta))),
        }
    result = {
        "gpu": torch.cuda.get_device_name(),
        "comparisons": results,
        "meaning": "Identical latent tensors and decoder weights; only FP32/FP16 arithmetic differs. Not a ground-truth accuracy measure.",
    }
    (output / "precision_comparison.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(replay(args.experiment, args.output), indent=2))


if __name__ == "__main__":
    main()
