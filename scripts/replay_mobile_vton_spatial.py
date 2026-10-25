"""Isolate tiled decoder error on the exact saved full-pipeline latents."""

from __future__ import annotations

import argparse
import copy
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
    from mobile_vton_spatial import install_spatial_tiling
    from PIL import Image

    config = json.loads((experiment / "config_compute_baseline.json").read_text())
    torch.set_num_threads(config["threads"])
    sys.path.insert(0, str(ROOT / config["upstream"]))
    from Mobile_VTON.models.autoencoders.vae import Decoder

    checkpoint = Path(config["checkpoint"]) / "vae_decoder"
    untiled = Decoder(**json.loads((checkpoint / "decoder.json").read_text()))
    untiled.load_state_dict(
        torch.load(checkpoint / "decoder.pt", map_location="cpu", weights_only=True)
    )
    untiled = (
        untiled.eval().requires_grad_(False).to(device="cuda", dtype=torch.float16)
    )
    tiled = copy.deepcopy(untiled)
    install_spatial_tiling(tiled, 32, separable=False)
    output.mkdir(parents=True, exist_ok=True)
    results = {}
    for variant in ("baseline", "spatial32", "spatial32_cfg", "decoder32_cfg"):
        folder = experiment / f"outputs_compute_{variant}"
        report = json.loads((folder / "metrics.json").read_text())
        latent = torch.load(
            folder / "decoder_latents.pt", weights_only=True, map_location="cpu"
        )
        latent_hash = hashlib.sha256(latent.float().numpy().tobytes()).hexdigest()
        if (
            latent_hash
            != report["decoder_optimization"]["events"][-1]["latent_sha256_fp32"]
        ):
            raise ValueError(f"Saved latent hash mismatch: {variant}")
        images, arrays = {}, {}
        for name, decoder in (("untiled", untiled), ("tiled32", tiled)):
            with torch.inference_mode():
                decoded = (
                    decoder(latent.to(device="cuda", dtype=torch.float16)).float().cpu()
                )
            if not torch.isfinite(decoded).all():
                raise ValueError(f"Nonfinite decoded image: {variant}/{name}")
            arrays[name] = decoded.numpy()
            pixels = (decoded / 2 + 0.5).clamp(0, 1).permute(0, 2, 3, 1)[0].numpy()
            images[name] = (pixels * 255).round().astype("uint8")
            Image.fromarray(images[name]).save(output / f"{variant}_{name}.png")
        expected = np.asarray(
            Image.open(ROOT / report["runs"][-1]["output"]).convert("RGB")
        )
        pipeline_mode = "untiled" if variant == "baseline" else "tiled32"
        matches = bool(np.array_equal(expected, images[pipeline_mode]))
        if not matches:
            raise ValueError(f"Decoder replay does not reproduce pipeline: {variant}")
        delta = (
            images["tiled32"].astype(np.float32) - images["untiled"].astype(np.float32)
        ) / 255
        raw_delta = arrays["tiled32"] - arrays["untiled"]
        mse = float(np.mean(delta**2))
        results[variant] = {
            "latent_sha256_fp32": latent_hash,
            "replay_matches_pipeline": matches,
            "raw_decoder_mae": float(np.mean(np.abs(raw_delta))),
            "image_identical": bool(
                np.array_equal(images["untiled"], images["tiled32"])
            ),
            "image_mae_0_to_1": float(np.mean(np.abs(delta))),
            "image_psnr_db": -10 * math.log10(mse) if mse else None,
            "image_max_absolute_difference_0_to_1": float(np.max(np.abs(delta))),
        }
    result = {
        "gpu": torch.cuda.get_device_name(),
        "comparisons": results,
        "meaning": "Identical latent and FP16 weights. Only local convolution evaluation is tiled; GroupNorm sees the full image. Not ground-truth try-on accuracy.",
    }
    (output / "spatial_replay.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(replay(args.experiment, args.output), indent=2))


if __name__ == "__main__":
    main()
