"""Compare Pixel 4 component outputs with the original FP32 PyTorch fixtures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--component",
        required=True,
        choices=["decoder", "vae_encoder", "vae_encoder_portable", "garment", "tryon"],
    )
    parser.add_argument("--variant", required=True)
    parser.add_argument(
        "--experiment", type=Path, default=Path("experiments/2026-10-24-pixel4")
    )
    args = parser.parse_args()
    root, label = args.experiment, f"{args.component}_{args.variant}"
    output = root / f"outputs_{label}"
    metrics = json.loads((output / f"{label}_metrics.json").read_text())
    export = json.loads(
        (root / "exports" / f"{args.component}_export.json").read_text()
    )
    shapes = export.get("outputs", {"image": [1, 3, 1024, 768]})
    comparisons = {}
    for name, shape in shapes.items():
        actual_path = output / f"{label}_{name}.f32"
        if not actual_path.exists():
            actual_path = output / f"{label}_output.f32"
        actual = np.fromfile(actual_path, dtype=np.float32).reshape(shape)
        ref_name = (
            "decoder_reference.f32"
            if args.component == "decoder"
            else f"{args.component}_reference_{name}.f32"
        )
        reference = np.fromfile(root / "exports" / ref_name, dtype=np.float32).reshape(
            shape
        )
        if not np.isfinite(actual).all():
            raise ValueError(f"Nonfinite output: {actual_path}")
        error = actual - reference
        comparisons[name] = {
            "shape": shape,
            "mae": float(np.abs(error).mean()),
            "rmse": float(np.sqrt(np.mean(error**2))),
            "max_absolute_error": float(np.abs(error).max()),
            "relative_rmse": float(
                np.sqrt(np.mean(error**2))
                / max(float(np.sqrt(np.mean(reference**2))), 1e-12)
            ),
        }
        if args.component == "decoder":
            for image_name, pixels in [
                ("pixel4_decoder", actual),
                ("desktop_fp32_decoder", reference),
            ]:
                pixels = (
                    ((pixels[0].transpose(1, 2, 0) / 2 + 0.5).clip(0, 1) * 255)
                    .round()
                    .astype("uint8")
                )
                Image.fromarray(pixels).save(output / f"{image_name}.png")
    backends = metrics.pop("operator_output_backends")
    metrics["operator_output_backend_counts"] = {
        str(v): list(backends.values()).count(v) for v in set(backends.values())
    }
    metrics["mean_seconds"] = float(
        np.mean(metrics["inference_and_output_download_seconds"])
    )
    metrics["numerical_comparison"] = comparisons
    metrics["scope"] = (
        "Component test only. Export fixtures are not a full generation on the phone. Output backend reports tensor placement, not a guarantee that all neural operations ran on GPU."
    )
    (output / "comparison.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
