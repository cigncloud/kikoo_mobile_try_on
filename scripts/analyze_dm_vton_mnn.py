"""Summarize synchronized DM-VTON runs and render fixed-input comparisons."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]


def values(path: Path) -> np.ndarray:
    result = np.fromfile(path, np.float32)
    if result.size != 3 * 256 * 192 or not np.isfinite(result).all():
        raise ValueError(f"Invalid image tensor: {path}")
    return result


def error(actual: np.ndarray, reference: np.ndarray) -> dict:
    difference = actual.astype(np.float64) - reference.astype(np.float64)
    rmse = float(np.sqrt(np.mean(difference**2)))
    return {
        "rmse": rmse,
        "relative_rmse": rmse
        / float(np.sqrt(np.mean(reference.astype(np.float64) ** 2))),
        "max_absolute_difference": float(np.abs(difference).max()),
    }


def picture(tensor: np.ndarray) -> Image.Image:
    pixels = tensor.reshape(3, 256, 192).transpose(1, 2, 0)
    return Image.fromarray(np.uint8(np.clip((pixels + 1) / 2, 0, 1) * 255 + 0.5))


def comparison(root: Path, output: str, columns: list[tuple[str, str]]) -> None:
    canvas = Image.new("RGB", (192 * len(columns), 284), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (title, filename) in enumerate(columns):
        draw.text((index * 192 + 6, 6), title, fill="black")
        canvas.paste(picture(values(root / filename)), (index * 192, 28))
    canvas.save(root / output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=ROOT / "experiments/2026-10-25-dm_vton"
    )
    args = parser.parse_args()
    root = args.root
    reference = values(root / "dmvton_reference_tryon.f32")
    report = {
        "model": "DM-VTON PF student",
        "upstream_revision": "5dcd04ee3f5b1fb392df1184c887c1a7e910c72e",
        "resolution_hw": [256, 192],
        "weights": "Unquantized FP32 MNN files; runtime precision differs by run",
        "mnn_model_bytes": sum(
            (root / f"dmvton_{stage}.mnn").stat().st_size
            for stage in ("warp", "generator")
        ),
        "kernel_cache_bytes": sum(
            path.stat().st_size for path in (root / "kernel_cache").glob("*.cache")
        ),
        "cuda_vs_export_reference": error(
            values(root / "cuda_fixtures/tryon.f32"), reference
        ),
        "runs": {},
        "scope": "One person/garment, full-body resize and manual upper-body crop. Not a dataset quality evaluation. Latency includes both stages, uploads, downloads and host bridge; excludes JPEG/PNG and ADB. Process RSS is not dedicated GPU VRAM.",
        "quality_verdict": "Not passed: garment misalignment, shoulder/neck artifacts and original-shirt remnants occur in the original CUDA model as well as the phone output.",
    }
    names = [
        "host_chain",
        "pixel4_opencl_low",
        "pixel4_opencl_high",
        "pixel4_upper_body_high",
        "pixel4_upper_body_cached",
    ]
    for name in names:
        data = json.loads((root / name / "metrics.json").read_text())
        timings = [run for run in data["runs"] if not run["warmup"]]
        actual = values(root / name / "tryon.f32")
        first = values(root / name / "first_tryon.f32")
        target = (
            values(root / "cuda_upper_body/tryon.f32")
            if "upper_body" in name
            else reference
        )
        report["runs"][name] = {
            "mean_seconds": float(np.mean([run["total_seconds"] for run in timings])),
            "mean_warp_seconds": float(
                np.mean([run["warp_seconds"] for run in timings])
            ),
            "mean_generator_seconds": float(
                np.mean([run["generator_seconds"] for run in timings])
            ),
            "measured_runs": len(timings),
            "cold_to_first_output_seconds": data["cold_to_first_output_seconds"],
            "process_peak_rss_bytes": data["process_peak_rss_bytes"],
            "process_peak_rss_mib": data["process_peak_rss_bytes"] / 2**20,
            "backend_requested": data["backend_requested"],
            "precision": data["precision"],
            "output_error": error(actual, target),
            "reference": "Original CUDA/CuPy"
            if "upper_body" in name
            else "FP32 PyTorch CPU with ATen correlation",
            "first_vs_last_max_absolute_difference": float(
                np.max(np.abs(first - actual))
            ),
            "first_forward_output_backend_counts": [
                model["first_forward_output_backend_counts"] for model in data["models"]
            ],
        }
        picture(actual).save(root / name / "result.png")
    comparison(
        root,
        "comparison.png",
        [
            ("Person", "dmvton_input_person.f32"),
            ("Garment", "dmvton_input_clothes.f32"),
            ("Original CUDA", "cuda_fixtures/tryon.f32"),
            ("Pixel 4 FP16", "pixel4_opencl_low/tryon.f32"),
            ("Pixel 4 FP32", "pixel4_opencl_high/tryon.f32"),
        ],
    )
    comparison(
        root,
        "comparison_upper_body.png",
        [
            ("Upper-body crop", "upper_body_inputs/dmvton_input_person.f32"),
            ("Garment", "dmvton_input_clothes.f32"),
            ("Original CUDA", "cuda_upper_body/tryon.f32"),
            ("Pixel 4 FP32", "pixel4_upper_body_cached/tryon.f32"),
        ],
    )
    (root / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
