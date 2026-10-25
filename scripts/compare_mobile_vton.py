"""Compare two completed Mobile-VTON runs without implying a ground-truth score."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]


def compare(baseline: Path, quantized: Path, output: Path) -> dict:
    reports = [
        json.loads((folder / "metrics.json").read_text())
        for folder in (baseline, quantized)
    ]
    if any(report["status"] != "passed" for report in reports):
        raise ValueError("Both inference runs must finish successfully")
    keys = (
        "model_revision",
        "upstream_revision",
        "width",
        "height",
        "steps",
        "seed",
        "guidance_scale",
        "scheduler_shift",
        "dtype",
        "device",
    )
    for key in keys:
        if reports[0]["config"][key] != reports[1]["config"][key]:
            raise ValueError(f"Comparison setting mismatch: {key}")
    for key in ("person", "garment"):
        if reports[0]["inputs"][key]["sha256"] != reports[1]["inputs"][key]["sha256"]:
            raise ValueError(f"Comparison input mismatch: {key}")
    if (
        reports[0]["embedding_cache"]["sha256"]
        != reports[1]["embedding_cache"]["sha256"]
    ):
        raise ValueError("Comparisons must use identical prompt embeddings")
    images = [
        Image.open(ROOT / report["runs"][0]["output"]).convert("RGB")
        for report in reports
    ]
    arrays = [np.asarray(image, dtype=np.float32) / 255 for image in images]
    delta = arrays[1] - arrays[0]
    mse = float(np.mean(delta**2))
    result = {
        "baseline_metrics": str(baseline / "metrics.json"),
        "int4_metrics": str(quantized / "metrics.json"),
        "matched_settings": {key: reports[0]["config"][key] for key in keys},
        "same_input_hashes": True,
        "same_embedding_cache_sha256": reports[0]["embedding_cache"]["sha256"],
        "output_difference": {
            "mae_0_to_1": float(np.mean(np.abs(delta))),
            "rmse_0_to_1": math.sqrt(mse),
            "psnr_db": -10 * math.log10(mse) if mse else None,
            "meaning": "Difference from BF16 output; this is not accuracy against a reference try-on image.",
        },
        "models": {},
    }
    for label, report in zip(("bf16", "int4"), reports):
        timings = [run["pipeline_seconds"] for run in report["runs"]]
        result["models"][label] = {
            "mean_pipeline_seconds": float(np.mean(timings)),
            "pipeline_seconds_each_run": timings,
            "warmup_steps": report["config"]["warmup_steps"],
            "model_tensor_storage_bytes": report.get(
                "runtime_model_tensor_storage_bytes",
                report["model_tensor_storage_bytes"],
            ),
            "host_peak_rss_bytes": report["host_peak_rss_bytes"],
            "runtime": (
                "PyTorch CPU"
                if report["config"]["device"] == "cpu"
                else report.get("gpu")
            ),
        }
        if report["config"]["device"] == "cuda":
            for metric in ("cuda_peak_allocated_bytes", "cuda_peak_reserved_bytes"):
                result["models"][label][metric] = max(
                    run[metric] for run in report["runs"]
                )
            result["models"][label]["cuda_peak_allocated_bytes_before_benchmark"] = (
                report["cuda_peak_allocated_bytes_before_benchmark"]
            )
    old, new = (
        result["models"][name]["model_tensor_storage_bytes"]
        for name in ("bf16", "int4")
    )
    result["tensor_storage_reduction_percent"] = 100 * (1 - new / old)
    result["limits"] = [
        "One person/garment pair with fixed seed; repeated timings do not establish dataset quality or phone performance.",
        "Weight-only INT4 for both denoisers' Linear layers; other layers and activations keep their original dtype.",
        "Host peak RSS includes model loading and quantization when performed in-process.",
        "Desktop torchao checkpoint layouts are not Android/iOS deployment packages or phone memory measurements.",
        "Shared desktop hardware may experience external contention; timing samples and warmup settings are reported explicitly.",
    ]
    if reports[0]["config"]["device"] == "cuda":
        old, new = (
            result["models"][name]["cuda_peak_allocated_bytes"]
            for name in ("bf16", "int4")
        )
        result["cuda_peak_allocated_reduction_percent"] = 100 * (1 - new / old)
        result["limits"].append(
            "CUDA allocated/reserved bytes describe PyTorch's GPU allocator, excluding driver/context and other processes. Host RSS is reported separately."
        )
    output.mkdir(parents=True, exist_ok=True)
    labels = ("Person", "Garment", "BF16 (cached text)", "INT4 weights")
    sources = [
        Image.open(ROOT / reports[0]["inputs"][key]["path"]).convert("RGB")
        for key in ("person", "garment")
    ]
    tiles = sources + images
    tile_size = (384, 512)
    sheet = Image.new("RGB", (tile_size[0] * 4, tile_size[1] + 40), "white")
    draw = ImageDraw.Draw(sheet)
    for column, (label, tile) in enumerate(zip(labels, tiles)):
        sheet.paste(ImageOps.contain(tile, tile_size), (tile_size[0] * column, 40))
        draw.text((tile_size[0] * column + 10, 14), label, fill="black")
    sheet.save(output / "bf16_vs_int4.jpg", quality=95)
    native = Image.new("RGB", (images[0].width * 2, images[0].height + 40), "white")
    native_draw = ImageDraw.Draw(native)
    for column, (label, tile) in enumerate(zip(labels[2:], images)):
        native.paste(tile, (column * tile.width, 40))
        native_draw.text((column * tile.width + 10, 14), label, fill="black")
    native.save(output / "bf16_vs_int4_native.png")
    # The same normalized crop isolates garment detail in both outputs.
    w, h = images[0].size
    box = (int(w * 0.15), int(h * 0.04), int(w * 0.85), int(h * 0.56))
    crop = Image.new("RGB", (768, 552), "white")
    draw = ImageDraw.Draw(crop)
    for column, (label, tile) in enumerate(zip(labels[2:], images)):
        crop.paste(ImageOps.contain(tile.crop(box), (384, 512)), (column * 384, 40))
        draw.text((column * 384 + 10, 14), label, fill="black")
    crop.save(output / "garment_detail.jpg", quality=95)
    (output / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--quantized", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(compare(args.baseline, args.quantized, args.output), indent=2))


if __name__ == "__main__":
    main()
