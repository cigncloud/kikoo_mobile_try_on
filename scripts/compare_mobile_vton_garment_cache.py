"""Validate matched resident CUDA runs with and without fixed-garment caches."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def compare(experiment: Path, output: Path) -> dict:
    names = ("resident_uncached", "resident_cached")
    reports = {
        name: json.loads(
            (experiment / f"outputs_garment_{name}/metrics.json").read_text()
        )
        for name in names
    }
    baseline = reports[names[0]]
    ignored = {"output_dir", "garment_cache"}
    variants = {}
    for name, report in reports.items():
        if report["status"] != "passed":
            raise ValueError(f"Incomplete run: {name}")
        if {k: v for k, v in report["config"].items() if k not in ignored} != {
            k: v for k, v in baseline["config"].items() if k not in ignored
        }:
            raise ValueError(f"Unmatched run configuration: {name}")
        if (
            report["config"]["memory_mode"] != "sequential_cfg"
            or not report["config"]["gpu_resident"]
        ):
            raise ValueError("Comparison requires all required models resident on CUDA")
        if report["gpu_residency_before"] != report["gpu_residency_after"]:
            raise ValueError(f"GPU residency changed during generation: {name}")
        if report["memory_optimization"]["events"]:
            raise ValueError(f"Unexpected phase reload/offload: {name}")
        for component in report["gpu_residency_before"].values():
            if component["phase_loaded"] or not all(
                device.startswith("cuda") for device in component["devices"]
            ):
                raise ValueError(f"Non-CUDA model residency: {name}")
        cached = name == "resident_cached"
        if ("image_encoder" in report["gpu_residency_before"]) == cached:
            raise ValueError("Image encoder must be absent only in the cached run")
        generations = report["config"]["runs"] + bool(report["config"]["warmup_steps"])
        if len(report["vae_optimization"]["events"]) != generations * (
            1 if cached else 2
        ):
            raise ValueError(f"VAE encode count mismatch: {name}")
        if not all(
            e["encoder_forward_count"] == 6
            for e in report["vae_optimization"]["events"]
        ):
            raise ValueError(f"VAE tiling changed: {name}")
        decoder_events = report["decoder_optimization"]["events"]
        if len(decoder_events) != generations or not all(
            e["finite"] and e["latent_sha256_fp32"] is None for e in decoder_events
        ):
            raise ValueError(
                "Online decoder must be finite without copying latent audits to CPU"
            )
        for key in ("person", "garment"):
            if report["inputs"][key]["sha256"] != baseline["inputs"][key]["sha256"]:
                raise ValueError("Input hashes mismatch")
        if report["embedding_cache"]["sha256"] != baseline["embedding_cache"]["sha256"]:
            raise ValueError("Text cache mismatch")
        for run in report["runs"]:
            path = ROOT / run["output"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != run["sha256"]:
                raise ValueError(f"Output hash mismatch: {path}")
        hashes = {run["sha256"] for run in report["runs"]}
        if len(hashes) != 1:
            raise ValueError(f"Nonrepeatable output: {name}")
        if hashes != {baseline["runs"][0]["sha256"]}:
            raise ValueError("Garment cache changed the image")
        seconds = [run["pipeline_seconds"] for run in report["runs"]]
        variants[name] = {
            "metrics": str(experiment / f"outputs_garment_{name}/metrics.json"),
            "pipeline_seconds": seconds,
            "mean_pipeline_seconds": statistics.mean(seconds),
            "median_pipeline_seconds": statistics.median(seconds),
            "cuda_peak_allocated_bytes_all_phases": report[
                "cuda_peak_allocated_bytes_all_phases"
            ],
            "cuda_peak_reserved_bytes_all_phases": report[
                "cuda_peak_reserved_bytes_all_phases"
            ],
            "cuda_peak_allocated_bytes_generation": max(
                r["cuda_peak_allocated_bytes"] for r in report["runs"]
            ),
            "cuda_peak_reserved_bytes_generation": max(
                r["cuda_peak_reserved_bytes"] for r in report["runs"]
            ),
            "host_peak_rss_bytes": report["host_peak_rss_bytes"],
            "host_current_rss_bytes_after_runs": [
                r["host_current_rss_bytes"] for r in report["runs"]
            ],
            "model_load_seconds": report["model_load_seconds"],
            "preprocess_and_text_cache_seconds": report[
                "preprocess_and_text_encode_seconds"
            ],
            "warmup_seconds": report["warmup_seconds"],
            "resident_models": report["gpu_residency_before"],
            "gpu_model_storage_bytes": sum(
                v["tensor_storage_bytes"]
                for v in report["gpu_residency_before"].values()
            ),
            "output_sha256": next(iter(hashes)),
            "output_identical": True,
        }
    previous = json.loads(
        (experiment / "outputs_compute_decoder32_cfg/metrics.json").read_text()
    )
    previous_hash = previous["runs"][0]["sha256"]
    if variants["resident_cached"]["output_sha256"] != previous_hash:
        raise ValueError(
            "Cached resident output changed from the previous phase-released setting"
        )
    precompute = json.loads(
        (experiment / "outputs_garment_precompute/metrics.json").read_text()
    )
    if precompute["status"] != "prepared":
        raise ValueError("Missing completed garment precomputation")
    cache = reports["resident_cached"]["garment_cache"]
    cache_path = Path(cache["path"])
    if (
        hashlib.sha256(cache_path.read_bytes()).hexdigest() != cache["sha256"]
        or cache["sha256"] != precompute["garment_cache"]["sha256"]
    ):
        raise ValueError("Garment cache artifact hash mismatch")
    if not all(v["device"].startswith("cuda") for v in cache["gpu_tensors"].values()):
        raise ValueError("Cached garment tensors must remain on CUDA")
    result = {
        "gpu": baseline["gpu"],
        "config": baseline["config"],
        "variants": variants,
        "precompute": precompute["garment_cache"],
        "cache_loading": cache,
        "same_output_as_previous_phase_release": True,
        "allocated_reduction_vs_resident_uncached_percent": 100
        * (
            1
            - variants["resident_cached"]["cuda_peak_allocated_bytes_all_phases"]
            / variants["resident_uncached"]["cuda_peak_allocated_bytes_all_phases"]
        ),
        "observed_mean_time_reduction_percent": 100
        * (
            1
            - variants["resident_cached"]["mean_pipeline_seconds"]
            / variants["resident_uncached"]["mean_pipeline_seconds"]
        ),
        "limitations": [
            "Desktop CUDA measurements, not Android/iOS runtime results.",
            "All online neural network operations run on CUDA and needed models remain resident; no CPU model offload. Host file IO, image preparation and final image transfer still use CPU.",
            "Pipeline time includes person VAE, denoising, decoding and output transfer; excludes cold model load, initial input preprocessing/cache verification, warmup and saving PNGs.",
            "Shared GPU and three repeats; time differences are observations, not isolated performance guarantees.",
            "CUDA allocated/reserved exclude driver/context. CPU RSS is separate and its peak need not coincide with the GPU peak.",
            "One pair and one cached garment; prior INT4 and VAE tiling quality limitations remain.",
            "VAE encoder remains necessary for person images, and GarmentNet remains necessary at each timestep. Its per-step features are not cached.",
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(compare(args.experiment, args.output), indent=2))


if __name__ == "__main__":
    main()
