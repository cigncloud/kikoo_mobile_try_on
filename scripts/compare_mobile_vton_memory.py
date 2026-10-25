"""Compare matched INT4 memory variants, including quality and phase residency."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]
MODES = ("baseline", "sequential_cfg", "phase_release", "combined")
KEYS = (
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
    "quantization",
    "group_size",
    "runs",
    "warmup_steps",
    "threads",
    "quantized_state_dir",
)


def compare(experiment: Path, output: Path, suite: str = "memory") -> dict:
    modes = {
        "memory": MODES,
        "vae": ("baseline", "encoder_only", "encoder_tiled512"),
        "decoder": ("fp32", "fp16", "fp16_vae_tiled512", "fp16_vae_tiled512_lowload"),
        "lifetimes": ("baseline", "release", "release_expandable"),
        "compute": (
            "baseline",
            "ffn256",
            "gqa",
            "combined",
            "spatial32",
            "spatial32_cfg",
            "decoder32_cfg",
        ),
    }[suite]
    baseline_mode = modes[0]
    report_paths = {
        mode: experiment / f"outputs_{suite}_{mode}/metrics.json" for mode in modes
    }
    if suite == "lifetimes":
        report_paths["baseline"] = (
            experiment / "outputs_decoder_fp16_vae_tiled512_lowload/metrics.json"
        )
    reports = {mode: json.loads(report_paths[mode].read_text()) for mode in modes}
    baseline = reports[baseline_mode]
    images = {}
    variants = {}
    for mode, report in reports.items():
        if report["status"] != "passed":
            raise ValueError(f"Incomplete experiment: {mode}")
        expected_mode = mode if suite == "memory" else "phase_release"
        if suite == "compute" and mode in ("spatial32_cfg", "decoder32_cfg"):
            expected_mode = "combined"
        if report["config"].get("memory_mode", "baseline") != expected_mode:
            raise ValueError(f"Memory mode mismatch: {mode}")
        if suite == "vae":
            expected_vae = (
                mode != "baseline",
                512 if mode == "encoder_tiled512" else 0,
            )
            actual_vae = (
                report["config"]["vae_encoder_only"],
                report["config"]["vae_tile_size"],
            )
            if actual_vae != expected_vae:
                raise ValueError(f"VAE configuration mismatch: {mode}")
        if suite == "decoder":
            expected = (
                True,
                512 if mode.startswith("fp16_vae_tiled512") else 0,
                "fp32" if mode == "fp32" else "fp16",
            )
            actual = (
                report["config"]["vae_encoder_only"],
                report["config"]["vae_tile_size"],
                report["config"]["decoder_dtype"],
            )
            if actual != expected:
                raise ValueError(f"Decoder configuration mismatch: {mode}")
            if report["config"].get("defer_quantized_transfer", False) != mode.endswith(
                "lowload"
            ):
                raise ValueError(f"Deferred transfer configuration mismatch: {mode}")
        if suite == "lifetimes":
            actual = report["config"].get("release_working_tensors", False)
            if actual != (mode != "baseline"):
                raise ValueError(f"Tensor lifetime configuration mismatch: {mode}")
            expected_allocator = (
                "expandable_segments:True" if mode == "release_expandable" else ""
            )
            if report.get("cuda_allocator_config", "") != expected_allocator:
                raise ValueError(f"CUDA allocator configuration mismatch: {mode}")
            ignored = {"output_dir", "release_working_tensors"}
            if {k: v for k, v in report["config"].items() if k not in ignored} != {
                k: v for k, v in baseline["config"].items() if k not in ignored
            }:
                raise ValueError(f"Unmatched lifetime experiment settings: {mode}")
        if suite == "compute":
            options = {
                "baseline": (0, False, 0),
                "ffn256": (256, False, 0),
                "gqa": (0, True, 0),
                "combined": (256, True, 0),
                "spatial32": (0, False, 32),
                "spatial32_cfg": (0, False, 32),
                "decoder32_cfg": (0, False, 32),
            }
            actual = (
                report["config"].get("ffn_chunk_size", 0),
                report["config"].get("native_gqa", False),
                report["config"].get("spatial_conv_rows", 0),
            )
            if actual != options[mode] or report["config"].get("profile_modules"):
                raise ValueError(f"Compute configuration mismatch: {mode}")
            if report["config"].get("spatial_decoder_only", False) != (
                mode == "decoder32_cfg"
            ):
                raise ValueError(f"Decoder-only configuration mismatch: {mode}")
            ignored = {
                "output_dir",
                "ffn_chunk_size",
                "native_gqa",
                "spatial_conv_rows",
                "memory_mode",
                "spatial_decoder_only",
            }
            if {k: v for k, v in report["config"].items() if k not in ignored} != {
                k: v for k, v in baseline["config"].items() if k not in ignored
            }:
                raise ValueError(f"Unmatched compute experiment settings: {mode}")
            if report.get("cuda_allocator_config") != "expandable_segments:True":
                raise ValueError(f"Allocator mismatch: {mode}")
        for key in KEYS:
            if report["config"][key] != baseline["config"][key]:
                raise ValueError(f"Setting mismatch in {mode}: {key}")
        for key in ("person", "garment"):
            if report["inputs"][key]["sha256"] != baseline["inputs"][key]["sha256"]:
                raise ValueError(f"Input mismatch in {mode}: {key}")
        if report["embedding_cache"]["sha256"] != baseline["embedding_cache"]["sha256"]:
            raise ValueError(f"Embedding mismatch in {mode}")
        for run in report["runs"]:
            path = ROOT / run["output"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != run["sha256"]:
                raise ValueError(f"Output hash mismatch: {path}")
        image = Image.open(ROOT / report["runs"][0]["output"]).convert("RGB")
        if image.size != (report["config"]["width"], report["config"]["height"]):
            raise ValueError(f"Output dimensions mismatch: {mode}")
        images[mode] = image
        reference = np.asarray(images[baseline_mode], dtype=np.float32) / 255
        delta = np.asarray(image, dtype=np.float32) / 255 - reference
        mse = float(np.mean(delta**2))
        runs = report["runs"]
        variants[mode] = {
            "metrics": str(report_paths[mode]),
            "pipeline_seconds": [run["pipeline_seconds"] for run in runs],
            "mean_pipeline_seconds": float(
                np.mean([run["pipeline_seconds"] for run in runs])
            ),
            "cuda_peak_allocated_bytes": max(
                run["cuda_peak_allocated_bytes"] for run in runs
            ),
            "cuda_peak_reserved_bytes": max(
                run["cuda_peak_reserved_bytes"] for run in runs
            ),
            "cuda_peak_before_benchmark_bytes": report[
                "cuda_peak_allocated_bytes_before_benchmark"
            ],
            "cuda_peak_allocated_bytes_all_phases": max(
                report["cuda_peak_allocated_bytes_before_benchmark"],
                max(run["cuda_peak_allocated_bytes"] for run in runs),
            ),
            "resident_model_tensor_storage_bytes": report[
                "runtime_model_tensor_storage_bytes"
            ],
            "host_peak_rss_bytes": report["host_peak_rss_bytes"],
            "cuda_allocator_config": report.get("cuda_allocator_config", ""),
            "repeat_identical": len({run["sha256"] for run in runs}) == 1,
            "identical_to_baseline": runs[0]["sha256"] == baseline["runs"][0]["sha256"],
            "difference_from_int4_baseline": {
                "mae_0_to_1": float(np.mean(np.abs(delta))),
                "psnr_db": -10 * math.log10(mse) if mse else None,
                "max_absolute_pixel_difference_0_to_1": float(np.max(np.abs(delta))),
            },
        }
        events = report.get("memory_optimization", {}).get("events", [])
        if expected_mode in ("phase_release", "combined"):
            for name in ("vae", "image_encoder", "vae_decoder"):
                actions = [
                    event["action"] for event in events if event["component"] == name
                ]
                generations = report["config"]["runs"] + bool(
                    report["config"]["warmup_steps"]
                )
                if (
                    actions.count("load") != generations
                    or actions.count("release") != generations
                ):
                    raise ValueError(f"Phase model lifecycle mismatch: {mode}/{name}")
            variants[mode]["phase_loads_and_releases_verified"] = True
        if suite in ("decoder", "lifetimes", "compute"):
            decode_events = report["decoder_optimization"]["events"]
            baseline_decode = baseline["decoder_optimization"]["events"]
            generations = report["config"]["runs"] + bool(
                report["config"]["warmup_steps"]
            )
            if len(decode_events) != generations or not all(
                event["finite"] for event in decode_events
            ):
                raise ValueError(f"Decoder did not complete every generation: {mode}")
            same_latents = [event["latent_sha256_fp32"] for event in decode_events] == [
                event["latent_sha256_fp32"] for event in baseline_decode
            ]
            if (suite == "lifetimes" or mode in ("fp32", "fp16")) and not same_latents:
                raise ValueError("Comparison requires identical decoder inputs")
            variants[mode]["decoder"] = {
                **report["components"]["vae_decoder"],
                (
                    "same_decoder_inputs_as_fp32"
                    if suite == "decoder"
                    else "same_decoder_inputs_as_baseline"
                ): same_latents,
                "events": decode_events,
            }
            phase_begin, phase_end = runs[0]["memory_event_range"]
            variants[mode]["first_generation_phase_events"] = events[
                phase_begin:phase_end
            ]
            if suite == "decoder" and mode.endswith("lowload"):
                previous = reports["fp16_vae_tiled512"]
                if [run["sha256"] for run in runs] != [
                    run["sha256"] for run in previous["runs"]
                ]:
                    raise ValueError(
                        "Deferred transfer must preserve generated outputs"
                    )
                variants[mode]["same_output_as_regular_loader"] = True
        if suite == "compute":
            if not variants[mode]["repeat_identical"]:
                raise ValueError(f"Compute variant was not repeatable: {mode}")
            variants[mode]["compute_optimization"] = report["compute_optimization"]
        if suite == "lifetimes":
            if (
                not variants[mode]["identical_to_baseline"]
                or not variants[mode]["repeat_identical"]
            ):
                raise ValueError(f"Tensor release changed the output: {mode}")
            if mode != "baseline":
                lifetime_events = report["working_tensor_release"]["events"]
                step_counts = (
                    [report["config"]["warmup_steps"]]
                    if report["config"]["warmup_steps"]
                    else []
                )
                step_counts += [report["config"]["steps"]] * report["config"]["runs"]
                expected_stages = []
                for steps in step_counts:
                    for step in range(steps):
                        expected_stages += [
                            ("step_before_release", step),
                            ("step_after_release", step),
                        ]
                    expected_stages += [
                        ("decode_before_release", None),
                        ("decode_after_release", None),
                    ]
                if [
                    (e["stage"], e["step"]) for e in lifetime_events
                ] != expected_stages:
                    raise ValueError("Incomplete tensor lifetime events")
                savings = [
                    before["cuda_allocated_bytes"] - after["cuda_allocated_bytes"]
                    for before, after in zip(
                        lifetime_events[::2], lifetime_events[1::2]
                    )
                ]
                if not all(value > 0 for value in savings):
                    raise ValueError("Tensor release did not reduce live allocations")
                variants[mode]["tensor_release"] = {
                    "verified_pairs": len(savings),
                    "min_freed_bytes": min(savings),
                    "max_freed_bytes": max(savings),
                    "events": lifetime_events,
                }
        if suite == "vae":
            vae_events = report["vae_optimization"]["events"]
            generations = report["config"]["runs"] + bool(
                report["config"]["warmup_steps"]
            )
            if len(vae_events) != 2 * generations:
                raise ValueError(
                    f"Expected person and garment encodes for every generation: {mode}"
                )
            expected_forwards = 6 if mode == "encoder_tiled512" else 1
            if any(
                event["encoder_forward_count"] != expected_forwards
                for event in vae_events
            ):
                raise ValueError(f"Unexpected tile count: {mode}")
            encode_begin, encode_end = runs[0]["vae_event_range"]
            first_measured_encodes = vae_events[encode_begin:encode_end]
            variants[mode]["vae"] = {
                **report["components"]["vae"],
                "encoder_forward_counts": [
                    event["encoder_forward_count"] for event in vae_events
                ],
                "last_person_encode": vae_events[-2],
                "last_garment_encode": vae_events[-1],
                "first_measured_encodes": first_measured_encodes,
            }
            phase_begin, phase_end = runs[0]["memory_event_range"]
            variants[mode]["first_generation_phase_events"] = events[
                phase_begin:phase_end
            ]
        variants[mode]["cuda_peak_reduction_percent"] = 100 * (
            1
            - variants[mode]["cuda_peak_allocated_bytes"]
            / variants[baseline_mode]["cuda_peak_allocated_bytes"]
        )
        variants[mode]["cuda_peak_all_phases_reduction_percent"] = 100 * (
            1
            - variants[mode]["cuda_peak_allocated_bytes_all_phases"]
            / variants[baseline_mode]["cuda_peak_allocated_bytes_all_phases"]
        )
    result = {
        "suite": suite,
        "baseline_variant": baseline_mode,
        "matched_settings": {key: baseline["config"][key] for key in KEYS},
        "gpu": baseline["gpu"],
        "embedding_sha256": baseline["embedding_cache"]["sha256"],
        "variants": variants,
        "limitations": [
            "Desktop CUDA experiment, not Android memory or speed measurement.",
            "All variants use the same existing weight-only INT4 checkpoints; quality defects from INT4 remain.",
            "One image pair and shared GPU; timing is diagnostic, not a device performance guarantee.",
            (
                "Sequential CFG retains upstream batched conditioning and garment features; it only splits denoiser forwards."
                if suite == "memory"
                else (
                    "Tensor release preserves the previously tiled INT4 output; prior tiling and quantization quality changes remain."
                    if suite in ("lifetimes", "compute")
                    else "Tiled VAE encoding changes the latent representation; matching shape alone does not establish image quality."
                )
            ),
            "Phase release reloads models each generation; reloads and releases are included in pipeline timing.",
            "CUDA allocated/reserved excludes driver/context; host RSS is separate and includes startup.",
            "Difference metrics compare against INT4 baseline, not a target try-on image or BF16.",
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    for native in (False, True):
        size = images[baseline_mode].size if native else (384, 512)
        sheet = Image.new("RGB", (size[0] * len(modes), size[1] + 40), "white")
        draw = ImageDraw.Draw(sheet)
        for index, mode in enumerate(modes):
            sheet.paste(ImageOps.contain(images[mode], size), (index * size[0], 40))
            draw.text((index * size[0] + 10, 12), mode, fill="black")
        sheet.save(output / ("comparison_native.png" if native else "comparison.jpg"))
    if suite == "compute":
        chosen = "decoder32_cfg"
        result["selected_variant_for_this_pair"] = chosen
        reference = reports["spatial32_cfg"]
        result["selected_matches_all_convolution_tiling"] = [
            run["sha256"] for run in reports[chosen]["runs"]
        ] == [run["sha256"] for run in reference["runs"]] and [
            e["latent_sha256_fp32"]
            for e in reports[chosen]["decoder_optimization"]["events"]
        ] == [
            e["latent_sha256_fp32"] for e in reference["decoder_optimization"]["events"]
        ]
        for filename, crop in (
            ("recommended_comparison.jpg", None),
            ("garment_detail.jpg", (220, 100, 580, 540)),
        ):
            tiles = [
                images[mode].crop(crop) if crop else images[mode]
                for mode in (baseline_mode, chosen)
            ]
            width, height = tiles[0].size
            sheet = Image.new("RGB", (2 * width, height + 40), "white")
            draw = ImageDraw.Draw(sheet)
            for index, (tile, label) in enumerate(
                zip(tiles, ("Previous best", "Decoder tiles + sequential CFG"))
            ):
                sheet.paste(tile, (index * width, 40))
                draw.text((index * width + 6, 12), label, fill="black")
            sheet.save(output / filename, quality=95)
    (output / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--suite",
        choices=("memory", "vae", "decoder", "lifetimes", "compute"),
        default="memory",
    )
    args = parser.parse_args()
    print(json.dumps(compare(args.experiment, args.output, args.suite), indent=2))


if __name__ == "__main__":
    main()
