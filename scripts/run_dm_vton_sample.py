"""Run one DM-VTON pair with the official CuPy correlation implementation."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "models/dm_vton/upstream"
sys.path.insert(0, str(UPSTREAM))


def tensor_image(path: Path, size: tuple[int, int]) -> torch.Tensor:
    image = (
        Image.open(path)
        .convert("RGB")
        .resize((size[1], size[0]), Image.Resampling.BICUBIC)
    )
    array = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 127.5 - 1
    return torch.from_numpy(array).unsqueeze(0)


def garment_mask(path: Path, size: tuple[int, int]) -> torch.Tensor:
    image = (
        Image.open(path)
        .convert("RGB")
        .resize((size[1], size[0]), Image.Resampling.NEAREST)
    )
    array = np.asarray(image, dtype=np.float32) / 255
    # The supplied garment has a white background; preserve non-white pixels.
    foreground = (array.min(axis=2) < 0.97).astype(np.float32)
    return torch.from_numpy(foreground[None, None])


def load(model, checkpoint: Path) -> None:
    # Official DM-VTON release checkpoints contain NumPy scalar metadata.
    # They are downloaded from the pinned GitHub release and are treated as
    # trusted artifacts for this local benchmark.
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = state.get("model", state)
    expected = model.state_dict()
    # The official loader intentionally copies only model keys; training
    # checkpoints may include a lazily-created correlation filter buffer.
    filtered = {name: state[name] for name in expected if name in state}
    missing = sorted(set(expected) - set(filtered))
    if missing:
        raise RuntimeError(f"Missing checkpoint tensors: {missing[:3]}")
    model.load_state_dict(filtered, strict=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--person",
        type=Path,
        default=ROOT / "experiments/2026-10-23-mobile_vton/inputs/person.jpg",
    )
    parser.add_argument(
        "--garment",
        type=Path,
        default=ROOT / "experiments/2026-10-23-mobile_vton/inputs/garment.jpg",
    )
    parser.add_argument("--warp", type=Path, default=Path("/tmp/dmvton_pf_warp.pt"))
    parser.add_argument(
        "--generator", type=Path, default=Path("/tmp/dmvton_pf_gen.1.pt")
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "experiments/2026-10-25-dm_vton"
    )
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument(
        "--fixtures", type=Path, help="Reuse exported person/clothes/edge .npy files"
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    # DM-VTON's pinned source targets CuPy 11's removed compile_with_cache API.
    # CuPy 14's RawModule preserves the same get_function launch contract.
    import cupy

    if not hasattr(cupy.cuda, "compile_with_cache"):
        cupy.cuda.compile_with_cache = lambda code: cupy.RawModule(code=code)
    from pipelines import DMVTONPipeline

    requested = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    )
    pipeline = DMVTONPipeline(align_corners=True)
    load(pipeline.warp_model, args.warp)
    load(pipeline.gen_model, args.generator)
    pipeline.eval().requires_grad_(False).to(requested)
    if args.runs < 1:
        raise ValueError("runs must be positive")
    if args.fixtures is not None:
        person, clothes, edge = (
            torch.from_numpy(np.load(args.fixtures / f"{name}.npy"))
            .float()
            .to(requested)
            for name in ("person", "clothes", "edge")
        )
    else:
        person = tensor_image(args.person, (256, 192)).to(requested)
        clothes = tensor_image(args.garment, (256, 192)).to(requested)
        edge = garment_mask(args.garment, (256, 192)).to(requested)
    with torch.inference_mode():
        # Warm-up and synchronize before timing.
        pipeline(person, clothes, edge, phase="test")
        if requested.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        timings = []
        output = None
        for _ in range(args.runs):
            start = time.perf_counter()
            output, warped = pipeline(person, clothes, edge, phase="test")
            if requested.type == "cuda":
                torch.cuda.synchronize()
            timings.append(time.perf_counter() - start)
    output = output[0].detach().cpu().float()
    warped = warped[0].detach().cpu().float()
    output.numpy().tofile(args.output / "tryon.f32")
    warped.numpy().tofile(args.output / "warped_cloth.f32")
    array = (
        ((output.permute(1, 2, 0).numpy() / 2 + 0.5).clip(0, 1) * 255)
        .round()
        .astype("uint8")
    )
    Image.fromarray(array).save(args.output / "result.png")
    params = sum(p.numel() for p in pipeline.parameters())
    storage = sum(p.numel() * p.element_size() for p in pipeline.parameters()) + sum(
        b.numel() * b.element_size() for b in pipeline.buffers()
    )
    report = {
        "status": "completed",
        "model": "DM-VTON PF student",
        "upstream_revision": "5dcd04ee3f5b1fb392df1184c887c1a7e910c72e",
        "device_requested": args.device,
        "device_selected": str(requested),
        "hardware": torch.cuda.get_device_name(requested)
        if requested.type == "cuda"
        else "CPU",
        "torch_cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated()
        if requested.type == "cuda"
        else None,
        "torch_cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved()
        if requested.type == "cuda"
        else None,
        "resolution": [256, 192],
        "warmup": 1,
        "runs": timings,
        "mean_seconds": float(np.mean(timings)),
        "fps": float(1 / np.mean(timings)),
        "parameters": params,
        "model_storage_bytes": storage,
        "weights": {
            "warp_bytes": args.warp.stat().st_size,
            "generator_bytes": args.generator.stat().st_size,
        },
        "input_note": "Garment foreground mask derived from near-white background for this sample; this is not the official VITON edge annotation.",
        "fixture_directory": str(args.fixtures) if args.fixtures is not None else None,
        "scope": "Single person/garment pair; functional speed and output smoke test, not FID/LPIPS evaluation.",
    }
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
