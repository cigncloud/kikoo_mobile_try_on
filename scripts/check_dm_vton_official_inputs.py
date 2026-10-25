"""Reproduce official DM-VTON test inputs and precompute fixed garment masks.

Run in the recorded vLLM CUDA container. This is a small visual reproduction,
not a replacement for the full VITON-Clean FID/LPIPS evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "models/dm_vton/upstream"))
sys.path.insert(0, str(ROOT / "scripts"))


def save_tensor(value: torch.Tensor, path: Path) -> Image.Image:
    rgb = ((value[0].detach().cpu().permute(1, 2, 0).numpy() + 1) * 127.5)
    image = Image.fromarray(rgb.clip(0, 255).round().astype(np.uint8))
    image.save(path)
    return image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/2026-10-24-dm_vton-preprocessing")
    parser.add_argument("--u2net-source", type=Path)
    parser.add_argument("--u2net-checkpoint", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this original-source control")
    if args.samples < 1:
        raise ValueError("samples must be positive")
    import cupy
    if not hasattr(cupy.cuda, "compile_with_cache"):
        cupy.cuda.compile_with_cache = lambda code: cupy.RawModule(code=code)
    from dataloader.viton_dataset import LoadVITONDataset
    from pipelines import DMVTONPipeline
    from run_dm_vton_sample import load

    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    pipeline = DMVTONPipeline(align_corners=True)
    load(pipeline.warp_model, Path("/tmp/dmvton_pf_warp.pt"))
    load(pipeline.gen_model, Path("/tmp/dmvton_pf_gen.1.pt"))
    pipeline.eval().requires_grad_(False).cuda()
    dataset = LoadVITONDataset(str(args.dataset), phase="test")
    count = min(args.samples, len(dataset))
    sheet = Image.new("RGB", (192 * 4, 280 * count), "white")
    draw = ImageDraw.Draw(sheet)
    rows = []
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode(), cupy.cuda.Device(0):
        for index in range(count):
            sample = dataset[index]
            tensors = [sample[k].unsqueeze(0).cuda() for k in ("image", "color", "edge")]
            person, clothes, edge = tensors
            assert person.shape == clothes.shape == (1, 3, 256, 192)
            assert edge.shape == (1, 1, 256, 192)
            start = time.perf_counter()
            result, warped = pipeline(*tensors, phase="test")
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            folder = output / "official" / f"{index:02d}"
            folder.mkdir(parents=True, exist_ok=True)
            for name, value in zip(("person", "clothes", "edge"), tensors):
                array = value.cpu().numpy().astype("<f4")
                np.save(folder / f"{name}.npy", array)
                array.tofile(folder / f"{name}.f32")
            result.cpu().numpy().astype("<f4").tofile(folder / "cuda_reference.f32")
            images = [save_tensor(t, folder / f"{name}.png") for t, name in zip((person, clothes, warped, result), ("person", "clothes", "warped", "result"))]
            Image.fromarray((edge[0, 0].cpu().numpy() * 255).round().astype(np.uint8)).save(folder / "edge.png")
            for column, (label, image) in enumerate(zip(("Person", "Garment", "Warp", "Original CUDA"), images)):
                x, y = column * 192, index * 280
                draw.text((x + 4, y + 3), f"{index:02d} {label}", fill="black")
                sheet.paste(image, (x, y + 24))
            rows.append({"index": index, "person": sample["p_name"], "garment": sample["c_name"], "seconds": elapsed})
            print(json.dumps(rows[-1]), flush=True)
    sheet.save(output / "official_comparison.jpg", quality=95)
    report = {"status": "completed", "scope": "First test_pairs entries in source order; visual control only; no FID/LPIPS", "dataset_count": len(dataset), "pairs": rows, "device": torch.cuda.get_device_name(), "torch": torch.__version__, "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated()}
    (output / "official_metrics.json").write_text(json.dumps(report, indent=2) + "\n")

    if args.u2net_source is not None:
        if args.u2net_checkpoint is None:
            parser.error("--u2net-source requires --u2net-checkpoint")
        spec = importlib.util.spec_from_file_location("official_dmvton_u2net", args.u2net_source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        model = module.U2NETP(3, 1).eval().cuda()
        model.load_state_dict(torch.load(args.u2net_checkpoint, map_location="cpu", weights_only=True), strict=True)
        mask_dir = ROOT / "weights/dmvton_catalog_masks"
        mask_dir.mkdir(parents=True, exist_ok=True)
        catalog = json.loads((ROOT / "android/app/src/main/assets/catalog.json").read_text())
        review = Image.new("RGB", (192 * 3, 280 * len(catalog)), "white")
        review_draw = ImageDraw.Draw(review)
        metadata = {"method": "Official KiseKloset U2NETP, RGB [-1,1], per-image min/max, >0.5", "checkpoint_sha256": hashlib.sha256(args.u2net_checkpoint.read_bytes()).hexdigest(), "source_sha256": hashlib.sha256(args.u2net_source.read_bytes()).hexdigest(), "garments": {}}
        with torch.inference_mode():
            for index, item in enumerate(catalog):
                path = ROOT / "android/app/src/main/assets" / item["image"]
                image = Image.open(path).convert("RGB")
                rgb = np.asarray(image, np.float32)
                tensor = torch.from_numpy(rgb.transpose(2, 0, 1) / 127.5 - 1).unsqueeze(0).cuda()
                pred = model(tensor)[:, 0]
                span = pred.max() - pred.min()
                if not torch.isfinite(pred).all() or span < 1e-6:
                    raise RuntimeError(f"Invalid garment segmentation: {item['id']}")
                mask = ((pred - pred.min()) / span > 0.5)[0].cpu().numpy()
                mask_image = Image.fromarray(mask.astype(np.uint8) * 255)
                mask_image.save(mask_dir / f"{item['id']}.png")
                isolated = Image.fromarray(np.where(mask[:, :, None], rgb, 255).astype(np.uint8))
                for column, tile in enumerate((image, mask_image.convert("RGB"), isolated)):
                    review.paste(tile, (column * 192, index * 280 + 24))
                review_draw.text((4, index * 280 + 4), item["id"] + " / mask / isolated", fill="black")
                metadata["garments"][item["id"]] = {"image_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "mask_sha256": hashlib.sha256((mask_dir / f"{item['id']}.png").read_bytes()).hexdigest(), "coverage": float(mask.mean())}
        review.save(output / "catalog_masks.jpg", quality=95)
        (mask_dir / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(json.dumps(metadata), flush=True)


if __name__ == "__main__":
    main()
