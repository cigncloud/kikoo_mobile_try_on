"""Bundle validated FP32 exports, official garment masks and mobile pose model."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "android/app/src/main/assets"
EXPERIMENT = ROOT / "experiments/2026-10-25-dm_vton"
OFFICIAL_CONTROL = ROOT / "experiments/2026-10-24-dm_vton-preprocessing/official/00"
POSE_SHA = "59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a"


def binary_annotation(image: Image.Image, size: tuple[int, int]) -> np.ndarray:
    """Use the supplied annotation, preserving white fabric and printed logos."""
    if image.size != size:
        raise ValueError("Garment and mask dimensions differ; align them before packaging")
    mask = np.asarray(image.convert("L")) > 127
    if not 0.02 < mask.mean() < 0.95:
        raise ValueError("Empty or implausible garment annotation")
    return mask


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path(os.environ.get("TRYON_VITON_DATASET", "/tmp/dmvton_official_dataset/VITON-Clean/VITON_test")))
    parser.add_argument("--pose-model", type=Path, default=ROOT / "weights/mediapipe/pose_landmarker_lite.task")
    args = parser.parse_args()
    for directory in ("models", "catalog", "preprocess"):
        (ASSETS / directory).mkdir(parents=True, exist_ok=True)
    models = []
    for stage in ("warp", "generator"):
        source = EXPERIMENT / f"dmvton_{stage}.mnn"
        target = ASSETS / "models" / source.name
        shutil.copy2(source, target)
        models.append({"file": source.name, "bytes": target.stat().st_size, "sha256": hashlib.sha256(target.read_bytes()).hexdigest()})
    (ASSETS / "models.json").write_text(json.dumps(models, indent=2) + "\n")
    if hashlib.sha256(args.pose_model.read_bytes()).hexdigest() != POSE_SHA:
        raise ValueError("Unexpected pose model checksum")
    shutil.copy2(args.pose_model, ASSETS / "preprocess/pose_landmarker_lite.task")
    (ASSETS / "preprocess/model.json").write_text(json.dumps({
        "file": "pose_landmarker_lite.task", "sha256": POSE_SHA,
        "source": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
        "runtime": "MediaPipe Tasks Vision 0.10.14, CPU delegate for Pixel 4 mask stability",
        "adaptation": "Pose Lite landmarks and segmentation replace web demo YOLOv7 + selfie segmentation; eye/hip crop and foreground threshold retained",
    }, indent=2) + "\n")
    definitions = [
        ("white_tee", "화이트 프린트 티", "라운드넥 · 반소매", "015095_1.jpg"),
        ("black_tee", "블랙 티셔츠", "베이직 · 반소매", "019243_1.jpg"),
        ("navy_hoodie", "네이비 후드", "단추 · 긴소매", "004727_1.jpg"),
    ]
    catalog = []
    for identifier, title, subtitle, filename in definitions:
        image = Image.open(args.dataset / "test_color" / filename).convert("RGB")
        if image.size != (192, 256):
            raise ValueError(f"Unexpected official garment size: {filename} {image.size}")
        mask_path = args.dataset / "test_edge" / filename
        mask = binary_annotation(Image.open(mask_path), image.size).astype(np.float32)
        array = np.asarray(image, np.float32).transpose(2, 0, 1) / 127.5 - 1
        image.save(ASSETS / "catalog" / f"{identifier}.png")
        Image.fromarray((mask * 255).astype(np.uint8)).save(ASSETS / "catalog" / f"{identifier}_mask.png")
        array.astype("<f4").tofile(ASSETS / "catalog" / f"{identifier}.f32")
        mask.astype("<f4").tofile(ASSETS / "catalog" / f"{identifier}_edge.f32")
        catalog.append({"id": identifier, "name": title, "description": subtitle,
            "image": f"catalog/{identifier}.png", "tensor": f"catalog/{identifier}.f32",
            "edge": f"catalog/{identifier}_edge.f32", "source": f"Official VITON-Clean test_color/{filename}",
            "mask_source": f"Official VITON-Clean test_edge/{filename}",
            "mask_method": "Supplied annotation >127; no RGB threshold or inferred segmentation",
            "annotation_sha256": hashlib.sha256(mask_path.read_bytes()).hexdigest()})
    expected = {str(item[k]).split("/")[-1] for item in catalog for k in ("image", "tensor", "edge")}
    expected.update(f"{item['id']}_mask.png" for item in catalog)
    for path in (ASSETS / "catalog").iterdir():
        if path.is_file() and path.name not in expected:
            path.unlink()
    (ASSETS / "catalog.json").write_text(json.dumps(catalog, ensure_ascii=False, indent=2) + "\n")
    test = ROOT / "android/app/src/androidTest/assets"
    test.mkdir(parents=True, exist_ok=True)
    for name in ("person", "clothes", "edge"):
        shutil.copy2(EXPERIMENT / f"dmvton_input_{name}.f32", test / f"{name}.f32")
    shutil.copy2(EXPERIMENT / "dmvton_reference_tryon.f32", test / "reference.f32")
    official = test / "official"
    official.mkdir(exist_ok=True)
    for name in ("person.f32", "clothes.f32", "edge.f32", "cuda_reference.f32", "person.png"):
        shutil.copy2(OFFICIAL_CONTROL / name, official / name)
    print(f"Prepared {len(models)} FP32 stages, Pose Lite and {len(catalog)} annotated garments")


if __name__ == "__main__":
    main()
