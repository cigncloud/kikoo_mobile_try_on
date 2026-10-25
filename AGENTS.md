# Repository Guidelines

## Project Structure & Module Organization

This repository is an empty scaffold for evaluating mobile virtual try-on (VTON) models. Use this layout:

- `models/<model_name>/`: model adapters, checkpoint notes, and inference wrappers.
- `experiments/<date>-<model>/`: fixed inputs, config, outputs, and a brief `README.md`.
- `scripts/`: preprocessing, benchmarking, export, and device automation.
- `tests/`: adapter and input/output contract smoke tests.
- `assets/`: small, licensed fixtures only; keep weights and generated datasets outside Git.
- `reports/`: comparison tables linked to experiment IDs.

## Build, Test, and Development Commands

No build system exists yet. A Python implementation should expose root commands such as:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_benchmark.py --model <name> --config configs/<name>.yaml
pytest -q
```

Record device, runtime, model revision, precision, resolution, latency, peak memory, and output path.

## Coding Style & Naming Conventions

Use Python 3.10+, four-space indentation, type hints on public functions, and small modules. Use `snake_case` for files, functions, and variables; `PascalCase` for classes; and IDs such as `fashn_vton_1_5`. Use Ruff/Black. Move repeatable notebook logic into `scripts/` or `models/`.

## Testing Guidelines

Use pytest with names such as `test_<behavior>.py`. Separate fast adapter tests from GPU benchmarks. Check seeds where supported, garment categories, dimensions, malformed inputs, and missing files. Never commit private or unlicensed person images.

## Commit & Pull Request Guidelines

Git history is unavailable in this scaffold. Use imperative Conventional Commit-style subjects, for example `feat: add FASHN adapter` or `bench: compare CatVTON latency`. PRs should state the model revision, license, hardware, preprocessing, metrics, and conclusion; link the paper or model card; and include only redistributable outputs.

## Mobile Evaluation & Model Governance

Run two tracks: an on-device preview using pose, segmentation, and garment-warping models exported to ONNX, Core ML, or TFLite; and a server-side generative path for high-fidelity images. Measure preprocessing and transfer as well as inference against RAM, latency, thermal, battery, offline, and privacy limits.

Track code, checkpoint, dataset, and output licenses independently. Start with **Mobile-VTON** (CVPR 2026, 415M parameters claimed, offline design) and **DM-VTON** (reported 37.8 MB GPU footprint), then verify Android/iOS runtimes because their public repos primarily expose research/PyTorch code. Use FASHN VTON 1.5 as a server baseline; compare CatVTON, IDM-VTON, and OOTDiffusion for quality and compression. Evaluate CatV2TON separately for video. Reproduce all reported metrics on target phones and check CC BY-NC-SA restrictions before commercial use.

## Model Downloads and Excluded Build Files

Large checkpoints, converted `.mnn`/`.onnx` networks, native `.so` libraries,
APK files, and build outputs are excluded from Git. Download and regenerate
them locally:

- DM-VTON PF release 1.0.0: <https://github.com/KiseKloset/DM-VTON/releases/tag/1.0.0>
  (`dmvton_pf_warp.pt` and `dmvton_pf_gen.pt`); export MNN stages with
  `scripts/export_dm_vton_mnn.py`.
- MediaPipe Pose Landmarker Lite: <https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task>.
  Save it under `weights/mediapipe/` and verify the SHA256 in
  `android/app/src/main/assets/preprocess/model.json`.
- VITON-Clean download and layout: use the pinned DM-VTON README/Colab links
  recorded in `reports/dm_vton_usage_audit.md`.
- Mobile-VTON source and Hugging Face checkpoint commands are in
  `models/mobile_vton/README.md`.

The repository stores provenance, scripts, small fixtures, and reports. Run
the build scripts after downloading the excluded inputs to recreate the APK.
