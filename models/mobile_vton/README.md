# Mobile-VTON evaluation adapter

Official source: https://github.com/tmllab/2026_CVPR_Mobile-VTON

Pinned code: `57410f20ac0e1703af795702e10473565feaa711`.
Pinned weights: `FlashStight/Mobile-VTON@0063147c171518b391fff5e44d04555c12370c35`.
Code and checkpoints are published for noncommercial research under CC BY-NC-SA 4.0.

The wrapper in `scripts/run_mobile_vton.py` follows upstream `inference.py` for
one person/garment pair, without requiring the full VITON-HD dataset. It uses
local checkpoints only, preserves the upstream source, uses a safe state-dict
load for the decoder, and records timings, peak memory, and component sizes.
Optional repeat checks seed both the initial noise and VAE sampling.

Weights are stored outside the repository at `/tmp/mobile_tryon/weights/mobile_vton`.
The Python dependencies currently come from the existing `comfyui` environment;
the experiment records their actual versions. No shared environment is modified.

```bash
git clone https://github.com/tmllab/2026_CVPR_Mobile-VTON.git models/mobile_vton/upstream
git -C models/mobile_vton/upstream checkout 57410f20ac0e1703af795702e10473565feaa711
hf download FlashStight/Mobile-VTON --revision 0063147c171518b391fff5e44d04555c12370c35 --cache-dir /tmp/mobile_tryon/hf_cache --local-dir /tmp/mobile_tryon/weights/mobile_vton --include 'checkpoint/*' --exclude 'checkpoint/.DS_Store'
mkdir -p experiments/2026-10-23-mobile_vton/inputs
curl --fail --location 'https://raw.githubusercontent.com/Zheng-Chong/CatVTON/7818397f25613beedb3d861a34769f607cfcf3b1/resource/demo/example/person/women/049713_0.jpg' --output experiments/2026-10-23-mobile_vton/inputs/person.jpg
curl --fail --location 'https://raw.githubusercontent.com/Zheng-Chong/CatVTON/7818397f25613beedb3d861a34769f607cfcf3b1/resource/demo/example/condition/upper/21514384_52353349_1000.jpg' --output experiments/2026-10-23-mobile_vton/inputs/garment.jpg
python scripts/audit_mobile_vton.py --manifest experiments/2026-10-23-mobile_vton/huggingface_manifest.json --weights-root /tmp/mobile_tryon/weights/mobile_vton --output experiments/2026-10-23-mobile_vton/checkpoint_audit.json
conda run --no-capture-output -n comfyui python scripts/run_mobile_vton.py --config experiments/2026-10-23-mobile_vton/config.json
```

Use `--device cpu --dtype fp32` only for functional checks. Desktop GPU timings
are not phone benchmarks. The public source currently contains no Android/iOS
runner or ready-to-use ONNX, Core ML, or TFLite model.

See `requirements-mobile-vton.txt` for the package versions actually tested. The
initial run uses the upstream maskless pipeline with an authored garment caption;
it does not claim to reproduce benchmark scores or the paper's mobile NPU build.

Pixel 4 component measurements and the input-refresh correction are documented
in [the Android experiment report](../../reports/mobile_vton_pixel4.md).
VAE FP32 matches the reference after the benchmark fix. A streamed-input Tryon
denoiser forward now completes on Pixel 4 in about 24.06 seconds at 1.46 GiB
RSS in the earlier component test. A 391MiB
MNN INT8 Tryon candidate was also generated: its host CPU single-forward
relative RMSE is 0.02690 versus 0.29655 for the 211MiB INT4 candidate, but the
Pixel 4 OpenCL run measured 25.46 seconds, 1.815GiB peak RSS, and relative RMSE
0.5144 in OpenCL Low mode. FP16 Tryon (731MiB) exceeded the Pixel 4 memory budget.
The latest structural INT4/FP16 candidate preserves 97 convolution weight tensors
and quantizes 341 with asymmetric INT4 block32. Its verified file is 287.72MiB.
On Pixel 4 OpenCL Low it measures 24.511 seconds over three forwards, 1.589GiB
peak RSS, and relative RMSE 0.08390 (host control: 0.08112). This is a structural
heuristic, not calibrated sensitivity analysis or final image quality validation.
Policy, conversion, and storage-audit instructions are in the experiment's
[`policies/README.md`](../../experiments/2026-10-24-pixel4/policies/README.md).
The same model in OpenCL Normal mode took 31.178 seconds, used 1.602GiB peak RSS,
and had relative RMSE 0.08466. Low remains the follow-up candidate on this fixture.

The subsequent full Pixel 4 trial completed one 768×1024 image with 28 steps and
56 sequential CFG forwards in **2,081 seconds**, at **1.603GiB peak RSS**.
Denoising forward/download alone took 1,647.47 seconds. This OpenCL implementation
is rejected for practical Pixel 4 latency. The per-timestep garment feature cache
also costs 1.723GiB per garment. See the [full trial](../../experiments/2026-10-24-pixel4/generation_hybrid_low/README.md)
for outputs, shared-cache FP32 controls, measurement exclusions, and remaining limits.
