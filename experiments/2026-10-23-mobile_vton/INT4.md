# Mobile-VTON INT4 comparison

This experiment uses `torchao==0.10.0` with group size 128. Only `nn.Linear`
weights in TryonNet and GarmentNet are quantized. Activations and other model
weights remain BF16; the small final decoder remains FP32. Both text encoders
are removed from runtime after caching the same prompt embeddings.

## Artifacts

- `outputs_int4_cpu/denoiser_int4.pt`: packed TryonNet weights and unquantized layers.
- `outputs_int4_cpu/denoiser_garment_int4.pt`: packed GarmentNet state.
- `outputs_int4_cpu/metrics.json`: layer coverage, exact storage, artifact hashes.
- `outputs_bf16_cached/embeddings.pt`: shared BF16 CPU text embeddings for this description.
- `outputs_int4_cpu_384/`: 384×512 INT4 inference using reloaded packed checkpoints.
- `outputs_bf16_cpu_384/`: matched BF16 CPU inference.
- `outputs_comparison/`: side-by-side images, garment crops, and comparison metrics.

The `outputs_int4_cpu/` states use torchao's native **CPU** packed layout. They need this
PyTorch/torchao environment and the original architecture/configuration. They
are not TFLite, Core ML, ONNX, or an Android application. CUDA quantization must
start from the original weights to obtain a CUDA-compatible layout.

## Reproduce

Prepare the text cache once (omit `--prepare-only` to generate an image):

```bash
.venv/bin/python scripts/run_mobile_vton.py --config experiments/2026-10-23-mobile_vton/config.json --device cpu --dtype bf16 --embedding-cache experiments/2026-10-23-mobile_vton/outputs_bf16_cached/embeddings.pt --output-dir experiments/2026-10-23-mobile_vton/outputs_bf16_cached --prepare-only
```

Create packed checkpoints using `config_int4_cpu.json` with `--prepare-only`.
For the matched CPU comparison, run:

```bash
.venv/bin/python scripts/run_mobile_vton.py --config experiments/2026-10-23-mobile_vton/config_int4_cpu.json --width 384 --height 512 --quantized-state-dir experiments/2026-10-23-mobile_vton/outputs_int4_cpu --output-dir experiments/2026-10-23-mobile_vton/outputs_int4_cpu_384
.venv/bin/python scripts/run_mobile_vton.py --config experiments/2026-10-23-mobile_vton/config_int4_cpu.json --quantization none --width 384 --height 512 --output-dir experiments/2026-10-23-mobile_vton/outputs_bf16_cpu_384
.venv/bin/python scripts/compare_mobile_vton.py --baseline experiments/2026-10-23-mobile_vton/outputs_bf16_cpu_384 --quantized experiments/2026-10-23-mobile_vton/outputs_int4_cpu_384 --output experiments/2026-10-23-mobile_vton/outputs_comparison
.venv/bin/python -m pytest -q tests/test_mobile_vton_quantization.py
```

Both image runs use seed 42, 28 steps, guidance 2.5, the same person and garment,
and identical cached text tensors. For 384×512, the wrapper calls upstream
`set_sample_size_rsd` on both denoisers; otherwise the original pipeline leaves
GarmentNet's rotary embeddings at the training resolution and fails with a
shape mismatch. The same resize applies to BF16 and INT4. This is an exploratory
lower-resolution test, not the paper's native-resolution quality protocol.
The initial full-resolution CPU attempt was stopped
after roughly 40 seconds per denoising step. Its checkpoints are valid; its
metrics explicitly say `interrupted`. Native-resolution GPU results were later
completed as documented below. No phone speed/RAM measurement is claimed.
The earlier 17.1-second GPU BF16 run is a
separate experiment and must not be used to claim an INT4 speedup here.
The two 384×512 generation jobs overlap on a shared CPU host. Their elapsed
times are diagnostic only; use serialized repeated runs for a speed benchmark.

## Result

Both 28-step runs completed. At 384×512, model tensor storage fell from 1.938GB
to 0.885GB, and whole-process peak RSS from 4.284GB to 2.836GB. Relative to the
BF16 output, MAE was 0.02362 and PSNR 27.99dB. These are output differences,
not ground-truth accuracy scores. This pair shows stronger neck/chest artifacts
in INT4; quality acceptance remains open. See the
[completed report](../../reports/mobile_vton_int4.md) and
[comparison image](outputs_comparison/bf16_vs_int4.jpg).

## Native resolution in the vLLM GPU container

BF16 and INT4 completed sequentially at **768×1024**, 28 steps, seed 42,
guidance 2.5, batch 1, one warmup step and two measured runs per precision.
They use identical cached text embeddings. No resolution adapter was applied.
The GPU is an RTX 3080 Laptop 16GB. This is a shared desktop GPU test.

The image is pinned to
`vllm/vllm-openai@sha256:04563c302537a91aa49ebdfbceda96111c5712275999b7e8804fa598f0b5641d`.
It runs the existing `.venv` and read-only `comfyui` Python environment with
PyTorch 2.7.1+cu126 and torchao 0.10.0, rather than launching `vllm serve`.
The local launcher requires these existing dependencies and the weights/cache
under `/tmp/mobile_tryon`; it is not a self-contained model image.

```bash
bash scripts/run_mobile_vton_in_vllm.sh --config experiments/2026-10-23-mobile_vton/config_bf16_gpu_native.json
bash scripts/run_mobile_vton_in_vllm.sh --config experiments/2026-10-23-mobile_vton/config_int4_gpu_native.json
bash scripts/run_mobile_vton_in_vllm.sh --config experiments/2026-10-23-mobile_vton/config_int4_gpu_native.json --quantized-state-dir experiments/2026-10-23-mobile_vton/outputs_vllm_int4_native --output-dir experiments/2026-10-23-mobile_vton/outputs_vllm_int4_native_reloaded --runs 1
.venv/bin/python scripts/compare_mobile_vton.py --baseline experiments/2026-10-23-mobile_vton/outputs_vllm_bf16_native --quantized experiments/2026-10-23-mobile_vton/outputs_vllm_int4_native --output experiments/2026-10-23-mobile_vton/outputs_vllm_comparison_native
```

CUDA checkpoints use `TensorCoreTiledLayout`, with 626,148,367 bytes across
the two denoisers. Their reload produced the exact original INT4 PNG hash.
The loader resolves `cuda` to an explicit GPU index before deserializing.
Minimal images without Git validate the upstream revision from `.git/HEAD`.

| Measurement | BF16 + cached text | INT4 + cached text |
| --- | ---: | ---: |
| Mean pipeline time | 14.981 s | 39.113 s |
| Model tensor storage | 1.947 GB | 0.973 GB |
| CUDA peak allocated | 3.14 GiB | 2.24 GiB |
| CUDA peak reserved | 3.93 GiB | 3.27 GiB |

GPU allocation fell 28.6%, but INT4 was slower in this configuration. These
allocator metrics exclude CUDA context/driver memory; host RSS is separate.
This pair retains garment shape but loses much of the printed graphic under
INT4, so quality is not accepted. MAE 0.01279 and PSNR 32.19 dB are differences
from BF16, not ground-truth accuracy.

[Native-size comparison](outputs_vllm_comparison_native/bf16_vs_int4_native.png)
· [Comparison JSON](outputs_vllm_comparison_native/comparison.json)
