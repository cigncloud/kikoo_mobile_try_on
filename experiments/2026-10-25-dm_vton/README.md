# DM-VTON → MNN → Pixel 4

Measured on 2026-10-24 (UTC+8); this directory is the experiment ID.
See [the report](../../reports/dm_vton_pixel4.md) for scope, quality failures,
precision differences and memory definitions.

## Artifacts

- `metrics.json`: initial CUDA smoke test; `cuda_fixtures/` repeats with the exact exported tensors.
- `export.json`: static input/output contract and pinned source revision.
- `host_chain/`: resident two-stage MNN CPU control.
- `pixel4_opencl_low/`, `pixel4_opencl_high/`: FP16 vs FP32 phone runs.
- `upper_body_inputs/crop.json`: manual crop coordinates.
- `pixel4_upper_body_high/`: cache creation; `pixel4_upper_body_cached/`: new process reusing it.
- `summary.json`: aggregated timings/errors; `environment.json`: revisions and SHA256.
- `comparison.png`, `comparison_upper_body.png`: original CUDA and phone output.
- `mnn_dm_vton_bench.cpp`: synchronized two-stage runner; optional final argument enables kernel caching.

Raw tensors, images, model files and binaries are local, ignored artifacts.
The garment mask is a near-white-background heuristic, not an official VITON annotation.

## Export and convert

Use the pinned official source in `models/dm_vton/upstream/` and official release
`1.0.0` checkpoints in `/tmp/dmvton_pf_warp.pt` and `/tmp/dmvton_pf_gen.1.pt`.
These pickle checkpoints are loaded as trusted official artifacts. Run export in
the project environment with PyTorch/ONNX; the CUDA smoke test additionally needs CuPy.

```bash
.venv/bin/python scripts/export_dm_vton_mnn.py
/tmp/mobile_tryon_mnn_host/MNNConvert -f ONNX \
  --modelFile experiments/2026-10-25-dm_vton/dmvton_warp.onnx \
  --MNNModel experiments/2026-10-25-dm_vton/dmvton_warp.mnn --bizCode dm_vton_mobile
/tmp/mobile_tryon_mnn_host/MNNConvert -f ONNX \
  --modelFile experiments/2026-10-25-dm_vton/dmvton_generator.onnx \
  --MNNModel experiments/2026-10-25-dm_vton/dmvton_generator.mnn --bizCode dm_vton_mobile
```

Source revision: `5dcd04ee3f5b1fb392df1184c887c1a7e910c72e`.
MNN 3.6.1 and the patched Android library are the same as the preceding Pixel 4
experiment; its `mnn_disable_buffer_winograd.patch` is applied.

## Build and run

```bash
export DMVTON_MNN_SOURCE=/tmp/MNN-3.6.1
export DMVTON_NDK=/home/michaelk/Android/Sdk/ndk/27.1.12297006
"$DMVTON_NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android26-clang++" \
  -std=c++17 -O2 -static-libstdc++ \
  experiments/2026-10-25-dm_vton/mnn_dm_vton_bench.cpp \
  -I"$DMVTON_MNN_SOURCE/include" -I"$DMVTON_MNN_SOURCE/source" \
  -I"$DMVTON_MNN_SOURCE/3rd_party" \
  -L/tmp/mobile_tryon_mnn_android/OFF/arm64-v8a -lMNN -llog \
  -o experiments/2026-10-25-dm_vton/mnn_dm_vton_bench_cached
```

Install the binary, `libMNN.so`, two `.mnn` files and three `dmvton_input_*.f32`
files under `/data/local/tmp/dm_vton_20261025/`. On the phone:

```sh
cd /data/local/tmp/dm_vton_20261025
mkdir -p output kernel_cache
LD_LIBRARY_PATH=. ./mnn_dm_vton_bench_cached . . output 3 1 3 kernel_cache
```

Arguments are model directory, input directory, output directory, backend
(`3` OpenCL, `0` CPU), precision (`1` High/FP32, `2` Low/FP16), measured runs,
and optional cache directory. A warm-up precedes measured runs. Run a second
process using the same cache to measure cached startup. Use different output
directories to retain both measurements. Delete/rebuild caches when model,
precision, runtime or device changes; caches are not portable model artifacts.

Original CUDA control used the pinned vLLM image recorded in `environment.json`:

```bash
python scripts/run_dm_vton_sample.py --fixtures experiments/2026-10-25-dm_vton \
  --output experiments/2026-10-25-dm_vton/cuda_fixtures
.venv/bin/python scripts/analyze_dm_vton_mnn.py
.venv/bin/pytest -q tests
```

The analyzer expects all named baseline and phone run directories above.
