# Pixel 4 component experiment — 2026-10-24

See [the report](../../reports/mobile_vton_pixel4.md) and [summary.json](summary.json).
One full on-phone VTON generation now completed: **2,081s (34m41s)** and
**1.603GiB peak RSS**, at 768×1024 with 28 steps and 56 CFG forwards.
See [the generation trial](generation_hybrid_low/README.md), its
[comparison](generation_hybrid_low/comparison.json) and
[images](generation_hybrid_low/comparison.png). Historical folders remain separate
component tests. This Pixel 4 OpenCL path is rejected for practical latency.

## Runtime and artifacts

- Pixel 4, Android 13, Adreno 640; native arm64 CLI, MNN 3.6.1 OpenCL buffer mode.
- Mobile-VTON revisions and model hashes are in `environment.json`.
- MNN uses `mnn_disable_buffer_winograd.patch` to avoid the observed Winograd crash.
- `exports/` contains ignored ONNX/MNN models and fixed FP32 reference fixtures.
- `outputs_*_refreshed/` contains corrected-input VAE/garment measurements and the earlier Tryon memory-pressure failure.
- `outputs_tryon_opencl_buffer_streamed/` contains the successful streamed-input Tryon forward (three repeats).
- `outputs_tryon_host_int8_control/` contains the host CPU control for the 391MiB INT8 Tryon model; it is not a phone measurement.
- `outputs_tryon_int8_opencl_streamed/` contains the completed Pixel 4 INT8 OpenCL forward: 25.46s, 1.815GiB peak RSS, and relative RMSE 0.5144.
- The 731MiB FP16 Tryon attempt was stopped by the phone memory monitor at about 2.25GiB RSS. INT8 fits the memory monitor but its OpenCL Low output error is too large for selection.
- `outputs_tryon_structural_int4_fp16_block32_opencl/` holds the latest mixed candidate: 24.511s over three forwards, 1.589GiB peak RSS, relative RMSE 0.08390. Its host control is in the matching `_host/` folder (relative RMSE 0.08112).
- `outputs_tryon_structural_int4_fp16_block32_opencl_normal/` holds the same model with Normal arithmetic: 31.178s over three forwards, 1.602GiB peak RSS, relative RMSE 0.08466. Low remains the follow-up candidate; final image quality is unverified.
- The mixed model is 287.72MiB, with 97 FP16 and 341 INT4 convolution weight tensors. [Policy and rebuild instructions](policies/README.md) explain actual storage verification and the structural heuristic.
- `outputs_tryon_hybrid_top8_skip_q_opencl/` retains an unsuccessful preliminary INT8/FP32 exclusion trial (relative RMSE 0.53436). Its layer-size proxy incorrectly omitted depthwise groups; it is not a sensitivity-calibrated policy.
- Older output folders retain historical runs whose inputs were not refreshed.

## Input refresh correction

The original benchmark uploaded inputs only once. Repeated forwards then changed
the VAE output. Refreshing every input before each measured forward restores
agreement with PyTorch. Previous large VAE errors do not establish a converter bug.

`mnn_component_bench_refreshed.cpp` is the corrected-input source used for the
first phone reruns. The latest `mnn_component_bench.cpp` reads one input at a
time on each forward; its streamed binary was run on Pixel 4 for the Tryon test.

## Reproduce corrected-input measurements

With the pinned MNN library, exported models, fixtures and executable installed
under `/data/local/tmp/mobile_tryon_20261024/`:

```bash
adb -s 98211FFAZ004ME devices -l
adb -s 98211FFAZ004ME shell 'cd /data/local/tmp/mobile_tryon_20261024 && LD_LIBRARY_PATH=. sh run_component_refreshed.sh vae_encoder_opencl_buffer_fp32_refreshed.json'
python scripts/analyze_pixel4_component.py --component vae_encoder --variant opencl_buffer_fp32_refreshed
```

Pull the named `.f32`, metrics and logs from the device into the corresponding
`outputs_<label>/` folder before analysis. Run commands from the repository root.
Use a new manifest/output label for additional measurements to preserve old logs.
The monitor stops the benchmark when system available RAM falls below 128MiB.
Numerical comparisons use the final saved measured forward against FP32 fixtures;
three timing repeats do not establish accuracy across inputs or timesteps.

The model exports are experimental. MNN INT4 quantizes more layer types than
the desktop TorchAO configuration; the single Tryon forward is not a generated
image quality validation. The host INT8 control has relative RMSE 0.02690 versus
0.29655 for INT4, while the Pixel 4 OpenCL Low INT8 run measured 0.5144.
The structural INT4/FP16 candidate reduces phone error to 0.08390 with slightly
more memory than INT4. Block size and retained layers changed together, so this
experiment does not isolate the cause of improvement. Final image quality,
multiple timesteps and the complete scheduler/CFG pipeline remain unverified.

Runtime `precision=2` (Low) uses FP16 storage/compute in OpenCL kernels;
`precision=0` (Normal) requests FP16 storage/FP32 compute. The Normal comparison
uses separate manifests and outputs with the `_normal` suffix. Neither setting
changes the stored INT4/FP16 model file. Backend-3 output placement alone does
not prove every operation executed exclusively on the GPU.
