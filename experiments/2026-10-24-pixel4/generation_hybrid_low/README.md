# Pixel 4 full-generation trial

This experiment runs person VAE encoding, 28 FlowMatch Euler steps with 56
sequential CFG forwards, and the final decoder on the phone. Resolution is
768×1024, guidance 2.5, shift 3, seed 42. `generation.json` pins the exact
timesteps, sigmas, input provenance, and model choices. `status.json` reports
whether the phone run actually completed; prepared inputs are not a completion.

## Result

One complete phone generation finished in **2,081s (34m41s)** with **1.603GiB
peak RSS**. VAE, denoising and decoding wall times were 292.35s, 1,747.78s and
37.50s. Denoising forward/download alone took 1,647.47s, so startup removal
cannot resolve the latency. This implementation is rejected for practical use
on Pixel 4. Outputs stayed finite; first CFG/Euler and VAE blending checks were
exact. The final image tensor relative RMSE against the shared-cache FP32
control is 0.16192, which is not a perceptual quality score.

See [the generated image](pixel4_result.png), [side-by-side comparison](comparison.png)
and [full measurements](comparison.json). This is a single pair/seed trial.

## Models and cache

- Tryon: the verified structural INT4/FP16 block32 model, OpenCL Low precision.
- Person VAE: FP32, 512-pixel tiles with 25% overlap. Three additional exports
  handle exact 512×384, 256×512, and 256×384 boundaries without padding.
- Decoder: the existing FP16 export, OpenCL Low.
- Fixed garments: original FP32 GarmentNet features for all 28 timesteps,
  serialized as FP16. The cache is **1,849,688,064 bytes (1.723GiB) per garment**
  for this exact latent/seed/scheduler, excluding other assets. It is a test
  strategy, not a practical catalog-storage solution.
- Posterior parameters and projected image/text embeddings come from the pinned
  earlier BF16 cache, promoted to FP32. Both phone and FP32 CUDA control share
  these inputs and the same FP16 features. This is not a fully uncompressed
  conditioning baseline, nor bitwise RNG equivalence to prior BF16/CUDA runs.

## Prepare and build

From the repository root:

```bash
.venv/bin/python scripts/prepare_pixel4_generation.py --stage prepare
.venv/bin/python scripts/prepare_pixel4_generation.py --stage vae
# Run --stage garment in the pinned vLLM CUDA container, as recorded in the report.
```

Build `../mnn_generation.cpp` against the pinned patched MNN 3.6.1 headers and
Android library, using NDK 27.1.12297006, API 26, `-O2 -std=c++17`,
`-ffp-contract=off`, and `-static-libstdc++`. The last flag avoids an undeployed
`libc++_shared.so` dependency. The executable only needs `libMNN.so` and Android
system libraries. `input_inventory.json` records generated inputs/model hashes.

## Run and compare

Push the generated folder, models, runtime, executable and shell wrapper under
`/data/local/tmp/mobile_tryon_20261024/`, then:

```bash
adb -s 98211FFAZ004ME shell 'cd /data/local/tmp/mobile_tryon_20261024 && sh run_generation.sh'
# Run scripts/compare_pixel4_generation.py --stage reference in the pinned CUDA container.
# Pull root .f32 outputs, .json metrics and .log files, without re-pulling data/models.
.venv/bin/python scripts/compare_pixel4_generation.py --stage analyze
```

Preserve existing results before rerunning. The wrapper stops below 128MiB
system MemAvailable and runs each stage in a separate process. Fixed-shape
sessions release serialized model buffers after their first forward. Every
forward refreshes inputs; features are read one at a time.

## Measurement scope

Online wall time includes model/session creation, all neural forwards, input
uploads, cache reads, CPU CFG/Euler/posterior/blending, tensor downloads, and
checkpoint writes. It excludes offline garment preparation, ADB transfers,
JPEG resizing/normalization, and host PNG encoding. No APK or camera path is
tested. Neural stages request OpenCL; tensor-backend audits do not prove every
kernel's execution device. Peak RSS includes visible GPU mappings but may omit
driver allocations. Sequential stage peaks are combined using their maximum.

The FP32 CUDA control is a separate desktop comparison, not phone latency.
Analysis verifies first-step CFG/Euler math and VAE stitching independently,
then saves `pixel4_result.png`, `comparison.png`, and numerical comparisons.
