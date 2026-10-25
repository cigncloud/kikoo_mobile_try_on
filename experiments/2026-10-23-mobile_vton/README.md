# Mobile-VTON first inference test

Latest fixed-garment results: the [garment cache and GPU residency report](../../reports/mobile_vton_garment_cache.md)
compares all models kept on CUDA with and without precomputed garment VAE/DINO
conditioning. Cached generation uses 1.071GiB peak allocated, 1.199GiB peak
reserved and averages 42.389s over three 768×1024/28-step runs on the shared GPU.
Outputs match the previous result exactly. Use
`config_garment_resident_cached.json` after building `config_garment_precompute.json`;
the new cache file is about 1.05MB. The previous 1.052GiB configuration below
releases phase models, so its residency policy differs.

Status: **passed on the desktop GPU; phone inference not tested**.

`config.json` pins inputs, source, weights, and generation settings. The complete
initial findings are in [the report](../../reports/mobile_vton_initial.md).

For weight-only INT4, see [the INT4 guide](INT4.md) and
[comparison report](../../reports/mobile_vton_int4.md).

The memory reduction experiment is documented in the
[memory report](../../reports/mobile_vton_memory.md). It verified that loading
the VAE/image encoder/decoder only for their phases reduced desktop CUDA peak
from 2.229GiB to 1.906GiB with an identical output hash. Sequential CFG did not
reduce peak memory and remains opt-in.

The [VAE experiment](../../reports/mobile_vton_vae.md) compares encoder-only
loading and 512px tiled encoding. The encoder-only output is identical, and
tiling lowers the encoding phase peak to 1.062GiB, but the final decoder keeps
the overall peak near 1.899GiB. Configs are `config_vae_*.json`; outputs and
matched comparisons are in `outputs_vae_*/`.

The [final decoder experiment](../../reports/mobile_vton_decoder.md) verifies
FP16 against identical saved latents. Combined with VAE tiling and loading the
packed CUDA states before transferring original weights, it reduces peak CUDA
allocation including startup to 1.428GiB. Use
`config_decoder_fp16_vae_tiled512_lowload.json` for that experimental setting;
VAE tiling still changes image detail. FP16 without VAE tiling is recorded in
`config_decoder_fp16.json`.

The [tensor lifetime experiment](../../reports/mobile_vton_lifetimes.md) further
reduces peak CUDA allocation to 1.263GiB with identical output PNGs on this pair.
Use `config_lifetimes_release_expandable.json` with
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`; the container wrapper forwards
this variable. Peak reserved memory during generation falls to 1.469GiB.
`config_lifetimes_release.json` uses the default allocator and reaches 1.270GiB
allocated. These are desktop CUDA measurements; phone RAM remains untested.

The [compute memory experiment](../../reports/mobile_vton_compute.md) profiles
the remaining highs and compares FFN chunking, native GQA and local convolution
tiling. FFN/GQA keep identical images but do not lower the overall high. Combining
32-row convolution tiles with sequential CFG reaches 1.052GiB allocated; unlike
whole-decoder tiling, GroupNorm still sees full images. See the report for output
differences and the selected config. All `config_compute_*.json` variants keep
768×1024 output and 28 denoising steps.

Native-resolution GPU BF16 and INT4 outputs are in `outputs_vllm_*_native/`;
the full-size comparison is `outputs_vllm_comparison_native/bf16_vs_int4_native.png`.
The pinned vLLM image provides CUDA for our PyTorch runner. The vLLM inference
server is not started. See `INT4.md` for the offline container commands.

- `outputs/metrics.json`: actual package versions, tensor counts, timing, memory, hashes, and limits.
- `outputs/result_01.png`, `outputs/result_02.png`: repeated 768×1024 generations.
- `outputs/comparison_01.jpg`: person, garment, and generated result side by side.
- `huggingface_manifest.json`, `checkpoint_audit.json`: revision-pinned official file metadata and hash checks.
- `sources.json`: source URLs and fixture rights notes.
- `paper_claims.json`: author-reported mobile numbers, separated from our measurements.
- `phone_preflight.txt`: read-only inspection of the connected Pixel 4.

Inputs and raster outputs stay local and are excluded from Git. See
[the model guide](../../models/mobile_vton/README.md) to download inputs/weights
and rerun. No private user photograph, paid API, or external upload was used.
