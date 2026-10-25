# Structural INT4/FP16 trial

This policy is a structural heuristic, not calibrated sensitivity analysis. It
preserves spatial convolutions, tensors with at most 65,536 stored weights,
time embeddings, and self-attention keys. Other weights use asymmetric INT4
with block size 32. All counts come from serialized weights, including groups.

`tryon_structural_int4_fp16_block32_summary.json` records each layer and reason.
The serialized result has 97 FP16 and 341 INT4 convolution weight tensors;
`../tryon_structural_int4_fp16_block32_weights.tsv` is the storage audit.

## Rebuild

Run from the repository root with the pinned MNN 3.6.1 checkout and converter.
Large models and the converter guide are local, ignored artifacts in `exports/`.
The guide was produced using zero inputs to enumerate layers only: **the zero
fixtures and any auto-quant `rate` are not calibration or accuracy evidence**.
Evaluation uses the original seeded FP32 fixtures, separately.

```bash
g++ -O2 -std=c++17 experiments/2026-10-24-pixel4/mnn_weight_inventory.cpp \
  -I/tmp/mobile_tryon_mnn_3.6.1/schema/current \
  -I/tmp/mobile_tryon_mnn_3.6.1/3rd_party/flatbuffers/include \
  -o /tmp/mnn_weight_inventory

root=experiments/2026-10-24-pixel4
/tmp/mnn_weight_inventory "$root/exports/tryon_float.mnn" > "$root/tryon_float_weights.tsv"
python scripts/build_mnn_hybrid_policy.py \
  --guide "$root/exports/tryon_auto_int8.mnn.json" \
  --inventory "$root/tryon_float_weights.tsv" \
  --output "$root/policies/tryon_structural_int4_fp16_block32.json" \
  --summary "$root/policies/tryon_structural_int4_fp16_block32_summary.json"

/tmp/mobile_tryon_mnn_host/MNNConvert -f MNN \
  --modelFile "$root/exports/tryon_float.mnn" \
  --MNNModel "$root/exports/tryon_structural_int4_fp32_block32.mnn" \
  --bizCode pixel4_tryon_structural --weightQuantBits 4 \
  --compressionParamsFile "$root/policies/tryon_structural_int4_fp16_block32.json"
/tmp/mobile_tryon_mnn_host/MNNConvert -f MNN \
  --modelFile "$root/exports/tryon_structural_int4_fp32_block32.mnn" \
  --MNNModel "$root/exports/tryon_structural_int4_fp16_block32.mnn" \
  --bizCode pixel4_tryon_structural --fp16
/tmp/mnn_weight_inventory "$root/exports/tryon_structural_int4_fp16_block32.mnn" \
  > "$root/tryon_structural_int4_fp16_block32_weights.tsv"
```

MNN clamps `bits=16` to 8. Here `bits=0` excludes a weight from quantization.
The second conversion packs these excluded weights and constants into FP16
while preserving already-quantized INT4 weights. Applying `--fp16` to the float
source before quantization would skip the intended quantization path.

## Device execution

Use `manifests/tryon_structural_int4_fp16_block32_opencl.json` (Low) or
`manifests/tryon_structural_int4_fp16_block32_opencl_normal.json` (Normal).
They share identical model and inputs, with distinct output paths. Low requests
FP16 storage/compute; Normal requests FP16 storage/FP32 compute in OpenCL kernels.
These are runtime choices, separate from serialized weight precision.
See the parent README for push/pull, monitoring, and measurement scope.
