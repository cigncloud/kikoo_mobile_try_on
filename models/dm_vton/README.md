# DM-VTON PF student

Official source: https://github.com/KiseKloset/DM-VTON
Checkpoint download: https://github.com/KiseKloset/DM-VTON/releases/tag/1.0.0
Revision: `5dcd04ee3f5b1fb392df1184c887c1a7e910c72e`; checkpoints: release `1.0.0`.
`upstream/` is an ignored source copy. Official PF warp/gen checkpoints live
outside Git in `/tmp/`; download `dmvton_pf_warp.pt` and `dmvton_pf_gen.pt`
from release 1.0.0. Their SHA256 is recorded in the experiment environment.
Converted MNN/ONNX files are generated locally and excluded from Git.

The public model uses CuPy correlation. `scripts/run_dm_vton_sample.py` runs the
original CUDA path; `scripts/export_dm_vton_mnn.py` substitutes an exportable
ATen cost volume, then exports separate warp and generator stages.

Pixel 4 OpenCL FP32 at 192×256: about 1.70 seconds per image, 347MiB peak RSS,
1.90 seconds to first output when reusing kernel caches. Two FP32 MNN files
total 37.72MB; these are not quantized weights. This sample fails visual quality
checks, including in the original CUDA implementation.

See [the measured report](../../reports/dm_vton_pixel4.md) and
[experiment](../../experiments/2026-10-25-dm_vton/README.md).
The upstream code is CC BY-NC-SA 4.0 and marked for academic use; do not treat
benchmark success as commercial deployment approval.
