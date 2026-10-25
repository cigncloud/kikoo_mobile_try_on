# Selfie PC smoke test

Input: `/home/michaelk/Downloads/Screenshot 2026-10-19 at 00.17.54.png`
(357×428). The photo is a front-facing portrait with the face and shoulders
visible and the hips outside the frame. It was cropped to `(63,94)-(294,403)`
to approximate the app's eye/shoulder selfie fallback and resized to 192×256.

The official DM-VTON PF student ran on an NVIDIA GeForce RTX 3080 Laptop GPU.
Black tee and navy hoodie outputs are in `pc_comparison.jpg`; timing and model
details are in `metrics.json`. Each run used a near-white-background garment
mask and is a functional smoke test, not a quality or FID/LPIPS evaluation.

The fallback crop keeps the face visible, but the model still produces large
garment shapes and neck/torso artifacts on this out-of-distribution portrait.
The result confirms that this photo can pass pose/crop preprocessing; it does
not establish product-quality fitting for arbitrary selfies.
