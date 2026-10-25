# 입어보기 · Android prototype

Offline photo try-on app for the Pixel 4. The launcher name is **입어보기** and
the package is `com.mobiletryon.demo`. Android 10+ / arm64 is required; the tested
device is Pixel 4, Android 13. This is a research prototype using DM-VTON PF
student, not a finished fitting product.

## Use on the phone

1. Open **입어보기** and tap **카메라로 촬영** or **사진 선택**.
2. Take a full-resolution photo in the system camera, or select an existing
   image with the Android document picker. Camera output goes to a private
   temporary JPEG through a non-exported provider with a one-URI grant.
   Cancelled captures are deleted. No camera, broad gallery or Internet
   permission is requested by this app.
3. Use a front-facing photo of one person with the face and shoulders visible.
   Hips improve alignment, but are optional for a close selfie. Pose Lite
   detects the upper body, whitens the background and aligns a 3:4 crop
   automatically. Photos with a covered face are rejected.
4. Choose **화이트 프린트 티**, **블랙 티셔츠** or **네이비 후드** and tap
   **이 옷 입어보기**.
5. Compare against the original photo, choose another garment, or save the PNG to
   `Pictures/MobileTryOn` using **사진에 저장**.

The first initialization compiles GPU kernels and takes about 47 seconds.
Later runs reuse private kernel caches. Model inference uses OpenCL FP32;
Pose Lite uses the MediaPipe CPU delegate for stable float segmentation masks on
Pixel 4. Alignment and background
restoration use Android bitmap operations. The network runs at 192×256, then
the result is placed back into the original photo (decoded at up to 1600 pixels
on its longest side). See [preprocessing validation](../reports/dm_vton_preprocessing.md)
for measured latency, memory and output images. General photo quality remains
an evaluation task; correct input handling does not guarantee a convincing fit.

## Build

Use JDK 17, Android SDK 35, NDK 27.1.12297006, Gradle 8.11.1 and AGP 8.6.1.
The Gradle project uses platform Java UI and MediaPipe Tasks Vision 0.10.14.

```bash
# From the repository root, after downloading checkpoints and preparing exports:
bash scripts/build_android_app.sh
adb install -r android/app/build/outputs/apk/debug/app-debug.apk
adb shell am start -n com.mobiletryon.demo/.MainActivity
```

APK files, native `.so` libraries, converted `.mnn` models, and the Pose Lite
`.task` file are excluded from Git. Download the original inputs and regenerate
them locally. DM-VTON PF checkpoints come from
<https://github.com/KiseKloset/DM-VTON/releases/tag/1.0.0>; run
`scripts/export_dm_vton_mnn.py` to create the MNN stages. Download the Pose Lite
task from <https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task>
to `weights/mediapipe/pose_landmarker_lite.task`. The root
[`README.md`](../README.md) lists the complete provenance and VITON-Clean link.

`prepare_android_assets.py` copies the locally generated FP32 `.mnn` exports from
`experiments/2026-10-25-dm_vton/`, prepares three official VITON-Clean garments
with their supplied foreground annotations, and verifies the Pose Lite model
checksum. Set `TRYON_VITON_DATASET` to the extracted `VITON_test` directory;
the pose task defaults to `weights/mediapipe/pose_landmarker_lite.task`.
Garment masks are prepared once, preserving white fabric and printed logos.
Large artifacts and
private test fixtures are ignored. Models are verified before extraction into
the app's private no-backup directory. Cache directory names include both
model hashes and the runtime/precision version.

`build_android_native.sh` links JNI against the validated, Winograd-patched
MNN 3.6.1 arm64 runtime. Set `TRYON_NDK` / `TRYON_MNN_LIB` to override the local
defaults. Public MNN headers and its license are in `vendor/`; the library is
stripped before packaging. See the preceding experiment for rebuilding MNN.

Configure `local.properties` with `sdk.dir`. Optional ignored `local-build.env`
can export `GRADLE_USER_HOME` and `TRYON_GRADLE_BIN` for an existing offline
installation. The debug signing key is generated locally and is ignored.

## Validate

```bash
cd android
bash ./gradlew assembleDebug assembleDebugAndroidTest lintDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk
adb install -r app/build/outputs/apk/androidTest/debug/app-debug-androidTest.apk
adb shell am instrument -w com.mobiletryon.demo.test/com.mobiletryon.demo.TryOnInstrumentation
adb shell run-as com.mobiletryon.demo cat files/instrumentation.json
```

Instrumentation requires real OpenCL FP32 sessions, compares raw output to the
saved PyTorch FP32 reference, verifies refreshed-input repeatability, rejects
bad dimensions and runs all three catalog garments. It also compares official
VITON-Clean tensors with original CUDA output, checks automatic alignment and
background restoration, and rejects a no-person photo. It never silently accepts
a CPU fallback. Pixel devices need the optional public `libOpenCL-pixel.so`
declaration in both application and instrumentation manifests.
It also verifies full-size camera-URI write/decode, capture cleanup and path
traversal rejection. System camera launch/return is checked separately on-device.

## Asset and runtime provenance

- DM-VTON official revision `5dcd04ee3f5b1fb392df1184c887c1a7e910c72e`, PF release
  `1.0.0`, CC BY-NC-SA 4.0 / academic use. Its license is packaged in assets.
- MNN 3.6.1 revision `d407447ed56c4121a11ccbd266dc184ca1ead0c2`, Apache 2.0.
- MediaPipe Pose Landmarker Lite float16 v1; model URL and SHA256 are recorded
  in `preprocess/model.json`. Its landmarks and segmentation replace the web
  demo's larger YOLOv7 and selfie models; its segmentation runs on CPU on
  Pixel 4 because the GPU delegate returned corrupted float masks.
- Garments and masks: official VITON-Clean `test_color` / `test_edge`, files
  `015095_1.jpg`, `019243_1.jpg`, `004727_1.jpg`. Names and annotation hashes
  are recorded in `catalog.json`. These are research fixtures; their commercial
  redistribution rights have not been established.
- No private person image is bundled in the main APK. Numerical fixtures live
  only in the separate, ignored instrumentation assets/test APK.
