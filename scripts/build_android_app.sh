#!/usr/bin/env bash
set -euo pipefail
tryon_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
"$tryon_root/.venv/bin/python" "$tryon_root/scripts/prepare_android_assets.py"
bash "$tryon_root/scripts/build_android_native.sh"
if [[ ! -f "$tryon_root/android/debug.keystore" ]]; then
    keytool -genkeypair -keystore "$tryon_root/android/debug.keystore" \
        -storepass android -keypass android -alias androiddebugkey \
        -dname 'CN=Mobile TryOn Debug,O=Local Research,C=US' \
        -keyalg RSA -keysize 2048 -validity 10000
fi
cd "$tryon_root/android"
if [[ $# -eq 0 ]]; then
    bash ./gradlew --no-daemon assembleDebug
else
    bash ./gradlew "$@"
fi
mkdir -p artifacts
cp app/build/outputs/apk/debug/app-debug.apk artifacts/mobile-tryon-debug.apk
