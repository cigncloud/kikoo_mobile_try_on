#!/usr/bin/env bash
set -euo pipefail
tryon_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
tryon_ndk=${TRYON_NDK:-${ANDROID_NDK:-/home/michaelk/Android/Sdk/ndk/27.1.12297006}}
tryon_toolchain="$tryon_ndk/toolchains/llvm/prebuilt/linux-x86_64/bin"
tryon_libs="$tryon_root/android/app/src/main/jniLibs/arm64-v8a"
mkdir -p "$tryon_libs"
if [[ ! -f "$tryon_libs/libMNN.so" ]]; then
    tryon_mnn_lib=${TRYON_MNN_LIB:-/tmp/mobile_tryon_mnn_android/OFF/arm64-v8a/libMNN.so}
    cp "$tryon_mnn_lib" "$tryon_libs/libMNN.so"
    "$tryon_toolchain/llvm-strip" --strip-unneeded "$tryon_libs/libMNN.so"
fi
"$tryon_toolchain/aarch64-linux-android29-clang++" -std=c++17 -O2 -shared -fPIC \
    -static-libstdc++ -Wl,-z,max-page-size=16384 -Wl,--no-undefined \
    "$tryon_root/android/app/src/main/cpp/tryon_jni.cpp" \
    -I"$tryon_root/android/vendor/mnn" -L"$tryon_libs" -lMNN -llog \
    -o "$tryon_libs/libtryon_jni.so"
"$tryon_toolchain/llvm-strip" --strip-unneeded "$tryon_libs/libtryon_jni.so"
