#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_ENV="${MOBILE_VTON_PYTHON_ENV:-/home/michaelk/miniconda3/envs/comfyui}"
CACHE_ROOT="${MOBILE_VTON_CACHE_ROOT:-/tmp/mobile_tryon}"
IMAGE="vllm/vllm-openai@sha256:04563c302537a91aa49ebdfbceda96111c5712275999b7e8804fa598f0b5641d"

exec docker run --rm --network none --gpus all \
  --env PYTORCH_CUDA_ALLOC_CONF \
  --user "$(id -u):$(id -g)" \
  --workdir "$ROOT" \
  --volume "$ROOT:$ROOT:rw" \
  --volume "$PYTHON_ENV:$PYTHON_ENV:ro" \
  --volume "$CACHE_ROOT:$CACHE_ROOT:rw" \
  --entrypoint "$ROOT/.venv/bin/python" \
  "$IMAGE" -u "$ROOT/scripts/run_mobile_vton.py" "$@"
