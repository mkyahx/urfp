#!/usr/bin/env bash
set -euo pipefail

# Run from the script directory, no matter where this script is called.
cd "$(dirname "$0")"

# Change these two paths if your CUB images or desired mask output are elsewhere.
IMAGE_ROOT="${IMAGE_ROOT:-datasets/CUB/CUB_200_2011/images}"
OUTPUT_ROOT="${OUTPUT_ROOT:-datasets/CUB/CUB_200_2011_tokencut_masks}"

# TokenCut / DINO settings.
ARCH="${ARCH:-vit_small}"
PATCH_SIZE="${PATCH_SIZE:-16}"
WHICH_FEATURES="${WHICH_FEATURES:-k}"
TAU="${TAU:-0.2}"
EPS="${EPS:-1e-5}"
MAX_SIZE="${MAX_SIZE:-}"

# Extra args are forwarded to batch_cub_masks.py, e.g.:
#   bash run_cub_masks.sh --overwrite
#   bash run_cub_masks.sh --save-patch-mask
cmd=(
  python batch_cub_masks.py
  --image-root "$IMAGE_ROOT"
  --output-root "$OUTPUT_ROOT"
  --arch "$ARCH"
  --patch-size "$PATCH_SIZE"
  --which-features "$WHICH_FEATURES"
  --tau "$TAU"
  --eps "$EPS"
)

if [[ -n "$MAX_SIZE" ]]; then
  cmd+=(--max-size "$MAX_SIZE")
fi

"${cmd[@]}" "$@"
