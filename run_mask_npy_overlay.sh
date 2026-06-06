#!/usr/bin/env bash
set -euo pipefail

# Run from the script directory, no matter where this script is called.
cd "$(dirname "$0")"

# Edit these paths for your dataset.
IMAGE_ROOT="${IMAGE_ROOT:-datasets/CUB/CUB_200_2011/images}"
MASK_ROOT="${MASK_ROOT:-datasets/CUB/CUB_200_2011_tokencut_masks}"
OUTPUT_ROOT="${OUTPUT_ROOT:-datasets/CUB/CUB_200_2011_mask_overlays}"

# Visualization style.
OUTPUT_FORMAT="${OUTPUT_FORMAT:-jpg}"
ALPHA="${ALPHA:-0.45}"
MASK_COLOR="${MASK_COLOR:-255,64,64}"

python mask_npy_overlay.py \
  --image-root "$IMAGE_ROOT" \
  --mask-root "$MASK_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --output-format "$OUTPUT_FORMAT" \
  --alpha "$ALPHA" \
  --mask-color "$MASK_COLOR" \
  "$@"
