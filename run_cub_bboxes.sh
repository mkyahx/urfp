#!/usr/bin/env bash
set -euo pipefail

# Run from the script directory, no matter where this script is called.
cd "$(dirname "$0")"

IMAGE_ROOT="${IMAGE_ROOT:-datasets/CUB/CUB_200_2011/images}"
OUTPUT_ROOT="${BBOX_OUTPUT_ROOT:-${OUTPUT_ROOT:-datasets/CUB/CUB_200_2011_tokencut_bboxes}}"

ARCH="${ARCH:-vit_small}"
PATCH_SIZE="${PATCH_SIZE:-16}"
WHICH_FEATURES="${WHICH_FEATURES:-k}"
TAU="${TAU:-0.2}"
EPS="${EPS:-1e-5}"

# Extra args are forwarded to batch_cub_bboxes.py, e.g.:
#   TOKENCUT_ROOT=/path/to/TokenCut bash run_cub_bboxes.sh --limit 20 --profile
#   ARCH=vit_base PATCH_SIZE=16 bash run_cub_bboxes.sh --overwrite
python batch_cub_bboxes.py \
  --image-root "$IMAGE_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --arch "$ARCH" \
  --patch-size "$PATCH_SIZE" \
  --which-features "$WHICH_FEATURES" \
  --tau "$TAU" \
  --eps "$EPS" \
  "$@"
