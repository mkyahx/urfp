#!/usr/bin/env bash
set -euo pipefail

# Run from the TokenCut repository root, no matter where this script is called.
cd "$(dirname "$0")"

IMAGE_ROOT="${IMAGE_ROOT:-datasets/CUB/CUB_200_2011/images}"
MASK_ROOT="${MASK_ROOT:-datasets/CUB/CUB_200_2011_tokencut_masks}"
BBOX_ROOT="${BBOX_ROOT:-datasets/CUB/CUB_200_2011_tokencut_bboxes}"
OUTPUT_ROOT="${OUTPUT_ROOT:-datasets/CUB/CUB_200_2011_tokencut_previews}"
FORMAT="${FORMAT:-jpg}"
ALPHA="${ALPHA:-0.45}"

# Extra args are forwarded to visualize_cub_npy.py, e.g.:
#   bash run_visualize_cub_npy.sh --limit 200 --overwrite
#   FORMAT=png bash run_visualize_cub_npy.sh
python visualize_cub_npy.py \
  --image-root "$IMAGE_ROOT" \
  --mask-root "$MASK_ROOT" \
  --bbox-root "$BBOX_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --format "$FORMAT" \
  --alpha "$ALPHA" \
  "$@"
