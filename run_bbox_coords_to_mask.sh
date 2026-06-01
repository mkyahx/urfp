#!/usr/bin/env bash
set -euo pipefail

# Run from the script directory, no matter where this script is called.
cd "$(dirname "$0")"

# Edit these paths for your dataset.
IMAGE_ROOT="${IMAGE_ROOT:-datasets/CUB/CUB_200_2011/images}"
BBOX_ROOT="${BBOX_ROOT:-datasets/CUB/CUB_200_2011_tokencut_bboxes}"
OUTPUT_ROOT="${OUTPUT_ROOT:-datasets/CUB/CUB_200_2011_tokencut_bbox_masks}"

# TokenCut bbox npy is [xmin, ymin, xmax, ymax]. Use xywh only for x,y,width,height files.
BBOX_FORMAT="${BBOX_FORMAT:-xyxy}"

python bbox_coords_to_mask.py \
  --image-root "$IMAGE_ROOT" \
  --bbox-root "$BBOX_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --bbox-format "$BBOX_FORMAT" \
  "$@"
