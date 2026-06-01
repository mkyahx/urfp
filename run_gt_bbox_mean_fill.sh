#!/usr/bin/env bash
set -euo pipefail

# Run from the script directory, no matter where this script is called.
cd "$(dirname "$0")"

# Edit these paths for your dataset.
IMAGE_ROOT="${IMAGE_ROOT:-datasets/CUB/CUB_200_2011/images}"
IMAGES_TXT="${IMAGES_TXT:-datasets/CUB/CUB_200_2011/images.txt}"
BBOX_TXT="${BBOX_TXT:-datasets/CUB/CUB_200_2011/bounding_boxes.txt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-datasets/CUB/CUB_200_2011_gt_bbox_mean_fill}"

# CUB bounding_boxes.txt is x,y,width,height. Use xyxy if your txt is x1,y1,x2,y2.
BBOX_FORMAT="${BBOX_FORMAT:-xywh}"

# Leave empty to compute dataset global RGB mean from images.txt.
# Set to "123,117,104" to skip mean computation and use a fixed color.
MEAN_COLOR="${MEAN_COLOR:-}"

# Leave empty to keep each input image extension, or set jpg/png.
OUTPUT_EXTENSION="${OUTPUT_EXTENSION:-}"

cmd=(
  python gt_bbox_mean_fill.py
  --image-root "$IMAGE_ROOT"
  --images-txt "$IMAGES_TXT"
  --bbox-txt "$BBOX_TXT"
  --output-root "$OUTPUT_ROOT"
  --bbox-format "$BBOX_FORMAT"
)

if [[ -n "$MEAN_COLOR" ]]; then
  cmd+=(--mean-color "$MEAN_COLOR")
fi

if [[ -n "$OUTPUT_EXTENSION" ]]; then
  cmd+=(--output-extension "$OUTPUT_EXTENSION")
fi

"${cmd[@]}" "$@"
