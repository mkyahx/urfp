#!/usr/bin/env bash
set -euo pipefail

# Run from the script directory, no matter where this script is called.
cd "$(dirname "$0")"

# Edit these paths for your dataset.
IMAGE_ROOT="${IMAGE_ROOT:-datasets/CUB/CUB_200_2011/images}"
IMAGES_TXT="${IMAGES_TXT:-datasets/CUB/CUB_200_2011/images.txt}"
BBOX_TXT="${BBOX_TXT:-datasets/CUB/CUB_200_2011/bounding_boxes.txt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-datasets/CUB/CUB_200_2011_gt_bbox_masks}"

# CUB bounding_boxes.txt is x,y,width,height. Use xyxy for x1,y1,x2,y2 files.
BBOX_FORMAT="${BBOX_FORMAT:-xywh}"

python gt_bbox_txt_to_mask.py \
  --image-root "$IMAGE_ROOT" \
  --images-txt "$IMAGES_TXT" \
  --bbox-txt "$BBOX_TXT" \
  --output-root "$OUTPUT_ROOT" \
  --bbox-format "$BBOX_FORMAT" \
  "$@"
