#!/usr/bin/env bash
set -euo pipefail

# Run from the script directory, no matter where this script is called.
cd "$(dirname "$0")"

# Edit these paths for your dataset.
IMAGE_ROOT="${IMAGE_ROOT:-datasets/CUB/CUB_200_2011/images}"
BBOX_ROOT="${BBOX_ROOT:-datasets/CUB/CUB_200_2011_tokencut_bboxes}"
OUTPUT_ROOT="${OUTPUT_ROOT:-datasets/CUB/CUB_200_2011_generated_bbox_mean_fill}"

# Generated TokenCut bbox npy is x1,y1,x2,y2. Use xywh only for x,y,width,height files.
BBOX_FORMAT="${BBOX_FORMAT:-xyxy}"

# Leave empty to compute dataset global RGB mean from matched images.
# Set to "123,117,104" to skip mean computation and use a fixed color.
MEAN_COLOR="${MEAN_COLOR:-}"

# Leave empty to keep each input image extension, or set jpg/png.
OUTPUT_EXTENSION="${OUTPUT_EXTENSION:-}"

cmd=(
  python generated_bbox_mean_fill.py
  --image-root "$IMAGE_ROOT"
  --bbox-root "$BBOX_ROOT"
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
