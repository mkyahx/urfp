#!/usr/bin/env bash
set -euo pipefail

# Run from the script directory, no matter where this script is called.
cd "$(dirname "$0")"

# Edit these paths for the single file you want to inspect.
NPY_PATH="${NPY_PATH:-datasets/CUB/CUB_200_2011_tokencut_masks/001.Black_footed_Albatross/Black_Footed_Albatross_0001_796111.npy}"
IMAGE_PATH="${IMAGE_PATH:-datasets/CUB/CUB_200_2011/images/001.Black_footed_Albatross/Black_Footed_Albatross_0001_796111.jpg}"
OUTPUT_PATH="${OUTPUT_PATH:-datasets/CUB/CUB_200_2011_tokencut_previews/001.Black_footed_Albatross/Black_Footed_Albatross_0001_796111.jpg}"

# Optional: set this to draw a bbox on top of a mask preview.
BBOX_NPY="${BBOX_NPY:-}"

# Visualization style.
ALPHA="${ALPHA:-0.45}"
MASK_COLOR="${MASK_COLOR:-255,64,64}"
BBOX_COLOR="${BBOX_COLOR:-64,255,128}"
LINE_WIDTH="${LINE_WIDTH:-3}"

cmd=(
  python npy_to_image.py
  --npy-path "$NPY_PATH"
  --image-path "$IMAGE_PATH"
  --output-path "$OUTPUT_PATH"
  --alpha "$ALPHA"
  --mask-color "$MASK_COLOR"
  --bbox-color "$BBOX_COLOR"
  --line-width "$LINE_WIDTH"
)

if [[ -n "$BBOX_NPY" ]]; then
  cmd+=(--bbox-npy "$BBOX_NPY")
fi

"${cmd[@]}" "$@"
