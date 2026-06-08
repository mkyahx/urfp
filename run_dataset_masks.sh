#!/usr/bin/env bash
set -euo pipefail

# Run from the script directory, no matter where this script is called.
cd "$(dirname "$0")"

# Generic image-dataset entrypoint.
# Examples:
#   TOKENCUT_ROOT=/path/to/TokenCut DATASET_NAME=stanford_cars IMAGE_ROOT=/data/stanford_cars/cars_train bash run_dataset_masks.sh
#   TOKENCUT_ROOT=/path/to/TokenCut DATASET_NAME=aircraft IMAGE_ROOT=/data/fgvc-aircraft/images bash run_dataset_masks.sh
DATASET_NAME="${DATASET_NAME:-dataset}"
IMAGE_ROOT="${IMAGE_ROOT:-datasets/${DATASET_NAME}/images}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/${DATASET_NAME}_tokencut_masks}"

# TokenCut / DINO settings.
ARCH="${ARCH:-vit_small}"
PATCH_SIZE="${PATCH_SIZE:-16}"
WHICH_FEATURES="${WHICH_FEATURES:-k}"
TAU="${TAU:-0.2}"
EPS="${EPS:-1e-5}"

python batch_cub_masks.py \
  --image-root "$IMAGE_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --arch "$ARCH" \
  --patch-size "$PATCH_SIZE" \
  --which-features "$WHICH_FEATURES" \
  --tau "$TAU" \
  --eps "$EPS" \
  "$@"
