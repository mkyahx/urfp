#!/usr/bin/env bash
set -euo pipefail

# Run from the script directory, no matter where this script is called.
cd "$(dirname "$0")"

# Pass single-image visualization args directly, for example:
#   bash run_npy_to_image.sh --npy-path mask.npy --image-path image.jpg --output-path preview.jpg
#   bash run_npy_to_image.sh --npy-path bbox.npy --image-path image.jpg --output-path bbox_preview.jpg
python npy_to_image.py "$@"
