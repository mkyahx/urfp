"""
Convert 4-value bbox .npy files to H x W binary bbox-mask .npy files.

Input bbox npy format defaults to [xmin, ymin, xmax, ymax]. The output mirrors
the image directory tree and stores uint8 masks where bbox pixels are 1 and the
complement is 0.
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch-convert 4-value bbox npy files to 0/1 mask npy files."
    )
    parser.add_argument(
        "--image-root",
        required=True,
        type=Path,
        help="Root image folder used to determine output mask size.",
    )
    parser.add_argument(
        "--bbox-root",
        required=True,
        type=Path,
        help="Root folder containing 4-value bbox .npy files.",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        type=Path,
        help="Root folder for output bbox-mask .npy files.",
    )
    parser.add_argument(
        "--bbox-format",
        default="xyxy",
        choices=["xyxy", "xywh"],
        help="Interpret bbox values as x1,y1,x2,y2 or x,y,w,h.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate output masks that already exist.",
    )
    parser.add_argument(
        "--limit",
        default=None,
        type=int,
        help="Only process the first N matched images after sorting.",
    )
    return parser.parse_args()


def iter_images(image_root):
    for path in sorted(image_root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def bbox_to_xyxy(bbox, bbox_format, image_size):
    width, height = image_size
    bbox = np.asarray(bbox, dtype=np.float32).reshape(-1)
    if bbox.size != 4:
        raise ValueError(f"Expected bbox with 4 values, got shape {bbox.shape}")

    if bbox_format == "xywh":
        x1, y1, box_width, box_height = bbox.tolist()
        x2 = x1 + box_width
        y2 = y1 + box_height
    else:
        x1, y1, x2, y2 = bbox.tolist()

    left = int(np.floor(x1))
    top = int(np.floor(y1))
    right = int(np.ceil(x2))
    bottom = int(np.ceil(y2))

    left = max(0, min(width, left))
    right = max(0, min(width, right))
    top = max(0, min(height, top))
    bottom = max(0, min(height, bottom))

    if right < left:
        left, right = right, left
    if bottom < top:
        top, bottom = bottom, top

    return left, top, right, bottom


def bbox_to_mask(bbox, bbox_format, image_size):
    width, height = image_size
    left, top, right, bottom = bbox_to_xyxy(bbox, bbox_format, image_size)
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[top:bottom, left:right] = 1
    return mask


def main():
    args = parse_args()
    image_root = args.image_root.resolve()
    bbox_root = args.bbox_root.resolve()
    output_root = args.output_root.resolve()

    if not image_root.exists():
        raise FileNotFoundError(f"Image root does not exist: {image_root}")
    if not bbox_root.exists():
        raise FileNotFoundError(f"Bbox root does not exist: {bbox_root}")

    matched = []
    missing_bboxes = 0
    for image_path in iter_images(image_root):
        rel_path = image_path.relative_to(image_root)
        bbox_path = bbox_root / rel_path.with_suffix(".npy")
        if bbox_path.exists():
            matched.append((image_path, bbox_path, rel_path))
        else:
            missing_bboxes += 1

    if args.limit is not None:
        matched = matched[: args.limit]
    if not matched:
        raise RuntimeError(f"No matching image/bbox pairs found under: {image_root}")

    converted = 0
    for image_path, bbox_path, rel_path in tqdm(matched, desc="Converting bbox npy"):
        output_path = output_root / rel_path.with_suffix(".npy")
        if output_path.exists() and not args.overwrite:
            continue

        with Image.open(image_path) as image:
            image_size = image.size

        bbox = np.load(bbox_path)
        mask = bbox_to_mask(bbox, args.bbox_format, image_size)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, mask)
        converted += 1

    print(f"Saved bbox masks to: {output_root}")
    print(f"Converted {converted} files.")
    if missing_bboxes:
        print(f"Skipped {missing_bboxes} images without matching bbox npy files.")


if __name__ == "__main__":
    main()
