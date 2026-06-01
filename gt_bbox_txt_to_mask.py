"""
Convert ground-truth bbox txt annotations to H x W binary bbox-mask .npy files.

The output mirrors images.txt paths and stores uint8 masks where bbox pixels are
1 and the complement is 0. The bbox txt format is:

    <index> <x> <y> <w> <h>

by default. Use --bbox-format xyxy for <index> <x1> <y1> <x2> <y2>.
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert GT bbox txt annotations to 0/1 bbox-mask npy files."
    )
    parser.add_argument(
        "--image-root",
        required=True,
        type=Path,
        help="Root image folder, usually CUB_200_2011/images.",
    )
    parser.add_argument(
        "--images-txt",
        required=True,
        type=Path,
        help="Mapping txt: each line is '<index> <relative_image_path>'.",
    )
    parser.add_argument(
        "--bbox-txt",
        required=True,
        type=Path,
        help="BBox txt: each line is '<index> <x> <y> <w> <h>' by default.",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        type=Path,
        help="Root folder for output bbox-mask .npy files.",
    )
    parser.add_argument(
        "--bbox-format",
        default="xywh",
        choices=["xywh", "xyxy"],
        help="Interpret bbox values as x,y,w,h or x1,y1,x2,y2.",
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
        help="Only process the first N images after sorting by index.",
    )
    return parser.parse_args()


def read_images_txt(path):
    mapping = {}
    with open(path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                raise ValueError(f"Bad images.txt line {line_number}: {line}")
            mapping[int(parts[0])] = Path(parts[1])
    return mapping


def read_bbox_txt(path):
    mapping = {}
    with open(path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                raise ValueError(f"Bad bbox txt line {line_number}: {line}")
            image_id = int(parts[0])
            mapping[image_id] = np.asarray(
                [float(value) for value in parts[1:]],
                dtype=np.float32,
            )
    return mapping


def bbox_to_xyxy(bbox, bbox_format, image_size):
    width, height = image_size
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
    images_txt = args.images_txt.resolve()
    bbox_txt = args.bbox_txt.resolve()
    output_root = args.output_root.resolve()

    if not image_root.exists():
        raise FileNotFoundError(f"Image root does not exist: {image_root}")
    if not images_txt.exists():
        raise FileNotFoundError(f"images.txt does not exist: {images_txt}")
    if not bbox_txt.exists():
        raise FileNotFoundError(f"bbox txt does not exist: {bbox_txt}")

    image_mapping = read_images_txt(images_txt)
    bbox_mapping = read_bbox_txt(bbox_txt)
    image_ids = sorted(image_id for image_id in image_mapping if image_id in bbox_mapping)
    if args.limit is not None:
        image_ids = image_ids[: args.limit]
    if not image_ids:
        raise RuntimeError("No matching image ids found between images.txt and bbox txt.")

    converted = 0
    for image_id in tqdm(image_ids, desc="Writing GT bbox masks"):
        rel_path = image_mapping[image_id]
        image_path = image_root / rel_path
        output_path = output_root / rel_path.with_suffix(".npy")

        if output_path.exists() and not args.overwrite:
            continue

        with Image.open(image_path) as image:
            image_size = image.size

        mask = bbox_to_mask(bbox_mapping[image_id], args.bbox_format, image_size)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, mask)
        converted += 1

    print(f"Saved GT bbox masks to: {output_root}")
    print(f"Converted {converted} files.")


if __name__ == "__main__":
    main()
