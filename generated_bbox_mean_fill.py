"""
Create mean-filled images from generated bbox .npy files.

For each image, pixels inside the generated bbox are kept unchanged and the
complement area is filled with a pure RGB background color. If --mean-color is
not provided, the color is computed as the global RGB mean over images that have
matching bbox .npy files.
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fill the complement of generated bboxes with global mean color."
    )
    parser.add_argument(
        "--image-root",
        required=True,
        type=Path,
        help="Root image folder, usually CUB_200_2011/images.",
    )
    parser.add_argument(
        "--bbox-root",
        required=True,
        type=Path,
        help="Root folder containing generated bbox .npy files.",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        type=Path,
        help="Root folder for mean-filled output images.",
    )
    parser.add_argument(
        "--bbox-format",
        default="xyxy",
        choices=["xyxy", "xywh"],
        help="Interpret bbox values as x1,y1,x2,y2 or x,y,w,h.",
    )
    parser.add_argument(
        "--mean-color",
        default=None,
        help="Optional RGB mean color such as '123,117,104'.",
    )
    parser.add_argument(
        "--output-extension",
        default=None,
        choices=[None, "jpg", "jpeg", "png"],
        help="Optional output extension. Defaults to each input image extension.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate output images that already exist.",
    )
    parser.add_argument(
        "--limit",
        default=None,
        type=int,
        help="Only process the first N matched images after sorting.",
    )
    return parser.parse_args()


def parse_rgb(value):
    parts = [int(part.strip()) for part in value.split(",")]
    if len(parts) != 3 or any(part < 0 or part > 255 for part in parts):
        raise ValueError(f"Expected RGB color like 123,117,104, got: {value}")
    return np.asarray(parts, dtype=np.uint8)


def iter_images(image_root):
    for path in sorted(image_root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def get_matched_pairs(image_root, bbox_root):
    pairs = []
    for image_path in iter_images(image_root):
        rel_path = image_path.relative_to(image_root)
        bbox_path = bbox_root / rel_path.with_suffix(".npy")
        if bbox_path.exists():
            pairs.append((image_path, bbox_path, rel_path))
    return pairs


def compute_global_mean(pairs):
    total = np.zeros(3, dtype=np.float64)
    count = 0

    for image_path, _, _ in tqdm(pairs, desc="Computing global mean"):
        with Image.open(image_path) as image:
            array = np.asarray(image.convert("RGB"), dtype=np.float64)
        total += array.reshape(-1, 3).sum(axis=0)
        count += array.shape[0] * array.shape[1]

    if count == 0:
        raise RuntimeError("No pixels found while computing global mean.")
    return np.rint(total / count).astype(np.uint8)


def load_bbox(bbox_path):
    bbox = np.asarray(np.load(bbox_path), dtype=np.float32).reshape(-1)
    if bbox.size != 4:
        raise ValueError(f"Expected bbox with 4 values in {bbox_path}, got {bbox.shape}")
    return bbox


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


def output_path_for(output_root, rel_path, output_extension):
    if output_extension is None:
        return output_root / rel_path
    return output_root / rel_path.with_suffix(f".{output_extension}")


def save_image(image, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() in {".jpg", ".jpeg"}:
        image.save(output_path, quality=95)
    else:
        image.save(output_path)


def main():
    args = parse_args()
    image_root = args.image_root.resolve()
    bbox_root = args.bbox_root.resolve()
    output_root = args.output_root.resolve()

    if not image_root.exists():
        raise FileNotFoundError(f"Image root does not exist: {image_root}")
    if not bbox_root.exists():
        raise FileNotFoundError(f"Bbox root does not exist: {bbox_root}")

    pairs = get_matched_pairs(image_root, bbox_root)
    if args.limit is not None:
        pairs = pairs[: args.limit]
    if not pairs:
        raise RuntimeError(f"No matching image/bbox pairs found under: {image_root}")

    if args.mean_color is None:
        mean_color = compute_global_mean(pairs)
    else:
        mean_color = parse_rgb(args.mean_color)
    print(f"Using mean color RGB: {mean_color.tolist()}")

    for image_path, bbox_path, rel_path in tqdm(pairs, desc="Writing mean-filled images"):
        output_path = output_path_for(output_root, rel_path, args.output_extension)

        if output_path.exists() and not args.overwrite:
            continue

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            array = np.asarray(image)

        left, top, right, bottom = bbox_to_xyxy(
            load_bbox(bbox_path),
            args.bbox_format,
            image.size,
        )

        filled = np.empty_like(array)
        filled[:, :] = mean_color
        filled[top:bottom, left:right] = array[top:bottom, left:right]
        save_image(Image.fromarray(filled, mode="RGB"), output_path)

    print(f"Saved mean-filled images to: {output_root}")


if __name__ == "__main__":
    main()
