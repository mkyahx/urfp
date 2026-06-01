"""
Create GT-bbox mean-filled images.

For each image, pixels inside the ground-truth bbox are kept unchanged and the
complement area is filled with a pure RGB background color. If --mean-color is
not provided, the color is computed as the global RGB mean over images.txt.
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fill the complement of GT bboxes with global mean color."
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
        help="Root folder for mean-filled output images.",
    )
    parser.add_argument(
        "--bbox-format",
        default="xywh",
        choices=["xywh", "xyxy"],
        help="Interpret bbox values as x,y,w,h or x1,y1,x2,y2.",
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
        help="Only process the first N images after sorting by index.",
    )
    return parser.parse_args()


def parse_rgb(value):
    parts = [int(part.strip()) for part in value.split(",")]
    if len(parts) != 3 or any(part < 0 or part > 255 for part in parts):
        raise ValueError(f"Expected RGB color like 123,117,104, got: {value}")
    return np.asarray(parts, dtype=np.uint8)


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
            image_id = int(parts[0])
            mapping[image_id] = Path(parts[1])
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
            mapping[image_id] = np.asarray([float(value) for value in parts[1:]], dtype=np.float32)
    return mapping


def compute_global_mean(image_root, image_paths):
    total = np.zeros(3, dtype=np.float64)
    count = 0

    for rel_path in tqdm(image_paths, desc="Computing global mean"):
        image_path = image_root / rel_path
        with Image.open(image_path) as image:
            array = np.asarray(image.convert("RGB"), dtype=np.float64)
        total += array.reshape(-1, 3).sum(axis=0)
        count += array.shape[0] * array.shape[1]

    if count == 0:
        raise RuntimeError("No pixels found while computing global mean.")
    return np.rint(total / count).astype(np.uint8)


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

    rel_paths = [image_mapping[image_id] for image_id in image_ids]
    if args.mean_color is None:
        mean_color = compute_global_mean(image_root, rel_paths)
    else:
        mean_color = parse_rgb(args.mean_color)
    print(f"Using mean color RGB: {mean_color.tolist()}")

    for image_id in tqdm(image_ids, desc="Writing mean-filled images"):
        rel_path = image_mapping[image_id]
        image_path = image_root / rel_path
        output_path = output_path_for(output_root, rel_path, args.output_extension)

        if output_path.exists() and not args.overwrite:
            continue

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            array = np.asarray(image)

        left, top, right, bottom = bbox_to_xyxy(
            bbox_mapping[image_id],
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
