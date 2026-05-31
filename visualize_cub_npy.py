"""
Visualize generated CUB TokenCut .npy masks and optional bbox .npy files.

The output mirrors the image directory tree and writes .jpg previews. Each
preview overlays the mask on the original image and draws the bbox if available.
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create jpg/png previews from TokenCut mask and bbox npy files."
    )
    parser.add_argument(
        "--image-root",
        required=True,
        type=Path,
        help="Root image folder, usually CUB_200_2011/images.",
    )
    parser.add_argument(
        "--mask-root",
        required=True,
        type=Path,
        help="Root folder containing mask .npy files.",
    )
    parser.add_argument(
        "--bbox-root",
        default=None,
        type=Path,
        help="Optional root folder containing bbox .npy files.",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        type=Path,
        help="Root folder for visualization images.",
    )
    parser.add_argument(
        "--format",
        default="jpg",
        choices=["jpg", "png"],
        help="Output preview image format.",
    )
    parser.add_argument(
        "--alpha",
        default=0.45,
        type=float,
        help="Mask overlay opacity in [0, 1].",
    )
    parser.add_argument(
        "--limit",
        default=None,
        type=int,
        help="Only visualize the first N images after sorting.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate previews that already exist.",
    )
    return parser.parse_args()


def iter_images(image_root):
    for path in sorted(image_root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def normalize_mask(mask, image_size):
    mask = np.asarray(mask)
    if mask.ndim > 2:
        mask = np.squeeze(mask)
    if mask.ndim != 2:
        raise ValueError(f"Expected a 2D mask, got shape {mask.shape}")

    mask = mask > 0
    width, height = image_size
    if mask.shape != (height, width):
        mask_image = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
        mask_image = mask_image.resize((width, height), Image.Resampling.NEAREST)
        mask = np.asarray(mask_image) > 0
    return mask


def overlay_mask(image, mask, alpha):
    image_arr = np.asarray(image.convert("RGB")).astype(np.float32)
    color = np.array([255, 64, 64], dtype=np.float32)
    blended = image_arr.copy()
    blended[mask] = image_arr[mask] * (1.0 - alpha) + color * alpha
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8), mode="RGB")


def draw_bbox(image, bbox):
    bbox = np.asarray(bbox, dtype=np.float32).reshape(-1)
    if bbox.size != 4:
        raise ValueError(f"Expected bbox with 4 values, got shape {bbox.shape}")

    draw = ImageDraw.Draw(image)
    x1, y1, x2, y2 = bbox.tolist()
    width, height = image.size
    x1 = max(0, min(width - 1, x1))
    x2 = max(0, min(width - 1, x2))
    y1 = max(0, min(height - 1, y1))
    y2 = max(0, min(height - 1, y2))

    for offset in range(3):
        draw.rectangle(
            [x1 - offset, y1 - offset, x2 + offset, y2 + offset],
            outline=(64, 255, 128),
        )


def main():
    args = parse_args()
    image_root = args.image_root.resolve()
    mask_root = args.mask_root.resolve()
    bbox_root = args.bbox_root.resolve() if args.bbox_root is not None else None
    output_root = args.output_root.resolve()

    if not image_root.exists():
        raise FileNotFoundError(f"Image root does not exist: {image_root}")
    if not mask_root.exists():
        raise FileNotFoundError(f"Mask root does not exist: {mask_root}")
    if bbox_root is not None and not bbox_root.exists():
        raise FileNotFoundError(f"Bbox root does not exist: {bbox_root}")

    image_paths = list(iter_images(image_root))
    if args.limit is not None:
        image_paths = image_paths[: args.limit]
    if not image_paths:
        raise RuntimeError(f"No images found under: {image_root}")

    output_root.mkdir(parents=True, exist_ok=True)
    missing_masks = 0
    missing_bboxes = 0

    for image_path in tqdm(image_paths, desc="Writing previews"):
        rel_path = image_path.relative_to(image_root)
        mask_path = mask_root / rel_path.with_suffix(".npy")
        bbox_path = bbox_root / rel_path.with_suffix(".npy") if bbox_root else None
        output_path = output_root / rel_path.with_suffix(f".{args.format}")

        if output_path.exists() and not args.overwrite:
            continue
        if not mask_path.exists():
            missing_masks += 1
            continue

        image = Image.open(image_path).convert("RGB")
        mask = normalize_mask(np.load(mask_path), image.size)
        preview = overlay_mask(image, mask, args.alpha)

        if bbox_path is not None:
            if bbox_path.exists():
                draw_bbox(preview, np.load(bbox_path))
            else:
                missing_bboxes += 1

        output_path.parent.mkdir(parents=True, exist_ok=True)
        preview.save(output_path, quality=95)

    print(f"Saved previews to: {output_root}")
    if missing_masks:
        print(f"Skipped {missing_masks} images with missing masks.")
    if missing_bboxes:
        print(f"Rendered {missing_bboxes} previews without bbox overlays.")


if __name__ == "__main__":
    main()
