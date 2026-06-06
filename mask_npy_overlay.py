"""
Batch-visualize 0/1 mask .npy files as red translucent overlays on images.

The output mirrors the image directory tree. For each image, the script looks
for a mask npy with the same relative path and a .npy suffix.
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
RESAMPLE_NEAREST = getattr(Image, "Resampling", Image).NEAREST


def parse_args():
    parser = argparse.ArgumentParser(
        description="Overlay 0/1 mask npy files on images and save previews."
    )
    parser.add_argument(
        "--image-root",
        required=True,
        type=Path,
        help="Root image folder.",
    )
    parser.add_argument(
        "--mask-root",
        required=True,
        type=Path,
        help="Root folder containing 0/1 mask .npy files.",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        type=Path,
        help="Root folder for output visualization images.",
    )
    parser.add_argument(
        "--output-format",
        default="jpg",
        choices=["jpg", "jpeg", "png"],
        help="Output image format.",
    )
    parser.add_argument(
        "--alpha",
        default=0.45,
        type=float,
        help="Red overlay opacity in [0, 1].",
    )
    parser.add_argument(
        "--mask-color",
        default="255,64,64",
        help="RGB overlay color, for example 255,64,64.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate visualization files that already exist.",
    )
    parser.add_argument(
        "--limit",
        default=None,
        type=int,
        help="Only process the first N images after sorting.",
    )
    return parser.parse_args()


def parse_rgb(value):
    parts = [int(part.strip()) for part in value.split(",")]
    if len(parts) != 3 or any(part < 0 or part > 255 for part in parts):
        raise ValueError(f"Expected RGB color like 255,64,64, got: {value}")
    return np.asarray(parts, dtype=np.float32)


def iter_images(image_root):
    for path in sorted(image_root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def load_mask(mask_path, image_size):
    mask = np.asarray(np.load(mask_path))
    if mask.ndim > 2:
        mask = np.squeeze(mask)
    if mask.ndim != 2:
        raise ValueError(f"Expected a 2D mask in {mask_path}, got shape {mask.shape}")

    mask = mask > 0
    width, height = image_size
    if mask.shape != (height, width):
        mask_image = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
        mask_image = mask_image.resize((width, height), RESAMPLE_NEAREST)
        mask = np.asarray(mask_image) > 0
    return mask


def overlay_mask(image, mask, alpha, color):
    alpha = max(0.0, min(1.0, alpha))
    image_array = np.asarray(image.convert("RGB"), dtype=np.float32)
    output = image_array.copy()
    output[mask] = image_array[mask] * (1.0 - alpha) + color * alpha
    return Image.fromarray(np.clip(output, 0, 255).astype(np.uint8), mode="RGB")


def main():
    args = parse_args()
    image_root = args.image_root.resolve()
    mask_root = args.mask_root.resolve()
    output_root = args.output_root.resolve()
    mask_color = parse_rgb(args.mask_color)

    if not image_root.exists():
        raise FileNotFoundError(f"Image root does not exist: {image_root}")
    if not mask_root.exists():
        raise FileNotFoundError(f"Mask root does not exist: {mask_root}")

    image_paths = list(iter_images(image_root))
    if args.limit is not None:
        image_paths = image_paths[: args.limit]
    if not image_paths:
        raise RuntimeError(f"No images found under: {image_root}")

    missing_masks = 0
    written = 0
    for image_path in tqdm(image_paths, desc="Writing mask overlays"):
        rel_path = image_path.relative_to(image_root)
        mask_path = mask_root / rel_path.with_suffix(".npy")
        output_path = output_root / rel_path.with_suffix(f".{args.output_format}")

        if output_path.exists() and not args.overwrite:
            continue
        if not mask_path.exists():
            missing_masks += 1
            continue

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            mask = load_mask(mask_path, image.size)
            preview = overlay_mask(image, mask, args.alpha, mask_color)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.suffix.lower() in {".jpg", ".jpeg"}:
            preview.save(output_path, quality=95)
        else:
            preview.save(output_path)
        written += 1

    print(f"Saved mask overlays to: {output_root}")
    print(f"Wrote {written} files.")
    if missing_masks:
        print(f"Skipped {missing_masks} images without matching mask npy files.")


if __name__ == "__main__":
    main()
