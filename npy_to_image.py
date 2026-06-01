"""
Convert one TokenCut .npy file to one png/jpg preview.

Supported inputs:
  - 2D mask npy: saved as grayscale, or overlaid on --image-path.
  - bbox npy with 4 values: drawn on --image-path.
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


RESAMPLE_NEAREST = getattr(Image, "Resampling", Image).NEAREST


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert a single TokenCut npy mask/bbox to png or jpg."
    )
    parser.add_argument(
        "--npy-path",
        required=True,
        type=Path,
        help="Input .npy file. Use a 2D mask or a bbox array with 4 values.",
    )
    parser.add_argument(
        "--output-path",
        required=True,
        type=Path,
        help="Output .png/.jpg path.",
    )
    parser.add_argument(
        "--image-path",
        default=None,
        type=Path,
        help="Optional original image for mask overlay or bbox drawing.",
    )
    parser.add_argument(
        "--bbox-npy",
        default=None,
        type=Path,
        help="Optional bbox .npy to draw when --npy-path is a mask.",
    )
    parser.add_argument(
        "--alpha",
        default=0.45,
        type=float,
        help="Mask overlay opacity in [0, 1].",
    )
    parser.add_argument(
        "--mask-color",
        default="255,64,64",
        help="RGB mask overlay color, for example 255,64,64.",
    )
    parser.add_argument(
        "--bbox-color",
        default="64,255,128",
        help="RGB bbox outline color, for example 64,255,128.",
    )
    parser.add_argument(
        "--line-width",
        default=3,
        type=int,
        help="BBox outline width in pixels.",
    )
    return parser.parse_args()


def parse_rgb(value):
    parts = [int(part.strip()) for part in value.split(",")]
    if len(parts) != 3 or any(part < 0 or part > 255 for part in parts):
        raise ValueError(f"Expected RGB color like 255,64,64, got: {value}")
    return tuple(parts)


def load_optional_image(image_path):
    if image_path is None:
        return None
    if not image_path.exists():
        raise FileNotFoundError(f"Image does not exist: {image_path}")
    return Image.open(image_path).convert("RGB")


def is_bbox_array(array):
    squeezed = np.squeeze(np.asarray(array))
    return squeezed.ndim == 1 and squeezed.size == 4


def normalize_mask(mask, image_size=None):
    mask = np.asarray(mask)
    if mask.ndim > 2:
        mask = np.squeeze(mask)
    if mask.ndim != 2:
        raise ValueError(f"Expected a 2D mask, got shape {mask.shape}")

    mask = mask > 0
    if image_size is None:
        return mask

    width, height = image_size
    if mask.shape != (height, width):
        mask_image = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
        mask_image = mask_image.resize((width, height), RESAMPLE_NEAREST)
        mask = np.asarray(mask_image) > 0
    return mask


def mask_to_grayscale(mask):
    return Image.fromarray(mask.astype(np.uint8) * 255, mode="L")


def overlay_mask(image, mask, alpha, color):
    image_arr = np.asarray(image.convert("RGB")).astype(np.float32)
    color_arr = np.array(color, dtype=np.float32)
    blended = image_arr.copy()
    blended[mask] = image_arr[mask] * (1.0 - alpha) + color_arr * alpha
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8), mode="RGB")


def draw_bbox(image, bbox, color, line_width):
    bbox = np.asarray(bbox, dtype=np.float32).reshape(-1)
    if bbox.size != 4:
        raise ValueError(f"Expected bbox with 4 values, got shape {bbox.shape}")

    draw = ImageDraw.Draw(image)
    width, height = image.size
    x1, y1, x2, y2 = bbox.tolist()
    x1 = max(0, min(width - 1, x1))
    x2 = max(0, min(width - 1, x2))
    y1 = max(0, min(height - 1, y1))
    y2 = max(0, min(height - 1, y2))

    for offset in range(max(1, line_width)):
        draw.rectangle(
            [x1 - offset, y1 - offset, x2 + offset, y2 + offset],
            outline=color,
        )


def main():
    args = parse_args()
    npy_path = args.npy_path.resolve()
    output_path = args.output_path.resolve()
    image = load_optional_image(args.image_path.resolve() if args.image_path else None)
    mask_color = parse_rgb(args.mask_color)
    bbox_color = parse_rgb(args.bbox_color)

    if not npy_path.exists():
        raise FileNotFoundError(f"Npy file does not exist: {npy_path}")

    array = np.load(npy_path)

    if is_bbox_array(array):
        if image is None:
            raise ValueError("A bbox npy requires --image-path for visualization.")
        preview = image.copy()
        draw_bbox(preview, array, bbox_color, args.line_width)
    else:
        image_size = image.size if image is not None else None
        mask = normalize_mask(array, image_size)
        if image is None:
            preview = mask_to_grayscale(mask)
        else:
            preview = overlay_mask(image, mask, args.alpha, mask_color)
            if args.bbox_npy is not None:
                bbox_path = args.bbox_npy.resolve()
                if not bbox_path.exists():
                    raise FileNotFoundError(f"Bbox npy does not exist: {bbox_path}")
                draw_bbox(preview, np.load(bbox_path), bbox_color, args.line_width)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() in {".jpg", ".jpeg"}:
        preview = preview.convert("RGB")
        preview.save(output_path, quality=95)
    else:
        preview.save(output_path)
    print(f"Saved preview to: {output_path}")


if __name__ == "__main__":
    main()
