"""
Generate TokenCut masks and bounding boxes for a CUB-style image directory.

The outputs mirror the input directory tree and replace image extensions with
.npy files. For example:

    CUB_200_2011/images/001.Black_footed_Albatross/foo.jpg

becomes:

    output_masks/001.Black_footed_Albatross/foo.npy
    output_bboxes/001.Black_footed_Albatross/foo.npy
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as pth_transforms
from tqdm import tqdm

from networks import get_model
from object_discovery import ncut


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

TRANSFORM = pth_transforms.Compose(
    [
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch-generate TokenCut masks for CUB images."
    )
    parser.add_argument(
        "--image-root",
        required=True,
        type=Path,
        help="Root image folder, usually CUB_200_2011/images.",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        type=Path,
        help="Root folder for output .npy masks.",
    )
    parser.add_argument(
        "--bbox-output-root",
        default=None,
        type=Path,
        help=(
            "Root folder for output bbox .npy files. Defaults to "
            "<output-root>_bboxes."
        ),
    )
    parser.add_argument(
        "--arch",
        default="vit_small",
        choices=[
            "vit_tiny",
            "vit_small",
            "vit_base",
            "moco_vit_small",
            "moco_vit_base",
            "mae_vit_base",
        ],
        help="Model architecture.",
    )
    parser.add_argument(
        "--patch-size",
        default=16,
        type=int,
        help="Patch resolution of the model.",
    )
    parser.add_argument(
        "--which-features",
        default="k",
        choices=["k", "q", "v"],
        help="Which ViT features to use for TokenCut.",
    )
    parser.add_argument(
        "--tau",
        default=0.2,
        type=float,
        help="Threshold for graph construction.",
    )
    parser.add_argument(
        "--eps",
        default=1e-5,
        type=float,
        help="Small graph edge weight used by TokenCut.",
    )
    parser.add_argument(
        "--no-binary-graph",
        action="store_true",
        help="Use similarity scores as graph weights instead of a binary graph.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate masks that already exist.",
    )
    parser.add_argument(
        "--save-patch-mask",
        action="store_true",
        help="Save the raw patch-grid mask instead of upsampling to image size.",
    )
    return parser.parse_args()


def iter_images(image_root):
    for path in sorted(image_root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def load_image_tensor(image_path):
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        original_size = image.size  # PIL: (width, height)
        tensor = TRANSFORM(image)
    return tensor, original_size


def pad_to_patch_multiple(img, patch_size):
    _, height, width = img.shape
    padded_height = int(np.ceil(height / patch_size) * patch_size)
    padded_width = int(np.ceil(width / patch_size) * patch_size)
    padded = torch.zeros((img.shape[0], padded_height, padded_width), dtype=img.dtype)
    padded[:, :height, :width] = img
    return padded


def get_vit_features(model, img, arch, which_features):
    if "vit" not in arch:
        raise ValueError("This batch mask script currently supports ViT models only.")

    feat_out = {}

    def hook_fn_forward_qkv(module, input, output):
        feat_out["qkv"] = output

    handle = model._modules["blocks"][-1]._modules["attn"]._modules[
        "qkv"
    ].register_forward_hook(hook_fn_forward_qkv)
    try:
        attentions = model.get_last_selfattention(img[None, :, :, :])
    finally:
        handle.remove()

    nb_im = attentions.shape[0]
    nh = attentions.shape[1]
    nb_tokens = attentions.shape[2]
    qkv = (
        feat_out["qkv"]
        .reshape(nb_im, nb_tokens, 3, nh, -1 // nh)
        .permute(2, 0, 3, 1, 4)
    )
    q, k, v = qkv[0], qkv[1], qkv[2]
    k = k.transpose(1, 2).reshape(nb_im, nb_tokens, -1)
    q = q.transpose(1, 2).reshape(nb_im, nb_tokens, -1)
    v = v.transpose(1, 2).reshape(nb_im, nb_tokens, -1)

    if which_features == "k":
        return k
    if which_features == "q":
        return q
    if which_features == "v":
        return v
    raise ValueError(f"Unsupported feature type: {which_features}")


def upsample_mask(mask, original_size):
    width, height = original_size
    mask_tensor = torch.from_numpy(mask).float()[None, None, :, :]
    upsampled = F.interpolate(mask_tensor, size=(height, width), mode="nearest")
    return upsampled[0, 0].cpu().numpy().astype(np.uint8)


def main():
    args = parse_args()
    image_root = args.image_root.resolve()
    output_root = args.output_root.resolve()
    bbox_output_root = (
        args.bbox_output_root.resolve()
        if args.bbox_output_root is not None
        else output_root.with_name(f"{output_root.name}_bboxes")
    )

    if not image_root.exists():
        raise FileNotFoundError(f"Image root does not exist: {image_root}")

    image_paths = list(iter_images(image_root))
    if not image_paths:
        raise RuntimeError(f"No images found under: {image_root}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = get_model(args.arch, args.patch_size, device)

    output_root.mkdir(parents=True, exist_ok=True)
    bbox_output_root.mkdir(parents=True, exist_ok=True)

    for image_path in tqdm(image_paths, desc="Generating masks"):
        rel_path = image_path.relative_to(image_root)
        output_path = output_root / rel_path.with_suffix(".npy")
        bbox_output_path = bbox_output_root / rel_path.with_suffix(".npy")

        if output_path.exists() and bbox_output_path.exists() and not args.overwrite:
            continue

        img, original_size = load_image_tensor(image_path)
        init_image_size = img.shape
        img = pad_to_patch_multiple(img, args.patch_size)

        if device.type == "cuda":
            img = img.cuda(non_blocking=True)

        w_featmap = img.shape[-2] // args.patch_size
        h_featmap = img.shape[-1] // args.patch_size
        scales = [args.patch_size, args.patch_size]

        with torch.no_grad():
            feats = get_vit_features(model, img, args.arch, args.which_features)
            bbox, _, foreground, _, _, _ = ncut(
                feats,
                [w_featmap, h_featmap],
                scales,
                init_image_size,
                args.tau,
                args.eps,
                im_name=str(rel_path),
                no_binary_graph=args.no_binary_graph,
            )

        mask = foreground.astype(np.uint8)
        if not args.save_patch_mask:
            mask = upsample_mask(mask, original_size)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, mask)
        bbox_output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(bbox_output_path, np.asarray(bbox, dtype=np.float32))

    print(f"Saved masks to: {output_root}")
    print(f"Saved bounding boxes to: {bbox_output_root}")


if __name__ == "__main__":
    main()
