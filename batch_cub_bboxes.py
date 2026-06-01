"""
Generate TokenCut bounding boxes for a CUB-style image directory.

The output mirrors the input directory tree and replaces image extensions with
.npy files. Each .npy stores [xmin, ymin, xmax, ymax] in image coordinates.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as pth_transforms
from tqdm import tqdm


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

TRANSFORM = pth_transforms.Compose(
    [
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)


def find_tokencut_root():
    script_dir = Path(__file__).resolve().parent
    candidates = []

    env_root = os.environ.get("TOKENCUT_ROOT")
    if env_root:
        candidates.append(Path(env_root).expanduser())

    candidates.extend(
        [
            script_dir,
            script_dir / "TokenCut",
            script_dir.parent / "TokenCut",
        ]
    )

    for candidate in candidates:
        candidate = candidate.resolve()
        if (candidate / "networks.py").exists() and (
            candidate / "object_discovery.py"
        ).exists():
            return candidate

    searched = "\n".join(f"  - {path}" for path in candidates)
    raise ModuleNotFoundError(
        "Could not find TokenCut source files `networks.py` and "
        "`object_discovery.py`.\n"
        "Run this script from the TokenCut repo root, or set:\n"
        "  TOKENCUT_ROOT=/path/to/TokenCut\n"
        f"Searched:\n{searched}"
    )


TOKENCUT_ROOT = find_tokencut_root()
sys.path.insert(0, str(TOKENCUT_ROOT))

from networks import get_model
from object_discovery import ncut


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch-generate TokenCut bbox npy files for CUB images."
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
        help="Root folder for output bbox .npy files.",
    )
    parser.add_argument(
        "--arch",
        default="vit_small",
        choices=["vit_small", "vit_base"],
        help="DINO ViT architecture.",
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
        help="Regenerate bbox files that already exist.",
    )
    parser.add_argument(
        "--limit",
        default=None,
        type=int,
        help="Only process the first N images after sorting.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Print per-image timing for load, ViT forward, NCut, and save.",
    )
    return parser.parse_args()


def iter_images(image_root):
    for path in sorted(image_root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def load_image_tensor(image_path):
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        tensor = TRANSFORM(image)
    return tensor


def pad_to_patch_multiple(img, patch_size):
    _, height, width = img.shape
    padded_height = int(np.ceil(height / patch_size) * patch_size)
    padded_width = int(np.ceil(width / patch_size) * patch_size)
    padded = torch.zeros((img.shape[0], padded_height, padded_width), dtype=img.dtype)
    padded[:, :height, :width] = img
    return padded


def get_vit_features(model, img, arch, which_features):
    if "vit" not in arch:
        raise ValueError("This bbox script supports ViT models only.")

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


def main():
    args = parse_args()
    image_root = args.image_root.resolve()
    output_root = args.output_root.resolve()

    if not image_root.exists():
        raise FileNotFoundError(f"Image root does not exist: {image_root}")

    image_paths = list(iter_images(image_root))
    if args.limit is not None:
        image_paths = image_paths[: args.limit]
    if not image_paths:
        raise RuntimeError(f"No images found under: {image_root}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")
    print(f"Using TokenCut root: {TOKENCUT_ROOT}")
    model = get_model(args.arch, args.patch_size, device)

    output_root.mkdir(parents=True, exist_ok=True)

    for image_path in tqdm(image_paths, desc="Generating bboxes"):
        rel_path = image_path.relative_to(image_root)
        output_path = output_root / rel_path.with_suffix(".npy")

        if output_path.exists() and not args.overwrite:
            continue

        load_start = time.perf_counter()
        img = load_image_tensor(image_path)
        init_image_size = img.shape
        img = pad_to_patch_multiple(img, args.patch_size)

        if device.type == "cuda":
            img = img.cuda(non_blocking=True)

        w_featmap = img.shape[-2] // args.patch_size
        h_featmap = img.shape[-1] // args.patch_size
        scales = [args.patch_size, args.patch_size]

        with torch.no_grad():
            vit_start = time.perf_counter()
            feats = get_vit_features(model, img, args.arch, args.which_features)
            ncut_start = time.perf_counter()
            bbox, _, _, _, _, _ = ncut(
                feats,
                [w_featmap, h_featmap],
                scales,
                init_image_size,
                args.tau,
                args.eps,
                im_name=str(rel_path),
                no_binary_graph=args.no_binary_graph,
            )
            post_ncut_time = time.perf_counter()

        save_start = time.perf_counter()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, np.asarray(bbox, dtype=np.float32))

        if args.profile:
            save_end = time.perf_counter()
            print(
                f"{rel_path}: "
                f"load={vit_start - load_start:.3f}s "
                f"vit={ncut_start - vit_start:.3f}s "
                f"ncut={post_ncut_time - ncut_start:.3f}s "
                f"save={save_end - save_start:.3f}s "
                f"total={save_end - load_start:.3f}s "
                f"patches={w_featmap * h_featmap}"
            )

    print(f"Saved bounding boxes to: {output_root}")


if __name__ == "__main__":
    main()
