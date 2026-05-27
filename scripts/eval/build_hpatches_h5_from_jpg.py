#!/usr/bin/env python3
"""Build HPatches-style cached .h5 from a folder of JPG/PNG images (plan A).

Each source image yields ``warps_per_image`` pairs:
  - original_img : float32 [1, H, W] in [0, 1]
  - warped_img   : cv2.warpPerspective(original, H)
  - homography   : float64 [3, 3], maps original pixel (x, y) -> warped pixel (x, y)

Then evaluate repeatability, MMA, and homography metrics with::

  ./bin/silk-cli mode=run-hpatches-tests-silk-cached-480-grey \\
    '+mode.loader.dataset.filepath=/path/to/out.h5' \\
    mode.model.checkpoint_path=checkpoints/pvgg-4.ckpt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

import cv2
import numpy as np
import torch
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "lib") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "lib"))

from silk.datasets.cached import CachedDataset
from silk.transforms.abstract import NamedContext


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _list_images(image_dir: Path, recursive: bool) -> List[Path]:
    pattern = "**/*" if recursive else "*"
    paths = [
        p
        for p in image_dir.glob(pattern)
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]
    return sorted(paths)


def _resize_min_side(
    image: np.ndarray,
    min_side: int,
    force_min_max: bool = True,
    no_resize: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Match HPatches test.yaml min_img_size + divisibility by 8.

    ``min_side``: target length for the **shorter** image edge (HPatches 默认 480).
    With ``force_min_max=True``, every image is scaled so min(H,W)==min_side.

    ``no_resize``: keep native resolution; only pad H,W to multiples of 8 (SiLK 需要).
    """
    h0, w0 = image.shape[:2]
    r = 1.0
    if no_resize:
        pass
    elif force_min_max or min(h0, w0) < min_side:
        r = float(min_side) / float(min(h0, w0))

    h1 = max(1, int(round(h0 * r)))
    w1 = max(1, int(round(w0 * r)))
    h1 += (8 - h1 % 8) % 8
    w1 += (8 - w1 % 8) % 8

    if (h1, w1) != (h0, w0):
        image = cv2.resize(image, (w1, h1), interpolation=cv2.INTER_AREA)

    # H_resize: maps coords in original full-res -> resized image (xy homogeneous)
    sx = w1 / float(w0)
    sy = h1 / float(h0)
    h_resize = np.array(
        [[sx, 0.0, 0.0], [0.0, sy, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return image, h_resize


def _t_translate(tx: float, ty: float) -> np.ndarray:
    return np.array([[1.0, 0.0, tx], [0.0, 1.0, ty], [0.0, 0.0, 1.0]], dtype=np.float64)


def _t_scale(s: float) -> np.ndarray:
    return np.array([[s, 0.0, 0.0], [0.0, s, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def _t_rotate(angle_rad: float) -> np.ndarray:
    c, s = float(np.cos(angle_rad)), float(np.sin(angle_rad))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def sample_random_homography(
    width: int,
    height: int,
    rng: np.random.Generator,
    *,
    scale_range: Tuple[float, float] = (0.65, 0.95),
    z_rot_deg: float = 22.0,
    trans_frac: float = 0.15,
    perspective: float = 5e-4,
) -> np.ndarray:
    """Return H (3x3) mapping original image pixels -> warped image pixels."""
    cx, cy = width / 2.0, height / 2.0
    angle = rng.uniform(-z_rot_deg, z_rot_deg) * np.pi / 180.0
    scale = rng.uniform(scale_range[0], scale_range[1])
    tx = rng.uniform(-trans_frac, trans_frac) * width
    ty = rng.uniform(-trans_frac, trans_frac) * height

    h_center = (
        _t_translate(cx, cy)
        @ _t_rotate(angle)
        @ _t_scale(scale)
        @ _t_translate(-cx, -cy)
        @ _t_translate(tx, ty)
    )

    # mild perspective (image-center anchored)
    p1 = rng.uniform(-perspective, perspective)
    p2 = rng.uniform(-perspective, perspective)
    h_persp = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [p1, p2, 1.0],
        ],
        dtype=np.float64,
    )
    return h_persp @ h_center


def _load_gray_float01(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"failed to read image: {path}")
    return img.astype(np.float32) / 255.0


def build_pairs(
    image_paths: Iterable[Path],
    *,
    min_side: int,
    warps_per_image: int,
    seed: int,
    no_resize: bool = False,
) -> Iterable[NamedContext]:
    rng = np.random.default_rng(seed)

    for path in image_paths:
        gray = _load_gray_float01(path)
        gray, _ = _resize_min_side(
            gray, min_side=min_side, force_min_max=True, no_resize=no_resize
        )
        h, w = gray.shape

        for _ in range(warps_per_image):
            h_mat = sample_random_homography(w, h, rng)
            warped = cv2.warpPerspective(
                gray,
                h_mat,
                (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0.0,
            )

            original_t = torch.from_numpy(gray).unsqueeze(0).float()
            warped_t = torch.from_numpy(warped).unsqueeze(0).float()
            homography_t = torch.from_numpy(h_mat).double()

            yield NamedContext(
                {
                    "original_img": original_t,
                    "warped_img": warped_t,
                    "homography": homography_t,
                    "source_path": str(path),
                }
            )


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image-dir",
        type=Path,
        required=True,
        help="Directory containing source images.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output .h5 path (e.g. outputs/custom_hpatches_480.h5).",
    )
    parser.add_argument(
        "--min-side",
        type=int,
        default=480,
        help="Scale so min(H,W) equals this when resizing (see --no-resize).",
    )
    parser.add_argument(
        "--no-resize",
        action="store_true",
        help="Keep native resolution; only pad to H,W %% 8 == 0 (for e.g. 768x432).",
    )
    parser.add_argument(
        "--warps-per-image",
        type=int,
        default=5,
        help="Synthetic warps per source image (HPatches uses 5 per sequence).",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=-1,
        help="Limit number of source images (-1 = all).",
    )
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    image_dir = args.image_dir.expanduser().resolve()
    if not image_dir.is_dir():
        print(f"error: not a directory: {image_dir}", file=sys.stderr)
        return 1

    paths = _list_images(image_dir, args.recursive)
    if not paths:
        print(f"error: no images under {image_dir}", file=sys.stderr)
        return 1

    if args.max_images is not None and args.max_images >= 0:
        paths = paths[: args.max_images]

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    pairs = build_pairs(
        paths,
        min_side=args.min_side,
        warps_per_image=args.warps_per_image,
        seed=args.seed,
        no_resize=args.no_resize,
    )
    n = args.warps_per_image
    total = len(paths) * n

    print(
        f"Building {total} pairs ({len(paths)} images x {n} warps) -> {output}"
    )
    CachedDataset.from_iterable(
        str(output),
        tqdm(pairs, total=total, desc="write h5"),
    )
    print(f"Done: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
