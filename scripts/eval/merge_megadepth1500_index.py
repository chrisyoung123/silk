#!/usr/bin/env python3
"""Merge LoFTR per-scene MegaDepth-1500 npz files into one SiLK index npz.

LoFTR stores test metadata as one npz per scene (e.g. ``0015_0.1_0.3.npz``) under
``megadepth_test_1500_scene_info/``. SiLK's ``MegaDepth1500Pairs`` expects a single
file with global indices:

  - image_paths
  - intrinsics
  - poses
  - pair_infos   # LoFTR format: ((i, j), overlap, ...)

Usage (from repo root)::

  python scripts/eval/merge_megadepth1500_index.py \\
    --scene-info-dir /path/to/megadepth_test_1500_scene_info \\
    --scene-list /path/to/megadepth_test_1500_scene_info/megadepth_test_1500.txt \\
    --output /path/to/megadepth_test_1500/index/megadepth_test_1500.npz

If ``--scene-list`` is omitted, every ``*.npz`` in ``--scene-info-dir`` is merged
(excluding the output file name when it lives in the same folder).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np


def _to_str_array(paths: Sequence) -> np.ndarray:
    out = []
    for p in paths:
        if p is None:
            out.append(None)
        else:
            out.append(str(p))
    return np.asarray(out, dtype=object)


def _normalize_intrinsic(value) -> np.ndarray:
    if value is None:
        return np.eye(3, dtype=np.float32)
    return np.asarray(value, dtype=np.float32)


def _normalize_pose(value) -> np.ndarray:
    if value is None:
        return np.eye(4, dtype=np.float32)
    return np.asarray(value, dtype=np.float32)


def _load_scene_list(path: Path) -> List[str]:
    lines = path.read_text().strip().splitlines()
    return [line.strip() for line in lines if line.strip() and not line.strip().startswith("#")]


def _discover_scene_npz(scene_info_dir: Path, output_path: Path) -> List[Path]:
    out_resolved = output_path.resolve()
    files = sorted(scene_info_dir.glob("*.npz"))
    return [p for p in files if p.resolve() != out_resolved]


def _remap_pair_infos(pair_infos, offset: int) -> List:
    remapped = []
    for item in pair_infos:
        pair, overlap, *rest = item
        i0, i1 = int(pair[0]), int(pair[1])
        remapped.append(((i0 + offset, i1 + offset), overlap, *rest))
    return remapped


def merge_megadepth1500(
    scene_npz_paths: Sequence[Path],
    min_overlap: float = 0.0,
) -> dict:
    image_paths: List[str] = []
    intrinsics: List[np.ndarray] = []
    poses: List[np.ndarray] = []
    pair_infos: List = []

    for scene_path in scene_npz_paths:
        scene = np.load(scene_path, allow_pickle=True)
        try:
            n_images = len(scene["image_paths"])
            offset = len(image_paths)

            image_paths.extend(scene["image_paths"])
            intrinsics.extend(_normalize_intrinsic(k) for k in scene["intrinsics"])
            poses.extend(_normalize_pose(p) for p in scene["poses"])

            scene_pairs = scene["pair_infos"]
            if min_overlap > 0.0:
                scene_pairs = [p for p in scene_pairs if float(p[1]) > min_overlap]
            pair_infos.extend(_remap_pair_infos(scene_pairs, offset))

            print(
                f"  {scene_path.name}: {n_images} images, "
                f"{len(scene_pairs)} pairs (offset {offset})"
            )
        finally:
            if hasattr(scene, "close"):
                scene.close()

    return {
        "image_paths": _to_str_array(image_paths),
        "intrinsics": np.asarray(intrinsics, dtype=np.float32),
        "poses": np.asarray(poses, dtype=np.float32),
        "pair_infos": np.asarray(pair_infos, dtype=object),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scene-info-dir",
        type=Path,
        required=True,
        help="Directory with LoFTR scene npz files (e.g. megadepth_test_1500_scene_info).",
    )
    parser.add_argument(
        "--scene-list",
        type=Path,
        default=None,
        help="Text file listing scene stems (default: all *.npz in scene-info-dir).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path, e.g. .../megadepth_test_1500/index/megadepth_test_1500.npz",
    )
    parser.add_argument(
        "--min-overlap",
        type=float,
        default=0.0,
        help="Drop pairs with overlap <= this value (LoFTR test uses 0).",
    )
    args = parser.parse_args(argv)

    scene_info_dir = args.scene_info_dir.expanduser().resolve()
    if not scene_info_dir.is_dir():
        print(f"error: not a directory: {scene_info_dir}", file=sys.stderr)
        return 1

    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.scene_list is not None:
        scene_list_path = args.scene_list.expanduser().resolve()
        if not scene_list_path.is_file():
            print(f"error: scene list not found: {scene_list_path}", file=sys.stderr)
            return 1
        scene_npz_paths = [
            scene_info_dir / f"{stem}.npz" for stem in _load_scene_list(scene_list_path)
        ]
    else:
        scene_npz_paths = _discover_scene_npz(scene_info_dir, output_path)

    missing = [p for p in scene_npz_paths if not p.is_file()]
    if missing:
        for p in missing:
            print(f"error: missing scene npz: {p}", file=sys.stderr)
        return 1

    print(f"Merging {len(scene_npz_paths)} scene file(s) -> {output_path}")
    merged = merge_megadepth1500(scene_npz_paths, min_overlap=args.min_overlap)

    np.savez_compressed(output_path, **merged)

    print(
        f"Done: {len(merged['image_paths'])} images, "
        f"{len(merged['pair_infos'])} pairs"
    )
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
