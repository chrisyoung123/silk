#!/usr/bin/env python3
"""SiLK 单图特征缓存：仅存关键点 + 描述子（不做 match）。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


@dataclass
class ImageFeatures:
    image_path: str
    positions_xy: np.ndarray
    descriptors: np.ndarray
    image_hw: Tuple[int, int]
    points_filtered: bool = False
    n_keypoints_raw: int = 0

    def __post_init__(self) -> None:
        self.positions_xy = np.asarray(self.positions_xy, dtype=np.float64).reshape(-1, 2)
        self.descriptors = np.asarray(self.descriptors, dtype=np.float32)
        if self.positions_xy.shape[0] != self.descriptors.shape[0]:
            raise ValueError(
                f"positions/descriptors 行数不一致: "
                f"{self.positions_xy.shape[0]} vs {self.descriptors.shape[0]}"
            )
        if self.n_keypoints_raw <= 0:
            self.n_keypoints_raw = int(self.positions_xy.shape[0])


def cache_key_for_image(image_path: Path) -> str:
    return hashlib.sha256(str(image_path.expanduser().resolve()).encode()).hexdigest()[:16]


class FeatureCacheStore:
    """目录结构: <root>/features/<key>.npz + index.jsonl。"""

    def __init__(self, root: Path) -> None:
        self.root = root.expanduser().resolve()
        self.features_dir = self.root / "features"
        self.index_path = self.root / "index.jsonl"
        self.features_dir.mkdir(parents=True, exist_ok=True)

    def npz_path(self, image_path: Path) -> Path:
        return self.features_dir / f"{cache_key_for_image(image_path)}.npz"

    def has(
        self,
        image_path: Path,
        *,
        require_points_filtered: bool = False,
    ) -> bool:
        p = self.npz_path(image_path)
        if not p.is_file():
            return False
        if not require_points_filtered:
            return True
        try:
            d = np.load(p, allow_pickle=False)
            return bool(int(d.get("points_filtered", 0)))
        except (OSError, ValueError, KeyError):
            return False

    def save(self, feat: ImageFeatures, *, update_index: bool = True) -> Path:
        img = Path(feat.image_path).expanduser().resolve()
        out = self.npz_path(img)
        np.savez_compressed(
            out,
            image_path=str(img),
            positions_xy=feat.positions_xy,
            descriptors=feat.descriptors,
            image_h=np.int32(feat.image_hw[0]),
            image_w=np.int32(feat.image_hw[1]),
            points_filtered=np.int8(1 if feat.points_filtered else 0),
            n_keypoints_raw=np.int32(feat.n_keypoints_raw),
        )
        if update_index:
            self._append_index(img, out, feat)
        return out

    def rewrite_index(self, rows: List[Dict[str, Any]]) -> None:
        with self.index_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def list_cached_npz(self) -> List[Path]:
        return sorted(self.features_dir.glob("*.npz"))

    def load_from_npz(self, npz_path: Path) -> ImageFeatures:
        d = np.load(npz_path, allow_pickle=False)
        n_raw = int(d["n_keypoints_raw"]) if "n_keypoints_raw" in d else int(d["positions_xy"].shape[0])
        filtered = bool(int(d["points_filtered"])) if "points_filtered" in d else False
        return ImageFeatures(
            image_path=str(d["image_path"]),
            positions_xy=d["positions_xy"],
            descriptors=d["descriptors"],
            image_hw=(int(d["image_h"]), int(d["image_w"])),
            points_filtered=filtered,
            n_keypoints_raw=n_raw,
        )

    def _append_index(self, image_path: Path, npz: Path, feat: ImageFeatures) -> None:
        row = {
            "image": str(image_path),
            "npz": str(npz),
            "n_keypoints": int(feat.positions_xy.shape[0]),
            "n_keypoints_raw": int(feat.n_keypoints_raw),
            "points_filtered": bool(feat.points_filtered),
        }
        with self.index_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def load(self, image_path: Path) -> ImageFeatures:
        p = self.npz_path(image_path)
        if not p.is_file():
            raise FileNotFoundError(f"特征缓存不存在: {p}")
        d = np.load(p, allow_pickle=False)
        n_raw = int(d["n_keypoints_raw"]) if "n_keypoints_raw" in d else int(d["positions_xy"].shape[0])
        filtered = bool(int(d["points_filtered"])) if "points_filtered" in d else False
        return ImageFeatures(
            image_path=str(d["image_path"]),
            positions_xy=d["positions_xy"],
            descriptors=d["descriptors"],
            image_hw=(int(d["image_h"]), int(d["image_w"])),
            points_filtered=filtered,
            n_keypoints_raw=n_raw,
        )

    def read_meta(self) -> Dict[str, Any]:
        meta_path = self.root / "meta.json"
        if not meta_path.is_file():
            return {}
        with meta_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def points_filtered_in_meta(self) -> bool:
        return bool(self.read_meta().get("points_filtered"))

    def write_meta(self, meta: Dict[str, Any]) -> None:
        meta_path = self.root / "meta.json"
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)


def unique_images_from_covis_nodes(nodes: Iterable[Dict[str, Any]]) -> List[Path]:
    seen: set[str] = set()
    out: List[Path] = []
    for node in nodes:
        p = Path(node["image"]).expanduser().resolve()
        key = str(p)
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return sorted(out)


def unique_images_from_covis_paths_txt(txt_path: Path) -> List[Path]:
    """从 covis_paths.txt 提取唯一图像路径（每行两列 path_u path_v）。"""
    seen: set[str] = set()
    out: List[Path] = []
    for raw in txt_path.expanduser().resolve().read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        for part in parts[:2]:
            p = Path(part.strip()).expanduser().resolve()
            key = str(p)
            if key in seen:
                continue
            seen.add(key)
            out.append(p)
    return sorted(out)


def load_covis_unique_images(
    *,
    data_dir: Optional[Path] = None,
    covis_adj_json: Optional[Path] = None,
    covis_paths_txt: Optional[Path] = None,
) -> List[Path]:
    """优先 covis_adj.json，否则 covis_paths.txt；均未指定时从 data_dir/covis_output/ 查找。"""
    if covis_adj_json is not None and Path(covis_adj_json).is_file():
        from lib.geometry.absolute_pose_from_covis import load_covis_adjacency

        _, nodes = load_covis_adjacency(Path(covis_adj_json).expanduser().resolve())
        return unique_images_from_covis_nodes(nodes)
    if covis_paths_txt is not None and Path(covis_paths_txt).is_file():
        return unique_images_from_covis_paths_txt(Path(covis_paths_txt))
    if data_dir is not None:
        base = Path(data_dir).expanduser().resolve()
        adj = base / "covis_output" / "covis_adj.json"
        txt = base / "covis_output" / "covis_paths.txt"
        if adj.is_file():
            return load_covis_unique_images(covis_adj_json=adj)
        if txt.is_file():
            return load_covis_unique_images(covis_paths_txt=txt)
    raise FileNotFoundError(
        "找不到共视图像列表：请提供 covis_adj.json 或 covis_paths.txt（或通过 --data-dir 自动查找）"
    )


def pt_path_for_image(pt_dir: Path, image_path: Path) -> Path:
    return pt_dir.expanduser().resolve() / f"{Path(image_path).name}.pt"


def load_from_silk_features_pt(
    pt_path: Path,
    *,
    image_path: Optional[Path] = None,
) -> ImageFeatures:
    """读取 bin/silk-features 输出的 .pt。"""
    import cv2
    import torch

    pt_path = pt_path.expanduser().resolve()
    data = torch.load(str(pt_path), map_location="cpu")
    img = Path(image_path or data["image"]).expanduser().resolve()
    positions = np.asarray(data["positions"], dtype=np.float64).reshape(-1, 2)
    descriptors = np.asarray(data["descriptors"], dtype=np.float32)
    im = cv2.imread(str(img), cv2.IMREAD_COLOR)
    if im is None:
        raise FileNotFoundError(f"无法读取图像以获取尺寸: {img}")
    h, w = im.shape[:2]
    return ImageFeatures(
        image_path=str(img),
        positions_xy=positions,
        descriptors=descriptors,
        image_hw=(h, w),
        points_filtered=False,
        n_keypoints_raw=int(positions.shape[0]),
    )
