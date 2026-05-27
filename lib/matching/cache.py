# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""图像对匹配缓存：描述子匹配（极线前）与极线 inlier（极线后）。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


def pair_cache_key(img0: Path, img1: Path, *, matcher: str = "silk") -> str:
    a, b = sorted(
        [str(img0.expanduser().resolve()), str(img1.expanduser().resolve())]
    )
    payload = f"{matcher}|{a}|{b}".encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def epipolar_cache_key(
    img0: Path,
    img1: Path,
    *,
    matcher: str,
    camera_model: str,
    ransac_threshold: float,
    min_inliers: int,
) -> str:
    a, b = sorted(
        [str(img0.expanduser().resolve()), str(img1.expanduser().resolve())]
    )
    payload = (
        f"epi|{matcher}|{camera_model}|{ransac_threshold:.6g}|{min_inliers}|{a}|{b}"
    ).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


class MatchPairCacheStore:
    """目录: <root>/pairs/<key>.npz + index.jsonl。"""

    def __init__(self, root: Path) -> None:
        self.root = root.expanduser().resolve()
        self.pairs_dir = self.root / "pairs"
        self.index_path = self.root / "index.jsonl"
        self.pairs_dir.mkdir(parents=True, exist_ok=True)

    def npz_path(self, img0: Path, img1: Path, *, matcher: str) -> Path:
        return self.pairs_dir / f"{pair_cache_key(img0, img1, matcher=matcher)}.npz"

    def has(self, img0: Path, img1: Path, *, matcher: str) -> bool:
        return self.npz_path(img0, img1, matcher=matcher).is_file()

    def load(
        self,
        img0: Path,
        img1: Path,
        *,
        matcher: str,
    ) -> Optional[Dict[str, Any]]:
        p = self.npz_path(img0, img1, matcher=matcher)
        if not p.is_file():
            return None
        d = np.load(p, allow_pickle=False)
        return {
            "img0": str(d["img0"]),
            "img1": str(d["img1"]),
            "matcher": str(d["matcher"]),
            "m0": np.asarray(d["m0"], dtype=np.float64),
            "m1": np.asarray(d["m1"], dtype=np.float64),
            "idx0": np.asarray(d["idx0"], dtype=np.int64),
            "idx1": np.asarray(d["idx1"], dtype=np.int64),
            "n_keypoints0": int(d["n_keypoints0"]),
            "n_keypoints1": int(d["n_keypoints1"]),
            "n_matches": int(d["n_matches"]),
            "cache_path": str(p),
        }

    def save(
        self,
        img0: Path,
        img1: Path,
        *,
        matcher: str,
        m0: np.ndarray,
        m1: np.ndarray,
        idx0: np.ndarray,
        idx1: np.ndarray,
        n_keypoints0: int,
        n_keypoints1: int,
    ) -> Path:
        img0 = img0.expanduser().resolve()
        img1 = img1.expanduser().resolve()
        out = self.npz_path(img0, img1, matcher=matcher)
        np.savez_compressed(
            out,
            img0=str(img0),
            img1=str(img1),
            matcher=str(matcher),
            m0=np.asarray(m0, dtype=np.float64),
            m1=np.asarray(m1, dtype=np.float64),
            idx0=np.asarray(idx0, dtype=np.int64),
            idx1=np.asarray(idx1, dtype=np.int64),
            n_keypoints0=np.int32(n_keypoints0),
            n_keypoints1=np.int32(n_keypoints1),
            n_matches=np.int32(m0.shape[0]),
        )
        row = {
            "img0": str(img0),
            "img1": str(img1),
            "matcher": matcher,
            "npz": str(out),
            "n_matches": int(m0.shape[0]),
            "n_keypoints0": int(n_keypoints0),
            "n_keypoints1": int(n_keypoints1),
        }
        with self.index_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return out


class EpipolarInlierCacheStore:
    """极线 RANSAC 后的 inlier 缓存。目录: <root>/pairs/<key>.npz + index.jsonl。"""

    def __init__(self, root: Path) -> None:
        self.root = root.expanduser().resolve()
        self.pairs_dir = self.root / "pairs"
        self.index_path = self.root / "index.jsonl"
        self.pairs_dir.mkdir(parents=True, exist_ok=True)

    def npz_path(
        self,
        img0: Path,
        img1: Path,
        *,
        matcher: str,
        camera_model: str,
        ransac_threshold: float,
        min_inliers: int,
    ) -> Path:
        key = epipolar_cache_key(
            img0,
            img1,
            matcher=matcher,
            camera_model=camera_model,
            ransac_threshold=ransac_threshold,
            min_inliers=min_inliers,
        )
        return self.pairs_dir / f"{key}.npz"

    def has(
        self,
        img0: Path,
        img1: Path,
        *,
        matcher: str,
        camera_model: str,
        ransac_threshold: float,
        min_inliers: int,
    ) -> bool:
        return self.npz_path(
            img0,
            img1,
            matcher=matcher,
            camera_model=camera_model,
            ransac_threshold=ransac_threshold,
            min_inliers=min_inliers,
        ).is_file()

    def load(
        self,
        img0: Path,
        img1: Path,
        *,
        matcher: str,
        camera_model: str,
        ransac_threshold: float,
        min_inliers: int,
    ) -> Optional[Dict[str, Any]]:
        p = self.npz_path(
            img0,
            img1,
            matcher=matcher,
            camera_model=camera_model,
            ransac_threshold=ransac_threshold,
            min_inliers=min_inliers,
        )
        if not p.is_file():
            return None
        d = np.load(p, allow_pickle=False)
        reason = str(d["reason"]) if "reason" in d.files and str(d["reason"]) else None
        return {
            "img0": str(d["img0"]),
            "img1": str(d["img1"]),
            "matcher": str(d["matcher"]),
            "camera_model": str(d["camera_model"]),
            "ransac_threshold": float(d["ransac_threshold"]),
            "min_inliers": int(d["min_inliers"]),
            "m0": np.asarray(d["m0"], dtype=np.float64),
            "m1": np.asarray(d["m1"], dtype=np.float64),
            "n_matches_pre_epi": int(d["n_matches_pre_epi"]),
            "n_epipolar_inliers": int(d["n_epipolar_inliers"]),
            "n_used": int(d["n_used"]),
            "n_keypoints0": int(d["n_keypoints0"]),
            "n_keypoints1": int(d["n_keypoints1"]),
            "ok": bool(int(d["ok"])),
            "reason": reason,
            "cache_path": str(p),
        }

    def save(
        self,
        img0: Path,
        img1: Path,
        *,
        matcher: str,
        camera_model: str,
        ransac_threshold: float,
        min_inliers: int,
        m0: np.ndarray,
        m1: np.ndarray,
        n_matches_pre_epi: int,
        n_epipolar_inliers: int,
        n_used: int,
        n_keypoints0: int,
        n_keypoints1: int,
        ok: bool,
        reason: Optional[str] = None,
    ) -> Path:
        img0 = img0.expanduser().resolve()
        img1 = img1.expanduser().resolve()
        out = self.npz_path(
            img0,
            img1,
            matcher=matcher,
            camera_model=camera_model,
            ransac_threshold=ransac_threshold,
            min_inliers=min_inliers,
        )
        np.savez_compressed(
            out,
            img0=str(img0),
            img1=str(img1),
            matcher=str(matcher),
            camera_model=str(camera_model),
            ransac_threshold=np.float64(ransac_threshold),
            min_inliers=np.int32(min_inliers),
            m0=np.asarray(m0, dtype=np.float64),
            m1=np.asarray(m1, dtype=np.float64),
            n_matches_pre_epi=np.int32(n_matches_pre_epi),
            n_epipolar_inliers=np.int32(n_epipolar_inliers),
            n_used=np.int32(n_used),
            n_keypoints0=np.int32(n_keypoints0),
            n_keypoints1=np.int32(n_keypoints1),
            ok=np.int8(1 if ok else 0),
            reason=str(reason or ""),
        )
        row = {
            "img0": str(img0),
            "img1": str(img1),
            "matcher": matcher,
            "camera_model": camera_model,
            "ransac_threshold": float(ransac_threshold),
            "min_inliers": int(min_inliers),
            "npz": str(out),
            "n_matches_pre_epi": int(n_matches_pre_epi),
            "n_epipolar_inliers": int(n_epipolar_inliers),
            "n_used": int(n_used),
            "ok": bool(ok),
        }
        with self.index_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return out
