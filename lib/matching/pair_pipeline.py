# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""图像对匹配管道：描述子匹配 → 缓存 → 极线 inlier 过滤 → 极线缓存。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

from lib.geometry.epipolar import filter_matches_epipolar
from lib.matching.cache import EpipolarInlierCacheStore, MatchPairCacheStore

DescriptorMatchFn = Callable[[Path, Path], Dict[str, Any]]


@dataclass(frozen=True)
class EpipolarConfig:
    camera_model: str
    ransac_threshold: float
    min_inliers: int
    min_matches: int


def epipolar_cache_params(cfg: EpipolarConfig) -> Dict[str, Any]:
    return {
        "camera_model": str(cfg.camera_model),
        "ransac_threshold": float(cfg.ransac_threshold),
        "min_inliers": int(cfg.min_inliers),
    }


def meta_from_epipolar_cache(
    cached: Dict[str, Any],
    img0: Path,
    img1: Path,
    matcher: str,
) -> Dict[str, Any]:
    meta: Dict[str, Any] = {
        "img0": str(img0),
        "img1": str(img1),
        "matcher": matcher,
        "n_matches": int(cached.get("n_matches_pre_epi", 0)),
        "n_keypoints0": int(cached.get("n_keypoints0", 0)),
        "n_keypoints1": int(cached.get("n_keypoints1", 0)),
        "n_matches_pre_epi": int(cached.get("n_matches_pre_epi", 0)),
        "n_epipolar_inliers": int(cached.get("n_epipolar_inliers", 0)),
        "n_used": int(cached.get("n_used", 0)),
        "ok_pre_epi": int(cached.get("n_matches_pre_epi", 0)) > 0,
        "ok": bool(cached.get("ok", False)),
        "epipolar_cache_hit": True,
        "epipolar_cache_path": cached.get("cache_path"),
        "from_cache": True,
    }
    reason = cached.get("reason")
    if reason:
        meta["reason"] = reason
    elif meta["ok"]:
        meta.pop("reason", None)
    return meta


def match_pair_with_epipolar(
    img0: Path,
    img1: Path,
    *,
    descriptor_match_fn: DescriptorMatchFn,
    K: np.ndarray,
    D: np.ndarray,
    epipolar: EpipolarConfig,
    matcher: str = "silk",
    match_cache: Optional[MatchPairCacheStore] = None,
    epi_cache: Optional[EpipolarInlierCacheStore] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """描述子匹配（SiLK/LightGlue）→ match 缓存 → 极线 RANSAC → epi 缓存。"""
    img0 = img0.expanduser().resolve()
    img1 = img1.expanduser().resolve()
    matcher = str(matcher).lower()
    epi_params = epipolar_cache_params(epipolar)

    if epi_cache is not None:
        epi_cached = epi_cache.load(img0, img1, matcher=matcher, **epi_params)
        if epi_cached is not None:
            meta = meta_from_epipolar_cache(epi_cached, img0, img1, matcher)
            m0 = np.asarray(epi_cached["m0"], dtype=np.float64)
            m1 = np.asarray(epi_cached["m1"], dtype=np.float64)
            return m0, m1, meta

    meta: Dict[str, Any] = {
        "img0": str(img0),
        "img1": str(img1),
        "matcher": matcher,
        "n_matches": 0,
        "epipolar_cache_hit": False,
    }

    cached = None
    if match_cache is not None:
        cached = match_cache.load(img0, img1, matcher=matcher)

    if cached is not None:
        m0 = cached["m0"]
        m1 = cached["m1"]
        meta["n_matches"] = int(cached["n_matches"])
        meta["n_keypoints0"] = int(cached["n_keypoints0"])
        meta["n_keypoints1"] = int(cached["n_keypoints1"])
        meta["match_cache_hit"] = True
        meta["match_cache_path"] = cached.get("cache_path")
        meta["from_cache"] = True
    else:
        match = descriptor_match_fn(img0, img1)
        meta["n_matches"] = int(match.get("n_matches", 0))
        meta["n_keypoints0"] = int(match.get("n_keypoints0", 0))
        meta["n_keypoints1"] = int(match.get("n_keypoints1", 0))
        meta["matcher"] = str(match.get("matcher", matcher))
        meta["from_cache"] = bool(match.get("from_cache", False))
        meta["match_cache_hit"] = False

        if not match.get("ok") and match.get("reason") in (
            "cache_miss_no_model",
            "match_failed",
            "filter_empty",
            "imread_failed",
        ):
            meta["reason"] = match.get("reason", "match_failed")
            return np.zeros((0, 2)), np.zeros((0, 2)), meta

        m0 = np.asarray(match["m0"], dtype=np.float64)
        m1 = np.asarray(match["m1"], dtype=np.float64)
        idx0 = np.asarray(match.get("idx0", np.zeros((0,), dtype=np.int64)))
        idx1 = np.asarray(match.get("idx1", np.zeros((0,), dtype=np.int64)))

        if match_cache is not None and m0.shape[0] > 0:
            cache_path = match_cache.save(
                img0,
                img1,
                matcher=matcher,
                m0=m0,
                m1=m1,
                idx0=idx0,
                idx1=idx1,
                n_keypoints0=int(meta["n_keypoints0"]),
                n_keypoints1=int(meta["n_keypoints1"]),
            )
            meta["match_cache_path"] = str(cache_path)

    m0, m1, epi_meta = filter_matches_epipolar(
        m0,
        m1,
        K,
        D,
        camera_model=epipolar.camera_model,
        ransac_threshold=epipolar.ransac_threshold,
        min_inliers=epipolar.min_inliers,
        min_matches=epipolar.min_matches,
        mask_mode="recover_pose",
    )
    meta.update(epi_meta)

    if epi_cache is not None:
        epi_path = epi_cache.save(
            img0,
            img1,
            matcher=matcher,
            m0=m0,
            m1=m1,
            n_matches_pre_epi=int(meta.get("n_matches_pre_epi", 0)),
            n_epipolar_inliers=int(meta.get("n_epipolar_inliers", 0)),
            n_used=int(meta.get("n_used", 0)),
            n_keypoints0=int(meta.get("n_keypoints0", 0)),
            n_keypoints1=int(meta.get("n_keypoints1", 0)),
            ok=bool(meta.get("ok", False)),
            reason=meta.get("reason"),
            **epi_params,
        )
        meta["epipolar_cache_path"] = str(epi_path)

    return m0, m1, meta
