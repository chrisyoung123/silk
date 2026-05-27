#!/usr/bin/env python3
"""covis 匹配管道：单文件缓存 + 安全极线（不修改 lib/matching/pair_pipeline）。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

from epipolar_safe import filter_matches_epipolar
from lib.matching.pair_pipeline import EpipolarConfig, epipolar_cache_params, meta_from_epipolar_cache
from sequence_pair_cache import SequencePairCacheStore

DescriptorMatchFn = Callable[[Path, Path], Dict[str, Any]]


def match_pair_with_epipolar(
    img0: Path,
    img1: Path,
    *,
    descriptor_match_fn: DescriptorMatchFn,
    K: np.ndarray,
    D: np.ndarray,
    epipolar: EpipolarConfig,
    matcher: str = "silk",
    pair_cache: Optional[SequencePairCacheStore] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """描述子匹配 → 单 npz 缓存 → 极线 RANSAC。"""
    img0 = img0.expanduser().resolve()
    img1 = img1.expanduser().resolve()
    matcher = str(matcher).lower()
    epi_params = epipolar_cache_params(epipolar)

    if pair_cache is not None:
        epi_cached = pair_cache.load_epipolar(img0, img1, matcher=matcher, **epi_params)
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
    if pair_cache is not None:
        cached = pair_cache.load_match(img0, img1, matcher=matcher)

    m0_raw: Optional[np.ndarray] = None
    m1_raw: Optional[np.ndarray] = None
    idx0: Optional[np.ndarray] = None
    idx1: Optional[np.ndarray] = None

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

        m0_raw = np.asarray(match["m0"], dtype=np.float64)
        m1_raw = np.asarray(match["m1"], dtype=np.float64)
        idx0 = np.asarray(match.get("idx0", np.zeros((0,), dtype=np.int64)))
        idx1 = np.asarray(match.get("idx1", np.zeros((0,), dtype=np.int64)))
        m0 = m0_raw
        m1 = m1_raw

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

    if pair_cache is not None:
        cache_path = pair_cache.save_pair(
            img0,
            img1,
            matcher=matcher,
            m0_raw=m0_raw,
            m1_raw=m1_raw,
            idx0=idx0,
            idx1=idx1,
            n_keypoints0=int(meta.get("n_keypoints0", 0)),
            n_keypoints1=int(meta.get("n_keypoints1", 0)),
            m0_epi=m0,
            m1_epi=m1,
            n_matches_pre_epi=int(meta.get("n_matches_pre_epi", 0)),
            n_epipolar_inliers=int(meta.get("n_epipolar_inliers", 0)),
            n_used=int(meta.get("n_used", 0)),
            ok=bool(meta.get("ok", False)),
            reason=meta.get("reason"),
            **epi_params,
        )
        meta["epipolar_cache_path"] = str(cache_path)
        meta["match_cache_path"] = str(cache_path)
        meta["cache_path"] = str(cache_path)

    return m0, m1, meta
