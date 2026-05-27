# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Adaptive Non-Maximal Suppression (ANMS) for spatially diverse keypoints."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def adaptive_nms(
    xy: np.ndarray,
    scores: Optional[np.ndarray] = None,
    *,
    top_k: int = 2000,
    min_radius: float = 0.0,
) -> np.ndarray:
    """
    Efficient ANMS：按 score 排序，保留 suppression radius 最大的 top_k 点。

    无 score 时视为均匀 score，仍可获得空间均匀分布（LocalBA 共面退化缓解）。

    Returns
    -------
    keep : (N,) bool
    """
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    n = xy.shape[0]
    if n == 0:
        return np.zeros(0, dtype=bool)
    if top_k <= 0 or n <= top_k:
        return np.ones(n, dtype=bool)

    if scores is None:
        sc = np.ones(n, dtype=np.float64)
    else:
        sc = np.asarray(scores, dtype=np.float64).reshape(-1)
        if sc.shape[0] != n:
            raise ValueError(f"scores length {sc.shape[0]} != xy rows {n}")

    order = np.argsort(-sc)
    xy_s = xy[order]
    radii = np.empty(n, dtype=np.float64)
    radii[order[0]] = np.inf
    for rank in range(1, n):
        i = order[rank]
        d = np.linalg.norm(xy_s[:rank] - xy_s[rank], axis=1)
        radii[i] = float(np.min(d)) if d.size else np.inf

    if min_radius > 0:
        valid = radii >= float(min_radius)
        if not np.any(valid):
            valid = radii >= float(np.max(radii[np.isfinite(radii)]) * 0.5)
    else:
        valid = np.isfinite(radii)

    cand_idx = np.nonzero(valid)[0]
    cand_r = radii[cand_idx]
    pick_local = np.argsort(-cand_r)[:top_k]
    pick = cand_idx[pick_local]

    keep = np.zeros(n, dtype=bool)
    keep[pick] = True
    return keep


def filter_keypoints_anms(
    positions_xy: np.ndarray,
    descriptors: np.ndarray,
    scores: Optional[np.ndarray] = None,
    *,
    top_k: int = 2000,
    min_radius: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """对关键点/描述子应用 ANMS，返回 (pos, desc, keep_mask)。"""
    pos = np.asarray(positions_xy, dtype=np.float64).reshape(-1, 2)
    desc = np.asarray(descriptors)
    keep = adaptive_nms(pos, scores, top_k=top_k, min_radius=min_radius)
    return pos[keep], desc[keep], keep
