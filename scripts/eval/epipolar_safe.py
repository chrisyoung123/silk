#!/usr/bin/env python3
"""极线过滤（recoverPose），兼容 OpenCV 4.x 多解 Essential 矩阵形状。"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Tuple

import cv2
import numpy as np

from lib.geometry.epipolar import (
    MaskMode,
    epipolar_essential_mask,
    undistort_normalized,
)

MaskMode = Literal["recover_pose", "essential"]


def _essential_matrix_candidates(E: np.ndarray) -> List[np.ndarray]:
    """Normalize findEssentialMat outputs to a list of 3x3 matrices."""
    if E is None:
        return []

    E = np.asarray(E, dtype=np.float64)
    if E.size == 0:
        return []

    if E.ndim == 1 and E.size == 9:
        E = E.reshape(3, 3)

    if E.ndim != 2:
        return []

    rows, cols = E.shape
    if rows == 3 and cols == 3:
        return [E]
    if rows % 3 == 0 and cols == 3:
        return [E[i * 3 : (i + 1) * 3, :] for i in range(rows // 3)]
    if rows == 3 and cols % 3 == 0:
        return [E[:, i * 3 : (i + 1) * 3] for i in range(cols // 3)]

    return []


def estimate_pose_recover_pose(
    pts0_dist: np.ndarray,
    pts1_dist: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
    camera_model: str,
    pixel_threshold: float,
    min_inliers: int,
) -> Optional[Tuple[np.ndarray, np.ndarray, int, int, np.ndarray]]:
    """去畸变 + Essential + recoverPose（安全处理 E 的多形状输出）。"""
    pts0_dist = np.asarray(pts0_dist, dtype=np.float64).reshape(-1, 2)
    pts1_dist = np.asarray(pts1_dist, dtype=np.float64).reshape(-1, 2)
    if pts0_dist.shape[0] < 5:
        return None
    if pts0_dist.shape != pts1_dist.shape:
        raise ValueError("matches 应为 Nx2 且行数一致")

    pts0 = undistort_normalized(pts0_dist, K, D, camera_model)
    pts1 = undistort_normalized(pts1_dist, K, D, camera_model)

    avg_focal = (float(K[0, 0]) + float(K[1, 1])) / 2.0
    norm_thresh = float(pixel_threshold) / max(avg_focal, 1e-6)

    E, mask = cv2.findEssentialMat(
        pts0,
        pts1,
        cameraMatrix=np.eye(3),
        method=cv2.RANSAC,
        prob=0.999,
        threshold=norm_thresh,
    )
    E_candidates = _essential_matrix_candidates(E)
    if not E_candidates:
        return None

    best_n_rec = -1
    best_n_epi = int(np.count_nonzero(mask)) if mask is not None else 0
    best_Rt = None
    best_rec_mask: Optional[np.ndarray] = None
    for Ei in E_candidates:
        if Ei.shape != (3, 3):
            continue
        mask_in = mask.copy() if mask is not None else None
        try:
            n_rec, R, t, rec_mask = cv2.recoverPose(
                Ei,
                pts0,
                pts1,
                cameraMatrix=np.eye(3),
                mask=mask_in,
            )
        except cv2.error:
            continue
        n_rec = int(n_rec)
        if n_rec > best_n_rec:
            best_n_rec = n_rec
            best_Rt = (R, t)
            best_rec_mask = rec_mask

    if best_Rt is None:
        return None
    R, t = best_Rt
    if best_n_epi < min_inliers:
        return None
    if not (np.all(np.isfinite(R)) and np.all(np.isfinite(t))):
        return None
    if best_rec_mask is None:
        rec_bool = np.zeros(pts0.shape[0], dtype=bool)
    else:
        rec_bool = np.asarray(best_rec_mask, dtype=np.uint8).ravel() > 0
    return R, t, best_n_epi, best_n_rec, rec_bool


def filter_matches_epipolar(
    m0: np.ndarray,
    m1: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
    *,
    camera_model: str,
    ransac_threshold: float,
    min_inliers: int,
    min_matches: int,
    mask_mode: MaskMode = "recover_pose",
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """极线 RANSAC；recoverPose 失败时回退到 essential mask。"""
    m0 = np.asarray(m0, dtype=np.float64).reshape(-1, 2)
    m1 = np.asarray(m1, dtype=np.float64).reshape(-1, 2)
    meta: Dict[str, Any] = {
        "n_matches_pre_epi": int(m0.shape[0]),
        "ok_pre_epi": bool(m0.shape[0] >= min_matches),
    }

    if m0.shape[0] == 0:
        meta["n_used"] = 0
        meta["ok"] = False
        meta.setdefault("reason", "match_failed")
        return m0, m1, meta

    if mask_mode == "essential":
        mask = epipolar_essential_mask(
            m0, m1, K, D, camera_model, ransac_threshold
        )
        if mask is not None:
            meta["n_epipolar_inliers"] = int(np.count_nonzero(mask))
            if int(np.count_nonzero(mask)) >= min_inliers:
                m0 = m0[mask]
                m1 = m1[mask]
        meta["n_used"] = int(m0.shape[0])
        meta["ok"] = m0.shape[0] >= min_matches
        if not meta["ok"]:
            meta["reason"] = "too_few_after_filter"
        else:
            meta.pop("reason", None)
        return m0, m1, meta

    est = estimate_pose_recover_pose(
        m0, m1, K, D, camera_model, ransac_threshold, min_inliers
    )
    if est is not None:
        _, _, n_epi, _, mask = est
        meta["n_epipolar_inliers"] = int(n_epi)
        if mask is not None and int(np.count_nonzero(mask)) >= min_inliers:
            m0 = m0[mask]
            m1 = m1[mask]
    else:
        mask = epipolar_essential_mask(
            m0, m1, K, D, camera_model, ransac_threshold
        )
        if mask is not None:
            meta["n_epipolar_inliers"] = int(np.count_nonzero(mask))
            meta["epipolar_fallback"] = "essential"
            if int(np.count_nonzero(mask)) >= min_inliers:
                m0 = m0[mask]
                m1 = m1[mask]

    meta["n_used"] = int(m0.shape[0])
    meta["ok"] = m0.shape[0] >= min_matches
    if not meta["ok"]:
        meta["reason"] = "too_few_after_filter"
    else:
        meta.pop("reason", None)
    return m0, m1, meta
