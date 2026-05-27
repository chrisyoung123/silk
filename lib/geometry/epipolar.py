# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""极线几何：去畸变 + EssentialMat RANSAC + recoverPose inlier 过滤。"""

from __future__ import annotations

from typing import Any, Dict, Literal, Optional, Tuple

import cv2
import numpy as np

MaskMode = Literal["recover_pose", "essential"]


def undistort_normalized(
    pts_dist: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
    camera_model: str,
) -> np.ndarray:
    """将畸变像素点转为归一化平面坐标 (Nx2)，供 findEssentialMat 使用。"""
    pts = np.asarray(pts_dist, dtype=np.float64).reshape(-1, 1, 2)
    k1, k2, k3, k4 = [float(D[i, 0]) for i in range(4)]

    if camera_model == "fisheye":
        return cv2.fisheye.undistortPoints(pts, K, D).reshape(-1, 2)

    if camera_model == "pinhole_plumb":
        D5 = np.array([k1, k2, 0.0, 0.0, k3], dtype=np.float64).reshape(5, 1)
        return cv2.undistortPoints(pts, K, D5).reshape(-1, 2)

    if camera_model == "pinhole_rational8":
        D8 = np.array([k1, k2, 0.0, 0.0, k3, k4, 0.0, 0.0], dtype=np.float64).reshape(8, 1)
        return cv2.undistortPoints(pts, K, D8).reshape(-1, 2)

    raise ValueError(f"未知 camera_model: {camera_model}")


def epipolar_essential_mask(
    m0: np.ndarray,
    m1: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
    camera_model: str,
    ransac_threshold_px: float,
) -> Optional[np.ndarray]:
    """findEssentialMat RANSAC 内点 mask（长度 N 的 bool）；失败返回 None。"""
    m0 = np.asarray(m0, dtype=np.float64).reshape(-1, 2)
    m1 = np.asarray(m1, dtype=np.float64).reshape(-1, 2)
    if m0.shape[0] < 5:
        return None
    pts0 = undistort_normalized(m0, K, D, camera_model)
    pts1 = undistort_normalized(m1, K, D, camera_model)
    avg_focal = (float(K[0, 0]) + float(K[1, 1])) / 2.0
    norm_thresh = float(ransac_threshold_px) / max(avg_focal, 1e-6)
    E, mask = cv2.findEssentialMat(
        pts0,
        pts1,
        cameraMatrix=np.eye(3),
        method=cv2.RANSAC,
        prob=0.999,
        threshold=norm_thresh,
    )
    if E is None or mask is None:
        return None
    return (mask.ravel() > 0)


def estimate_pose_recover_pose(
    pts0_dist: np.ndarray,
    pts1_dist: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
    camera_model: str,
    pixel_threshold: float,
    min_inliers: int,
) -> Optional[Tuple[np.ndarray, np.ndarray, int, int, np.ndarray]]:
    """去畸变 + Essential + recoverPose。

    成功返回 (R, t, n_epipolar_inliers, n_recover_pose_inliers, recover_inlier_mask)。
    recover_inlier_mask 与输入 pts 行对齐。
    """
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
    if E is None:
        return None

    if E.ndim == 2:
        E_candidates = [E]
    else:
        E_candidates = [E[3 * i : 3 * (i + 1), :] for i in range(E.shape[0] // 3)]

    best_n_rec = -1
    best_n_epi = int(np.count_nonzero(mask)) if mask is not None else 0
    best_Rt = None
    best_rec_mask: Optional[np.ndarray] = None
    for Ei in E_candidates:
        mask_in = mask.copy() if mask is not None else None
        n_rec, R, t, rec_mask = cv2.recoverPose(
            Ei,
            pts0,
            pts1,
            cameraMatrix=np.eye(3),
            mask=mask_in,
        )
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
    """在匹配点上做极线 RANSAC，返回过滤后的 m0/m1 与统计 meta。"""
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

    meta["n_used"] = int(m0.shape[0])
    meta["ok"] = m0.shape[0] >= min_matches
    if not meta["ok"]:
        meta["reason"] = "too_few_after_filter"
    else:
        meta.pop("reason", None)
    return m0, m1, meta
