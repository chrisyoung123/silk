# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""由共视邻帧已知位姿 + 2D 匹配，经地面平面反投影构建 2D-3D，PnP 解算目标帧绝对相机位姿。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


def load_covis_adjacency(adj_json: Path) -> Tuple[Dict[str, List[str]], List[Dict[str, str]]]:
    """读取 covisibility_2d_poses.py 输出的 covis_adj.json。"""
    with adj_json.open("r", encoding="utf-8") as f:
        data = json.load(f)
    nodes = data.get("nodes") or []
    adj_raw = data.get("adjacency") or {}
    id_to_image: Dict[int, str] = {}
    for node in nodes:
        nid = int(node["id"])
        id_to_image[nid] = str(Path(node["image"]).expanduser().resolve())

    adj_by_path: Dict[str, List[str]] = {}
    for key, nbr_ids in adj_raw.items():
        src_id = int(key)
        if src_id not in id_to_image:
            continue
        src = id_to_image[src_id]
        nbrs: List[str] = []
        for j in nbr_ids:
            jid = int(j)
            if jid in id_to_image:
                nbrs.append(id_to_image[jid])
        adj_by_path[src] = sorted(set(nbrs))
    return adj_by_path, nodes


def T_world_cam_from_planar_extra(
    extra_path: Path,
    Tvc: np.ndarray,
    *,
    pose_x_key: str = "value.pose.x",
    pose_y_key: str = "value.pose.y",
    pose_theta_key: str = "value.pose.theta",
    theta_degrees: bool = True,
    pose_xy_scale: float = 0.001,
    body_z_world: float = 0.0,
    load_pose_from_extra,
    T_world_body_from_planar_pose,
) -> np.ndarray:
    """p_world = T_world_cam @ p_cam；与 eval_pairs_pose_auc 一致。"""
    x, y, th = load_pose_from_extra(
        extra_path,
        pose_x_key,
        pose_y_key,
        pose_theta_key,
        theta_degrees,
        pose_xy_scale,
    )
    T_wb = T_world_body_from_planar_pose(x, y, th, body_z_world)
    return T_wb @ np.asarray(Tvc, dtype=np.float64)


def pixel_to_cam_ray(
    uv: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
    camera_model: str,
) -> np.ndarray:
    """畸变像素 -> 相机系非归一化射线方向 (Nx3)。"""
    pts = np.asarray(uv, dtype=np.float64).reshape(-1, 1, 2)
    if camera_model == "fisheye":
        norm = cv2.fisheye.undistortPoints(pts, K, D).reshape(-1, 2)
    elif camera_model == "pinhole_plumb":
        k1, k2, k3 = [float(D[i, 0]) for i in range(3)]
        D5 = np.array([k1, k2, 0.0, 0.0, k3], dtype=np.float64).reshape(5, 1)
        norm = cv2.undistortPoints(pts, K, D5).reshape(-1, 2)
    elif camera_model == "pinhole_rational8":
        k1, k2, k3, k4 = [float(D[i, 0]) for i in range(4)]
        D8 = np.array([k1, k2, 0.0, 0.0, k3, k4, 0.0, 0.0], dtype=np.float64).reshape(8, 1)
        norm = cv2.undistortPoints(pts, K, D8).reshape(-1, 2)
    else:
        raise ValueError(f"未知 camera_model: {camera_model}")
    rays = np.concatenate([norm, np.ones((norm.shape[0], 1), dtype=np.float64)], axis=1)
    return rays


def lift_points_ground_plane(
    uv_neighbor: np.ndarray,
    T_world_cam_neighbor: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
    camera_model: str,
    ground_z: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    邻帧已知 T_world_cam，将其像素通过地面平面 z=ground_z 反投影到世界 3D。

    Returns
    -------
    X_world : (M, 3)
    valid : (N,) bool，与输入 uv_neighbor 对齐
    """
    uv = np.asarray(uv_neighbor, dtype=np.float64).reshape(-1, 2)
    n = uv.shape[0]
    valid = np.zeros(n, dtype=bool)
    if n == 0:
        return np.zeros((0, 3), dtype=np.float64), valid

    T = np.asarray(T_world_cam_neighbor, dtype=np.float64).reshape(4, 4)
    R = T[:3, :3]
    O = T[:3, 3]
    rays_cam = pixel_to_cam_ray(uv, K, D, camera_model)
    dirs_world = (R @ rays_cam.T).T

    X_out = np.zeros((n, 3), dtype=np.float64)
    for i in range(n):
        dz = float(dirs_world[i, 2])
        if abs(dz) < 1e-9:
            continue
        s = (float(ground_z) - float(O[2])) / dz
        if s <= 1e-6:
            continue
        X = O + s * dirs_world[i]
        X_out[i] = X
        valid[i] = True
    return X_out[valid], valid


def solve_pnp_absolute_pose(
    object_points_world: np.ndarray,
    image_points_target: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
    camera_model: str,
    *,
    min_points: int = 6,
    reprojection_threshold: float = 8.0,
    confidence: float = 0.999,
) -> Optional[Dict[str, Any]]:
    """OpenCV PnP RANSAC；返回 T_world_cam（p_world = T @ p_cam）及内点统计。"""
    obj = np.asarray(object_points_world, dtype=np.float64).reshape(-1, 3)
    img = np.asarray(image_points_target, dtype=np.float64).reshape(-1, 2)
    if obj.shape[0] != img.shape[0] or obj.shape[0] < min_points:
        return None

    if camera_model == "fisheye":
        ok, rvec, tvec, inliers = cv2.fisheye.solvePnPRansac(
            obj.reshape(-1, 1, 3),
            img.reshape(-1, 1, 2),
            K,
            D,
            iterationsCount=200,
            reprojectionError=float(reprojection_threshold),
            confidence=float(confidence),
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
    else:
        if camera_model == "pinhole_plumb":
            k1, k2, k3 = [float(D[i, 0]) for i in range(3)]
            dist = np.array([k1, k2, 0.0, 0.0, k3], dtype=np.float64)
        elif camera_model == "pinhole_rational8":
            k1, k2, k3, k4 = [float(D[i, 0]) for i in range(4)]
            dist = np.array([k1, k2, 0.0, 0.0, k3, k4, 0.0, 0.0], dtype=np.float64)
        else:
            raise ValueError(camera_model)
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            obj,
            img,
            K,
            dist,
            iterationsCount=200,
            reprojectionError=float(reprojection_threshold),
            confidence=float(confidence),
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

    if not ok:
        return None

    R, _ = cv2.Rodrigues(rvec)
    T_w2c = np.eye(4, dtype=np.float64)
    T_w2c[:3, :3] = np.asarray(R, dtype=np.float64).reshape(3, 3)
    T_w2c[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)
    T_w_c = np.linalg.inv(T_w2c)

    n_inl = int(inliers.size) if inliers is not None else 0
    if n_inl < min_points:
        return None

    return {
        "T_world_cam": T_w_c,
        "T_world_to_cam": T_w2c,
        "n_inliers": n_inl,
        "n_points": int(obj.shape[0]),
        "inlier_indices": inliers.reshape(-1).tolist() if inliers is not None else [],
    }


def rotation_error_deg(R_est: np.ndarray, R_gt: np.ndarray) -> float:
    trace = float(np.trace(R_est @ R_gt.T))
    trace = float(np.clip((trace - 1.0) / 2.0, -1.0, 1.0))
    return float(np.rad2deg(np.arccos(trace)))


def position_xy_error(T_est: np.ndarray, T_gt: np.ndarray) -> float:
    d = np.asarray(T_est[:3, 3], dtype=np.float64) - np.asarray(T_gt[:3, 3], dtype=np.float64)
    return float(np.linalg.norm(d[:2]))


def T_to_serializable(T: np.ndarray) -> List[List[float]]:
    return np.asarray(T, dtype=np.float64).reshape(4, 4).tolist()
