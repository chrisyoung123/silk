# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
LocalBA.md 管道：两已知参考相机三角化 → PnP 初始化第三相机 → 平面车体局部 BA。

约定与 eval_pairs_pose_auc 一致：
  p_body = Tvc @ p_cam
  p_world = T_world_body @ p_body = T_world_cam @ p_cam
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

try:
    from scipy.optimize import least_squares
except ImportError:  # pragma: no cover
    least_squares = None


@dataclass
class PlanarBody:
    x: float
    y: float
    theta: float

    def as_vec(self) -> np.ndarray:
        return np.array([self.x, self.y, self.theta], dtype=np.float64)


@dataclass
class RefBodyState:
    """参考帧 BA 状态：平面 x,y,θ + 小扰动 dz, roll, pitch（body 系）。"""

    x: float
    y: float
    theta: float
    dz: float = 0.0
    roll: float = 0.0
    pitch: float = 0.0

    def as_vec(self) -> np.ndarray:
        return np.array(
            [self.x, self.y, self.theta, self.dz, self.roll, self.pitch],
            dtype=np.float64,
        )

    @classmethod
    def from_planar(cls, body: PlanarBody) -> "RefBodyState":
        return cls(x=body.x, y=body.y, theta=body.theta)

    def to_dict(self) -> Dict[str, float]:
        return {
            "x": self.x,
            "y": self.y,
            "theta": self.theta,
            "dz": self.dz,
            "roll": self.roll,
            "pitch": self.pitch,
            "roll_deg": float(math.degrees(self.roll)),
            "pitch_deg": float(math.degrees(self.pitch)),
        }


@dataclass
class VisualTrack:
    """一条三视图（或两视图）轨迹：世界 3D 点 + 各图像观测。"""

    X_world: np.ndarray
    observations: Dict[str, np.ndarray] = field(default_factory=dict)


def T_world_body_from_planar(x: float, y: float, theta: float, z_world: float = 0.0) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    T = np.eye(4, dtype=np.float64)
    T[0, 0], T[0, 1] = c, -s
    T[1, 0], T[1, 1] = s, c
    T[0, 3], T[1, 3], T[2, 3] = float(x), float(y), float(z_world)
    return T


def _R_body_roll_pitch(roll: float, pitch: float) -> np.ndarray:
    """body 系小角度 roll(X)、pitch(Y) 旋转矩阵。"""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float64)
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float64)
    return ry @ rx


def T_world_body_from_ref_state(ref: RefBodyState, z_world: float = 0.0) -> np.ndarray:
    T_planar = T_world_body_from_planar(ref.x, ref.y, ref.theta, z_world + ref.dz)
    if abs(ref.roll) < 1e-12 and abs(ref.pitch) < 1e-12:
        return T_planar
    R_pert = _R_body_roll_pitch(ref.roll, ref.pitch)
    T_pert = np.eye(4, dtype=np.float64)
    T_pert[:3, :3] = R_pert
    return T_planar @ T_pert


def T_world_cam_from_ref_body(ref: RefBodyState, Tvc: np.ndarray, z_world: float = 0.0) -> np.ndarray:
    return T_world_body_from_ref_state(ref, z_world) @ np.asarray(Tvc, dtype=np.float64)


def T_world_cam_from_planar_body(
    body: PlanarBody,
    Tvc: np.ndarray,
    z_world: float = 0.0,
) -> np.ndarray:
    return T_world_body_from_planar(body.x, body.y, body.theta, z_world) @ np.asarray(
        Tvc, dtype=np.float64
    )


def planar_body_from_T_world_cam(
    T_world_cam: np.ndarray,
    Tvc: np.ndarray,
    z_world: float = 0.0,
) -> PlanarBody:
    Tvc = np.asarray(Tvc, dtype=np.float64).reshape(4, 4)
    T_wb = np.asarray(T_world_cam, dtype=np.float64) @ np.linalg.inv(Tvc)
    x, y = float(T_wb[0, 3]), float(T_wb[1, 3])
    theta = float(math.atan2(T_wb[1, 0], T_wb[0, 0]))
    _ = z_world
    return PlanarBody(x=x, y=y, theta=theta)


def body_baseline_m(b0: PlanarBody, b1: PlanarBody) -> float:
    return float(math.hypot(b0.x - b1.x, b0.y - b1.y))


def projection_matrix_normalized(T_world_cam: np.ndarray) -> np.ndarray:
    """世界坐标 -> 相机归一化平面 的 3x4 投影矩阵（不含 K，配合去畸变像素）。"""
    T_c_w = np.linalg.inv(np.asarray(T_world_cam, dtype=np.float64).reshape(4, 4))
    return T_c_w[:3, :].copy()


def _undistort_pixels_normalized(
    uv: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
    camera_model: str,
) -> np.ndarray:
    """畸变像素 -> 归一化平面坐标 (Nx2)，与 PnP/BA 使用的相机模型一致。"""
    from lib.geometry.absolute_pose_from_covis import pixel_to_cam_ray

    rays = pixel_to_cam_ray(np.asarray(uv, dtype=np.float64).reshape(-1, 2), K, D, camera_model)
    return rays[:, :2]


def triangulate_points_dlt(
    T_w_c0: np.ndarray,
    T_w_c1: np.ndarray,
    uv0: np.ndarray,
    uv1: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
    camera_model: str,
) -> np.ndarray:
    """DLT 三角化：畸变像素先去畸变，再在归一化平面三角化。返回 Nx3 世界坐标。"""
    n = uv0.shape[0]
    if n == 0:
        return np.zeros((0, 3), dtype=np.float64)
    P0 = projection_matrix_normalized(T_w_c0)
    P1 = projection_matrix_normalized(T_w_c1)
    pts0 = _undistort_pixels_normalized(uv0, K, D, camera_model).T.reshape(2, -1)
    pts1 = _undistort_pixels_normalized(uv1, K, D, camera_model).T.reshape(2, -1)
    hom = cv2.triangulatePoints(P0, P1, pts0, pts1)
    hom = hom / hom[3:4, :]
    return hom[:3, :].T.reshape(n, 3)


def depth_in_cam(T_world_cam: np.ndarray, X_world: np.ndarray) -> np.ndarray:
    T_c_w = np.linalg.inv(np.asarray(T_world_cam, dtype=np.float64).reshape(4, 4))
    X = np.asarray(X_world, dtype=np.float64).reshape(-1, 3)
    ones = np.ones((X.shape[0], 1), dtype=np.float64)
    Xc = (T_c_w @ np.concatenate([X, ones], axis=1).T).T[:, :3]
    return Xc[:, 2]


def filter_triangulated_points(
    X_world: np.ndarray,
    T_refs: Sequence[np.ndarray],
    *,
    depth_min: float = 0.5,
    depth_max: float = 10.0,
) -> np.ndarray:
    """LocalBA 情况 B：过滤无效/远景深度。"""
    X = np.asarray(X_world, dtype=np.float64).reshape(-1, 3)
    valid = np.ones(X.shape[0], dtype=bool)
    for T in T_refs:
        z = depth_in_cam(T, X)
        valid &= np.isfinite(z) & (z > 1e-3) & (z >= depth_min) & (z <= depth_max)
    return valid


def project_world_to_pixel(
    X_world: np.ndarray,
    T_world_cam: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
    camera_model: str,
) -> np.ndarray:
    X = np.asarray(X_world, dtype=np.float64).reshape(-1, 3)
    T_c_w = np.linalg.inv(np.asarray(T_world_cam, dtype=np.float64).reshape(4, 4))
    R = T_c_w[:3, :3]
    t = T_c_w[:3, 3].reshape(3, 1)
    rvec, _ = cv2.Rodrigues(R)
    rvec = np.ascontiguousarray(rvec.reshape(3, 1), dtype=np.float64)
    tvec = np.ascontiguousarray(t.reshape(3, 1), dtype=np.float64)
    obj = X.reshape(-1, 1, 3)
    if camera_model == "fisheye":
        uv, _ = cv2.fisheye.projectPoints(obj, rvec, tvec, K, D)
    else:
        if camera_model == "pinhole_plumb":
            k1, k2, k3 = [float(D[i, 0]) for i in range(3)]
            dist = np.array([k1, k2, 0.0, 0.0, k3], dtype=np.float64)
        elif camera_model == "pinhole_rational8":
            k1, k2, k3, k4 = [float(D[i, 0]) for i in range(4)]
            dist = np.array([k1, k2, 0.0, 0.0, k3, k4, 0.0, 0.0], dtype=np.float64)
        else:
            raise ValueError(camera_model)
        uv, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    return uv.reshape(-1, 2)


def _nearest_indices(anchor: np.ndarray, pool: np.ndarray, max_dist_px: float) -> Tuple[np.ndarray, np.ndarray]:
    """对每个 anchor[i]，在 pool 中找最近邻；返回 (idx, dist)，无效为 -1 / inf。"""
    a = np.asarray(anchor, dtype=np.float64).reshape(-1, 2)
    p = np.asarray(pool, dtype=np.float64).reshape(-1, 2)
    n = a.shape[0]
    idx = np.full(n, -1, dtype=np.int64)
    dist = np.full(n, np.inf, dtype=np.float64)
    if p.shape[0] == 0:
        return idx, dist
    for i in range(n):
        d = np.linalg.norm(p - a[i], axis=1)
        j = int(np.argmin(d))
        if d[j] <= max_dist_px:
            idx[i] = j
            dist[i] = float(d[j])
    return idx, dist


def build_triview_tracks(
    ref0_key: str,
    ref1_key: str,
    target_key: str,
    uv12_0: np.ndarray,
    uv12_1: np.ndarray,
    uv13_0: np.ndarray,
    uv13_t: np.ndarray,
    uv23_1: np.ndarray,
    uv23_t: np.ndarray,
    T_w_c0: np.ndarray,
    T_w_c1: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
    camera_model: str,
    *,
    associate_px: float = 3.0,
    c23_agree_px: float = 4.0,
    depth_min: float = 0.5,
    depth_max: float = 10.0,
) -> Tuple[List[VisualTrack], Dict[str, Any]]:
    """
    LocalBA 三视图轨迹：C1–C2 三角化 + C1–C3 / C2–C3 匹配交叉验证 u3。

    Returns
    -------
    tracks, stats  stats 含 n_triangulated, n_c13, n_c23_agree 等。
    """
    stats: Dict[str, Any] = {
        "n_match_c12": int(uv12_0.shape[0]),
        "n_match_c13": int(uv13_0.shape[0]),
        "n_match_c23": int(uv23_1.shape[0]),
        "n_triangulated": 0,
        "n_depth_ok": 0,
        "n_c13_assoc": 0,
        "n_c23_assoc": 0,
        "n_c23_agree": 0,
        "n_tracks": 0,
        "mean_c13_assoc_dist_px": float("nan"),
        "mean_c23_assoc_dist_px": float("nan"),
        "mean_c23_agree_dist_px": float("nan"),
    }
    c13_dists: List[float] = []
    c23_dists: List[float] = []
    c23_agree_dists: List[float] = []
    if uv12_0.shape[0] == 0:
        return [], stats

    X = triangulate_points_dlt(T_w_c0, T_w_c1, uv12_0, uv12_1, K, D, camera_model)
    stats["n_triangulated"] = int(X.shape[0])
    valid = filter_triangulated_points(X, [T_w_c0, T_w_c1], depth_min=depth_min, depth_max=depth_max)
    stats["n_depth_ok"] = int(np.count_nonzero(valid))
    if not np.any(valid):
        return [], stats

    X = X[valid]
    u0 = uv12_0[valid]
    u1 = uv12_1[valid]

    j13, d13 = _nearest_indices(u0, uv13_0, associate_px)
    k23, d23 = _nearest_indices(u1, uv23_1, associate_px)

    tracks: List[VisualTrack] = []
    for i in range(X.shape[0]):
        if j13[i] < 0 or k23[i] < 0:
            continue
        stats["n_c13_assoc"] += 1
        stats["n_c23_assoc"] += 1
        c13_dists.append(float(d13[i]))
        c23_dists.append(float(d23[i]))
        u3_c1 = uv13_t[j13[i]]
        u3_c2 = uv23_t[k23[i]]
        agree_dist = float(np.linalg.norm(u3_c1 - u3_c2))
        if agree_dist > c23_agree_px:
            continue
        stats["n_c23_agree"] += 1
        c23_agree_dists.append(agree_dist)
        u3 = 0.5 * (u3_c1 + u3_c2)
        tracks.append(
            VisualTrack(
                X_world=X[i].copy(),
                observations={
                    ref0_key: u0[i].copy(),
                    ref1_key: u1[i].copy(),
                    target_key: u3.copy(),
                },
            )
        )
    stats["n_tracks"] = len(tracks)
    if c13_dists:
        stats["mean_c13_assoc_dist_px"] = float(np.mean(c13_dists))
    if c23_dists:
        stats["mean_c23_assoc_dist_px"] = float(np.mean(c23_dists))
    if c23_agree_dists:
        stats["mean_c23_agree_dist_px"] = float(np.mean(c23_agree_dists))
    return tracks, stats


def triangulate_multiview_svd(
    cam_keys: Sequence[str],
    observations: Dict[str, np.ndarray],
    T_world_cams: Dict[str, np.ndarray],
    K: np.ndarray,
    D: np.ndarray,
    camera_model: str,
) -> Optional[np.ndarray]:
    """多视图 SVD 三角化（Multicamera.md §1）；观测像素先去畸变到归一化平面。"""
    rows: List[np.ndarray] = []
    for ck in cam_keys:
        if ck not in observations or ck not in T_world_cams:
            continue
        uv = np.asarray(observations[ck], dtype=np.float64).reshape(1, 2)
        uv_norm = _undistort_pixels_normalized(uv, K, D, camera_model).reshape(2)
        P = projection_matrix_normalized(T_world_cams[ck])
        u, v = float(uv_norm[0]), float(uv_norm[1])
        rows.append(u * P[2, :] - P[0, :])
        rows.append(v * P[2, :] - P[1, :])
    if len(rows) < 4:
        return None
    A = np.stack(rows, axis=0)
    _, _, vh = np.linalg.svd(A)
    Xh = vh[-1]
    if abs(float(Xh[3])) < 1e-9:
        return None
    X = (Xh[:3] / Xh[3]).astype(np.float64)
    return X


def refine_tracks_multiview(
    tracks: List[VisualTrack],
    cam_keys: Sequence[str],
    T_world_cams: Dict[str, np.ndarray],
    K: np.ndarray,
    D: np.ndarray,
    camera_model: str,
    *,
    depth_min: float = 0.5,
    depth_max: float = 10.0,
    max_reproj_px: float = 6.0,
) -> Tuple[List[VisualTrack], Dict[str, int]]:
    """用多视图三角化精炼已有轨迹的 X_world。"""
    stats = {"n_refined": 0, "n_rejected": 0}
    out: List[VisualTrack] = []
    ref_keys = [k for k in cam_keys if k in T_world_cams]
    for tr in tracks:
        obs_keys = [k for k in tr.observations if k in T_world_cams]
        if len(obs_keys) < 2:
            out.append(tr)
            continue
        X = triangulate_multiview_svd(
            obs_keys, tr.observations, T_world_cams, K, D, camera_model
        )
        if X is None:
            stats["n_rejected"] += 1
            out.append(tr)
            continue
        valid = filter_triangulated_points(
            X.reshape(1, 3),
            [T_world_cams[k] for k in obs_keys],
            depth_min=depth_min,
            depth_max=depth_max,
        )
        if not bool(valid[0]):
            stats["n_rejected"] += 1
            out.append(tr)
            continue
        ok = True
        for ck in obs_keys:
            uv_proj = project_world_to_pixel(X, T_world_cams[ck], K, D, camera_model)[0]
            if float(np.linalg.norm(tr.observations[ck] - uv_proj)) > max_reproj_px:
                ok = False
                break
        if not ok:
            stats["n_rejected"] += 1
            out.append(tr)
            continue
        out.append(VisualTrack(X_world=X.copy(), observations=dict(tr.observations)))
        stats["n_refined"] += 1
    return out, stats


def augment_tracks_with_extra_refs(
    tracks: List[VisualTrack],
    bridge_ref: str,
    target_key: str,
    extra_ref_keys: Sequence[str],
    match_bridge_extra: Dict[str, Tuple[np.ndarray, np.ndarray]],
    match_extra_target: Dict[str, Tuple[np.ndarray, np.ndarray]],
    *,
    associate_px: float = 3.0,
) -> Tuple[List[VisualTrack], Dict[str, int]]:
    """
    将额外已知相机观测加入轨迹（经 bridge_ref 关联）。
    match_bridge_extra[r] = (uv on bridge, uv on r)
    match_extra_target[r] = (uv on r, uv on target)
    """
    stats = {
        "n_extra_obs_added": 0,
        "n_tracks_augmented": 0,
        "mean_associate_dist_px": float("nan"),
    }
    assoc_dists: List[float] = []
    if not tracks or not extra_ref_keys:
        return tracks, stats

    augmented: List[VisualTrack] = []
    for tr in tracks:
        obs = dict(tr.observations)
        u_bridge = obs.get(bridge_ref)
        if u_bridge is None:
            augmented.append(tr)
            continue
        n_added = 0
        for rk in extra_ref_keys:
            if rk in obs:
                continue
            pair_be = match_bridge_extra.get(rk)
            pair_et = match_extra_target.get(rk)
            if pair_be is None or pair_et is None:
                continue
            uv_b_on_bridge, uv_on_r_from_bridge = pair_be
            uv_on_r_from_target, _uv_t_from_et = pair_et
            j, _ = _nearest_indices(u_bridge.reshape(1, 2), uv_b_on_bridge, associate_px)
            if j[0] < 0:
                continue
            u_r_expected = uv_on_r_from_bridge[j[0]]
            k, dist = _nearest_indices(u_r_expected.reshape(1, 2), uv_on_r_from_target, associate_px)
            if k[0] < 0:
                continue
            assoc_dists.append(float(dist[0]))
            obs[rk] = uv_on_r_from_target[k[0]].copy()
            n_added += 1
        if n_added > 0:
            stats["n_tracks_augmented"] += 1
            stats["n_extra_obs_added"] += n_added
        augmented.append(VisualTrack(X_world=tr.X_world.copy(), observations=obs))
    if assoc_dists:
        stats["mean_associate_dist_px"] = float(np.mean(assoc_dists))
    return augmented, stats


def _copy_tracks(tracks: Sequence[VisualTrack]) -> List[VisualTrack]:
    return [
        VisualTrack(
            X_world=tr.X_world.copy(),
            observations={k: v.copy() for k, v in tr.observations.items()},
        )
        for tr in tracks
    ]


def filter_tracks_by_reprojection(
    tracks: Sequence[VisualTrack],
    cam_keys: Sequence[str],
    T_world_cams: Dict[str, np.ndarray],
    K: np.ndarray,
    D: np.ndarray,
    camera_model: str,
    *,
    max_reproj_px: float,
    mode: str = "track",
    target_key: Optional[str] = None,
    min_observations_per_track: int = 1,
) -> Tuple[List[VisualTrack], Dict[str, Any]]:
    """
    按重投影误差过滤 track 或单条观测。

    mode='track': 任一观测超阈值则丢弃整条 track。
    mode='observation': 剔除超阈值观测；剩余观测不足则丢弃 track。
    mode='target_observation': 仅剔除 target 相机上的超阈值观测。
    """
    stats: Dict[str, Any] = {
        "mode": mode,
        "max_reproj_px": float(max_reproj_px),
        "n_tracks_in": len(tracks),
        "n_obs_in": sum(len(tr.observations) for tr in tracks),
        "n_tracks_removed": 0,
        "n_obs_removed": 0,
        "n_tracks_out": 0,
        "removed_track_indices": [],
        "removed_observations": [],
    }
    if max_reproj_px <= 0:
        return _copy_tracks(tracks), stats

    out: List[VisualTrack] = []
    for ti, tr in enumerate(tracks):
        obs_errors: List[Tuple[str, float]] = []
        for ck in cam_keys:
            if ck not in tr.observations:
                continue
            uv_proj = project_world_to_pixel(tr.X_world, T_world_cams[ck], K, D, camera_model)[0]
            err = float(np.linalg.norm(tr.observations[ck] - uv_proj))
            obs_errors.append((ck, err))

        if mode == "track":
            bad = [(ck, err) for ck, err in obs_errors if err > max_reproj_px]
            if bad:
                stats["n_tracks_removed"] += 1
                stats["removed_track_indices"].append(ti)
                for ck, err in bad:
                    stats["removed_observations"].append(
                        {"track_idx": ti, "cam_key": ck, "reproj_px": err}
                    )
                continue
            out.append(
                VisualTrack(
                    X_world=tr.X_world.copy(),
                    observations={k: v.copy() for k, v in tr.observations.items()},
                )
            )
            continue

        keep_obs: Dict[str, np.ndarray] = {}
        for ck, err in obs_errors:
            drop = err > max_reproj_px
            if mode == "target_observation":
                drop = drop and target_key is not None and ck == target_key
            if drop:
                stats["n_obs_removed"] += 1
                stats["removed_observations"].append(
                    {"track_idx": ti, "cam_key": ck, "reproj_px": err}
                )
            else:
                keep_obs[ck] = tr.observations[ck].copy()
        if len(keep_obs) >= min_observations_per_track:
            out.append(VisualTrack(X_world=tr.X_world.copy(), observations=keep_obs))
        else:
            stats["n_tracks_removed"] += 1
            stats["removed_track_indices"].append(ti)

    stats["n_tracks_out"] = len(out)
    stats["n_obs_out"] = sum(len(tr.observations) for tr in out)
    return out, stats


def _per_camera_reproj_rmse(
    tracks: Sequence[VisualTrack],
    cam_keys: Sequence[str],
    T_world_cams: Dict[str, np.ndarray],
    K: np.ndarray,
    D: np.ndarray,
    camera_model: str,
) -> Dict[str, float]:
    out: Dict[str, List[float]] = {k: [] for k in cam_keys}
    for tr in tracks:
        for ck in cam_keys:
            if ck not in tr.observations:
                continue
            uv_proj = project_world_to_pixel(tr.X_world, T_world_cams[ck], K, D, camera_model)[0]
            err = float(np.linalg.norm(tr.observations[ck] - uv_proj))
            out[ck].append(err)
    return {
        k: float(np.sqrt(np.mean(np.square(v)))) if v else float("nan")
        for k, v in out.items()
    }


def build_tracks_ref_pair_target(
    ref0_key: str,
    ref1_key: str,
    target_key: str,
    uv_ref0: np.ndarray,
    uv_ref1: np.ndarray,
    uv_tgt_on_ref0: np.ndarray,
    uv_tgt: np.ndarray,
    T_w_c0: np.ndarray,
    T_w_c1: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
    camera_model: str,
    *,
    associate_px: float = 3.0,
    depth_min: float = 0.5,
    depth_max: float = 10.0,
) -> Tuple[List[VisualTrack], Dict[str, Any]]:
    """仅 C1 桥接（无 C2–C3 校验）；C2–C3 匹配不足时 fallback。"""
    stats: Dict[str, Any] = {
        "n_match_c12": int(uv_ref0.shape[0]),
        "n_match_c13": int(uv_tgt_on_ref0.shape[0]),
        "n_triangulated": 0,
        "n_depth_ok": 0,
        "n_c13_assoc": 0,
        "n_tracks": 0,
        "mean_c13_assoc_dist_px": float("nan"),
    }
    c13_dists: List[float] = []
    if uv_ref0.shape[0] == 0:
        return [], stats

    X = triangulate_points_dlt(T_w_c0, T_w_c1, uv_ref0, uv_ref1, K, D, camera_model)
    stats["n_triangulated"] = int(X.shape[0])
    valid = filter_triangulated_points(X, [T_w_c0, T_w_c1], depth_min=depth_min, depth_max=depth_max)
    stats["n_depth_ok"] = int(np.count_nonzero(valid))
    if not np.any(valid):
        return [], stats

    X = X[valid]
    u0 = uv_ref0[valid]
    u1 = uv_ref1[valid]
    j13, d13 = _nearest_indices(u0, uv_tgt_on_ref0, associate_px)

    tracks: List[VisualTrack] = []
    for i in range(X.shape[0]):
        if j13[i] < 0:
            continue
        stats["n_c13_assoc"] += 1
        c13_dists.append(float(d13[i]))
        tracks.append(
            VisualTrack(
                X_world=X[i].copy(),
                observations={
                    ref0_key: u0[i].copy(),
                    ref1_key: u1[i].copy(),
                    target_key: uv_tgt[j13[i]].copy(),
                },
            )
        )
    stats["n_tracks"] = len(tracks)
    if c13_dists:
        stats["mean_c13_assoc_dist_px"] = float(np.mean(c13_dists))
    return tracks, stats


def select_ref_pair(
    neighbor_paths: Sequence[str],
    body_priors: Dict[str, PlanarBody],
    *,
    min_baseline_m: float,
) -> Optional[Tuple[str, str, float]]:
    """在邻帧中选基线最大的参考对；基线仍不足则返回 None（情况 A）。"""
    paths = [p for p in neighbor_paths if p in body_priors]
    best: Optional[Tuple[str, str, float]] = None
    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            d = body_baseline_m(body_priors[paths[i]], body_priors[paths[j]])
            if best is None or d > best[2]:
                best = (paths[i], paths[j], d)
    if best is None or best[2] < min_baseline_m:
        return None
    return best


def solve_pnp_from_tracks(
    tracks: Sequence[VisualTrack],
    target_key: str,
    K: np.ndarray,
    D: np.ndarray,
    camera_model: str,
    *,
    min_points: int = 6,
    reprojection_threshold: float = 8.0,
    confidence: float = 0.999,
) -> Optional[Dict[str, Any]]:
    from lib.geometry.absolute_pose_from_covis import solve_pnp_absolute_pose

    if len(tracks) == 0:
        return None
    obj = np.stack([t.X_world for t in tracks], axis=0)
    img = np.stack([t.observations[target_key] for t in tracks], axis=0)
    return solve_pnp_absolute_pose(
        obj,
        img,
        K,
        D,
        camera_model,
        min_points=min_points,
        reprojection_threshold=reprojection_threshold,
        confidence=confidence,
    )


def _wrap_angle(delta: float) -> float:
    d = (delta + math.pi) % (2.0 * math.pi) - math.pi
    return d


def _build_cam_var_layout(
    keys: Sequence[str],
    prior_keys: Sequence[str],
    *,
    use_ref6: bool,
    use_target6: bool,
) -> Tuple[Dict[str, Tuple[int, int, str]], int]:
    """返回 cam_key -> (start, end, kind)，kind 为 'ref' | 'target6' | 'planar'。"""
    prior_set = set(prior_keys)
    layout: Dict[str, Tuple[int, int, str]] = {}
    offset = 0
    target_key = keys[-1] if keys else None
    for k in keys:
        if k == target_key and use_target6:
            layout[k] = (offset, offset + 6, "target6")
            offset += 6
        elif k in prior_set and use_ref6:
            layout[k] = (offset, offset + 6, "ref")
            offset += 6
        else:
            layout[k] = (offset, offset + 3, "planar")
            offset += 3
    return layout, offset


def local_ba_planar(
    tracks: Sequence[VisualTrack],
    cam_keys: Sequence[str],
    body_init: Dict[str, PlanarBody],
    body_prior: Dict[str, PlanarBody],
    Tvc: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
    camera_model: str,
    *,
    z_world: float = 0.0,
    prior_weight: float = 10.0,
    prior_weight_xy: Optional[float] = None,
    prior_weight_yaw: Optional[float] = None,
    prior_weight_attitude: float = 10.0,
    ref_extra_dofs: bool = True,
    target_extra_dofs: bool = True,
    ref_xy_max_m: float = 0.05,
    ref_yaw_max_deg: float = 5.0,
    ref_dz_max_m: float = 0.1,
    ref_angle_max_deg: float = 1.0,
    target_xy_max_m: float = 0.15,
    target_yaw_max_deg: float = 15.0,
    target_dz_max_m: float = 0.1,
    target_angle_max_deg: float = 1.0,
    sigma_xy_m: float = 0.03,
    sigma_yaw_deg: float = 1.5,
    optimize_points: bool = False,
    max_optimize_points: int = 150,
    huber_px: float = 2.0,
) -> Optional[Dict[str, Any]]:
    """
    平面车体 Local BA：视觉重投影 + 参考帧先验。

    ref_extra_dofs=True（方案 B）时，参考帧优化 [x,y,θ,dz,roll,pitch]（6 DOF），
    target_extra_dofs=True 时目标帧同样 6 DOF（小范围 dz/roll/pitch）；否则目标仅 [x,y,θ]。
    xy 强先验 + 硬边界，dz/roll/pitch 小范围优化。
    """
    if least_squares is None:
        return None
    if len(tracks) < 4:
        return None

    w_xy = float(prior_weight_xy if prior_weight_xy is not None else 500.0)
    w_yaw = float(prior_weight_yaw if prior_weight_yaw is not None else max(prior_weight, 50.0))
    w_att = float(prior_weight_attitude)
    sigma_yaw = math.radians(max(sigma_yaw_deg, 1e-6))
    sigma_angle = math.radians(max(ref_angle_max_deg, 1e-6))
    sigma_dz = max(ref_dz_max_m, 1e-6)
    sigma_xy = max(sigma_xy_m, 1e-6)
    yaw_max = math.radians(max(ref_yaw_max_deg, 1e-6))

    Tvc = np.asarray(Tvc, dtype=np.float64).reshape(4, 4)
    keys = list(cam_keys)
    n_body = len(keys)
    n_tracks = min(len(tracks), max_optimize_points) if optimize_points else len(tracks)
    use_tracks = list(tracks[:n_tracks])

    prior_keys = [k for k in keys if k in body_prior and k != keys[-1]]
    target_key = keys[-1]
    use_ref6 = bool(ref_extra_dofs and prior_keys)
    use_target6 = bool(target_extra_dofs)
    layout, n_body_vars = _build_cam_var_layout(
        keys, prior_keys, use_ref6=use_ref6, use_target6=use_target6
    )
    n_point_vars = 3 * n_tracks if optimize_points else 0
    n_vars = n_body_vars + n_point_vars

    x0 = np.zeros(n_vars, dtype=np.float64)
    lb = np.full(n_vars, -np.inf, dtype=np.float64)
    ub = np.full(n_vars, np.inf, dtype=np.float64)

    ref_init: Dict[str, RefBodyState] = {}
    ref_prior: Dict[str, RefBodyState] = {}
    target_yaw_max = math.radians(max(target_yaw_max_deg, 1e-6))
    target_angle = math.radians(max(target_angle_max_deg, 1e-6))

    for k in keys:
        b0 = body_init[k]
        if use_ref6 and k in prior_keys:
            ref = RefBodyState.from_planar(b0)
            ref_init[k] = ref
            ref_prior[k] = RefBodyState.from_planar(body_prior[k])
            s, e, _ = layout[k]
            x0[s:e] = ref.as_vec()
            p = ref_prior[k]
            lb[s : s + 2] = [p.x - ref_xy_max_m, p.y - ref_xy_max_m]
            ub[s : s + 2] = [p.x + ref_xy_max_m, p.y + ref_xy_max_m]
            lb[s + 2] = p.theta - yaw_max
            ub[s + 2] = p.theta + yaw_max
            lb[s + 3] = -ref_dz_max_m
            ub[s + 3] = ref_dz_max_m
            lb[s + 4 : s + 6] = -sigma_angle
            ub[s + 4 : s + 6] = sigma_angle
        elif use_target6 and k == target_key:
            ref = RefBodyState.from_planar(b0)
            ref_init[k] = ref
            ref_prior[k] = ref
            s, e, _ = layout[k]
            x0[s:e] = ref.as_vec()
            p = ref_prior[k]
            lb[s : s + 2] = [p.x - target_xy_max_m, p.y - target_xy_max_m]
            ub[s : s + 2] = [p.x + target_xy_max_m, p.y + target_xy_max_m]
            lb[s + 2] = p.theta - target_yaw_max
            ub[s + 2] = p.theta + target_yaw_max
            lb[s + 3] = -target_dz_max_m
            ub[s + 3] = target_dz_max_m
            lb[s + 4 : s + 6] = -target_angle
            ub[s + 4 : s + 6] = target_angle
        else:
            s, e, _ = layout[k]
            x0[s:e] = b0.as_vec()
    if optimize_points:
        for i, tr in enumerate(use_tracks):
            x0[n_body_vars + 3 * i : n_body_vars + 3 * i + 3] = tr.X_world

    obs_list: List[Tuple[int, str, np.ndarray]] = []
    for ti, tr in enumerate(use_tracks):
        for ck in keys:
            if ck in tr.observations:
                obs_list.append((ti, ck, tr.observations[ck]))

    sw_xy = math.sqrt(max(w_xy, 1e-6))
    sw_yaw = math.sqrt(max(w_yaw, 1e-6))
    sw_att = math.sqrt(max(w_att, 1e-6))

    def _T_world_cam_for_key(k: str, vec: np.ndarray) -> np.ndarray:
        s, e, kind = layout[k]
        if kind in ("ref", "target6"):
            ref = RefBodyState(
                x=float(vec[s]),
                y=float(vec[s + 1]),
                theta=float(vec[s + 2]),
                dz=float(vec[s + 3]),
                roll=float(vec[s + 4]),
                pitch=float(vec[s + 5]),
            )
            return T_world_cam_from_ref_body(ref, Tvc, z_world)
        body = PlanarBody(x=float(vec[s]), y=float(vec[s + 1]), theta=float(vec[s + 2]))
        return T_world_cam_from_planar_body(body, Tvc, z_world)

    def residuals(vec: np.ndarray) -> np.ndarray:
        T_cams = {k: _T_world_cam_for_key(k, vec) for k in keys}
        res: List[float] = []
        for ti, ck, uv_obs in obs_list:
            if optimize_points:
                X = vec[n_body_vars + 3 * ti : n_body_vars + 3 * ti + 3]
            else:
                X = use_tracks[ti].X_world
            uv_proj = project_world_to_pixel(X, T_cams[ck], K, D, camera_model)[0]
            res.extend([float(uv_obs[0] - uv_proj[0]), float(uv_obs[1] - uv_proj[1])])

        if use_ref6:
            for k in prior_keys:
                s, e, _ = layout[k]
                ref = RefBodyState(
                    x=float(vec[s]),
                    y=float(vec[s + 1]),
                    theta=float(vec[s + 2]),
                    dz=float(vec[s + 3]),
                    roll=float(vec[s + 4]),
                    pitch=float(vec[s + 5]),
                )
                p = ref_prior[k]
                res.append(sw_xy * (ref.x - p.x) / sigma_xy)
                res.append(sw_xy * (ref.y - p.y) / sigma_xy)
                res.append(sw_yaw * _wrap_angle(ref.theta - p.theta) / sigma_yaw)
                res.append(sw_att * ref.dz / sigma_dz)
                res.append(sw_att * ref.roll / sigma_angle)
                res.append(sw_att * ref.pitch / sigma_angle)
        elif not use_target6:
            sw = math.sqrt(max(prior_weight, 1e-6))
            for k in prior_keys:
                s, e, _ = layout[k]
                b = PlanarBody(x=float(vec[s]), y=float(vec[s + 1]), theta=float(vec[s + 2]))
                p = body_prior[k]
                res.append(sw * (b.x - p.x))
                res.append(sw * (b.y - p.y))
                res.append(sw * _wrap_angle(b.theta - p.theta))
        if use_target6:
            s, e, _ = layout[target_key]
            ref = RefBodyState(
                x=float(vec[s]),
                y=float(vec[s + 1]),
                theta=float(vec[s + 2]),
                dz=float(vec[s + 3]),
                roll=float(vec[s + 4]),
                pitch=float(vec[s + 5]),
            )
            p = ref_prior[target_key]
            sigma_tgt_dz = max(target_dz_max_m, 1e-6)
            res.append(sw_att * ref.dz / sigma_tgt_dz)
            res.append(sw_att * ref.roll / target_angle)
            res.append(sw_att * ref.pitch / target_angle)
            _ = p
        return np.asarray(res, dtype=np.float64)

    result = least_squares(
        residuals,
        x0,
        bounds=(lb, ub),
        method="trf",
        loss="huber",
        f_scale=float(huber_px),
        max_nfev=100,
    )

    bodies_out: Dict[str, PlanarBody] = {}
    ref_bodies_out: Dict[str, RefBodyState] = {}
    T_out: Dict[str, np.ndarray] = {}
    for k in keys:
        s, e, kind = layout[k]
        if kind in ("ref", "target6"):
            ref = RefBodyState(
                x=float(result.x[s]),
                y=float(result.x[s + 1]),
                theta=float(result.x[s + 2]),
                dz=float(result.x[s + 3]),
                roll=float(result.x[s + 4]),
                pitch=float(result.x[s + 5]),
            )
            ref_bodies_out[k] = ref
            bodies_out[k] = PlanarBody(x=ref.x, y=ref.y, theta=ref.theta)
            T_out[k] = T_world_cam_from_ref_body(ref, Tvc, z_world)
        else:
            b = PlanarBody(
                x=float(result.x[s]),
                y=float(result.x[s + 1]),
                theta=float(result.x[s + 2]),
            )
            bodies_out[k] = b
            T_out[k] = T_world_cam_from_planar_body(b, Tvc, z_world)

    final_res = np.asarray(residuals(result.x), dtype=np.float64)
    n_vis = 2 * len(obs_list)
    vis_rmse = float(np.sqrt(np.mean(np.square(final_res[:n_vis])))) if n_vis else float("nan")

    per_cam_rmse = _per_camera_reproj_rmse(use_tracks, keys, T_out, K, D, camera_model)
    prior_rmse = float("nan")
    if len(final_res) > n_vis:
        prior_rmse = float(np.sqrt(np.mean(np.square(final_res[n_vis:]))))

    body_init_out = {k: {"x": b.x, "y": b.y, "theta": b.theta} for k, b in body_init.items()}
    ref_init_out = {k: v.to_dict() for k, v in ref_init.items()} if (use_ref6 or use_target6) else {}
    ref_out_dict = {k: v.to_dict() for k, v in ref_bodies_out.items()} if (use_ref6 or use_target6) else {}

    return {
        "body": {k: {"x": b.x, "y": b.y, "theta": b.theta} for k, b in bodies_out.items()},
        "ref_body": ref_out_dict,
        "body_init": body_init_out,
        "ref_body_init": ref_init_out,
        "T_world_cam": T_out,
        "T_world_cam_init": {
            k: (_T_world_cam_for_key(k, x0) if k in layout else T_out[k]) for k in keys
        },
        "cost": float(result.cost),
        "vis_reproj_rmse_px": vis_rmse,
        "per_camera_reproj_rmse_px": {
            Path(k).name if "/" in k or "\\" in k else k: v for k, v in per_cam_rmse.items()
        },
        "prior_rmse": prior_rmse,
        "n_observations": len(obs_list),
        "n_cameras": n_body,
        "optimize_points": optimize_points,
        "ref_extra_dofs": use_ref6,
        "target_extra_dofs": use_target6,
        "success": bool(result.success),
        "n_iterations": int(getattr(result, "nfev", 0)),
        "tracks_X_world": [tr.X_world.tolist() for tr in use_tracks],
    }


def local_ba_planar_robust(
    tracks: Sequence[VisualTrack],
    cam_keys: Sequence[str],
    body_init: Dict[str, PlanarBody],
    body_prior: Dict[str, PlanarBody],
    Tvc: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
    camera_model: str,
    *,
    outlier_reproj_px: Optional[float] = 15.0,
    outlier_max_passes: int = 2,
    outlier_mode: str = "track",
    min_tracks: int = 4,
    **ba_kwargs: Any,
) -> Optional[Dict[str, Any]]:
    """
    Local BA + 迭代重投影外点剔除后再优化。

    每轮 BA 结束后用当前位姿评估重投影；剔除超阈值 track/观测后，
    以上一轮 BA 输出为初值再跑 BA，直至无新外点或达到 max_passes。
    """
    use_tracks = _copy_tracks(tracks)
    cur_init = {
        k: PlanarBody(x=b.x, y=b.y, theta=b.theta) for k, b in body_init.items()
    }
    target_key = list(cam_keys)[-1] if cam_keys else None
    outlier_enabled = outlier_reproj_px is not None and float(outlier_reproj_px) > 0
    max_passes = max(1, int(outlier_max_passes)) if outlier_enabled else 1
    filter_log: List[Dict[str, Any]] = []

    ba: Optional[Dict[str, Any]] = None
    ba_prev: Optional[Dict[str, Any]] = None
    for pass_idx in range(max_passes):
        ba = local_ba_planar(
            use_tracks,
            cam_keys,
            cur_init,
            body_prior,
            Tvc,
            K,
            D,
            camera_model,
            **ba_kwargs,
        )
        if ba is None:
            return ba_prev
        ba_prev = ba

        if not outlier_enabled or pass_idx >= max_passes - 1:
            break

        filtered, fstats = filter_tracks_by_reprojection(
            use_tracks,
            cam_keys,
            ba["T_world_cam"],
            K,
            D,
            camera_model,
            max_reproj_px=float(outlier_reproj_px),
            mode=outlier_mode,
            target_key=target_key,
        )
        fstats["pass"] = pass_idx + 1
        filter_log.append(fstats)

        if fstats["n_tracks_removed"] == 0 and fstats.get("n_obs_removed", 0) == 0:
            break
        if len(filtered) < min_tracks:
            fstats["aborted"] = "too_few_tracks"
            break

        use_tracks = filtered
        cur_init = {
            k: PlanarBody(
                x=float(v["x"]),
                y=float(v["y"]),
                theta=float(v["theta"]),
            )
            for k, v in ba["body"].items()
        }

    if ba is None:
        return None

    ba["outlier_filter"] = {
        "enabled": outlier_enabled,
        "max_reproj_px": float(outlier_reproj_px) if outlier_enabled else None,
        "mode": outlier_mode,
        "max_passes": max_passes,
        "n_filter_passes": len(filter_log),
        "passes": filter_log,
        "n_tracks_final": len(use_tracks),
    }
    if target_key is not None:
        tname = (
            Path(target_key).name
            if "/" in target_key or "\\" in target_key
            else target_key
        )
        per_cam = ba.get("per_camera_reproj_rmse_px") or {}
        trmse = per_cam.get(tname)
        if trmse is not None and np.isfinite(trmse) and outlier_enabled:
            ba["outlier_filter"]["target_reproj_rmse_px"] = float(trmse)
            ba["outlier_filter"]["target_suspicious"] = bool(
                trmse > float(outlier_reproj_px) * 2.0
            )
    return ba
