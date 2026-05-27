# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
位姿真值 → 图像对稠密对应，供 SiLK 阶段二（策略梯度）与带位姿的数据集使用。

支持：
  - 4×4 相对位姿 T_0to1（MegaDepth / ScanNet / MASt3R npz）
  - 平面 body 位姿 (x, y, theta) + 相机外参 Tvc（与 eval_pairs_pose_auc 一致）
  - 3×3 单应矩阵 H（HPatches 等）
  - 可选深度图（ScanNet / MegaDepth depth）
"""

from __future__ import annotations

import json
import math
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch

from silk.losses.info_nce import (
    keep_mutual_correspondences_only,
    positions_to_unidirectional_correspondence,
)


class PoseConvention(str, Enum):
    W2C = "w2c"
    C2W = "c2w"


def _as_4x4(T: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
    if isinstance(T, np.ndarray):
        T = torch.as_tensor(T, dtype=torch.float32)
    if T.shape == (3, 4):
        bottom = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=T.device, dtype=T.dtype)
        T = torch.cat([T, bottom], dim=0)
    assert T.shape == (4, 4), f"expected 4x4, got {tuple(T.shape)}"
    return T


def T_world_body_from_planar_pose(
    x: float,
    y: float,
    theta_rad: float,
    z_world: float = 0.0,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Body 在世界系：水平 (x,y)、高度 z_world、仅 yaw。"""
    c, s = math.cos(theta_rad), math.sin(theta_rad)
    T = torch.eye(4, device=device, dtype=torch.float32)
    T[0, 0], T[0, 1] = c, -s
    T[1, 0], T[1, 1] = s, c
    T[0, 3], T[1, 3], T[2, 3] = float(x), float(y), float(z_world)
    return T


def load_Tvc_from_extrinsic_json(
    data: Dict[str, Any],
    camera_model: str = "fisheye",
) -> np.ndarray:
    """从外参 JSON 读取 Tvc，满足 p_body = Tvc @ p_cam。"""
    if camera_model == "fisheye":
        node = data.get("Tvc_fish")
        if node is None:
            raise KeyError("鱼眼外参 JSON 缺少 Tvc_fish")
        if isinstance(node, dict) and "Tvc" in node:
            return np.asarray(node["Tvc"], dtype=np.float64).reshape(4, 4)
        return np.asarray(node, dtype=np.float64).reshape(4, 4)

    node = data.get("Tvc")
    if node is None:
        raise KeyError("针孔外参 JSON 缺少 Tvc")
    if isinstance(node, dict) and "Tvc" in node:
        return np.asarray(node["Tvc"], dtype=np.float64).reshape(4, 4)
    return np.asarray(node, dtype=np.float64).reshape(4, 4)


def T_world_cam_from_body_and_Tvc(
    T_world_body: Union[np.ndarray, torch.Tensor],
    Tvc: Union[np.ndarray, torch.Tensor],
) -> torch.Tensor:
    T_wb = _as_4x4(
        T_world_body
        if isinstance(T_world_body, torch.Tensor)
        else torch.as_tensor(T_world_body, dtype=torch.float32)
    )
    Tvc = _as_4x4(
        Tvc if isinstance(Tvc, torch.Tensor) else torch.as_tensor(Tvc, dtype=torch.float32)
    )
    return T_wb @ Tvc


def planar_body_poses_to_T_cam0_cam1(
    x0: float,
    y0: float,
    th0: float,
    x1: float,
    y1: float,
    th1: float,
    Tvc: Union[np.ndarray, torch.Tensor],
    body_z_world: float = 0.0,
    pose_xy_scale: float = 1.0,
) -> torch.Tensor:
    """由两张图的平面 body 位姿 + 共用 Tvc 得到 T_cam0_to_cam1（4×4）。"""
    Tvc = torch.as_tensor(Tvc, dtype=torch.float32)
    T_wc0 = T_world_cam_from_body_and_Tvc(
        T_world_body_from_planar_pose(
            x0 * pose_xy_scale, y0 * pose_xy_scale, th0, body_z_world
        ),
        Tvc,
    )
    T_wc1 = T_world_cam_from_body_and_Tvc(
        T_world_body_from_planar_pose(
            x1 * pose_xy_scale, y1 * pose_xy_scale, th1, body_z_world
        ),
        Tvc,
    )
    return torch.linalg.inv(T_wc0) @ T_wc1


def T_cam0_to_cam1_from_poses(
    pose0: Union[np.ndarray, torch.Tensor],
    pose1: Union[np.ndarray, torch.Tensor],
    convention: Union[str, PoseConvention] = PoseConvention.W2C,
) -> torch.Tensor:
    """由两张图的 4×4 外参计算 cam0→cam1。"""
    p0 = _as_4x4(pose0)
    p1 = _as_4x4(pose1)
    if isinstance(convention, str):
        convention = PoseConvention(convention)
    if convention == PoseConvention.W2C:
        return p1 @ torch.linalg.inv(p0)
    return torch.linalg.inv(p1) @ p0


@torch.no_grad()
def warp_image_points_constant_depth(
    points_xy: torch.Tensor,
    depth: torch.Tensor,
    K0: torch.Tensor,
    K1: torch.Tensor,
    T_0to1: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    将 image0 上的点 (x,y) 经深度 warp 到 image1。

    Parameters
    ----------
    points_xy : (B, L, 2)  OpenCV 顺序 (x, y)，像素坐标
    depth : (B, L)  各点深度（米或与 T,t 同单位）
    K0, K1 : (B, 3, 3)
    T_0to1 : (B, 4, 4)  p_cam1 = T_0to1 @ p_cam0（齐次）

    Returns
    -------
    warped_xy : (B, L, 2)
    valid : (B, L) bool
    """
    B, L, _ = points_xy.shape
    device = points_xy.device
    R = T_0to1[:, :3, :3]
    t = T_0to1[:, :3, 3:4]

    uv = points_xy
    z = depth.clamp(min=1e-4)
    ones = torch.ones(B, L, 1, device=device, dtype=uv.dtype)
    pix_h = torch.cat([uv * z.unsqueeze(-1), z.unsqueeze(-1)], dim=-1)

    K0_inv = torch.linalg.inv(K0)
    cam0 = torch.einsum("bij,bkj->bki", K0_inv, pix_h)

    cam1 = torch.einsum("bij,bkj->bki", R, cam0) + t.transpose(1, 2)
    z1 = cam1[..., 2].clamp(min=1e-4)
    proj = torch.einsum("bij,bkj->bki", K1, cam1)
    warped = proj[..., :2] / z1.unsqueeze(-1)

    valid = z1 > 1e-4
    return warped, valid


@torch.no_grad()
def sample_depth_at_points(
    depth_map: torch.Tensor,
    points_xy: torch.Tensor,
) -> torch.Tensor:
    """depth_map (B,H,W)，points_xy (B,L,2) 像素 (x,y)，返回 (B,L)。"""
    B, L, _ = points_xy.shape
    x = points_xy[..., 0].round().long().clamp(0, depth_map.shape[-1] - 1)
    y = points_xy[..., 1].round().long().clamp(0, depth_map.shape[-2] - 1)
    depth = torch.stack(
        [depth_map[b, y[b], x[b]] for b in range(B)],
        dim=0,
    )
    return depth


@torch.no_grad()
def warp_points_homography(
    points_xy: torch.Tensor,
    H_0to1: torch.Tensor,
) -> torch.Tensor:
    """points_xy (B,L,2) 经 3×3 单应 H warp。"""
    B, L, _ = points_xy.shape
    ones = torch.ones(B, L, 1, device=points_xy.device, dtype=points_xy.dtype)
    pts = torch.cat([points_xy, ones], dim=-1)
    H = H_0to1
    if H.dim() == 2:
        H = H.unsqueeze(0).expand(B, -1, -1)
    warped_h = torch.einsum("bij,bkj->bki", H, pts)
    warped = warped_h[..., :2] / warped_h[..., 2:3].clamp(min=1e-8)
    return warped


def compute_pair_correspondences(
    positions_image_xy: torch.Tensor,
    desc_width: int,
    desc_height: int,
    cell_size: float = 1.0,
    *,
    homography: Optional[torch.Tensor] = None,
    T_0to1: Optional[torch.Tensor] = None,
    K0: Optional[torch.Tensor] = None,
    K1: Optional[torch.Tensor] = None,
    depth0: Optional[torch.Tensor] = None,
    default_depth: float = 5.0,
    image_shape: Optional[Tuple[int, int]] = None,
    coord_mapping=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    由位姿真值计算稠密格点对应（与 SiLKRandomHomographies._get_corr 输出格式一致）。

    positions_image_xy : (B, L, 2)  图像坐标 (x, y)
    返回 corr_forward, corr_backward : (B, L) int64，-1 表示无对应
    """
    B, L, _ = positions_image_xy.shape
    device = positions_image_xy.device

    if homography is not None:
        H = homography
        if H.dim() == 2:
            H = H.unsqueeze(0).expand(B, -1, -1)
        warped_fwd = warp_points_homography(positions_image_xy, H)
        valid_fwd = torch.ones(B, L, dtype=torch.bool, device=device)
    elif T_0to1 is not None and K0 is not None and K1 is not None:
        if depth0 is not None and depth0.dim() == 3:
            depth_at_pts = sample_depth_at_points(depth0, positions_image_xy)
            depth_at_pts = torch.where(
                depth_at_pts > 1e-6,
                depth_at_pts,
                torch.full_like(depth_at_pts, float(default_depth)),
            )
        else:
            depth_at_pts = torch.full(
                (B, L),
                float(default_depth),
                device=device,
                dtype=positions_image_xy.dtype,
            )

        warped_fwd, valid_fwd = warp_image_points_constant_depth(
            positions_image_xy,
            depth_at_pts,
            K0,
            K1,
            T_0to1,
        )
        valid_bwd = valid_fwd
    else:
        raise ValueError(
            "need homography or (T_0to1, K0, K1) for pose-based correspondences"
        )

    if image_shape is not None:
        h, w = image_shape[-2], image_shape[-1]
        in0 = (
            (positions_image_xy[..., 0] >= 0)
            & (positions_image_xy[..., 0] < w)
            & (positions_image_xy[..., 1] >= 0)
            & (positions_image_xy[..., 1] < h)
        )
        in1 = (
            (warped_fwd[..., 0] >= 0)
            & (warped_fwd[..., 0] < w)
            & (warped_fwd[..., 1] >= 0)
            & (warped_fwd[..., 1] < h)
        )
        valid_fwd = valid_fwd & in0 & in1

    warped_fwd_desc = warped_fwd
    if coord_mapping is not None:
        warped_fwd_desc = coord_mapping.apply(warped_fwd)

    corr_forward = positions_to_unidirectional_correspondence(
        warped_fwd_desc,
        desc_width,
        desc_height,
        cell_size,
        ordering="xy",
    )
    corr_forward = torch.where(
        valid_fwd,
        corr_forward,
        torch.full_like(corr_forward, -1),
    )

    corr_backward = torch.full_like(corr_forward, -1)
    for b in range(B):
        for i in range(L):
            j = int(corr_forward[b, i].item())
            if j >= 0:
                corr_backward[b, j] = i

    corr_forward, corr_backward = keep_mutual_correspondences_only(
        corr_forward,
        corr_backward,
    )

    return corr_forward, corr_backward


def load_pose_from_extra_json(
    path: Union[str, Path],
    pose_x_key: str = "value.pose.x",
    pose_y_key: str = "value.pose.y",
    pose_theta_key: str = "value.pose.theta",
    theta_degrees: bool = False,
    pose_xy_scale: float = 1.0,
) -> Tuple[float, float, float]:
    """读取 *_extra.json 中的平面位姿（与 eval_pairs_pose_auc 键路径一致）。"""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    def _get(dotted: str) -> float:
        cur: Any = data
        for part in dotted.split("."):
            cur = cur[part]
        return float(cur)

    x = _get(pose_x_key) * float(pose_xy_scale)
    y = _get(pose_y_key) * float(pose_xy_scale)
    th = _get(pose_theta_key)
    if theta_degrees:
        th = math.radians(th)
    return x, y, th
