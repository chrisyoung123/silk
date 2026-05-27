# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
MASt3R batch_manifest / 通用 pairs JSON，支持多种位姿字段。

每项可包含（优先级从高到低）：
  - ``T_0to1`` : 4×4 相对位姿（列表或 npz 路径）
  - ``pose0`` + ``pose1`` : 4×4 外参 + ``pose_convention``
  - ``extra0`` / ``extra1`` + 全局 ``extrinsic_json``（平面 x,y,theta）
  - 仅 path0/path1：自动查找 ``*_extra.json``（需 extrinsic_json）

manifest 顶层键：
  - ``pairs`` : 列表（必须）
  - ``intrinsics`` : 可选共用 K（3×3）
  - ``poses`` : 可选 (N,4,4)，与 ``pose_index0`` / ``pose_index1`` 联用
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

from silk.datasets.pose_pairs.common import PosePairSample, load_image_chw
from silk.geometry.pose_correspondence import (
    PoseConvention,
    T_cam0_to_cam1_from_poses,
    load_Tvc_from_extrinsic_json,
    load_pose_from_extra_json,
    planar_body_poses_to_T_cam0_cam1,
)


def _as_tensor_4x4(obj: Any) -> torch.Tensor:
    arr = np.asarray(obj, dtype=np.float32)
    if arr.shape == (3, 4):
        arr = np.vstack([arr, [0, 0, 0, 1]])
    return torch.from_numpy(arr.reshape(4, 4))


class MASt3RManifestPosePairs(torch.utils.data.Dataset):
    def __init__(
        self,
        manifest_json: str,
        pairs_root: Optional[str] = None,
        extrinsic_json: Optional[str] = None,
        camera_json: Optional[str] = None,
        pose_convention: str = "w2c",
        pose_x_key: str = "value.pose.x",
        pose_y_key: str = "value.pose.y",
        pose_theta_key: str = "value.pose.theta",
        theta_degrees: bool = False,
        pose_xy_scale: float = 0.001,
        body_z_world: float = 0.0,
        camera_model: str = "fisheye",
        as_gray: bool = False,
        max_pairs: int = -1,
        default_depth: float = 5.0,
    ) -> None:
        super().__init__()
        self._root = Path(pairs_root) if pairs_root else None
        self._pose_convention = pose_convention
        self._pose_x_key = pose_x_key
        self._pose_y_key = pose_y_key
        self._pose_theta_key = pose_theta_key
        self._theta_degrees = theta_degrees
        self._pose_xy_scale = pose_xy_scale
        self._body_z_world = body_z_world
        self._as_gray = as_gray
        self._default_depth = default_depth

        path = Path(manifest_json)
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        self._pairs: List[Dict[str, Any]] = list(data["pairs"])
        if max_pairs is not None and max_pairs >= 0:
            self._pairs = self._pairs[:max_pairs]

        self._global_poses: Optional[np.ndarray] = None
        if "poses" in data:
            self._global_poses = np.asarray(data["poses"], dtype=np.float32)

        self._global_K: Optional[torch.Tensor] = None
        if "intrinsics" in data:
            K = np.asarray(data["intrinsics"], dtype=np.float32)
            if K.ndim == 2:
                K = K[None]
            self._global_K = torch.from_numpy(K[0])

        self._Tvc: Optional[np.ndarray] = None
        if extrinsic_json is not None:
            with Path(extrinsic_json).open("r", encoding="utf-8") as f:
                ext = json.load(f)
            self._Tvc = load_Tvc_from_extrinsic_json(ext, camera_model=camera_model)

        if camera_json is not None and self._global_K is None:
            from silk.datasets.pose_pairs.json_planar import JsonPlanarPosePairs

            self._global_K = JsonPlanarPosePairs._load_K_from_camera_json(camera_json)

        # 允许 manifest 同目录下的 poses.npz（MASt3R 导出）
        npz_path = path.parent / "poses.npz"
        if self._global_poses is None and npz_path.is_file():
            with np.load(npz_path, allow_pickle=True) as z:
                if "poses" in z:
                    self._global_poses = np.asarray(z["poses"], dtype=np.float32)
                if "intrinsics" in z and self._global_K is None:
                    self._global_K = torch.from_numpy(
                        np.asarray(z["intrinsics"][0], dtype=np.float32)
                    )

    def _resolve(self, p: str) -> Path:
        path = Path(p)
        if not path.is_absolute() and self._root is not None:
            path = self._root / path
        return path

    def _extra_for(self, img_path: Path, item: Dict[str, Any], key: str) -> Path:
        if key in item:
            return self._resolve(item[key])
        return img_path.parent / f"{img_path.stem}_extra.json"

    def _T_from_item(self, item: Dict[str, Any], p0: Path, p1: Path) -> torch.Tensor:
        if "T_0to1" in item:
            return _as_tensor_4x4(item["T_0to1"])

        if "pose0" in item and "pose1" in item:
            return T_cam0_to_cam1_from_poses(
                item["pose0"],
                item["pose1"],
                convention=item.get("pose_convention", self._pose_convention),
            )

        if self._global_poses is not None:
            i0 = int(item["pose_index0"])
            i1 = int(item["pose_index1"])
            return T_cam0_to_cam1_from_poses(
                self._global_poses[i0],
                self._global_poses[i1],
                convention=self._pose_convention,
            )

        if self._Tvc is not None:
            e0 = self._extra_for(p0, item, "extra0")
            e1 = self._extra_for(p1, item, "extra1")
            x0, y0, th0 = load_pose_from_extra_json(
                e0,
                self._pose_x_key,
                self._pose_y_key,
                self._pose_theta_key,
                self._theta_degrees,
                self._pose_xy_scale,
            )
            x1, y1, th1 = load_pose_from_extra_json(
                e1,
                self._pose_x_key,
                self._pose_y_key,
                self._pose_theta_key,
                self._theta_degrees,
                self._pose_xy_scale,
            )
            return planar_body_poses_to_T_cam0_cam1(
                x0,
                y0,
                th0,
                x1,
                y1,
                th1,
                self._Tvc,
                body_z_world=self._body_z_world,
                pose_xy_scale=1.0,
            )

        raise RuntimeError(
            f"pair 缺少位姿：需提供 T_0to1、pose0/1、pose_index* 或 extra+extrinsic_json: {item}"
        )

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        item = self._pairs[index]
        p0 = self._resolve(item["path0"])
        p1 = self._resolve(item["path1"])

        T_0to1 = self._T_from_item(item, p0, p1)
        image0 = load_image_chw(str(p0), as_gray=self._as_gray)
        image1 = load_image_chw(str(p1), as_gray=self._as_gray)

        K0 = K1 = self._global_K
        if "K0" in item:
            K0 = _as_tensor_4x4(item["K0"])[:3, :3]
        if "K1" in item:
            K1 = _as_tensor_4x4(item["K1"])[:3, :3]

        sample = PosePairSample(
            image0=image0,
            image1=image1,
            T_0to1=T_0to1,
            K0=K0,
            K1=K1,
            meta={
                "path0": str(p0),
                "path1": str(p1),
                "index": item.get("index", index),
            },
        )
        return sample.to_dict()
