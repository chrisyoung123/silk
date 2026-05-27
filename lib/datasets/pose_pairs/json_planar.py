# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
自定义 / MASt3R 风格 pairs JSON + 平面位姿 extra.json + 相机内外参。

JSON 格式（与 eval_pairs_pose_auc 兼容）::

    {
      "pairs": [
        {"path0": "a.jpg", "path1": "b.jpg"},
        ...
      ]
    }

每张图位姿默认从 ``<stem>_extra.json`` 读取 value.pose.{x,y,theta}；
也可用项内 ``extra0`` / ``extra1`` 覆盖路径。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.utils.data

from silk.datasets.pose_pairs.common import PosePairSample, load_image_chw
from silk.geometry.pose_correspondence import (
    load_Tvc_from_extrinsic_json,
    load_pose_from_extra_json,
    planar_body_poses_to_T_cam0_cam1,
)


def _extra_json_for_image(image_path: Path) -> Path:
    return image_path.parent / f"{image_path.stem}_extra.json"


class JsonPlanarPosePairs(torch.utils.data.Dataset):
    def __init__(
        self,
        pairs_json: str,
        extrinsic_json: str,
        camera_json: Optional[str] = None,
        pairs_root: Optional[str] = None,
        pose_x_key: str = "value.pose.x",
        pose_y_key: str = "value.pose.y",
        pose_theta_key: str = "value.pose.theta",
        theta_degrees: bool = False,
        pose_xy_scale: float = 0.001,
        body_z_world: float = 0.0,
        camera_model: str = "fisheye",
        as_gray: bool = False,
        max_pairs: int = -1,
        shared_intrinsics: bool = True,
    ) -> None:
        super().__init__()
        self._pairs_root = Path(pairs_root) if pairs_root else None
        self._pose_x_key = pose_x_key
        self._pose_y_key = pose_y_key
        self._pose_theta_key = pose_theta_key
        self._theta_degrees = theta_degrees
        self._pose_xy_scale = pose_xy_scale
        self._body_z_world = body_z_world
        self._as_gray = as_gray

        with Path(pairs_json).open("r", encoding="utf-8") as f:
            data = json.load(f)
        self._pairs: List[Dict[str, Any]] = list(data["pairs"])
        if max_pairs is not None and max_pairs >= 0:
            self._pairs = self._pairs[:max_pairs]

        with Path(extrinsic_json).open("r", encoding="utf-8") as f:
            ext_data = json.load(f)
        self._Tvc = load_Tvc_from_extrinsic_json(ext_data, camera_model=camera_model)

        self._K0: Optional[torch.Tensor] = None
        self._K1: Optional[torch.Tensor] = None
        if camera_json is not None:
            K = self._load_K_from_camera_json(camera_json)
            self._K0 = K
            self._K1 = K.clone() if shared_intrinsics else K.clone()

    @staticmethod
    def _load_K_from_camera_json(path: str) -> torch.Tensor:
        with Path(path).open("r", encoding="utf-8") as f:
            raw = json.load(f)

        def req(key: str) -> float:
            return float(raw[key])

        fx = req("Camera.fx")
        fy = req("Camera.fy")
        cx = req("Camera.cx")
        cy = req("Camera.cy")
        K = np.array(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        return torch.from_numpy(K)

    def _resolve(self, p: str) -> Path:
        path = Path(p)
        if not path.is_absolute() and self._pairs_root is not None:
            path = self._pairs_root / path
        return path

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        item = self._pairs[index]
        p0 = self._resolve(item["path0"])
        p1 = self._resolve(item["path1"])

        extra0 = item.get("extra0")
        extra1 = item.get("extra1")
        extra0_path = Path(extra0) if extra0 else _extra_json_for_image(p0)
        extra1_path = Path(extra1) if extra1 else _extra_json_for_image(p1)

        x0, y0, th0 = load_pose_from_extra_json(
            extra0_path,
            self._pose_x_key,
            self._pose_y_key,
            self._pose_theta_key,
            self._theta_degrees,
            self._pose_xy_scale,
        )
        x1, y1, th1 = load_pose_from_extra_json(
            extra1_path,
            self._pose_x_key,
            self._pose_y_key,
            self._pose_theta_key,
            self._theta_degrees,
            self._pose_xy_scale,
        )

        T_0to1 = planar_body_poses_to_T_cam0_cam1(
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

        image0 = load_image_chw(str(p0), as_gray=self._as_gray)
        image1 = load_image_chw(str(p1), as_gray=self._as_gray)

        sample = PosePairSample(
            image0=image0,
            image1=image1,
            T_0to1=T_0to1,
            K0=self._K0,
            K1=self._K1,
            meta={"path0": str(p0), "path1": str(p1), "index": index},
        )
        return sample.to_dict()
