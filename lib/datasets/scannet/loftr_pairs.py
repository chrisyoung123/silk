# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from pathlib import Path
from typing import Tuple

import numpy as np
import skimage.io as io
import torch
import torch.utils.data


def _load_image_tensor(path: str, as_gray: bool) -> torch.Tensor:
    image = io.imread(path, as_gray=as_gray)
    image = torch.as_tensor(image, dtype=torch.float32)

    if image.ndim == 2:
        image = image.unsqueeze(0)
    else:
        image = image.permute(2, 0, 1)

    if image.max() > 1.0:
        image = image / 255.0

    return image


class ScanNet1500Pairs(torch.utils.data.Dataset):
    """ScanNet test-1500 pair loader (LoFTR-style preprocessed data + test.npz).

    Expected layout under ``dataset_root``::

        scannet_test_1500/
            test.npz          # fields: name (N,4), rel_pose (optional)
            scene0707_00/
                color/{frame}.jpg
                pose/{frame}.txt
                intrinsic/intrinsic_color.txt

    ``name[i]`` is ``(scene_id, scene_sub_idx, frame0, frame1)`` as in LoFTR.
    Poses in ``pose/*.txt`` are 4x4 camera-to-world matrices.
    """

    def __init__(
        self,
        dataset_root: str,
        index_path: str = "",
        scenes_subdir: str = "scannet_test_1500",
        pose_convention: str = "c2w",
        max_pairs: int = -1,
        as_gray: bool = True,
    ) -> None:
        super().__init__()

        assert pose_convention in {"w2c", "c2w"}

        self._dataset_root = Path(dataset_root)
        self._scenes_root = self._dataset_root / scenes_subdir
        self._pose_convention = pose_convention
        self._as_gray = as_gray

        if index_path:
            self._index_path = Path(index_path)
        else:
            self._index_path = self._scenes_root / "test.npz"

        if not self._index_path.is_file():
            raise FileNotFoundError(f"ScanNet index not found: {self._index_path}")
        if not self._scenes_root.is_dir():
            raise FileNotFoundError(f"ScanNet scenes dir not found: {self._scenes_root}")

        with np.load(self._index_path, allow_pickle=True) as f:
            self._names = np.asarray(f["name"], dtype=np.uint16)

        if max_pairs is not None and max_pairs >= 0:
            self._names = self._names[:max_pairs]

    def __len__(self) -> int:
        return len(self._names)

    @staticmethod
    def _scene_name(scene_id: int, scene_sub_idx: int) -> str:
        return f"scene{int(scene_id):04d}_{int(scene_sub_idx):02d}"

    def _scene_dir(self, scene_id: int, scene_sub_idx: int) -> Path:
        return self._scenes_root / self._scene_name(scene_id, scene_sub_idx)

    def _image_path(self, scene_id: int, scene_sub_idx: int, frame_id: int) -> Path:
        scene_dir = self._scene_dir(scene_id, scene_sub_idx)
        return scene_dir / "color" / f"{int(frame_id)}.jpg"

    def _pose_path(self, scene_id: int, scene_sub_idx: int, frame_id: int) -> Path:
        scene_dir = self._scene_dir(scene_id, scene_sub_idx)
        return scene_dir / "pose" / f"{int(frame_id)}.txt"

    def _intrinsic_path(self, scene_id: int, scene_sub_idx: int) -> Path:
        scene_dir = self._scene_dir(scene_id, scene_sub_idx)
        return scene_dir / "intrinsic" / "intrinsic_color.txt"

    @staticmethod
    def _load_pose(path: Path) -> np.ndarray:
        pose = np.loadtxt(path, dtype=np.float64)
        if pose.shape == (3, 4):
            bottom = np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float64)
            pose = np.vstack([pose, bottom])
        return pose

    @staticmethod
    def _load_intrinsic(path: Path) -> np.ndarray:
        K = np.loadtxt(path, dtype=np.float64)
        return K[:3, :3]

    def _compute_relative_pose(self, pose0: np.ndarray, pose1: np.ndarray) -> np.ndarray:
        if self._pose_convention == "w2c":
            return pose1 @ np.linalg.inv(pose0)
        return np.linalg.inv(pose1) @ pose0

    def _load_pair(self, name_row: np.ndarray) -> Tuple[torch.Tensor, ...]:
        scene_id, scene_sub_idx, frame0, frame1 = (int(v) for v in name_row)

        image_path_0 = self._image_path(scene_id, scene_sub_idx, frame0)
        image_path_1 = self._image_path(scene_id, scene_sub_idx, frame1)

        if not image_path_0.is_file():
            raise RuntimeError(f"image file does not exist: {image_path_0}")
        if not image_path_1.is_file():
            raise RuntimeError(f"image file does not exist: {image_path_1}")

        image0 = _load_image_tensor(str(image_path_0), self._as_gray)
        image1 = _load_image_tensor(str(image_path_1), self._as_gray)

        intrinsic_path = self._intrinsic_path(scene_id, scene_sub_idx)
        K = self._load_intrinsic(intrinsic_path)
        K0 = torch.as_tensor(K, dtype=torch.float32)
        K1 = K0.clone()

        pose0 = self._load_pose(self._pose_path(scene_id, scene_sub_idx, frame0))
        pose1 = self._load_pose(self._pose_path(scene_id, scene_sub_idx, frame1))
        T_0to1 = torch.as_tensor(
            self._compute_relative_pose(pose0, pose1), dtype=torch.float32
        )

        return image0, image1, K0, K1, T_0to1

    def __getitem__(self, index: int):
        return self._load_pair(self._names[index])
