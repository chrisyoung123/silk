# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import skimage.io as io
import torch
import torch.utils.data


def _to_string(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _looks_like_pair(value) -> bool:
    if not isinstance(value, (tuple, list, np.ndarray)):
        return False
    if len(value) < 2:
        return False
    return np.issubdtype(type(value[0]), np.integer) and np.issubdtype(
        type(value[1]), np.integer
    )


def _extract_pair(value) -> Tuple[int, int]:
    # Typical shape used in LoFTR indices:
    # pair_infos[i] = ((idx0, idx1), overlap_score, ...)
    if _looks_like_pair(value):
        return int(value[0]), int(value[1])

    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _extract_pair(value.item())
        if value.size > 0:
            return _extract_pair(value[0])

    if isinstance(value, (tuple, list)) and len(value) > 0:
        return _extract_pair(value[0])

    raise RuntimeError(f"unable to parse pair info item: {value}")


class MegaDepth1500Pairs(torch.utils.data.Dataset):
    """MegaDepth pair dataset loader compatible with LoFTR test indices.

    Expected npz fields:
        - image_paths : list/array of relative image paths
        - intrinsics  : (N, 3, 3) camera intrinsics
        - poses       : (N, 4, 4) camera extrinsics
        - pair_infos  : pair metadata (commonly object array)
    """

    def __init__(
        self,
        dataset_root: str,
        index_path: str,
        pose_convention: str = "w2c",
        max_pairs: int = -1,
        as_gray: bool = True,
    ) -> None:
        super().__init__()

        assert pose_convention in {"w2c", "c2w"}

        self._dataset_root = Path(dataset_root)
        self._index_path = Path(index_path)
        self._pose_convention = pose_convention
        self._as_gray = as_gray

        with np.load(self._index_path, allow_pickle=True) as f:
            self._image_paths = [_to_string(p) for p in f["image_paths"]]
            self._intrinsics = np.asarray(f["intrinsics"], dtype=np.float32)
            self._poses = np.asarray(f["poses"], dtype=np.float32)
            pair_infos = f["pair_infos"]

        self._pairs = self._parse_pairs(pair_infos)

        if max_pairs is not None and max_pairs >= 0:
            self._pairs = self._pairs[:max_pairs]

    @staticmethod
    def _parse_pairs(pair_infos) -> List[Tuple[int, int]]:
        if (
            isinstance(pair_infos, np.ndarray)
            and pair_infos.ndim == 2
            and pair_infos.shape[1] >= 2
            and np.issubdtype(pair_infos.dtype, np.integer)
        ):
            return [(int(p[0]), int(p[1])) for p in pair_infos]

        if isinstance(pair_infos, np.ndarray):
            iterable: Iterable = pair_infos.tolist()
        else:
            iterable = pair_infos

        return [_extract_pair(item) for item in iterable]

    def __len__(self) -> int:
        return len(self._pairs)

    def _resolve_image_path(self, rel_or_abs_path: str) -> str:
        path = Path(rel_or_abs_path)
        if path.is_absolute():
            return str(path)
        return str(self._dataset_root / path)

    def _compute_relative_pose(self, pose0, pose1):
        # T_0to1 maps camera-0 coordinates to camera-1 coordinates.
        if self._pose_convention == "w2c":
            return pose1 @ np.linalg.inv(pose0)
        return np.linalg.inv(pose1) @ pose0

    def __getitem__(self, index: int):
        idx0, idx1 = self._pairs[index]

        image_path_0 = self._resolve_image_path(self._image_paths[idx0])
        image_path_1 = self._resolve_image_path(self._image_paths[idx1])

        if not os.path.exists(image_path_0):
            raise RuntimeError(f"image file does not exist: {image_path_0}")
        if not os.path.exists(image_path_1):
            raise RuntimeError(f"image file does not exist: {image_path_1}")

        image0 = io.imread(image_path_0, as_gray=self._as_gray)
        image1 = io.imread(image_path_1, as_gray=self._as_gray)

        image0 = torch.as_tensor(image0, dtype=torch.float32)
        image1 = torch.as_tensor(image1, dtype=torch.float32)

        # Ensure shape is CxHxW
        if image0.ndim == 2:
            image0 = image0.unsqueeze(0)
        else:
            image0 = image0.permute(2, 0, 1)

        if image1.ndim == 2:
            image1 = image1.unsqueeze(0)
        else:
            image1 = image1.permute(2, 0, 1)

        # skimage returns [0, 255] for uint8 inputs.
        if image0.max() > 1.0:
            image0 = image0 / 255.0
        if image1.max() > 1.0:
            image1 = image1 / 255.0

        K0 = torch.as_tensor(self._intrinsics[idx0], dtype=torch.float32)
        K1 = torch.as_tensor(self._intrinsics[idx1], dtype=torch.float32)

        pose0 = self._poses[idx0]
        pose1 = self._poses[idx1]
        T_0to1 = torch.as_tensor(
            self._compute_relative_pose(pose0, pose1), dtype=torch.float32
        )

        return image0, image1, K0, K1, T_0to1
