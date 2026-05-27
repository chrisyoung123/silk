# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""MegaDepth / ScanNet LoFTR 索引（image_paths + poses + intrinsics + pair_infos）。"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from silk.datasets.megadepth.loftr_pairs import MegaDepth1500Pairs
from silk.datasets.pose_pairs.common import PosePairSample
from silk.datasets.scannet.loftr_pairs import ScanNet1500Pairs


class LoFTRIndexPosePairs(torch.utils.data.Dataset):
    """统一封装 MegaDepth1500Pairs / ScanNet1500Pairs，输出阶段二所需字段。"""

    def __init__(
        self,
        backend: str = "megadepth",
        dataset_root: str = "",
        index_path: str = "",
        pose_convention: str = "w2c",
        max_pairs: int = -1,
        as_gray: bool = True,
        depth_root: Optional[str] = None,
        load_depth: bool = False,
        **backend_kwargs: Any,
    ) -> None:
        super().__init__()
        backend = backend.lower()
        if backend == "megadepth":
            self._inner = MegaDepth1500Pairs(
                dataset_root=dataset_root,
                index_path=index_path,
                pose_convention=pose_convention,
                max_pairs=max_pairs,
                as_gray=as_gray,
            )
        elif backend == "scannet":
            self._inner = ScanNet1500Pairs(
                dataset_root=dataset_root,
                index_path=index_path,
                pose_convention=pose_convention,
                max_pairs=max_pairs,
                as_gray=as_gray,
                **backend_kwargs,
            )
        else:
            raise ValueError(f"unknown backend {backend}, use megadepth or scannet")

        self._load_depth = load_depth
        self._depth_root = depth_root

    def __len__(self) -> int:
        return len(self._inner)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        image0, image1, K0, K1, T_0to1 = self._inner[index]
        depth0 = depth1 = None
        if self._load_depth and self._depth_root is not None:
            # 可选：由子类路径规则加载深度；未配置则跳过
            pass

        sample = PosePairSample(
            image0=image0,
            image1=image1,
            T_0to1=T_0to1,
            K0=K0,
            K1=K1,
            depth0=depth0,
            depth1=depth1,
            meta={"index": index, "backend": type(self._inner).__name__},
        )
        return sample.to_dict()
