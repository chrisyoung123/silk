# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Callable, Dict

import torch.utils.data


class PosePairTransformWrapper(torch.utils.data.Dataset):
    """对 image0/image1 施加相同随机变换（需在 transform 内分别调用）。"""

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        transform_image: Callable[[Any], Any],
    ) -> None:
        super().__init__()
        self._dataset = dataset
        self._transform_image = transform_image

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        item = self._dataset[index]
        item["image0"] = self._transform_image(item["image0"])
        item["image1"] = self._transform_image(item["image1"])
        return item
