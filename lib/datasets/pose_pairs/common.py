# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
from PIL import Image


@dataclass
class PosePairSample:
    """单对训练样本；阶段二 DataLoader 返回 dict，collate 后进入 Lightning。"""

    image0: torch.Tensor
    image1: torch.Tensor
    # 以下至少提供一种几何监督
    T_0to1: Optional[torch.Tensor] = None
    K0: Optional[torch.Tensor] = None
    K1: Optional[torch.Tensor] = None
    homography: Optional[torch.Tensor] = None
    depth0: Optional[torch.Tensor] = None
    depth1: Optional[torch.Tensor] = None
    meta: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        out = {
            "image0": self.image0,
            "image1": self.image1,
        }
        if self.T_0to1 is not None:
            out["T_0to1"] = self.T_0to1
        if self.K0 is not None:
            out["K0"] = self.K0
        if self.K1 is not None:
            out["K1"] = self.K1
        if self.homography is not None:
            out["homography"] = self.homography
        if self.depth0 is not None:
            out["depth0"] = self.depth0
        if self.depth1 is not None:
            out["depth1"] = self.depth1
        if self.meta is not None:
            out["meta"] = self.meta
        return out


def load_image_chw(
    path: str,
    as_gray: bool = False,
    rgb: bool = True,
) -> torch.Tensor:
    img = Image.open(path)
    if as_gray:
        img = img.convert("L")
    elif rgb:
        img = img.convert("RGB")
    t = torch.from_numpy(np.array(img))
    if t.ndim == 2:
        t = t.unsqueeze(-1)
    t = t.permute(2, 0, 1).float()
    if t.max() > 1.0:
        t = t / 255.0
    return t
