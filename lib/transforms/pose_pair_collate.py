# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List

import torch

from silk.transforms.abstract import NamedContext, Transform


class CollatePosePairBatch(Transform):
    """将 PosePair Dataset 的 list[dict] 合并为 NamedContext batch。"""

    def __call__(self, items: List[Dict[str, Any]]) -> NamedContext:
        if not items:
            raise ValueError("empty batch")

        keys = set(items[0].keys())
        for it in items[1:]:
            keys &= set(it.keys())

        batch: Dict[str, Any] = {}
        for key in sorted(keys):
            vals = [it[key] for it in items]
            if isinstance(vals[0], torch.Tensor):
                batch[key] = torch.stack(vals, dim=0)
            else:
                batch[key] = vals

        return NamedContext(batch)
