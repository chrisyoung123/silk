# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np
import torch


def _l2_normalize_descriptors(desc: np.ndarray) -> np.ndarray:
    d = np.asarray(desc, dtype=np.float32)
    n = np.linalg.norm(d, axis=1, keepdims=True)
    return d / np.maximum(n, 1e-6)


def lightglue_features_for_descriptor_dim(dim: int) -> str:
    """Map descriptor dimension to a LightGlue weights preset."""
    if dim == 256:
        return "superpoint"
    if dim == 128:
        return "disk"
    raise ValueError(
        f"unsupported descriptor dim {dim} for LightGlue; "
        "expected 128 (disk) or 256 (superpoint)"
    )


def create_lightglue_matcher(
    features: str = "disk",
    device: Optional[str] = None,
    *,
    flash: bool = True,
    depth_confidence: float = 0.95,
    width_confidence: float = 0.99,
) -> Any:
    """Create a LightGlue matcher.

    SiLK 128-dim descriptors use features='disk'; 256-dim use 'superpoint'.
    """
    try:
        from lightglue import LightGlue
    except ImportError as e:
        raise ImportError(
            "LightGlue is not installed. Run: "
            "pip install git+https://github.com/cvg/LightGlue.git"
        ) from e
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    return (
        LightGlue(
            features=features,
            flash=flash,
            depth_confidence=depth_confidence,
            width_confidence=width_confidence,
        )
        .eval()
        .to(dev)
    )


def match_lightglue(
    matcher: Any,
    pos0: np.ndarray,
    desc0: np.ndarray,
    pos1: np.ndarray,
    desc1: np.ndarray,
    *,
    image_hw0: Tuple[int, int],
    image_hw1: Tuple[int, int],
    device: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Match keypoints + descriptors with LightGlue."""
    dev = device or next(matcher.parameters()).device
    if desc0.shape[0] == 0 or desc1.shape[0] == 0:
        empty_i = np.zeros((0,), dtype=np.int64)
        empty = np.zeros((0, 2), dtype=np.float64)
        return empty, empty, empty_i, empty_i

    desc_dim = int(desc0.shape[-1])
    input_dim = int(getattr(getattr(matcher, "conf", None), "input_dim", desc_dim))
    if desc_dim != input_dim:
        expected = lightglue_features_for_descriptor_dim(desc_dim)
        raise ValueError(
            f"descriptor dim {desc_dim} does not match LightGlue input_dim {input_dim}; "
            f"try --lightglue-features {expected}"
        )

    kpts0 = torch.from_numpy(np.asarray(pos0, dtype=np.float32))[None].to(dev)
    kpts1 = torch.from_numpy(np.asarray(pos1, dtype=np.float32))[None].to(dev)
    d0 = torch.from_numpy(_l2_normalize_descriptors(desc0))[None].to(dev)
    d1 = torch.from_numpy(_l2_normalize_descriptors(desc1))[None].to(dev)
    h0, w0 = int(image_hw0[0]), int(image_hw0[1])
    h1, w1 = int(image_hw1[0]), int(image_hw1[1])
    size0 = torch.tensor([[w0, h0]], device=dev, dtype=torch.float32)
    size1 = torch.tensor([[w1, h1]], device=dev, dtype=torch.float32)

    data = {
        "image0": {
            "keypoints": kpts0,
            "descriptors": d0,
            "image_size": size0,
        },
        "image1": {
            "keypoints": kpts1,
            "descriptors": d1,
            "image_size": size1,
        },
    }
    with torch.inference_mode():
        pred = matcher(data)

    matches = pred["matches"][0].detach().cpu().numpy()
    if matches.size == 0:
        empty_i = np.zeros((0,), dtype=np.int64)
        empty = np.zeros((0, 2), dtype=np.float64)
        return empty, empty, empty_i, empty_i

    valid = (matches[:, 0] >= 0) & (matches[:, 1] >= 0)
    pairs = matches[valid]
    if pairs.shape[0] == 0:
        empty_i = np.zeros((0,), dtype=np.int64)
        empty = np.zeros((0, 2), dtype=np.float64)
        return empty, empty, empty_i, empty_i

    idx0 = pairs[:, 0].astype(np.int64)
    idx1 = pairs[:, 1].astype(np.int64)
    m0 = np.asarray(pos0, dtype=np.float64)[idx0]
    m1 = np.asarray(pos1, dtype=np.float64)[idx1]
    return m0, m1, idx0, idx1
