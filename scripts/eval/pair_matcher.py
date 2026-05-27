#!/usr/bin/env python3
"""图像对描述子匹配：SiLK mutual-NN 或 LightGlue（可选）。"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from silk.matching import (
    MatchResult,
    PairMatcher,
    create_lightglue_matcher,
    get_pair_matcher,
    match_lightglue,
)

_DEFAULT_SILK_MATCHER = get_pair_matcher("silk")


def match_silk_mutual_nn(
    pos0: np.ndarray,
    desc0: np.ndarray,
    pos1: np.ndarray,
    desc1: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """SiLK double-softmax / ratio-test 匹配，返回 m0,m1 及 idx0,idx1。"""
    out = _DEFAULT_SILK_MATCHER.match(pos0, desc0, pos1, desc1)
    return out.m0, out.m1, out.idx0, out.idx1


def match_descriptor_pair(
    pos0: np.ndarray,
    desc0: np.ndarray,
    pos1: np.ndarray,
    desc1: np.ndarray,
    *,
    matcher: str = "silk",
    lightglue: Any = None,
    image_hw0: Optional[Tuple[int, int]] = None,
    image_hw1: Optional[Tuple[int, int]] = None,
    lightglue_device: Optional[str] = None,
) -> Dict[str, Any]:
    """统一匹配接口，返回 m0,m1,idx0,idx1 及统计。"""
    name = str(matcher or "silk").lower()
    if isinstance(lightglue, PairMatcher):
        out = lightglue.match(
            pos0,
            desc0,
            pos1,
            desc1,
            image_hw0=image_hw0,
            image_hw1=image_hw1,
        )
        return out.as_dict()

    if name == "lightglue":
        if lightglue is None:
            raise ValueError("LightGlue matcher 未初始化")
        pair_matcher = PairMatcher(
            name,
            lightglue_model=lightglue,
            lightglue_device=lightglue_device,
        )
    else:
        pair_matcher = _DEFAULT_SILK_MATCHER

    out: MatchResult = pair_matcher.match(
        pos0,
        desc0,
        pos1,
        desc1,
        image_hw0=image_hw0,
        image_hw1=image_hw1,
    )
    return out.as_dict()


__all__ = [
    "PairMatcher",
    "create_lightglue_matcher",
    "get_pair_matcher",
    "match_descriptor_pair",
    "match_lightglue",
    "match_silk_mutual_nn",
]
