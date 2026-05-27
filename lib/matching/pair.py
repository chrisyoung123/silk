# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import torch

from silk.matching.lightglue import create_lightglue_matcher, match_lightglue

VALID_ENGINES = {"silk", "lightglue"}


@dataclass
class MatchResult:
    matcher: str
    m0: np.ndarray
    m1: np.ndarray
    idx0: np.ndarray
    idx1: np.ndarray
    n_keypoints0: int
    n_keypoints1: int
    distances: Optional[np.ndarray] = None

    @property
    def n_matches(self) -> int:
        return int(self.m0.shape[0])

    def as_dict(self) -> Dict[str, Any]:
        return {
            "matcher": self.matcher,
            "m0": self.m0,
            "m1": self.m1,
            "idx0": self.idx0,
            "idx1": self.idx1,
            "n_matches": self.n_matches,
            "n_keypoints0": self.n_keypoints0,
            "n_keypoints1": self.n_keypoints1,
            "distances": self.distances,
        }


class PairMatcher:
    """Stateful pair matcher for SiLK MNN or LightGlue."""

    def __init__(
        self,
        name: str,
        *,
        silk_fn: Optional[Callable[..., Any]] = None,
        lightglue_model: Any = None,
        lightglue_device: Optional[str] = None,
        return_distances: bool = False,
    ) -> None:
        self.name = str(name).lower()
        self._silk_fn = silk_fn
        self._lightglue = lightglue_model
        self._lightglue_device = lightglue_device
        self._return_distances = return_distances

    def match(
        self,
        pos0: np.ndarray,
        desc0: np.ndarray,
        pos1: np.ndarray,
        desc1: np.ndarray,
        *,
        image_hw0: Optional[Tuple[int, int]] = None,
        image_hw1: Optional[Tuple[int, int]] = None,
    ) -> MatchResult:
        pos0 = np.asarray(pos0, dtype=np.float64).reshape(-1, 2)
        pos1 = np.asarray(pos1, dtype=np.float64).reshape(-1, 2)
        desc0 = np.asarray(desc0)
        desc1 = np.asarray(desc1)
        n_kp0 = int(pos0.shape[0])
        n_kp1 = int(pos1.shape[0])

        if self.name == "lightglue":
            if self._lightglue is None:
                raise ValueError("LightGlue matcher is not initialized")
            if image_hw0 is None or image_hw1 is None:
                raise ValueError("LightGlue requires image_hw0 and image_hw1")
            m0, m1, idx0, idx1 = match_lightglue(
                self._lightglue,
                pos0,
                desc0,
                pos1,
                desc1,
                image_hw0=image_hw0,
                image_hw1=image_hw1,
                device=self._lightglue_device,
            )
            return MatchResult(
                matcher=self.name,
                m0=m0,
                m1=m1,
                idx0=idx0,
                idx1=idx1,
                n_keypoints0=n_kp0,
                n_keypoints1=n_kp1,
                distances=None,
            )

        if self._silk_fn is None:
            raise ValueError("SiLK matcher is not initialized")

        if desc0.shape[0] == 0 or desc1.shape[0] == 0:
            empty_i = np.zeros((0,), dtype=np.int64)
            empty = np.zeros((0, 2), dtype=np.float64)
            return MatchResult(
                matcher=self.name,
                m0=empty,
                m1=empty,
                idx0=empty_i,
                idx1=empty_i,
                n_keypoints0=n_kp0,
                n_keypoints1=n_kp1,
                distances=np.zeros((0,), dtype=np.float32),
            )

        d0 = torch.as_tensor(desc0)
        d1 = torch.as_tensor(desc1)
        distances = None
        if self._return_distances:
            matches, dist_t = self._silk_fn(d0, d1)
            distances = dist_t.detach().cpu().numpy().astype(np.float32)
        else:
            matches = self._silk_fn(d0, d1)

        n_match = int(matches.shape[0])
        if n_match <= 0:
            empty_i = np.zeros((0,), dtype=np.int64)
            empty = np.zeros((0, 2), dtype=np.float64)
            return MatchResult(
                matcher=self.name,
                m0=empty,
                m1=empty,
                idx0=empty_i,
                idx1=empty_i,
                n_keypoints0=n_kp0,
                n_keypoints1=n_kp1,
                distances=np.zeros((0,), dtype=np.float32),
            )

        idx0 = matches[:, 0].cpu().numpy().astype(np.int64)
        idx1 = matches[:, 1].cpu().numpy().astype(np.int64)
        m0 = pos0[idx0]
        m1 = pos1[idx1]
        return MatchResult(
            matcher=self.name,
            m0=m0,
            m1=m1,
            idx0=idx0,
            idx1=idx1,
            n_keypoints0=n_kp0,
            n_keypoints1=n_kp1,
            distances=distances,
        )


def get_pair_matcher(
    engine: str = "silk",
    *,
    postprocessing: str = "double-softmax",
    threshold: float = 0.6,
    temperature: float = 0.1,
    return_distances: bool = False,
    lightglue_features: str = "disk",
    lightglue_device: Optional[str] = None,
    lightglue_flash: bool = True,
    lightglue_depth_confidence: float = 0.95,
    lightglue_width_confidence: float = 0.99,
) -> PairMatcher:
    """Factory for image-pair matchers (SiLK MNN or LightGlue)."""
    name = str(engine or "silk").lower()
    if name not in VALID_ENGINES:
        raise ValueError(f"invalid engine {engine!r}, expected one of {VALID_ENGINES}")

    if name == "lightglue":
        model = create_lightglue_matcher(
            features=lightglue_features,
            device=lightglue_device,
            flash=lightglue_flash,
            depth_confidence=lightglue_depth_confidence,
            width_confidence=lightglue_width_confidence,
        )
        return PairMatcher(
            name,
            lightglue_model=model,
            lightglue_device=lightglue_device,
        )

    from silk.models.silk import matcher as silk_matcher_factory

    silk_fn = silk_matcher_factory(
        postprocessing=postprocessing,
        threshold=threshold,
        temperature=temperature,
        return_distances=return_distances,
    )
    return PairMatcher(
        name,
        silk_fn=silk_fn,
        return_distances=return_distances,
    )
