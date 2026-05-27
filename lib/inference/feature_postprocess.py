# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""SiLK sparse keypoint post-processing: sky/mask filtering and ANMS."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple, Union

import cv2
import numpy as np

from lib.geometry.anms import filter_keypoints_anms
from lib.inference.keypoint_filters import (
    exclude_mask_for_hw,
    non_exclude_mask_point_mask,
    non_sky_point_mask,
)


@dataclass
class PointFilterConfig:
    sky_seg_post: str = "minmax"
    sky_saliency_threshold: int = 128
    sky_erode_px: int = 0
    sky_class_id: int = 1
    exclude_mask_value: int = 255
    use_anms: bool = False
    anms_top_k: int = 2000
    anms_min_radius: float = 0.0


def silk_positions_to_xy(pts: np.ndarray) -> np.ndarray:
    """SiLK sparse_positions 经 from_feature_coords_to_image_coords 后为 (row,col)=(y,x)，转为 (x,y)。"""
    pts = np.asarray(pts, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] < 2:
        raise ValueError(f"期望关键点 shape 为 Nx(>=2)，当前为 {pts.shape}")
    pts_yx = pts[:, :2]
    return pts_yx[:, [1, 0]]


def point_filter_config_from_namespace(args: Any) -> PointFilterConfig:
    return PointFilterConfig(
        sky_seg_post=str(getattr(args, "sky_seg_post", "minmax")),
        sky_saliency_threshold=int(getattr(args, "sky_saliency_threshold", 128)),
        sky_erode_px=int(getattr(args, "sky_erode_px", 0)),
        sky_class_id=int(getattr(args, "sky_class_id", 1)),
        exclude_mask_value=int(getattr(args, "exclude_mask_value", 255)),
        use_anms=bool(getattr(args, "use_anms", False)),
        anms_top_k=int(getattr(args, "anms_top_k", 2000)),
        anms_min_radius=float(getattr(args, "anms_min_radius", 0.0)),
    )


def apply_point_filters(
    positions_xy: np.ndarray,
    descriptors: np.ndarray,
    *,
    img_path: Optional[Path] = None,
    im_bgr: Optional[np.ndarray] = None,
    sky_seg: Any = None,
    exclude_mask: Optional[np.ndarray] = None,
    config: Optional[PointFilterConfig] = None,
    scores: Optional[np.ndarray] = None,
) -> Union[
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray],
]:
    """对关键点应用天空 / exclude mask 过滤。"""
    cfg = config or PointFilterConfig()
    pos = np.asarray(positions_xy, dtype=np.float64).reshape(-1, 2)
    desc = np.asarray(descriptors, dtype=np.float32)
    keep = np.ones(pos.shape[0], dtype=bool)

    need_im = sky_seg is not None or exclude_mask is not None
    if need_im and im_bgr is None:
        if img_path is None:
            raise ValueError("apply_point_filters 需要 im_bgr 或 img_path")
        im_bgr = cv2.imread(str(Path(img_path).expanduser().resolve()), cv2.IMREAD_COLOR)
        if im_bgr is None:
            raise FileNotFoundError(f"无法读取图像: {img_path}")

    if sky_seg is not None:
        assert im_bgr is not None
        cls = sky_seg.predict(im_bgr)
        sal_thr: Optional[int] = (
            int(cfg.sky_saliency_threshold) if cfg.sky_seg_post == "minmax" else None
        )
        sky_keep = non_sky_point_mask(
            pos,
            cls,
            cfg.sky_class_id,
            saliency_threshold=sal_thr,
            sky_erode_px=int(cfg.sky_erode_px),
        )
        keep &= sky_keep

    if exclude_mask is not None:
        assert im_bgr is not None
        h, w = im_bgr.shape[:2]
        ex_m = exclude_mask_for_hw(exclude_mask, h, w)
        keep &= non_exclude_mask_point_mask(
            pos, ex_m, exclude_value=int(cfg.exclude_mask_value)
        )

    idx = np.nonzero(keep)[0]
    pos_out = pos[idx]
    desc_out = desc[idx]
    if scores is None:
        return pos_out, desc_out
    sc = np.asarray(scores, dtype=np.float64).reshape(-1)
    return pos_out, desc_out, sc[idx]


def maybe_anms_filter(
    positions_xy: np.ndarray,
    descriptors: np.ndarray,
    *,
    config: Optional[PointFilterConfig] = None,
    scores: Optional[np.ndarray] = None,
) -> Union[
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray],
]:
    """可选 ANMS：使关键点在图像上空间均匀分布。"""
    cfg = config or PointFilterConfig()
    pos = np.asarray(positions_xy, dtype=np.float64).reshape(-1, 2)
    desc = np.asarray(descriptors, dtype=np.float32)
    if not cfg.use_anms:
        if scores is None:
            return pos, desc
        return pos, desc, np.asarray(scores, dtype=np.float64).reshape(-1)
    pos, desc, keep = filter_keypoints_anms(
        pos,
        desc,
        scores=scores,
        top_k=int(cfg.anms_top_k),
        min_radius=float(cfg.anms_min_radius),
    )
    if scores is None:
        return pos, desc
    sc = np.asarray(scores, dtype=np.float64).reshape(-1)
    return pos, desc, sc[keep]
