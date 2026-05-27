# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Keypoint-level sky / exclude-mask filtering."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np


def load_exclude_mask_grayscale(path: Path) -> np.ndarray:
    m = cv2.imread(str(path.expanduser().resolve()), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(f"无法读取 exclude mask: {path}")
    return m


def exclude_mask_for_hw(mask_u8: np.ndarray, h: int, w: int) -> np.ndarray:
    if int(mask_u8.shape[0]) == h and int(mask_u8.shape[1]) == w:
        return mask_u8
    return cv2.resize(mask_u8, (w, h), interpolation=cv2.INTER_NEAREST)


def non_exclude_mask_point_mask(
    points_xy: np.ndarray,
    mask_u8: np.ndarray,
    *,
    exclude_value: int = 255,
) -> np.ndarray:
    """返回点级 bool 掩码：True 表示保留（mask 像素 != exclude_value）。"""
    pts = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    h, w = int(mask_u8.shape[0]), int(mask_u8.shape[1])
    keep = np.ones((pts.shape[0],), dtype=bool)
    ex = int(exclude_value)
    for i, p in enumerate(pts):
        x = int(round(float(p[0])))
        y = int(round(float(p[1])))
        if 0 <= x < w and 0 <= y < h:
            keep[i] = int(mask_u8[y, x]) != ex
    return keep


def sky_binary_u8(
    cls_map: np.ndarray,
    h: int,
    w: int,
    sky_class_id: int,
    *,
    saliency_threshold: Optional[int] = None,
) -> np.ndarray:
    """天空二值图 (H,W)，1=天空。尺寸与 (h,w) 对齐。"""
    interp = cv2.INTER_LINEAR if saliency_threshold is not None else cv2.INTER_NEAREST
    if int(cls_map.shape[0]) != h or int(cls_map.shape[1]) != w:
        cls_r = cv2.resize(cls_map.astype(np.uint8), (w, h), interpolation=interp)
    else:
        cls_r = np.asarray(cls_map, dtype=np.uint8)
    if saliency_threshold is not None:
        return (cls_r.astype(np.int32) > int(saliency_threshold)).astype(np.uint8)
    return (cls_r.astype(np.int32) == int(sky_class_id)).astype(np.uint8)


def erode_sky_binary(sky_u8: np.ndarray, erode_px: int) -> np.ndarray:
    """腐蚀天空区域（内缩），erode_px 为椭圆核半径（像素），0=不腐蚀。"""
    px = int(erode_px)
    if px <= 0 or not np.any(sky_u8):
        return sky_u8
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    return cv2.erode(sky_u8, k, iterations=1)


def non_sky_point_mask(
    points_xy: np.ndarray,
    cls_map: np.ndarray,
    sky_class_id: int,
    *,
    saliency_threshold: Optional[int] = None,
    sky_erode_px: int = 0,
) -> np.ndarray:
    """返回点级 bool 掩码：True 表示非天空点（可经腐蚀内缩天空边界）。"""
    pts = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    h, w = int(cls_map.shape[0]), int(cls_map.shape[1])
    sky = sky_binary_u8(
        cls_map, h, w, sky_class_id, saliency_threshold=saliency_threshold
    )
    sky = erode_sky_binary(sky, sky_erode_px)
    keep = np.zeros((pts.shape[0],), dtype=bool)
    for i, p in enumerate(pts):
        x = int(round(float(p[0])))
        y = int(round(float(p[1])))
        if 0 <= x < w and 0 <= y < h:
            keep[i] = int(sky[y, x]) == 0
    return keep
