# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from silk.backbones.silk.silk import from_feature_coords_to_image_coords
from silk.config.core import instantiate_and_ensure_is_instance
from silk.logger import LOG
from silk.models.silk import matcher


def _as_tensor(t: torch.Tensor) -> torch.Tensor:
    if isinstance(t, (tuple, list)):
        assert len(t) == 1
        return t[0]
    return t


def _resize_image_tensor(image: torch.Tensor, max_edge: int) -> Tuple[torch.Tensor, float]:
    if max_edge is None or max_edge <= 0:
        return image, 1.0

    _, h, w = image.shape
    long_edge = max(h, w)
    if long_edge <= max_edge:
        return image, 1.0

    scale = float(max_edge) / float(long_edge)
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))

    image = torch.nn.functional.interpolate(
        image.unsqueeze(0),
        size=(new_h, new_w),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)

    return image, scale


def _scale_intrinsics(K: torch.Tensor, scale: float) -> torch.Tensor:
    if abs(scale - 1.0) < 1e-8:
        return K
    K = K.clone()
    K[0, 0] *= scale
    K[1, 1] *= scale
    K[0, 2] *= scale
    K[1, 2] *= scale
    return K


def _preprocess_urr_center_crop(
    image: torch.Tensor, K: torch.Tensor, img_dim: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Match URR VideoDataset rgb_transform + K update (evaluate.py --img_dim)."""
    _, h, w = image.shape
    smaller_dim = float(h)
    crop_offset = (float(w) - float(h)) / 2.0

    new_h = int(img_dim)
    new_w = max(new_h, int(round(w * img_dim / smaller_dim)))

    image = torch.nn.functional.interpolate(
        image.unsqueeze(0),
        size=(new_h, new_w),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)

    if new_w > new_h:
        x0 = (new_w - new_h) // 2
        image = image[:, :, x0 : x0 + new_h]

    K = K.clone()
    K[0, 2] -= crop_offset
    K[:2, :] *= float(img_dim) / smaller_dim
    return image, K


def _preprocess_fixed_hw(
    image: torch.Tensor,
    K: torch.Tensor,
    size_hw: List[int],
    uniform_scale: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Target (H, W) = (480, 640) for LoFTR ScanNet-1500 (native images often 968 x 1296).

    When ``uniform_scale`` is true (default), scale by height then center-crop width so
  1296x968 is not non-uniformly stretched (sx != sy). Matches common SuperGlue practice
    better than independent x/y resize.
    """
    h_tgt, w_tgt = (int(size_hw[0]), int(size_hw[1]))
    _, h, w = image.shape

    if h == h_tgt and w == w_tgt:
        return image, K

    K = K.clone()

    if not uniform_scale:
        image = torch.nn.functional.interpolate(
            image.unsqueeze(0),
            size=(h_tgt, w_tgt),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        K[0, 0] *= float(w_tgt) / float(w)
        K[1, 1] *= float(h_tgt) / float(h)
        K[0, 2] *= float(w_tgt) / float(w)
        K[1, 2] *= float(h_tgt) / float(h)
        return image, K

    # Fix height to h_tgt (LoFTR: 480), preserve aspect ratio, crop/pad width to w_tgt.
    scale = float(h_tgt) / float(h)
    new_h = h_tgt
    new_w = max(1, int(round(w * scale)))

    image = torch.nn.functional.interpolate(
        image.unsqueeze(0),
        size=(new_h, new_w),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    K[:2, :] *= scale

    if new_w > w_tgt:
        x0 = (new_w - w_tgt) // 2
        image = image[:, :, x0 : x0 + w_tgt]
        K[0, 2] -= float(x0)
    elif new_w < w_tgt:
        pad_left = (w_tgt - new_w) // 2
        image = torch.nn.functional.pad(
            image, (pad_left, w_tgt - new_w - pad_left, 0, 0), mode="constant", value=0.0
        )
        K[0, 2] += float(pad_left)

    return image, K


def _apply_image_preprocess(
    image: torch.Tensor,
    K: torch.Tensor,
    preprocess_cfg,
    max_image_edge: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if preprocess_cfg is not None:
        mode = str(getattr(preprocess_cfg, "mode", "max_edge"))
        if mode == "urr_center_crop":
            img_dim = int(preprocess_cfg.img_dim)
            return _preprocess_urr_center_crop(image, K, img_dim)
        if mode == "fixed_hw":
            size_hw = list(preprocess_cfg.size_hw)
            uniform_scale = bool(getattr(preprocess_cfg, "uniform_scale", True))
            return _preprocess_fixed_hw(image, K, size_hw, uniform_scale=uniform_scale)
        if mode == "max_edge":
            max_edge = int(getattr(preprocess_cfg, "max_edge", max_image_edge))
        elif mode != "none":
            raise RuntimeError(f"unsupported image_preprocess.mode `{mode}`")

    scale = 1.0
    if max_image_edge is not None and max_image_edge > 0:
        image, scale = _resize_image_tensor(image, max_image_edge)
        K = _scale_intrinsics(K, scale)
    return image, K


def _extract_features(model: torch.nn.Module, image: torch.Tensor, device: str):
    image = image.unsqueeze(0).to(device)

    outputs = ["score", "sparse_positions", "sparse_descriptors"]
    if hasattr(model, "model_forward_flow"):
        scores, positions, descriptors = model.model_forward_flow(
            images=image, outputs=outputs
        )
    elif hasattr(model, "forward_flow"):
        scores, positions, descriptors = model.forward_flow(images=image, outputs=outputs)
    else:
        raise RuntimeError(
            "model should have `model_forward_flow` or `forward_flow` for SiLK evaluation"
        )

    positions = _as_tensor(positions)
    descriptors = _as_tensor(descriptors)

    return positions, descriptors


def _select_top_k(
    positions: torch.Tensor,
    descriptors: torch.Tensor,
    top_k: int,
):
    """Keep top-scoring sparse keypoints (score is in positions[:, 2] before coord transform)."""
    if top_k is None or top_k <= 0 or positions.shape[0] <= top_k:
        return positions, descriptors

    if positions.shape[1] < 3:
        return positions, descriptors

    k = min(int(top_k), int(positions.shape[0]))
    kp_scores = positions[:, 2]
    idx = torch.topk(kp_scores, k=k, largest=True).indices
    return positions[idx], descriptors[idx]


def _to_xy(points: torch.Tensor, ordering: str) -> torch.Tensor:
    if ordering == "xy":
        return points
    if ordering == "yx":
        return points[:, [1, 0]]
    raise RuntimeError(f"unsupported keypoint ordering `{ordering}`")


def _rotation_error_deg(R_est: np.ndarray, R_gt: np.ndarray) -> float:
    trace = np.trace(R_est @ R_gt.T)
    trace = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    return float(np.rad2deg(np.arccos(trace)))


def _translation_error_deg(t_est: np.ndarray, t_gt: np.ndarray) -> float:
    t_est = t_est.reshape(3)
    t_gt = t_gt.reshape(3)
    denom = np.linalg.norm(t_est) * np.linalg.norm(t_gt)
    if denom < 1e-9:
        return 180.0
    cos_val = np.clip(np.dot(t_est, t_gt) / denom, -1.0, 1.0)
    err = float(np.rad2deg(np.arccos(cos_val)))
    return min(err, 180.0 - err)


def _compute_auc(errors: Iterable[float], thresholds: Iterable[float]) -> Dict[str, float]:
    errors = np.asarray(list(errors), dtype=np.float32)
    if errors.size == 0:
        return {f"auc@{int(t)}": 0.0 for t in thresholds}

    sort_idx = np.argsort(errors)
    errors = errors[sort_idx]
    recall = (np.arange(len(errors)) + 1) / len(errors)
    errors = np.r_[0.0, errors]
    recall = np.r_[0.0, recall]

    auc = {}
    for t in thresholds:
        last_index = np.searchsorted(errors, t)
        rec = np.r_[recall[:last_index], recall[last_index - 1]]
        err = np.r_[errors[:last_index], t]
        auc[f"auc@{int(t)}"] = float(np.trapz(rec, x=err) / t)
    return auc


def _essential_matrix_candidates(E: np.ndarray) -> List[np.ndarray]:
    """Normalize findEssentialMat outputs to a list of 3x3 matrices."""
    if E is None:
        return []

    E = np.asarray(E, dtype=np.float64)
    if E.size == 0:
        return []

    if E.ndim == 1 and E.size == 9:
        E = E.reshape(3, 3)

    if E.ndim != 2:
        return []

    rows, cols = E.shape
    if rows == 3 and cols == 3:
        return [E]
    # Stacked hypotheses: (9, 3) row blocks or (3, 9) column blocks (OpenCV 4.x).
    if rows % 3 == 0 and cols == 3:
        return [E[i * 3 : (i + 1) * 3, :] for i in range(rows // 3)]
    if rows == 3 and cols % 3 == 0:
        return [E[:, i * 3 : (i + 1) * 3] for i in range(cols // 3)]

    return []


def _estimate_pose(
    points0_xy: np.ndarray,
    points1_xy: np.ndarray,
    K0: np.ndarray,
    K1: np.ndarray,
    pixel_threshold: float,
):
    if points0_xy.shape[0] < 5:
        return None

    pts0_norm = cv2.undistortPoints(points0_xy.reshape(-1, 1, 2), K0, None).reshape(-1, 2)
    pts1_norm = cv2.undistortPoints(points1_xy.reshape(-1, 1, 2), K1, None).reshape(-1, 2)

    avg_focal = (K0[0, 0] + K0[1, 1] + K1[0, 0] + K1[1, 1]) / 4.0
    norm_thresh = pixel_threshold / max(avg_focal, 1e-6)

    E, mask = cv2.findEssentialMat(
        pts0_norm,
        pts1_norm,
        cameraMatrix=np.eye(3),
        method=cv2.RANSAC,
        prob=0.999,
        threshold=norm_thresh,
    )
    E_candidates = _essential_matrix_candidates(E)
    if not E_candidates:
        return None

    best = None
    best_inliers = -1
    for Ei in E_candidates:
        if Ei.shape != (3, 3):
            continue
        try:
            n_inliers, R, t, recover_mask = cv2.recoverPose(
                Ei,
                pts0_norm,
                pts1_norm,
                cameraMatrix=np.eye(3),
                mask=mask,
            )
        except cv2.error:
            continue
        if n_inliers > best_inliers:
            best_inliers = int(n_inliers)
            best = (R, t, recover_mask)

    if best is None:
        return None
    return best[0], best[1], best_inliers


def main(config: DictConfig):
    dataset = instantiate_and_ensure_is_instance(config.mode.dataset, Dataset)
    loader = instantiate_and_ensure_is_instance(config.mode.loader, DataLoader)
    model = instantiate_and_ensure_is_instance(config.mode.model, torch.nn.Module)
    matcher_fn = instantiate_and_ensure_is_instance(config.mode.matcher, object)

    device = config.mode.device
    model = model.to(device).eval()

    keypoint_ordering = config.mode.keypoint_ordering
    max_edge = int(getattr(config.mode, "max_image_edge", 0))
    preprocess_cfg = getattr(config.mode, "image_preprocess", None)
    top_k = int(getattr(config.mode.keypoints, "top_k", -1))

    ransac_cfg = config.mode.ransac
    pixel_thresholds = list(getattr(ransac_cfg, "pixel_thresholds", []) or [])
    if not pixel_thresholds:
        pixel_thresholds = [float(ransac_cfg.pixel_threshold)]

    pair_cache = []

    for batch in tqdm(loader):
        image0, image1, K0, K1, T_0to1 = batch

        image0 = image0.squeeze(0)
        image1 = image1.squeeze(0)
        K0 = K0.squeeze(0)
        K1 = K1.squeeze(0)
        T_0to1 = T_0to1.squeeze(0)

        image0, K0 = _apply_image_preprocess(image0, K0, preprocess_cfg, max_edge)
        image1, K1 = _apply_image_preprocess(image1, K1, preprocess_cfg, max_edge)

        points0, desc0 = _extract_features(model, image0, device)
        points1, desc1 = _extract_features(model, image1, device)
        points0, desc0 = _select_top_k(points0, desc0, top_k)
        points1, desc1 = _select_top_k(points1, desc1, top_k)
        points0 = from_feature_coords_to_image_coords(model, points0[:, :2])
        points1 = from_feature_coords_to_image_coords(model, points1[:, :2])

        matches = matcher_fn(desc0, desc1)
        if matches.numel() == 0:
            pair_cache.append(None)
            continue

        m0 = points0[matches[:, 0]]
        m1 = points1[matches[:, 1]]

        m0_xy = _to_xy(m0, keypoint_ordering).detach().cpu().numpy().astype(np.float32)
        m1_xy = _to_xy(m1, keypoint_ordering).detach().cpu().numpy().astype(np.float32)
        K0_np = K0.detach().cpu().numpy().astype(np.float64)
        K1_np = K1.detach().cpu().numpy().astype(np.float64)
        T_gt = T_0to1.detach().cpu().numpy()
        R_gt = T_gt[:3, :3]
        t_gt = T_gt[:3, 3]

        pair_cache.append(
            {
                "m0_xy": m0_xy,
                "m1_xy": m1_xy,
                "K0_np": K0_np,
                "K1_np": K1_np,
                "R_gt": R_gt,
                "t_gt": t_gt,
                "n_matches": int(matches.shape[0]),
            }
        )

    total_pairs = len(dataset)
    metrics_out = {}
    best_auc = None
    best_stats = None

    for pixel_threshold in pixel_thresholds:
        pose_errors = []
        rotation_errors = []
        translation_errors = []
        match_count = []
        inlier_count = []
        valid_pairs = 0

        for item in pair_cache:
            if item is None:
                continue

            pose = _estimate_pose(
                item["m0_xy"],
                item["m1_xy"],
                item["K0_np"],
                item["K1_np"],
                pixel_threshold,
            )
            if pose is None:
                continue

            R_est, t_est, n_inliers = pose
            r_err = _rotation_error_deg(R_est, item["R_gt"])
            t_err = _translation_error_deg(t_est, item["t_gt"])
            p_err = max(r_err, t_err)

            valid_pairs += 1
            rotation_errors.append(r_err)
            translation_errors.append(t_err)
            pose_errors.append(p_err)
            match_count.append(item["n_matches"])
            inlier_count.append(int(n_inliers))

        auc = _compute_auc(pose_errors, config.mode.metrics.thresholds_deg)
        suffix = "" if len(pixel_thresholds) == 1 else f"_ransac{pixel_threshold:g}"
        metrics_out.update({f"{k}{suffix}": v for k, v in auc.items()})

        if best_auc is None or auc.get("auc@5", 0.0) > best_auc.get("auc@5", 0.0):
            best_auc = auc
            best_stats = {
                "valid_pair_rate": float(valid_pairs / max(total_pairs, 1)),
                "n_pairs_valid": int(valid_pairs),
                "mean_pose_error_deg": float(np.mean(pose_errors)) if pose_errors else float("inf"),
                "mean_rotation_error_deg": float(np.mean(rotation_errors))
                if rotation_errors
                else float("inf"),
                "mean_translation_error_deg": float(np.mean(translation_errors))
                if translation_errors
                else float("inf"),
                "mean_match_count": float(np.mean(match_count)) if match_count else 0.0,
                "mean_inlier_count": float(np.mean(inlier_count)) if inlier_count else 0.0,
                "ransac_pixel_threshold": float(pixel_threshold),
            }

    if best_stats is None:
        LOG.warning("no valid pair found for pose estimation")
        best_stats = {
            "valid_pair_rate": 0.0,
            "n_pairs_valid": 0,
            "mean_pose_error_deg": float("inf"),
            "mean_rotation_error_deg": float("inf"),
            "mean_translation_error_deg": float("inf"),
            "mean_match_count": 0.0,
            "mean_inlier_count": 0.0,
            "ransac_pixel_threshold": float(pixel_thresholds[0]),
        }

    return {
        "metrics": {
            **metrics_out,
            **(best_stats or {}),
            "n_pairs_total": int(total_pairs),
            "keypoint_top_k": int(top_k),
            "image_preprocess": str(
                getattr(preprocess_cfg, "mode", "max_edge")
                if preprocess_cfg is not None
                else "max_edge"
            ),
        }
    }
