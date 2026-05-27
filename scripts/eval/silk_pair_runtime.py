#!/usr/bin/env python3
"""单进程批量：一次加载 SiLK + 天空分割，循环评估多对图像。"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLES = _REPO_ROOT / "scripts" / "examples"
if str(_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES))

import silk_pair_rotation_error as spi  # noqa: E402
from common import SILK_MATCHER, get_model, load_images  # noqa: E402
from silk.backbones.silk.silk import from_feature_coords_to_image_coords  # noqa: E402
from lib.inference.feature_postprocess import (  # noqa: E402
    apply_point_filters as _apply_point_filters_shared,
    maybe_anms_filter as _maybe_anms_filter_shared,
    point_filter_config_from_namespace,
    silk_positions_to_xy,
)
from lib.inference.keypoint_filters import load_exclude_mask_grayscale  # noqa: E402
from sky_seg_onnx_infer import SkySegOnnxSession, providers_for_device  # noqa: E402

_EVAL_AUC_PATH = _REPO_ROOT / "scripts/eval/eval_pairs_pose_auc.py"
_FEATURE_CACHE_MOD: Any = None


def _feature_cache_module():
    global _FEATURE_CACHE_MOD
    if _FEATURE_CACHE_MOD is None:
        import silk_feature_cache as sfc  # noqa: WPS433

        _FEATURE_CACHE_MOD = sfc
    return _FEATURE_CACHE_MOD


def _load_pose_auc_module():
    name = "eval_pairs_pose_auc"
    spec = importlib.util.spec_from_file_location(name, _EVAL_AUC_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 {_EVAL_AUC_PATH}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@dataclass
class PairEvalRuntime:
    ckpt: Path
    model: Any
    pose_auc: Any
    K: np.ndarray
    D: np.ndarray
    args: argparse.Namespace
    gt_mode: str
    sky_seg: Optional[SkySegOnnxSession] = None
    exclude_mask: Optional[np.ndarray] = None
    vz: Optional[Any] = None
    T_vc: Optional[np.ndarray] = None
    R_gt_fixed: Optional[np.ndarray] = None
    gt_yaw_world_deg_fixed: Optional[float] = None
    feature_cache: Optional[Any] = None
    lightglue_matcher: Optional[Any] = None


def _count_gt_modes(args: argparse.Namespace) -> int:
    return sum(
        [
            args.extrinsic_json is not None,
            args.gt_rel_yaw_deg is not None,
            args.gt_rel_r_json is not None,
        ]
    )


def build_pair_runtime(args: argparse.Namespace) -> PairEvalRuntime:
    if _count_gt_modes(args) != 1:
        raise ValueError(
            "真值来源请三选一：--extrinsic-json / --gt-rel-yaw-deg / --gt-rel-r-json"
        )

    cache_only = bool(getattr(args, "cache_only", False))
    pose_auc = _load_pose_auc_module()
    K, D = pose_auc._load_camera_json(args.camera_json)

    gt_mode = "extra_json"
    T_vc = None
    R_gt_fixed = None
    gt_yaw_fixed = None

    if args.extrinsic_json is not None:
        gt_mode = "extra_json"
        with Path(args.extrinsic_json).open("r", encoding="utf-8") as f:
            ext_root = json.load(f)
        T_vc = pose_auc._load_Tvc_extrinsic(ext_root, args.camera_model)
    elif args.gt_rel_yaw_deg is not None:
        gt_mode = "yaw_deg"
        R_gt_fixed = spi._Rz_deg(float(args.gt_rel_yaw_deg))
        gt_yaw_fixed = spi._wrap_deg(float(args.gt_rel_yaw_deg))
    else:
        gt_mode = "r_matrix_json"
        R_gt_fixed = spi._load_R_gt_from_json(args.gt_rel_r_json)

    model: Any = None
    if cache_only:
        ckpt = Path(args.checkpoint).expanduser().resolve() if args.checkpoint else Path("cache-only")
        print("[runtime] cache-only：跳过 SiLK 模型加载", file=sys.stderr)
    else:
        ckpt = spi._resolve_checkpoint(_REPO_ROOT, args.checkpoint)
        if not ckpt.is_file():
            raise FileNotFoundError(f"找不到权重: {ckpt}")
        print(f"[runtime] 加载 SiLK: {ckpt}", file=sys.stderr)
        model = get_model(
            checkpoint=str(ckpt),
            default_outputs=("sparse_positions", "sparse_descriptors"),
            nms=int(getattr(args, "nms_dist", 9)),
            border=int(getattr(args, "border_dist", 20)),
            top_k=int(getattr(args, "detection_top_k", 20000)),
            threshold=float(getattr(args, "detection_threshold", 1.0)),
        )

    sky_seg: Optional[SkySegOnnxSession] = None
    if args.sky_seg_onnx is not None:
        seg_path = Path(args.sky_seg_onnx).expanduser().resolve()
        if not seg_path.is_file():
            raise FileNotFoundError(f"找不到天空分割 ONNX: {seg_path}")
        norm_style = "imagenet" if args.sky_seg_norm == "imagenet" else "mmseg"
        output_style = "minmax_u8" if args.sky_seg_post == "minmax" else "argmax"
        sky_device = getattr(args, "sky_seg_device", None)
        sky_providers = (
            providers_for_device(str(sky_device)) if sky_device is not None else None
        )
        print(f"[runtime] 加载天空分割: {seg_path}", file=sys.stderr)
        sky_seg = SkySegOnnxSession(
            seg_path,
            norm_style=norm_style,
            output_style=output_style,
            providers=sky_providers,
        )
        print(f"[runtime] 天空分割 providers: {sky_seg.providers}", file=sys.stderr)

    exclude_mask: Optional[np.ndarray] = None
    if args.exclude_mask_png is not None:
        mask_path = Path(args.exclude_mask_png).expanduser().resolve()
        if mask_path.is_file():
            print(f"[runtime] 加载 exclude mask: {mask_path}", file=sys.stderr)
            exclude_mask = load_exclude_mask_grayscale(mask_path)

    return PairEvalRuntime(
        ckpt=ckpt,
        model=model,
        pose_auc=pose_auc,
        K=K,
        D=D,
        args=args,
        gt_mode=gt_mode,
        sky_seg=sky_seg,
        exclude_mask=exclude_mask,
        vz=None,
        T_vc=T_vc,
        R_gt_fixed=R_gt_fixed,
        gt_yaw_world_deg_fixed=gt_yaw_fixed,
    )


def _gt_for_pair(
    rt: PairEvalRuntime, img0: Path, img1: Path
) -> Dict[str, Any]:
    args = rt.args
    if rt.gt_mode == "extra_json":
        ex0 = args.extra0 if args.extra0 is not None else spi._extra_json_for_image(img0)
        ex1 = args.extra1 if args.extra1 is not None else spi._extra_json_for_image(img1)
        if not ex0.is_file():
            raise FileNotFoundError(f"找不到 extra.json: {ex0}")
        if not ex1.is_file():
            raise FileNotFoundError(f"找不到 extra.json: {ex1}")
        x0, y0, th0 = rt.pose_auc._load_pose_from_extra(
            ex0,
            args.pose_x_key,
            args.pose_y_key,
            args.pose_theta_key,
            args.theta_degrees,
            args.pose_xy_scale,
        )
        x1, y1, th1 = rt.pose_auc._load_pose_from_extra(
            ex1,
            args.pose_x_key,
            args.pose_y_key,
            args.pose_theta_key,
            args.theta_degrees,
            args.pose_xy_scale,
        )
        T_WB0 = rt.pose_auc._T_world_body_from_planar_pose(x0, y0, th0, args.body_z_world)
        T_WB1 = rt.pose_auc._T_world_body_from_planar_pose(x1, y1, th1, args.body_z_world)
        T_WC0 = T_WB0 @ rt.T_vc
        T_WC1 = T_WB1 @ rt.T_vc
        T_c0_c1 = np.linalg.inv(T_WC0) @ T_WC1
        R_gt, t_gt = rt.pose_auc._gt_Rt_from_T_cam0_cam1(T_c0_c1)
        return {
            "R_gt": R_gt,
            "t_gt": t_gt,
            "T_WB0": T_WB0,
            "th0": th0,
            "th1": th1,
            "ex0": ex0,
            "ex1": ex1,
            "body_yaw0_deg": math.degrees(th0),
            "body_yaw1_deg": math.degrees(th1),
            "gt_yaw_world_deg": spi._relative_world_yaw_deg_from_thetas(th0, th1),
        }
    if rt.gt_mode == "yaw_deg":
        return {
            "R_gt": rt.R_gt_fixed,
            "t_gt": None,
            "T_WB0": None,
            "th0": None,
            "th1": None,
            "ex0": None,
            "ex1": None,
            "body_yaw0_deg": None,
            "body_yaw1_deg": None,
            "gt_yaw_world_deg": rt.gt_yaw_world_deg_fixed,
        }
    return {
        "R_gt": rt.R_gt_fixed,
        "t_gt": None,
        "T_WB0": None,
        "th0": None,
        "th1": None,
        "ex0": None,
        "ex1": None,
        "body_yaw0_deg": None,
        "body_yaw1_deg": None,
        "gt_yaw_world_deg": None,
    }


def _fail_out(rt: PairEvalRuntime, img0: Path, img1: Path, reason: str, **extra: Any) -> Dict[str, Any]:
    o: Dict[str, Any] = {
        "ok": False,
        "reason": reason,
        "checkpoint": str(rt.ckpt),
        "gt_mode": rt.gt_mode,
        "img0": str(img0),
        "img1": str(img1),
    }
    o.update(extra)
    return o


def _write_pair_viz(
    rt: PairEvalRuntime,
    img0: Path,
    img1: Path,
    viz_out: Path,
    out: Dict[str, Any],
    *,
    m0: np.ndarray,
    m1: np.ndarray,
    kp0_all: Optional[np.ndarray] = None,
    kp1_all: Optional[np.ndarray] = None,
    sky_cls0: Optional[np.ndarray] = None,
    sky_cls1: Optional[np.ndarray] = None,
    rec_inlier_mask: Optional[np.ndarray] = None,
    caption_lines: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """写入左右拼图匹配可视化；失败样本也可调用（caption 由调用方提供）。"""
    args = rt.args
    viz_path = viz_out.expanduser().resolve()
    viz_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if rt.vz is None:
            rt.vz = spi._load_viz_pairs_module(_REPO_ROOT)
        vz = rt.vz
        im0 = cv2.imread(str(img0), cv2.IMREAD_UNCHANGED)
        im1 = cv2.imread(str(img1), cv2.IMREAD_UNCHANGED)
        if im0 is None or im1 is None:
            out["viz_error"] = "imread_failed"
            return out
        if im0.ndim == 3 and im0.shape[2] == 4:
            im0 = cv2.cvtColor(im0, cv2.COLOR_BGRA2BGR)
        if im1.ndim == 3 and im1.shape[2] == 4:
            im1 = cv2.cvtColor(im1, cv2.COLOR_BGRA2BGR)
        m0v = np.asarray(m0, dtype=np.float64).reshape(-1, 2)
        m1v = np.asarray(m1, dtype=np.float64).reshape(-1, 2)
        has_kp = (
            kp0_all is not None
            and kp1_all is not None
            and kp0_all.size > 0
            and kp1_all.size > 0
        )
        if m0v.shape[0] == 0 and not has_kp:
            out["viz_error"] = "no_matches_to_draw"
            return out
        uniform: Optional[Tuple[int, int, int]] = (0, 220, 100)
        draw_mask: Optional[np.ndarray] = None
        if m0v.shape[0] > 0 and m1v.shape[0] > 0:
            if rec_inlier_mask is not None and int(np.count_nonzero(rec_inlier_mask)) > 0:
                m0v, m1v = m0v[rec_inlier_mask], m1v[rec_inlier_mask]
                uniform = (0, 220, 100)
            elif rec_inlier_mask is None and m0v.shape[0] >= 5:
                epi = vz._epipolar_mask(
                    m0v,
                    m1v,
                    rt.K,
                    rt.D,
                    args.camera_model,
                    args.ransac_threshold,
                )
                if epi is not None and int(np.count_nonzero(epi)) > 0:
                    draw_mask = epi
                    uniform = None
        w0_orig = int(im0.shape[1])
        max_edge = int(args.viz_max_edge)
        scale_for_viz = 1.0
        if max_edge > 0:
            im0, im1, m0v, m1v = vz._resize_pair_and_matches(im0, im1, m0v, m1v, max_edge)
            scale_for_viz = float(im0.shape[1]) / max(float(w0_orig), 1.0)
        im0_draw = im0.copy()
        im1_draw = im1.copy()
        if (
            sky_cls0 is not None
            and sky_cls1 is not None
            and args.viz_sky_style != "none"
        ):
            sal_viz = (
                int(args.sky_saliency_threshold)
                if args.sky_seg_post == "minmax"
                else None
            )
            spi._apply_sky_viz_to_image(
                im0_draw,
                sky_cls0,
                args.sky_class_id,
                args.viz_sky_style,
                args.viz_sky_mask_alpha,
                saliency_threshold=sal_viz,
                sky_erode_px=int(getattr(args, "sky_erode_px", 0)),
            )
            spi._apply_sky_viz_to_image(
                im1_draw,
                sky_cls1,
                args.sky_class_id,
                args.viz_sky_style,
                args.viz_sky_mask_alpha,
                saliency_threshold=sal_viz,
                sky_erode_px=int(getattr(args, "sky_erode_px", 0)),
            )
        panel = vz._draw_pair_panel(
            im0_draw,
            im1_draw,
            m0v,
            m1v,
            gap=int(args.viz_gap),
            max_draw=int(args.viz_max_draw),
            mask=draw_mask,
            seed=int(out.get("pair_idx", 0)),
            uniform_bgr=uniform,
        )
        if not getattr(args, "viz_no_keypoints", False) and has_kp:
            kp0v = np.asarray(kp0_all, dtype=np.float64).reshape(-1, 2) * scale_for_viz
            kp1v = np.asarray(kp1_all, dtype=np.float64).reshape(-1, 2) * scale_for_viz
            gap = int(args.viz_gap)
            w0_panel = int(im0.shape[1])
            h0, w0_im = int(im0.shape[0]), int(im0.shape[1])
            h1, w1_im = int(im1.shape[0]), int(im1.shape[1])
            sky0_v = sky1_v = None
            if sky_cls0 is not None and sky_cls1 is not None:
                sal_v = (
                    int(args.sky_saliency_threshold)
                    if args.sky_seg_post == "minmax"
                    else None
                )
                er_px = int(getattr(args, "sky_erode_px", 0))
                sky0_v = spi._erode_sky_binary(
                    spi._sky_binary_u8(
                        sky_cls0, h0, w0_im, args.sky_class_id, saliency_threshold=sal_v
                    ),
                    er_px,
                )
                sky1_v = spi._erode_sky_binary(
                    spi._sky_binary_u8(
                        sky_cls1, h1, w1_im, args.sky_class_id, saliency_threshold=sal_v
                    ),
                    er_px,
                )
            ex0_v = ex1_v = None
            ex_val = int(getattr(args, "exclude_mask_value", 255))
            if rt.exclude_mask is not None:
                ex0_v = spi._exclude_mask_for_hw(rt.exclude_mask, h0, w0_im)
                ex1_v = spi._exclude_mask_for_hw(rt.exclude_mask, h1, w1_im)
            spi._draw_keypoints_classified_inplace(
                panel,
                kp0v,
                sky_u8=sky0_v,
                exclude_u8=ex0_v,
                exclude_value=ex_val,
                radius=int(getattr(args, "viz_keypoint_radius", 1)),
                max_draw=int(getattr(args, "viz_keypoint_max_draw", 0)),
                seed=17,
            )
            kp1_on_panel = kp1v.copy()
            kp1_on_panel[:, 0] += float(w0_panel + gap)
            spi._draw_keypoints_classified_inplace(
                panel,
                kp1_on_panel,
                sky_u8=sky1_v,
                exclude_u8=ex1_v,
                exclude_value=ex_val,
                x_offset=float(w0_panel + gap),
                radius=int(getattr(args, "viz_keypoint_radius", 1)),
                max_draw=int(getattr(args, "viz_keypoint_max_draw", 0)),
                seed=29,
            )
        lines = list(caption_lines or [])
        if not lines:
            lines.append(f"n_matches={m0v.shape[0]}")
        cap = vz._caption(panel, lines)
        viz_save = spi._viz_save_path(viz_path, prefer_jpg=True)
        q = int(getattr(args, "viz_jpeg_quality", spi.VIZ_JPEG_QUALITY_DEFAULT))
        if spi._imwrite_viz_bgr(viz_save, cap, jpeg_quality=q):
            out["viz_out"] = str(viz_save)
        else:
            out["viz_error"] = "imwrite_failed"
    except (FileNotFoundError, RuntimeError, cv2.error, ValueError) as e:
        out["viz_error"] = str(e)
    return out


def extract_image_features(rt: PairEvalRuntime, img_path: Path) -> Any:
    """SiLK 前向：仅提取关键点与描述子（不做 match / 天空过滤）。"""
    sfc = _feature_cache_module()
    img_path = img_path.expanduser().resolve()
    im_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if im_bgr is None:
        raise FileNotFoundError(f"无法读取图像: {img_path}")
    h, w = im_bgr.shape[:2]

    t_in = load_images(str(img_path))
    sp, desc = rt.model(t_in)
    sp = from_feature_coords_to_image_coords(rt.model, sp)
    positions_xy = silk_positions_to_xy(sp[0].detach().cpu().numpy())
    descriptors = desc[0].detach().cpu().numpy().astype(np.float32)
    return sfc.ImageFeatures(
        image_path=str(img_path),
        positions_xy=positions_xy,
        descriptors=descriptors,
        image_hw=(h, w),
        points_filtered=False,
        n_keypoints_raw=int(positions_xy.shape[0]),
    )


def _apply_point_filters(
    rt: PairEvalRuntime,
    img_path: Path,
    positions_xy: np.ndarray,
    descriptors: np.ndarray,
    im_bgr: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """对关键点应用天空 / exclude mask 过滤。"""
    return _apply_point_filters_shared(
        positions_xy,
        descriptors,
        img_path=img_path,
        im_bgr=im_bgr,
        sky_seg=rt.sky_seg,
        exclude_mask=rt.exclude_mask,
        config=point_filter_config_from_namespace(rt.args),
    )


def extract_and_filter_image_features(rt: PairEvalRuntime, img_path: Path) -> Any:
    """SiLK 前向 + 天空/exclude 点过滤，供特征缓存阶段写入。"""
    sfc = _feature_cache_module()
    feat = extract_image_features(rt, img_path)
    if rt.sky_seg is None and rt.exclude_mask is None:
        return feat

    im_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if im_bgr is None:
        raise FileNotFoundError(f"无法读取图像: {img_path}")
    pos_f, desc_f = _apply_point_filters(
        rt, img_path, feat.positions_xy, feat.descriptors, im_bgr=im_bgr
    )
    pos_f, desc_f = _maybe_anms_filter(rt, pos_f, desc_f)
    return sfc.ImageFeatures(
        image_path=feat.image_path,
        positions_xy=pos_f,
        descriptors=desc_f,
        image_hw=feat.image_hw,
        points_filtered=True,
        n_keypoints_raw=int(feat.n_keypoints_raw),
    )


def _maybe_anms_filter(
    rt: PairEvalRuntime,
    positions_xy: np.ndarray,
    descriptors: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """可选 ANMS：使关键点在图像上空间均匀分布。"""
    return _maybe_anms_filter_shared(
        positions_xy,
        descriptors,
        config=point_filter_config_from_namespace(rt.args),
    )


def _filter_cached_features(
    rt: PairEvalRuntime,
    img_path: Path,
    positions_xy: np.ndarray,
    descriptors: np.ndarray,
    im_bgr: Optional[np.ndarray] = None,
    *,
    points_filtered: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """对缓存特征应用天空 / exclude mask（与 match_pair 一致）。"""
    if points_filtered or (rt.sky_seg is None and rt.exclude_mask is None):
        pos, desc = (
            np.asarray(positions_xy, dtype=np.float64).reshape(-1, 2),
            np.asarray(descriptors, dtype=np.float32),
        )
    else:
        pos, desc = _apply_point_filters(rt, img_path, positions_xy, descriptors, im_bgr=im_bgr)
    return _maybe_anms_filter(rt, pos, desc)


def _ensure_lightglue(rt: PairEvalRuntime) -> Any:
    if rt.lightglue_matcher is not None:
        return rt.lightglue_matcher
    from silk.matching import get_pair_matcher

    features = str(getattr(rt.args, "lightglue_features", "disk"))
    device = getattr(rt.args, "lightglue_device", None)
    rt.lightglue_matcher = get_pair_matcher(
        "lightglue",
        lightglue_features=features,
        lightglue_device=device,
        lightglue_flash=bool(getattr(rt.args, "lightglue_flash", True)),
        lightglue_depth_confidence=float(
            getattr(rt.args, "lightglue_depth_confidence", 0.95)
        ),
        lightglue_width_confidence=float(
            getattr(rt.args, "lightglue_width_confidence", 0.99)
        ),
    )
    print(f"[runtime] LightGlue features={features}", file=sys.stderr)
    return rt.lightglue_matcher


def _match_descriptors(
    rt: PairEvalRuntime,
    img0: Path,
    img1: Path,
    pos0: np.ndarray,
    desc0: np.ndarray,
    pos1: np.ndarray,
    desc1: np.ndarray,
    *,
    image_hw0: Optional[Tuple[int, int]] = None,
    image_hw1: Optional[Tuple[int, int]] = None,
) -> Dict[str, Any]:
    import pair_matcher as pm  # noqa: WPS433

    args = rt.args
    if desc0.shape[0] == 0 or desc1.shape[0] == 0:
        return {
            "ok": False,
            "reason": "filter_empty",
            "img0": str(img0),
            "img1": str(img1),
            "m0": np.zeros((0, 2), dtype=np.float64),
            "m1": np.zeros((0, 2), dtype=np.float64),
            "idx0": np.zeros((0,), dtype=np.int64),
            "idx1": np.zeros((0,), dtype=np.int64),
            "n_matches": 0,
            "n_keypoints0": int(pos0.shape[0]),
            "n_keypoints1": int(pos1.shape[0]),
            "matcher": str(getattr(args, "matcher", "silk")),
            "from_cache": True,
        }

    matcher_name = str(getattr(args, "matcher", "silk")).lower()
    lg = _ensure_lightglue(rt) if matcher_name == "lightglue" else None
    out = pm.match_descriptor_pair(
        pos0,
        desc0,
        pos1,
        desc1,
        matcher=matcher_name,
        lightglue=lg,
        image_hw0=image_hw0,
        image_hw1=image_hw1,
        lightglue_device=getattr(args, "lightglue_device", None),
    )
    n_match = int(out["n_matches"])
    return {
        "ok": n_match >= args.min_matches,
        "reason": None if n_match >= args.min_matches else "too_few_matches",
        "img0": str(img0),
        "img1": str(img1),
        "m0": out["m0"],
        "m1": out["m1"],
        "idx0": out["idx0"],
        "idx1": out["idx1"],
        "n_matches": n_match,
        "n_keypoints0": int(out["n_keypoints0"]),
        "n_keypoints1": int(out["n_keypoints1"]),
        "matcher": out["matcher"],
        "from_cache": True,
    }


def match_pair(
    rt: PairEvalRuntime,
    img0: Path,
    img1: Path,
) -> Dict[str, Any]:
    """SiLK 匹配两图；若 runtime.feature_cache 命中则跳过模型前向。"""
    args = rt.args
    img0 = img0.expanduser().resolve()
    img1 = img1.expanduser().resolve()
    cache = rt.feature_cache

    if cache is not None and cache.has(img0) and cache.has(img1):
        f0 = cache.load(img0)
        f1 = cache.load(img1)
        p0, d0 = _filter_cached_features(
            rt, img0, f0.positions_xy, f0.descriptors, points_filtered=f0.points_filtered
        )
        p1, d1 = _filter_cached_features(
            rt, img1, f1.positions_xy, f1.descriptors, points_filtered=f1.points_filtered
        )
        return _match_descriptors(
            rt,
            img0,
            img1,
            p0,
            d0,
            p1,
            d1,
            image_hw0=f0.image_hw,
            image_hw1=f1.image_hw,
        )

    if rt.model is None:
        miss = []
        if cache is None or not cache.has(img0):
            miss.append(img0.name)
        if cache is None or not cache.has(img1):
            miss.append(img1.name)
        return {
            "ok": False,
            "reason": "cache_miss_no_model",
            "img0": str(img0),
            "img1": str(img1),
            "missing_cache": miss,
            "m0": np.zeros((0, 2), dtype=np.float64),
            "m1": np.zeros((0, 2), dtype=np.float64),
            "n_matches": 0,
        }

    t_in = load_images(str(img0))
    t_im1 = load_images(str(img1))
    sp0, d0 = rt.model(t_in)
    sp1, d1 = rt.model(t_im1)
    sp0 = from_feature_coords_to_image_coords(rt.model, sp0)
    sp1 = from_feature_coords_to_image_coords(rt.model, sp1)
    kp0_all = spi._silk_positions_to_xy(sp0[0].detach().cpu().numpy())
    kp1_all = spi._silk_positions_to_xy(sp1[0].detach().cpu().numpy())

    sp0_used = sp0[0]
    sp1_used = sp1[0]
    d0_used = d0[0]
    d1_used = d1[0]
    im0_bgr: Optional[np.ndarray] = None
    im1_bgr: Optional[np.ndarray] = None

    if rt.sky_seg is not None or rt.exclude_mask is not None:
        im0_bgr = cv2.imread(str(img0), cv2.IMREAD_COLOR)
        im1_bgr = cv2.imread(str(img1), cv2.IMREAD_COLOR)
        if im0_bgr is None or im1_bgr is None:
            return {"ok": False, "reason": "imread_failed", "img0": str(img0), "img1": str(img1)}

    if rt.sky_seg is not None:
        assert im0_bgr is not None and im1_bgr is not None
        cls0 = rt.sky_seg.predict(im0_bgr)
        cls1 = rt.sky_seg.predict(im1_bgr)
        sal_thr: Optional[int] = (
            int(args.sky_saliency_threshold) if args.sky_seg_post == "minmax" else None
        )
        sky_erode_px = int(getattr(args, "sky_erode_px", 0))
        keep0 = spi._non_sky_point_mask(
            kp0_all, cls0, args.sky_class_id, saliency_threshold=sal_thr, sky_erode_px=sky_erode_px
        )
        keep1 = spi._non_sky_point_mask(
            kp1_all, cls1, args.sky_class_id, saliency_threshold=sal_thr, sky_erode_px=sky_erode_px
        )
        idx0 = np.nonzero(keep0)[0]
        idx1 = np.nonzero(keep1)[0]
        if idx0.size == 0 or idx1.size == 0:
            return {"ok": False, "reason": "sky_filter_empty", "img0": str(img0), "img1": str(img1)}
        idx0_t = sp0_used.new_tensor(idx0).long()
        idx1_t = sp1_used.new_tensor(idx1).long()
        sp0_used = sp0_used[idx0_t]
        sp1_used = sp1_used[idx1_t]
        d0_used = d0_used[idx0_t]
        d1_used = d1_used[idx1_t]

    if rt.exclude_mask is not None:
        assert im0_bgr is not None and im1_bgr is not None
        h0, w0 = im0_bgr.shape[:2]
        h1, w1 = im1_bgr.shape[:2]
        m0 = spi._exclude_mask_for_hw(rt.exclude_mask, h0, w0)
        m1 = spi._exclude_mask_for_hw(rt.exclude_mask, h1, w1)
        kp0_xy = spi._silk_positions_to_xy(sp0_used.detach().cpu().numpy())
        kp1_xy = spi._silk_positions_to_xy(sp1_used.detach().cpu().numpy())
        ex_val = int(getattr(args, "exclude_mask_value", 255))
        keep0 = spi._non_exclude_mask_point_mask(kp0_xy, m0, exclude_value=ex_val)
        keep1 = spi._non_exclude_mask_point_mask(kp1_xy, m1, exclude_value=ex_val)
        idx0 = np.nonzero(keep0)[0]
        idx1 = np.nonzero(keep1)[0]
        if idx0.size == 0 or idx1.size == 0:
            return {
                "ok": False,
                "reason": "exclude_mask_filter_empty",
                "img0": str(img0),
                "img1": str(img1),
            }
        idx0_t = sp0_used.new_tensor(idx0).long()
        idx1_t = sp1_used.new_tensor(idx1).long()
        sp0_used = sp0_used[idx0_t]
        sp1_used = sp1_used[idx1_t]
        d0_used = d0_used[idx0_t]
        d1_used = d1_used[idx1_t]

    if getattr(args, "use_anms", False):
        from lib.geometry.anms import adaptive_nms  # noqa: WPS433

        pos0 = spi._silk_positions_to_xy(sp0_used.detach().cpu().numpy())
        pos1 = spi._silk_positions_to_xy(sp1_used.detach().cpu().numpy())
        keep0 = adaptive_nms(
            pos0,
            top_k=int(getattr(args, "anms_top_k", 2000)),
            min_radius=float(getattr(args, "anms_min_radius", 0.0)),
        )
        keep1 = adaptive_nms(
            pos1,
            top_k=int(getattr(args, "anms_top_k", 2000)),
            min_radius=float(getattr(args, "anms_min_radius", 0.0)),
        )
        idx0 = np.nonzero(keep0)[0]
        idx1 = np.nonzero(keep1)[0]
        if idx0.size == 0 or idx1.size == 0:
            return {"ok": False, "reason": "anms_filter_empty", "img0": str(img0), "img1": str(img1)}
        idx0_t = sp0_used.new_tensor(idx0).long()
        idx1_t = sp1_used.new_tensor(idx1).long()
        sp0_used = sp0_used[idx0_t]
        sp1_used = sp1_used[idx1_t]
        d0_used = d0_used[idx0_t]
        d1_used = d1_used[idx1_t]

    pos0 = spi._silk_positions_to_xy(sp0_used.detach().cpu().numpy())
    pos1 = spi._silk_positions_to_xy(sp1_used.detach().cpu().numpy())
    desc0 = np.asarray(d0_used.detach().cpu().numpy(), dtype=np.float32)
    desc1 = np.asarray(d1_used.detach().cpu().numpy(), dtype=np.float32)
    if im0_bgr is not None and im1_bgr is not None:
        hw0 = (int(im0_bgr.shape[0]), int(im0_bgr.shape[1]))
        hw1 = (int(im1_bgr.shape[0]), int(im1_bgr.shape[1]))
    else:
        g0 = cv2.imread(str(img0), cv2.IMREAD_GRAYSCALE)
        g1 = cv2.imread(str(img1), cv2.IMREAD_GRAYSCALE)
        if g0 is None or g1 is None:
            return {"ok": False, "reason": "imread_failed", "img0": str(img0), "img1": str(img1)}
        hw0 = (int(g0.shape[0]), int(g0.shape[1]))
        hw1 = (int(g1.shape[0]), int(g1.shape[1]))

    out = _match_descriptors(
        rt,
        img0,
        img1,
        pos0,
        desc0,
        pos1,
        desc1,
        image_hw0=hw0,
        image_hw1=hw1,
    )
    out["from_cache"] = False
    return out


def evaluate_pair(
    rt: PairEvalRuntime,
    img0: Path,
    img1: Path,
    viz_out: Optional[Path] = None,
) -> Dict[str, Any]:
    args = rt.args
    img0 = img0.expanduser().resolve()
    img1 = img1.expanduser().resolve()

    try:
        gt = _gt_for_pair(rt, img0, img1)
    except (FileNotFoundError, KeyError, OSError, json.JSONDecodeError, TypeError, ValueError) as e:
        return _fail_out(rt, img0, img1, f"gt_error:{e}")

    R_gt = gt["R_gt"]
    t_gt = gt.get("t_gt")
    T_WB0 = gt.get("T_WB0")

    t_in = load_images(str(img0))
    t_im1 = load_images(str(img1))
    sp0, d0 = rt.model(t_in)
    sp1, d1 = rt.model(t_im1)
    sp0 = from_feature_coords_to_image_coords(rt.model, sp0)
    sp1 = from_feature_coords_to_image_coords(rt.model, sp1)
    kp0_all = spi._silk_positions_to_xy(sp0[0].detach().cpu().numpy())
    kp1_all = spi._silk_positions_to_xy(sp1[0].detach().cpu().numpy())

    sp0_used = sp0[0]
    sp1_used = sp1[0]
    d0_used = d0[0]
    d1_used = d1[0]
    sky_kept0 = sky_kept1 = None
    sky_removed0 = sky_removed1 = None
    sky_cls0 = sky_cls1 = None
    mask_removed0 = mask_removed1 = None
    im0_bgr: Optional[np.ndarray] = None
    im1_bgr: Optional[np.ndarray] = None

    if rt.sky_seg is not None or rt.exclude_mask is not None:
        im0_bgr = cv2.imread(str(img0), cv2.IMREAD_COLOR)
        im1_bgr = cv2.imread(str(img1), cv2.IMREAD_COLOR)
        if im0_bgr is None or im1_bgr is None:
            return _fail_out(rt, img0, img1, "imread_failed")

    if rt.sky_seg is not None:
        cls0 = rt.sky_seg.predict(im0_bgr)
        cls1 = rt.sky_seg.predict(im1_bgr)
        sky_cls0, sky_cls1 = cls0, cls1
        sal_thr: Optional[int] = (
            int(args.sky_saliency_threshold) if args.sky_seg_post == "minmax" else None
        )
        sky_erode_px = int(getattr(args, "sky_erode_px", 0))
        keep0 = spi._non_sky_point_mask(
            kp0_all,
            cls0,
            args.sky_class_id,
            saliency_threshold=sal_thr,
            sky_erode_px=sky_erode_px,
        )
        keep1 = spi._non_sky_point_mask(
            kp1_all,
            cls1,
            args.sky_class_id,
            saliency_threshold=sal_thr,
            sky_erode_px=sky_erode_px,
        )
        idx0 = np.nonzero(keep0)[0]
        idx1 = np.nonzero(keep1)[0]
        if idx0.size == 0 or idx1.size == 0:
            return _fail_out(rt, img0, img1, "sky_filter_empty")
        idx0_t = sp0_used.new_tensor(idx0).long()
        idx1_t = sp1_used.new_tensor(idx1).long()
        sp0_used = sp0_used[idx0_t]
        sp1_used = sp1_used[idx1_t]
        d0_used = d0_used[idx0_t]
        d1_used = d1_used[idx1_t]
        sky_kept0 = int(idx0.size)
        sky_kept1 = int(idx1.size)
        sky_removed0 = int(kp0_all.shape[0] - idx0.size)
        sky_removed1 = int(kp1_all.shape[0] - idx1.size)

    if rt.exclude_mask is not None:
        assert im0_bgr is not None and im1_bgr is not None
        h0, w0 = im0_bgr.shape[:2]
        h1, w1 = im1_bgr.shape[:2]
        m0 = spi._exclude_mask_for_hw(rt.exclude_mask, h0, w0)
        m1 = spi._exclude_mask_for_hw(rt.exclude_mask, h1, w1)
        kp0_xy = spi._silk_positions_to_xy(sp0_used.detach().cpu().numpy())
        kp1_xy = spi._silk_positions_to_xy(sp1_used.detach().cpu().numpy())
        ex_val = int(getattr(args, "exclude_mask_value", 255))
        keep0 = spi._non_exclude_mask_point_mask(kp0_xy, m0, exclude_value=ex_val)
        keep1 = spi._non_exclude_mask_point_mask(kp1_xy, m1, exclude_value=ex_val)
        idx0 = np.nonzero(keep0)[0]
        idx1 = np.nonzero(keep1)[0]
        if idx0.size == 0 or idx1.size == 0:
            return _fail_out(rt, img0, img1, "exclude_mask_filter_empty")
        n0_before = int(sp0_used.shape[0])
        n1_before = int(sp1_used.shape[0])
        idx0_t = sp0_used.new_tensor(idx0).long()
        idx1_t = sp1_used.new_tensor(idx1).long()
        sp0_used = sp0_used[idx0_t]
        sp1_used = sp1_used[idx1_t]
        d0_used = d0_used[idx0_t]
        d1_used = d1_used[idx1_t]
        mask_removed0 = n0_before - int(idx0.size)
        mask_removed1 = n1_before - int(idx1.size)

    matches = SILK_MATCHER(d0_used, d1_used)
    n_match = int(matches.shape[0])
    if n_match > 0:
        m0 = spi._silk_positions_to_xy(sp0_used[matches[:, 0]].detach().cpu().numpy())
        m1 = spi._silk_positions_to_xy(sp1_used[matches[:, 1]].detach().cpu().numpy())
    else:
        m0 = np.zeros((0, 2), dtype=np.float64)
        m1 = np.zeros((0, 2), dtype=np.float64)

    if n_match < args.min_matches:
        out_fail = _fail_out(
            rt,
            img0,
            img1,
            "too_few_matches",
            n_matches=n_match,
        )
        if viz_out is not None and n_match > 0:
            out_fail = _write_pair_viz(
                rt,
                img0,
                img1,
                viz_out,
                out_fail,
                m0=m0,
                m1=m1,
                kp0_all=kp0_all,
                kp1_all=kp1_all,
                sky_cls0=sky_cls0,
                sky_cls1=sky_cls1,
                caption_lines=[f"FAIL too_few_matches n={n_match}"],
            )
        return out_fail

    est = rt.pose_auc._estimate_pose_recover_pose(
        m0,
        m1,
        rt.K,
        rt.D,
        args.camera_model,
        args.ransac_threshold,
        args.min_inliers,
    )
    if est is None:
        out_fail = _fail_out(
            rt,
            img0,
            img1,
            "recover_pose_failed",
            n_matches=n_match,
        )
        if viz_out is not None and n_match > 0:
            out_fail = _write_pair_viz(
                rt,
                img0,
                img1,
                viz_out,
                out_fail,
                m0=m0,
                m1=m1,
                kp0_all=kp0_all,
                kp1_all=kp1_all,
                sky_cls0=sky_cls0,
                sky_cls1=sky_cls1,
                caption_lines=[f"FAIL recover_pose n_matches={n_match}"],
            )
        return out_fail

    R_est, t_est, n_epi, n_rec, rec_inlier_mask = est
    R_est_np = np.asarray(R_est, dtype=np.float64).reshape(3, 3)
    err_deg = float(rt.pose_auc._rotation_error_deg(R_est_np, R_gt))
    t_est_np = np.asarray(t_est, dtype=np.float64).reshape(3)
    trans_err_deg: Optional[float] = None
    if t_gt is not None:
        trans_err_deg = float(rt.pose_auc._translation_error_deg(t_est_np, t_gt))

    gt_yaw_world_deg = gt.get("gt_yaw_world_deg")
    if rt.gt_mode == "extra_json" and gt.get("th0") is not None:
        est_yaw_world_deg = spi._relative_world_yaw_deg_from_cam_est(
            R_est_np, t_est_np, T_WB0, rt.T_vc
        )
    else:
        est_yaw_world_deg = None

    world_yaw_error_deg: Optional[float] = None
    if est_yaw_world_deg is not None and gt_yaw_world_deg is not None:
        world_yaw_error_deg = spi._wrap_deg(est_yaw_world_deg - gt_yaw_world_deg)

    out: Dict[str, Any] = {
        "ok": True,
        "gt_mode": rt.gt_mode,
        "rotation_error_deg": err_deg,
        "translation_direction_error_deg": trans_err_deg,
        "estimated_yaw_world_deg": est_yaw_world_deg,
        "gt_yaw_world_deg": gt_yaw_world_deg,
        "world_yaw_error_deg": world_yaw_error_deg,
        "n_keypoints_img0": int(kp0_all.shape[0]),
        "n_keypoints_img1": int(kp1_all.shape[0]),
        "n_matches": int(matches.shape[0]),
        "n_epipolar_inliers": int(n_epi),
        "recover_pose_inliers": int(n_rec),
        "checkpoint": str(rt.ckpt),
        "img0": str(img0),
        "img1": str(img1),
    }
    if rt.gt_mode == "extra_json":
        out["extra0"] = str(gt["ex0"])
        out["extra1"] = str(gt["ex1"])
        out["extrinsic_json"] = str(args.extrinsic_json)
    if rt.sky_seg is not None:
        out["sky_seg_onnx"] = str(Path(args.sky_seg_onnx).resolve())
        out["n_keypoints_after_sky_img0"] = sky_kept0
        out["n_keypoints_after_sky_img1"] = sky_kept1
        out["n_keypoints_removed_sky_img0"] = sky_removed0
        out["n_keypoints_removed_sky_img1"] = sky_removed1
        out["sky_erode_px"] = int(getattr(args, "sky_erode_px", 0))
    if rt.exclude_mask is not None:
        out["exclude_mask_png"] = str(Path(args.exclude_mask_png).resolve())
        out["exclude_mask_value"] = int(getattr(args, "exclude_mask_value", 255))
        out["n_keypoints_removed_exclude_mask_img0"] = mask_removed0
        out["n_keypoints_removed_exclude_mask_img1"] = mask_removed1

    if viz_out is not None:
        cap_lines = [
            f"cam_rot_err={err_deg:.2f}deg",
            f"n_epi={n_epi} n_rec={n_rec} n_matches={n_match}",
        ]
        if world_yaw_error_deg is not None:
            cap_lines[0] += f" world_yaw_err={world_yaw_error_deg:.2f}deg"
        if trans_err_deg is not None:
            cap_lines[0] += f" trans_dir_err={trans_err_deg:.2f}deg"
        out = _write_pair_viz(
            rt,
            img0,
            img1,
            viz_out,
            out,
            m0=m0,
            m1=m1,
            kp0_all=kp0_all,
            kp1_all=kp1_all,
            sky_cls0=sky_cls0,
            sky_cls1=sky_cls1,
            rec_inlier_mask=rec_inlier_mask,
            caption_lines=cap_lines,
        )

    return out


def build_feature_extract_runtime(args: argparse.Namespace) -> PairEvalRuntime:
    """仅加载 SiLK + 可选 sky/mask，供特征缓存（不需要相机内参/外参）。"""
    ckpt = spi._resolve_checkpoint(_REPO_ROOT, args.checkpoint)
    if not ckpt.is_file():
        raise FileNotFoundError(f"找不到权重: {ckpt}")
    print(f"[runtime] 加载 SiLK（feature extract）: {ckpt}", file=sys.stderr)
    model = get_model(
        checkpoint=str(ckpt),
        default_outputs=("sparse_positions", "sparse_descriptors"),
        nms=int(getattr(args, "nms_dist", 9)),
        border=int(getattr(args, "border_dist", 20)),
        top_k=int(getattr(args, "detection_top_k", 20000)),
        threshold=float(getattr(args, "detection_threshold", 1.0)),
    )

    sky_seg: Optional[SkySegOnnxSession] = None
    if args.sky_seg_onnx is not None:
        seg_path = Path(args.sky_seg_onnx).expanduser().resolve()
        if not seg_path.is_file():
            raise FileNotFoundError(f"找不到天空分割 ONNX: {seg_path}")
        norm_style = "imagenet" if args.sky_seg_norm == "imagenet" else "mmseg"
        output_style = "minmax_u8" if args.sky_seg_post == "minmax" else "argmax"
        sky_device = getattr(args, "sky_seg_device", None)
        sky_providers = (
            providers_for_device(str(sky_device)) if sky_device is not None else None
        )
        print(f"[runtime] 加载天空分割: {seg_path}", file=sys.stderr)
        sky_seg = SkySegOnnxSession(
            seg_path,
            norm_style=norm_style,
            output_style=output_style,
            providers=sky_providers,
        )

    exclude_mask: Optional[np.ndarray] = None
    if args.exclude_mask_png is not None:
        mask_path = Path(args.exclude_mask_png).expanduser().resolve()
        if mask_path.is_file():
            print(f"[runtime] 加载 exclude mask: {mask_path}", file=sys.stderr)
            exclude_mask = load_exclude_mask_grayscale(mask_path)

    return PairEvalRuntime(
        ckpt=ckpt,
        model=model,
        pose_auc=None,
        K=np.empty((0, 0)),
        D=np.empty((0,)),
        args=args,
        gt_mode="feature_extract",
        sky_seg=sky_seg,
        exclude_mask=exclude_mask,
    )


def namespace_for_feature_extract(
    *,
    checkpoint: Optional[Path] = None,
    sky_seg_onnx: Optional[Path] = None,
    sky_seg_device: Optional[str] = None,
    sky_seg_norm: str = "imagenet",
    sky_seg_post: str = "minmax",
    sky_saliency_threshold: int = 128,
    sky_erode_px: int = 0,
    sky_class_id: int = 1,
    exclude_mask_png: Optional[Path] = None,
    exclude_mask_value: int = 255,
    nms_dist: int = 9,
    border_dist: int = 20,
    detection_top_k: int = 20000,
    detection_threshold: float = 1.0,
    use_anms: bool = False,
    anms_top_k: int = 2000,
    anms_min_radius: float = 8.0,
) -> argparse.Namespace:
    return argparse.Namespace(
        checkpoint=checkpoint,
        sky_seg_onnx=sky_seg_onnx,
        sky_seg_device=sky_seg_device,
        sky_class_id=sky_class_id,
        sky_seg_norm=sky_seg_norm,
        sky_seg_post=sky_seg_post,
        sky_saliency_threshold=sky_saliency_threshold,
        sky_erode_px=sky_erode_px,
        exclude_mask_png=exclude_mask_png,
        exclude_mask_value=exclude_mask_value,
        nms_dist=nms_dist,
        border_dist=border_dist,
        detection_top_k=detection_top_k,
        detection_threshold=detection_threshold,
        use_anms=use_anms,
        anms_top_k=anms_top_k,
        anms_min_radius=anms_min_radius,
    )


def namespace_for_batch(
    *,
    camera_json: Path,
    extrinsic_json: Path,
    camera_model: str = "fisheye",
    pose_xy_scale: float = 0.001,
    theta_degrees: bool = True,
    checkpoint: Optional[Path] = None,
    sky_seg_onnx: Optional[Path] = None,
    sky_seg_device: Optional[str] = None,
    sky_seg_norm: str = "imagenet",
    sky_seg_post: str = "minmax",
    sky_saliency_threshold: int = 128,
    sky_erode_px: int = 0,
    sky_class_id: int = 1,
    ransac_threshold: float = 0.5,
    min_matches: int = 10,
    min_inliers: int = 8,
    body_z_world: float = 0.0,
    viz_max_edge: int = 1400,
    viz_max_draw: int = 500,
    viz_gap: int = 16,
    viz_sky_style: str = "both",
    viz_sky_mask_alpha: float = 0.28,
    viz_no_keypoints: bool = False,
    exclude_mask_png: Optional[Path] = None,
    exclude_mask_value: int = 255,
    viz_jpeg_quality: int = 88,
    nms_dist: int = 9,
    border_dist: int = 20,
    detection_top_k: int = 20000,
    detection_threshold: float = 1.0,
    use_anms: bool = False,
    anms_top_k: int = 2000,
    anms_min_radius: float = 8.0,
    cache_only: bool = False,
    matcher: str = "silk",
    lightglue_features: str = "disk",
    lightglue_device: Optional[str] = None,
    lightglue_flash: bool = True,
    lightglue_depth_confidence: float = 0.95,
    lightglue_width_confidence: float = 0.99,
) -> argparse.Namespace:
    return argparse.Namespace(
        img0=None,
        img1=None,
        extra0=None,
        extra1=None,
        extrinsic_json=extrinsic_json,
        pose_x_key="value.pose.x",
        pose_y_key="value.pose.y",
        pose_theta_key="value.pose.theta",
        theta_degrees=theta_degrees,
        pose_xy_scale=pose_xy_scale,
        body_z_world=body_z_world,
        checkpoint=checkpoint,
        camera_json=camera_json,
        camera_model=camera_model,
        gt_rel_yaw_deg=None,
        gt_rel_r_json=None,
        ransac_threshold=ransac_threshold,
        min_matches=min_matches,
        min_inliers=min_inliers,
        sky_seg_onnx=sky_seg_onnx,
        sky_seg_device=sky_seg_device,
        sky_class_id=sky_class_id,
        sky_seg_norm=sky_seg_norm,
        sky_seg_post=sky_seg_post,
        sky_saliency_threshold=sky_saliency_threshold,
        sky_erode_px=sky_erode_px,
        viz_out=None,
        viz_max_edge=viz_max_edge,
        viz_max_draw=viz_max_draw,
        viz_gap=viz_gap,
        viz_no_keypoints=viz_no_keypoints,
        viz_keypoint_max_draw=0,
        viz_keypoint_radius=1,
        viz_debug_points=0,
        viz_sky_style=viz_sky_style,
        viz_sky_mask_alpha=viz_sky_mask_alpha,
        exclude_mask_png=exclude_mask_png,
        exclude_mask_value=exclude_mask_value,
        viz_jpeg_quality=viz_jpeg_quality,
        nms_dist=nms_dist,
        border_dist=border_dist,
        detection_top_k=detection_top_k,
        detection_threshold=detection_threshold,
        use_anms=use_anms,
        anms_top_k=anms_top_k,
        anms_min_radius=anms_min_radius,
        cache_only=cache_only,
        matcher=matcher,
        lightglue_features=lightglue_features,
        lightglue_device=lightglue_device,
        lightglue_flash=lightglue_flash,
        lightglue_depth_confidence=lightglue_depth_confidence,
        lightglue_width_confidence=lightglue_width_confidence,
    )
