#!/usr/bin/env python3
"""
基于 covis_adj.json 共视图 + LocalBA.md / Multicamera.md 管道，估计每帧绝对相机/车体位姿。

流程：
  1. 从共视邻帧中选质量最佳的参考对 C1,C2（匹配数 + 基线）
  2. 可选：追加更多共视参考相机（最多 max_ref_cameras），多视图三角化精炼 3D 点
  3. solvePnPRansac 初始化目标相机 C_target
  4. N 相机平面 Local BA（Huber 重投影 + 参考帧 body 先验）
  5. 生成 report.html + viz/<stem>_detail.html（匹配对与 BA 详情，按轨迹数优先）

特征缓存（推荐）：
  1. batch_cache_covis_silk_features.sh  →  SiLK 特征缓存
  2. batch_covis_absolute_pose.sh        →  读缓存做 descriptor match（可选 USE_ANMS=1）
"""

from __future__ import annotations

import argparse
import html
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_EVAL_DIR = _REPO_ROOT / "scripts" / "eval"
if str(_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_EVAL_DIR))

import silk_pair_runtime as spr  # noqa: E402
import silk_feature_cache as sfc  # noqa: E402
from lib.matching.pair_pipeline import EpipolarConfig  # noqa: E402
from covis_pair_pipeline import match_pair_with_epipolar  # noqa: E402
from sequence_pair_cache import SequencePairCacheStore  # noqa: E402

from lib.geometry.absolute_pose_from_covis import (  # noqa: E402
    T_to_serializable,
    T_world_cam_from_planar_extra,
    load_covis_adjacency,
    position_xy_error,
    rotation_error_deg,
)
from lib.geometry.local_ba_planar import (  # noqa: E402
    PlanarBody,
    VisualTrack,
    augment_tracks_with_extra_refs,
    body_baseline_m,
    build_triview_tracks,
    build_tracks_ref_pair_target,
    local_ba_planar,
    local_ba_planar_robust,
    planar_body_from_T_world_cam,
    refine_tracks_multiview,
    solve_pnp_from_tracks,
)

import covis_absolute_pose_viz as capviz  # noqa: E402


def _load_eval_pairs_pose_auc():
    path = _REPO_ROOT / "scripts/eval/eval_pairs_pose_auc.py"
    spec = importlib.util.spec_from_file_location("eval_pairs_pose_auc", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _extra_json_for_image(image_path: Path) -> Path:
    return image_path.parent / f"{image_path.stem}_extra.json"


def _resolve_target_image(
    all_images: Sequence[str],
    spec: str,
    data_dir: Optional[Path] = None,
) -> str:
    """
    在序列节点中解析单帧目标：支持绝对路径、相对 data_dir、文件名或 stem。
    """
    raw = spec.strip()
    if not raw:
        raise ValueError("target-image 不能为空")

    candidates: List[Path] = [Path(raw).expanduser()]
    if data_dir is not None:
        candidates.append((data_dir.expanduser().resolve() / raw))
        if not raw.lower().endswith((".jpg", ".jpeg", ".png")):
            for ext in (".jpg", ".jpeg", ".png"):
                candidates.append(data_dir.expanduser().resolve() / f"{raw}{ext}")

    resolved = {str(Path(p).expanduser().resolve()) for p in all_images}
    for cand in candidates:
        key = str(cand.resolve())
        if key in resolved:
            return key

    base = Path(raw).name
    stem = Path(raw).stem
    by_name = [s for s in all_images if Path(s).name == base]
    if len(by_name) == 1:
        return by_name[0]
    by_stem = [s for s in all_images if Path(s).stem == stem]
    if len(by_stem) == 1:
        return by_stem[0]
    if len(by_name) > 1 or len(by_stem) > 1:
        raise ValueError(f"target-image '{spec}' 在序列中不唯一")
    raise FileNotFoundError(
        f"target-image '{spec}' 不在 covis_adj.json 节点中；"
        f"可用 basename/stem/绝对路径"
    )


def _load_gt_T_world_cam(
    ep,
    image_path: Path,
    Tvc: np.ndarray,
    args: argparse.Namespace,
) -> Optional[np.ndarray]:
    extra = _extra_json_for_image(image_path)
    if not extra.is_file():
        return None
    try:
        return T_world_cam_from_planar_extra(
            extra,
            Tvc,
            pose_x_key=args.pose_x_key,
            pose_y_key=args.pose_y_key,
            pose_theta_key=args.pose_theta_key,
            theta_degrees=args.theta_degrees,
            pose_xy_scale=args.pose_xy_scale,
            body_z_world=args.body_z_world,
            load_pose_from_extra=ep._load_pose_from_extra,
            T_world_body_from_planar_pose=ep._T_world_body_from_planar_pose,
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError, OSError):
        return None


def _load_body_prior(
    ep,
    image_path: Path,
    args: argparse.Namespace,
) -> Optional[PlanarBody]:
    extra = _extra_json_for_image(image_path)
    if not extra.is_file():
        return None
    try:
        x, y, th = ep._load_pose_from_extra(
            extra,
            args.pose_x_key,
            args.pose_y_key,
            args.pose_theta_key,
            args.theta_degrees,
            args.pose_xy_scale,
        )
        return PlanarBody(x=float(x), y=float(y), theta=float(th))
    except (KeyError, TypeError, ValueError, json.JSONDecodeError, OSError):
        return None


def _epipolar_config(args: argparse.Namespace) -> EpipolarConfig:
    return EpipolarConfig(
        camera_model=str(args.camera_model),
        ransac_threshold=float(args.ransac_threshold),
        min_inliers=int(args.min_inliers),
        min_matches=int(args.min_matches),
    )


def _match_with_epipolar_filter(
    rt: spr.PairEvalRuntime,
    img0: Path,
    img1: Path,
    args: argparse.Namespace,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """描述子匹配 → 单文件缓存 → 极线 inlier（scripts/eval 层，不改 lib）。"""
    return match_pair_with_epipolar(
        img0,
        img1,
        descriptor_match_fn=lambda i0, i1: spr.match_pair(rt, i0, i1),
        K=rt.K,
        D=rt.D,
        epipolar=_epipolar_config(args),
        matcher=str(getattr(args, "matcher", "silk")),
        pair_cache=getattr(args, "_pair_cache", None),
    )


def _sorted_ref_pairs(
    neighbors: Sequence[str],
    body_priors: Dict[str, PlanarBody],
) -> List[Tuple[str, str, float]]:
    paths = [p for p in neighbors if p in body_priors]
    pairs: List[Tuple[str, str, float]] = []
    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            d = body_baseline_m(body_priors[paths[i]], body_priors[paths[j]])
            pairs.append((paths[i], paths[j], d))
    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs


def _pair_match_score(meta01: Dict, meta0t: Dict, meta1t: Dict, baseline: float) -> float:
    n12 = int(meta01.get("n_used", 0))
    n13 = int(meta0t.get("n_used", 0))
    n23 = int(meta1t.get("n_used", 0))
    return float(min(n12, n13, n23)) + 0.001 * float(baseline)


def _solve_one_image(
    rt: spr.PairEvalRuntime,
    ep,
    target: Path,
    neighbors: Sequence[str],
    gt_cache: Dict[str, Optional[np.ndarray]],
    body_cache: Dict[str, Optional[PlanarBody]],
    Tvc: np.ndarray,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    target = target.expanduser().resolve()
    tgt_key = str(target)
    max_ref_cameras = int(getattr(args, "max_ref_cameras", 4))
    multiview = bool(getattr(args, "multiview", True))
    rec: Dict[str, Any] = {
        "image": tgt_key,
        "n_covis_neighbors": len(neighbors),
        "n_covis_used": 0,
        "n_tracks": 0,
        "n_pnp_inliers": 0,
        "ref_pair_details": [],
        "solver": "multiview_local_ba" if multiview else "local_ba_triangulation",
    }

    if len(neighbors) < args.min_covis_neighbors:
        rec["status"] = "skipped_low_covis"
        rec["ok"] = False
        rec["reason"] = f"covis_neighbors={len(neighbors)}<{args.min_covis_neighbors}"
        return rec

    nbr_bodies = {p: body_cache.get(p) for p in neighbors if body_cache.get(p) is not None}
    if len(nbr_bodies) < 2:
        rec["status"] = "failed"
        rec["ok"] = False
        rec["reason"] = "insufficient_neighbor_body_priors"
        return rec

    # 预匹配：各共视邻帧 ↔ 目标（Multicamera.md 多视图 PnP 数据准备）
    nbr_match_target: Dict[str, Tuple[np.ndarray, np.ndarray, Dict[str, Any]]] = {}
    for nbr in neighbors:
        if nbr not in nbr_bodies or gt_cache.get(nbr) is None:
            continue
        m0, m1, meta = _match_with_epipolar_filter(rt, Path(nbr), target, args)
        nbr_match_target[nbr] = (m0, m1, meta)

    ref_pairs = _sorted_ref_pairs(neighbors, nbr_bodies)  # type: ignore[arg-type]
    tracks: List[VisualTrack] = []
    chosen_ref: Optional[Tuple[str, str, float]] = None
    chosen_detail: Optional[Dict[str, Any]] = None
    best_score = -1.0

    for ref0, ref1, baseline in ref_pairs:
        if baseline < args.min_baseline_m:
            rec["ref_pair_details"].append(
                {"ref0": ref0, "ref1": ref1, "baseline_m": baseline, "used": False, "reason": "baseline_too_small"}
            )
            continue

        p0, p1 = Path(ref0), Path(ref1)
        T0 = gt_cache.get(ref0)
        T1 = gt_cache.get(ref1)
        if T0 is None or T1 is None:
            rec["ref_pair_details"].append(
                {"ref0": ref0, "ref1": ref1, "baseline_m": baseline, "used": False, "reason": "missing_T_world_cam"}
            )
            continue

        m01_0, m01_1, meta01 = _match_with_epipolar_filter(rt, p0, p1, args)
        if ref0 in nbr_match_target:
            m0t_0, m0t_t, meta0t = nbr_match_target[ref0]
        else:
            m0t_0, m0t_t, meta0t = _match_with_epipolar_filter(rt, p0, target, args)
        if ref1 in nbr_match_target:
            m1t_1, m1t_t, meta1t = nbr_match_target[ref1]
        else:
            m1t_1, m1t_t, meta1t = _match_with_epipolar_filter(rt, p1, target, args)

        score = _pair_match_score(meta01, meta0t, meta1t, baseline)
        detail = {
            "ref0": ref0,
            "ref1": ref1,
            "baseline_m": baseline,
            "match_score": score,
            "match_c12": meta01,
            "match_c13": meta0t,
            "match_c23": meta1t,
        }
        min_ok = args.min_matches
        if m01_0.shape[0] < min_ok or m0t_0.shape[0] < min_ok or m1t_1.shape[0] < min_ok:
            detail["used"] = False
            detail["reason"] = "match_insufficient"
            rec["ref_pair_details"].append(detail)
            continue

        cand, tstats = build_triview_tracks(
            ref0,
            ref1,
            tgt_key,
            m01_0,
            m01_1,
            m0t_0,
            m0t_t,
            m1t_1,
            m1t_t,
            T0,
            T1,
            rt.K,
            rt.D,
            args.camera_model,
            associate_px=args.associate_px,
            c23_agree_px=args.c23_agree_px,
            depth_min=args.depth_min_m,
            depth_max=args.depth_max_m,
        )
        detail["track_stats"] = tstats
        detail["track_mode"] = "triview_c23"

        if len(cand) < args.min_pnp_points:
            cand_fb, fb_stats = build_tracks_ref_pair_target(
                ref0,
                ref1,
                tgt_key,
                m01_0,
                m01_1,
                m0t_0,
                m0t_t,
                T0,
                T1,
                rt.K,
                rt.D,
                args.camera_model,
                associate_px=args.associate_px,
                depth_min=args.depth_min_m,
                depth_max=args.depth_max_m,
            )
            detail["track_mode"] = "fallback_c1_bridge"
            detail["track_stats"] = fb_stats
            detail["n_tracks_fallback"] = len(cand_fb)
            cand = cand_fb

        detail["n_tracks"] = len(cand)
        detail["_m01"] = (m01_0, m01_1)
        detail["_m0t"] = (m0t_0, m0t_t)
        detail["_m1t"] = (m1t_1, m1t_t)

        if len(cand) >= args.min_pnp_points and score >= best_score:
            best_score = score
            tracks = cand
            chosen_ref = (ref0, ref1, baseline)
            chosen_detail = detail
            rec["track_mode"] = detail.get("track_mode")

        if len(cand) >= args.min_pnp_points:
            detail["used"] = True
        else:
            detail["used"] = False
            detail["reason"] = f"too_few_tracks={len(cand)}"
        rec["ref_pair_details"].append(detail)

    if chosen_ref is None or not tracks or chosen_detail is None:
        rec["status"] = "failed"
        rec["ok"] = False
        rec["reason"] = "triangulation_or_tracks_failed"
        return rec

    ref0, ref1, baseline = chosen_ref
    chosen_detail["used"] = True
    rec["ref0"] = ref0
    rec["ref1"] = ref1
    rec["ref_baseline_m"] = baseline
    rec["match_score"] = best_score
    ref_keys: List[str] = [ref0, ref1]

    # 多相机：追加更多共视参考帧观测 + 多视图三角化精炼
    if multiview and max_ref_cameras > 2 and len(nbr_match_target) > 2:
        others = [n for n in neighbors if n not in (ref0, ref1) and n in nbr_match_target]
        others.sort(
            key=lambda n: int(nbr_match_target[n][2].get("n_used", 0)),
            reverse=True,
        )
        extra_refs = others[: max(0, max_ref_cameras - 2)]
        match_bridge_extra: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        match_extra_target: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        for rk in extra_refs:
            m_be0, m_be1, _ = _match_with_epipolar_filter(rt, Path(ref0), Path(rk), args)
            match_bridge_extra[rk] = (m_be0, m_be1)
            mt0, mt1, _ = nbr_match_target[rk]
            match_extra_target[rk] = (mt0, mt1)
        tracks, aug_stats = augment_tracks_with_extra_refs(
            tracks,
            ref0,
            tgt_key,
            extra_refs,
            match_bridge_extra,
            match_extra_target,
            associate_px=args.associate_px,
        )
        ref_keys.extend(extra_refs)
        rec["augment_stats"] = aug_stats
        rec["extra_ref_cameras"] = extra_refs

        T_init = {k: gt_cache[k] for k in ref_keys if gt_cache.get(k) is not None}
        tracks, refine_stats = refine_tracks_multiview(
            tracks,
            ref_keys + [tgt_key],
            T_init,
            rt.K,
            rt.D,
            args.camera_model,
            depth_min=args.depth_min_m,
            depth_max=args.depth_max_m,
        )
        rec["refine_stats"] = refine_stats

    rec["n_covis_used"] = len(ref_keys)
    rec["ref_cameras"] = list(ref_keys)
    rec["n_tracks"] = len(tracks)

    pnp = solve_pnp_from_tracks(
        tracks,
        tgt_key,
        rt.K,
        rt.D,
        args.camera_model,
        min_points=args.min_pnp_points,
        reprojection_threshold=args.pnp_reproj_threshold,
        confidence=args.pnp_confidence,
    )
    if pnp is None:
        rec["status"] = "failed"
        rec["ok"] = False
        rec["reason"] = "pnp_init_failed"
        return rec

    T_pnp = pnp["T_world_cam"]
    rec["n_pnp_inliers"] = int(pnp["n_inliers"])
    rec["T_world_cam_pnp"] = T_to_serializable(T_pnp)

    tgt_key_s = tgt_key
    body_init = {k: nbr_bodies[k] for k in ref_keys}  # type: ignore[index]
    body_init[tgt_key_s] = planar_body_from_T_world_cam(T_pnp, Tvc, args.body_z_world)
    body_prior = {k: nbr_bodies[k] for k in ref_keys}  # type: ignore[index]
    cam_keys = ref_keys + [tgt_key_s]

    _outlier_px = (
        None
        if getattr(args, "no_ba_outlier_filter", False)
        else getattr(args, "ba_outlier_reproj_px", 15.0)
    )
    if _outlier_px is not None and float(_outlier_px) <= 0:
        _outlier_px = None

    ba = local_ba_planar_robust(
        tracks,
        cam_keys,
        body_init,
        body_prior,
        Tvc,
        rt.K,
        rt.D,
        args.camera_model,
        z_world=args.body_z_world,
        prior_weight=args.ba_prior_weight,
        prior_weight_xy=getattr(args, "ba_prior_weight_xy", None),
        prior_weight_yaw=getattr(args, "ba_prior_weight_yaw", None),
        prior_weight_attitude=getattr(args, "ba_prior_weight_attitude", 10.0),
        ref_extra_dofs=not getattr(args, "no_ba_ref_extra_dofs", False),
        target_extra_dofs=not getattr(args, "no_ba_target_extra_dofs", False),
        ref_xy_max_m=getattr(args, "ba_ref_xy_max_m", 0.05),
        ref_yaw_max_deg=getattr(args, "ba_ref_yaw_max_deg", 5.0),
        ref_dz_max_m=getattr(args, "ba_ref_dz_max_m", 0.1),
        ref_angle_max_deg=getattr(args, "ba_ref_angle_max_deg", 1.0),
        target_xy_max_m=getattr(args, "ba_target_xy_max_m", 0.15),
        target_yaw_max_deg=getattr(args, "ba_target_yaw_max_deg", 15.0),
        target_dz_max_m=getattr(args, "ba_target_dz_max_m", 0.1),
        target_angle_max_deg=getattr(args, "ba_target_angle_max_deg", 1.0),
        sigma_xy_m=getattr(args, "ba_sigma_xy_m", 0.03),
        sigma_yaw_deg=getattr(args, "ba_sigma_yaw_deg", 1.5),
        optimize_points=args.ba_optimize_points,
        max_optimize_points=args.ba_max_points,
        huber_px=args.ba_huber_px,
        outlier_reproj_px=_outlier_px,
        outlier_max_passes=getattr(args, "ba_outlier_max_passes", 2),
        outlier_mode=getattr(args, "ba_outlier_mode", "track"),
        min_tracks=4,
    )

    if ba is None:
        T_est = T_pnp
        rec["ba_applied"] = False
        rec["reason_ba"] = "ba_unavailable_or_failed"
    else:
        T_est = ba["T_world_cam"][tgt_key_s]
        rec["ba_applied"] = True
        rec["ba_cost"] = ba["cost"]
        rec["ba_vis_reproj_rmse_px"] = ba["vis_reproj_rmse_px"]
        rec["ba_detail"] = {
            "cost": ba["cost"],
            "vis_reproj_rmse_px": ba["vis_reproj_rmse_px"],
            "per_camera_reproj_rmse_px": ba.get("per_camera_reproj_rmse_px", {}),
            "prior_rmse": ba.get("prior_rmse"),
            "n_observations": ba.get("n_observations"),
            "n_cameras": ba.get("n_cameras"),
            "n_iterations": ba.get("n_iterations"),
            "ref_extra_dofs": ba.get("ref_extra_dofs"),
            "target_extra_dofs": ba.get("target_extra_dofs"),
            "body_init": ba.get("body_init", {}),
            "body_out": ba.get("body", {}),
            "ref_body_init": ba.get("ref_body_init", {}),
            "ref_body_out": ba.get("ref_body", {}),
            "outlier_filter": ba.get("outlier_filter"),
        }
        rec["body_planar"] = ba["body"].get(tgt_key_s)
        for rk in ref_keys:
            rec[f"body_planar_{Path(rk).stem}"] = ba["body"].get(rk)

    rec["status"] = "ok"
    rec["ok"] = True
    rec["T_world_cam"] = T_to_serializable(T_est)

    # 匹配可视化（优先展示质量最好的参考对）
    if getattr(args, "viz_matches", True) and getattr(args, "viz_dir", None):
        viz_root = Path(args.viz_dir)
        stem = target.stem
        m01_0, m01_1 = chosen_detail["_m01"]
        m0t_0, m0t_t = chosen_detail["_m0t"]
        m1t_1, m1t_t = chosen_detail["_m1t"]

        def _panel(tag, p0, p1, m0, m1, cap):
            cx0, cx1 = capviz.track_cross_uvs_for_pair(tracks, p0, p1)
            return (tag, Path(p0), Path(p1), m0, m1, cap, cx0, cx1)

        panels: List[Any] = [
            _panel(
                "c12",
                ref0,
                ref1,
                m01_0,
                m01_1,
                f"C1-C2 n={m01_0.shape[0]} baseline={baseline:.2f}m",
            ),
            _panel(
                "c13",
                ref0,
                target,
                m0t_0,
                m0t_t,
                f"C1-target n={m0t_0.shape[0]}",
            ),
            _panel(
                "c23",
                ref1,
                target,
                m1t_1,
                m1t_t,
                f"C2-target n={m1t_1.shape[0]}",
            ),
        ]
        for rk in rec.get("extra_ref_cameras") or []:
            if rk in nbr_match_target:
                mt0, mt1, meta = nbr_match_target[rk]
                panels.append(
                    _panel(
                        f"extra_{Path(rk).stem[:20]}",
                        rk,
                        target,
                        mt0,
                        mt1,
                        f"extra-target n={meta.get('n_used', 0)}",
                    )
                )
        pair_meta = capviz.write_match_viz_files(viz_root, stem, panels)
        for pm in pair_meta:
            pm["rel_path"] = f"{stem}/{Path(pm['path']).name}"
        rec["viz_pairs"] = pair_meta

        if ba is not None and ba.get("tracks_X_world"):
            body_gt_map: Dict[str, Dict[str, float]] = {}
            T_gt_map: Dict[str, np.ndarray] = {}
            for ck in cam_keys:
                bg = body_cache.get(ck)
                if bg is not None:
                    body_gt_map[ck] = {"x": bg.x, "y": bg.y, "theta": bg.theta}
                Tg = gt_cache.get(ck)
                if Tg is not None:
                    T_gt_map[ck] = Tg
            map_common = dict(
                tracks_X_world=ba["tracks_X_world"],
                cam_keys=cam_keys,
                ref_keys=ref_keys,
                target_key=tgt_key_s,
                body_out=ba.get("body", {}),
                body_gt=body_gt_map,
                T_world_cam_out=ba.get("T_world_cam"),
                T_world_cam_gt=T_gt_map if T_gt_map else None,
                Tvc=Tvc,
                involved_keys=cam_keys,
            )
            map_data = capviz.build_map_viz_data(
                **map_common,
                sequence_cams=getattr(args, "sequence_map_positions", None),
            )
            map_data_local = capviz.build_map_viz_data(
                **map_common,
                sequence_cams=None,
            )
            rec["map_viz"] = map_data
            rec["map_viz_local"] = map_data_local
            rec["map_viz_rel"] = capviz.write_map_viz_file(viz_root, stem, map_data)
            rec["map_viz_local_rel"] = capviz.write_map_viz_file(
                viz_root, stem, map_data_local, fname="map_local_xy.svg", local_only=True
            )

    T_gt = gt_cache.get(tgt_key)
    if T_gt is not None:
        rec["T_world_cam_gt"] = T_to_serializable(T_gt)
        rec["rotation_error_deg"] = rotation_error_deg(T_est[:3, :3], T_gt[:3, :3])
        rec["position_xy_error_m"] = position_xy_error(T_est, T_gt)
        gt_body = body_cache.get(tgt_key)
        if gt_body is not None and rec.get("body_planar"):
            bp = rec["body_planar"]
            rec["body_xy_error_m"] = float(
                math.hypot(bp["x"] - gt_body.x, bp["y"] - gt_body.y)
            )
            rec["body_yaw_error_deg"] = float(
                abs((bp["theta"] - gt_body.theta + math.pi) % (2 * math.pi) - math.pi)
                * 180.0
                / math.pi
            )
    return rec


def run_sequence(args: argparse.Namespace) -> Dict[str, Any]:
    covis_path = (
        args.covis_adj_json.expanduser().resolve()
        if args.covis_adj_json is not None
        else (args.data_dir / "covis_output" / "covis_adj.json").expanduser().resolve()
    )
    if not covis_path.is_file():
        raise FileNotFoundError(f"找不到共视 JSON: {covis_path}")

    adj, nodes = load_covis_adjacency(covis_path)
    all_images = sorted({str(Path(n["image"]).expanduser().resolve()) for n in nodes})

    target_only: Optional[str] = None
    if getattr(args, "target_image", None):
        target_only = _resolve_target_image(all_images, str(args.target_image), args.data_dir)
        print(f"单帧重定位: {Path(target_only).name}", file=sys.stderr)
        process_images = [target_only]
    else:
        process_images = all_images

    ep = _load_eval_pairs_pose_auc()
    with Path(args.extrinsic_json).open("r", encoding="utf-8") as f:
        ext_root = json.load(f)
    Tvc = ep._load_Tvc_extrinsic(ext_root, args.camera_model)

    gt_cache: Dict[str, Optional[np.ndarray]] = {}
    body_cache: Dict[str, Optional[PlanarBody]] = {}
    for img_str in all_images:
        gt_cache[img_str] = _load_gt_T_world_cam(ep, Path(img_str), Tvc, args)
        body_cache[img_str] = _load_body_prior(ep, Path(img_str), args)

    args.sequence_map_positions = capviz.build_sequence_cam_positions(body_cache)

    cache_points_filtered = False
    cache_only = bool(getattr(args, "cache_only", False))
    if args.feature_cache_dir is not None:
        cache_root_probe = args.feature_cache_dir.expanduser().resolve()
        if cache_root_probe.is_dir():
            cache_points_filtered = sfc.FeatureCacheStore(cache_root_probe).points_filtered_in_meta()
            cache_only = True

    ns = spr.namespace_for_batch(
        camera_json=args.camera_json.expanduser().resolve(),
        extrinsic_json=args.extrinsic_json.expanduser().resolve(),
        camera_model=args.camera_model,
        pose_xy_scale=args.pose_xy_scale,
        theta_degrees=args.theta_degrees,
        checkpoint=args.checkpoint.expanduser().resolve() if args.checkpoint else None,
        ransac_threshold=args.ransac_threshold,
        min_matches=args.min_matches,
        min_inliers=args.min_inliers,
        body_z_world=args.body_z_world,
        use_anms=bool(getattr(args, "use_anms", False)),
        anms_top_k=int(getattr(args, "anms_top_k", 2000)),
        anms_min_radius=float(getattr(args, "anms_min_radius", 8.0)),
        cache_only=cache_only,
        matcher=str(getattr(args, "matcher", "silk")),
        lightglue_features=str(getattr(args, "lightglue_features", "disk")),
        lightglue_device=getattr(args, "lightglue_device", None),
    )
    ns.pose_x_key = args.pose_x_key
    ns.pose_y_key = args.pose_y_key
    ns.pose_theta_key = args.pose_theta_key

    if getattr(args, "match_cache_dir", None) is not None:
        mc_root = args.match_cache_dir.expanduser().resolve()
        mc_root.mkdir(parents=True, exist_ok=True)
        epi_cfg = _epipolar_config(args)
        if not getattr(args, "no_epipolar_cache", False):
            args._pair_cache = SequencePairCacheStore(
                mc_root,
                camera_model=epi_cfg.camera_model,
                ransac_threshold=epi_cfg.ransac_threshold,
                min_inliers=epi_cfg.min_inliers,
            )
            print(f"序列匹配缓存（单文件）: {mc_root / 'covis_edges.npz'}", file=sys.stderr)
        else:
            args._pair_cache = None
            print(f"匹配缓存已禁用", file=sys.stderr)
    else:
        args._pair_cache = None

    runtime = spr.build_pair_runtime(ns)
    if args.feature_cache_dir is not None:
        cache_root = args.feature_cache_dir.expanduser().resolve()
        if not cache_root.is_dir():
            raise FileNotFoundError(f"特征缓存目录不存在: {cache_root}")
        runtime.feature_cache = sfc.FeatureCacheStore(cache_root)
        print(f"使用特征缓存: {cache_root}", file=sys.stderr)

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    viz_dir = out_dir / "viz" if getattr(args, "viz_matches", True) else None
    args.viz_dir = viz_dir

    records: List[Dict[str, Any]] = []
    n_skip = n_fail = n_ok = 0
    rot_errs: List[float] = []
    xy_errs: List[float] = []

    for idx, img_str in enumerate(process_images):
        target = Path(img_str)
        neighbors = adj.get(img_str, [])
        print(
            f"[{idx + 1}/{len(process_images)}] {target.name} "
            f"covis={len(neighbors)}",
            file=sys.stderr,
        )
        try:
            rec = _solve_one_image(
                runtime, ep, target, neighbors, gt_cache, body_cache, Tvc, args
            )
        except Exception as e:
            print(
                f"警告: {target.name} 解算异常 ({type(e).__name__}: {e})",
                file=sys.stderr,
            )
            rec = {
                "image": str(target.expanduser().resolve()),
                "n_covis_neighbors": len(neighbors),
                "n_covis_used": 0,
                "n_tracks": 0,
                "n_pnp_inliers": 0,
                "ref_pair_details": [],
                "solver": "local_ba_triangulation",
                "status": "failed",
                "ok": False,
                "reason": f"exception:{type(e).__name__}:{e}",
            }
        records.append(rec)
        if rec.get("status") == "skipped_low_covis":
            n_skip += 1
        elif rec.get("ok"):
            n_ok += 1
            if rec.get("rotation_error_deg") is not None:
                rot_errs.append(float(rec["rotation_error_deg"]))
            if rec.get("position_xy_error_m") is not None:
                xy_errs.append(float(rec["position_xy_error_m"]))
        else:
            n_fail += 1

    pair_cache = getattr(args, "_pair_cache", None)
    if pair_cache is not None:
        pair_cache.flush()
        print(f"匹配缓存已落盘: {pair_cache.archive_path}", file=sys.stderr)

    summary = {
        "seq_id": args.seq_id,
        "covis_adj_json": str(covis_path),
        "n_images_total": len(all_images),
        "n_processed": len(process_images),
        "target_image": target_only,
        "n_skipped_low_covis": n_skip,
        "n_pose_ok": n_ok,
        "n_pose_failed": n_fail,
        "min_covis_neighbors": args.min_covis_neighbors,
        "feature_cache_dir": str(args.feature_cache_dir) if args.feature_cache_dir else None,
        "feature_cache_points_filtered": cache_points_filtered,
        "max_ref_cameras": int(getattr(args, "max_ref_cameras", 4)),
        "multiview": bool(getattr(args, "multiview", True)),
        "use_anms": bool(getattr(args, "use_anms", False)),
        "matcher": str(getattr(args, "matcher", "silk")),
        "match_cache_dir": str(args.match_cache_dir) if getattr(args, "match_cache_dir", None) else None,
        "mean_rotation_error_deg": float(np.mean(rot_errs)) if rot_errs else None,
        "mean_position_xy_error_m": float(np.mean(xy_errs)) if xy_errs else None,
    }

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    clean_records = [_sanitize_record(r) for r in records]

    jsonl_path = out_dir / "per_image.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for rec in clean_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    summary_path = out_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump({"summary": summary, "records": clean_records}, f, indent=2, ensure_ascii=False)

    report_path = out_dir / "report.html"
    capviz.write_sequence_report_html(report_path, args.seq_id, summary, records, covis_path)

    if viz_dir is not None:
        viz_max = 1 if target_only else int(getattr(args, "viz_max_detail", 80))
        ok_sorted = sorted(
            [r for r in records if r.get("ok")],
            key=lambda r: (r.get("n_tracks", 0), r.get("match_score", 0)),
            reverse=True,
        )
        detail_recs = records if target_only else ok_sorted[:viz_max]
        for rec in detail_recs:
            stem = Path(rec["image"]).stem
            capviz.write_image_detail_html(
                viz_dir / f"{stem}_detail.html",
                rec,
                rel_viz_prefix="",
            )

    result = {"summary": summary, "out_dir": str(out_dir), "report_html": str(report_path)}
    if target_only and records:
        result["record"] = _sanitize_record(records[0])
        rec0 = records[0]
        print(
            json.dumps(
                {
                    "target": Path(target_only).name,
                    "status": rec0.get("status"),
                    "ok": rec0.get("ok"),
                    "rotation_error_deg": rec0.get("rotation_error_deg"),
                    "position_xy_error_m": rec0.get("position_xy_error_m"),
                    "n_tracks": rec0.get("n_tracks"),
                    "n_covis_used": rec0.get("n_covis_used"),
                    "detail_html": str(viz_dir / f"{Path(target_only).stem}_detail.html")
                    if viz_dir is not None
                    else None,
                },
                indent=2,
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
    return result


def _sanitize_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    """去掉不可 JSON 序列化的内部字段（如匹配坐标数组）。"""
    out = dict(rec)
    details = []
    for d in rec.get("ref_pair_details") or []:
        dd = {k: v for k, v in d.items() if not str(k).startswith("_")}
        details.append(dd)
    out["ref_pair_details"] = details
    return out


def _write_html_report(
    path: Path,
    seq_id: str,
    summary: Dict[str, Any],
    records: List[Dict[str, Any]],
    covis_path: Path,
) -> None:
    """Deprecated: 使用 covis_absolute_pose_viz.write_sequence_report_html。"""
    capviz.write_sequence_report_html(path, seq_id, summary, records, covis_path)


def _summarize_multi(root: Path, summary_html: Path) -> int:
    root = root.expanduser().resolve()
    seq_results: List[Tuple[str, Dict[str, Any]]] = []
    for p in sorted(root.glob("*/summary.json")):
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        seq_results.append((p.parent.name, data.get("summary", data)))
    if not seq_results:
        print(f"警告：{root} 下无 */summary.json", file=sys.stderr)
        return 1
    rows = []
    for seq_id, s in seq_results:
        rows.append(
            f"<tr><td>{html.escape(seq_id)}</td>"
            f"<td>{s.get('n_pose_ok', 0)}/{s.get('n_images_total', 0)}</td>"
            f"<td>{s.get('n_skipped_low_covis', 0)}</td>"
            f"<td>{s.get('n_pose_failed', 0)}</td>"
            f"<td>{s.get('mean_rotation_error_deg', '')}</td></tr>"
        )
    doc = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8"/>
<title>多序列共视绝对位姿汇总</title>
<style>table{{border-collapse:collapse}}td,th{{border:1px solid #ccc;padding:6px}}</style>
</head><body><h1>多序列汇总</h1>
<table><tr><th>序列</th><th>成功/总数</th><th>跳过</th><th>失败</th><th>平均旋转误差°</th></tr>
{"".join(rows)}</table></body></html>"""
    summary_html.parent.mkdir(parents=True, exist_ok=True)
    summary_html.write_text(doc, encoding="utf-8")
    print(f"写入: {summary_html}", file=sys.stderr)
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--summarize-root", type=Path, default=None, help="多序列汇总 HTML")
    ap.add_argument("--seq-id", default=None)
    ap.add_argument("--data-dir", type=Path, default=None, help="序列根目录（含 covis_output/）")
    ap.add_argument("--covis-adj-json", type=Path, default=None)
    ap.add_argument("--camera-json", type=Path, default=None)
    ap.add_argument("--extrinsic-json", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument(
        "--target-image",
        default=None,
        help="仅评估该目标帧（basename/stem/绝对路径）；用于单帧重定位",
    )
    ap.add_argument("--min-covis-neighbors", type=int, default=3)
    ap.add_argument(
        "--max-ref-cameras",
        type=int,
        default=4,
        help="Local BA 最多使用的已知参考相机数（含 C1,C2；Multicamera.md）",
    )
    ap.add_argument(
        "--no-multiview",
        action="store_true",
        help="禁用多相机扩展，仅使用两参考帧三视图",
    )
    ap.add_argument("--min-baseline-m", type=float, default=0.1, help="参考对最小基线（米），LocalBA 情况 A")
    ap.add_argument("--associate-px", type=float, default=3.0, help="ref 上关联 target 匹配的像素阈值")
    ap.add_argument(
        "--c23-agree-px",
        type=float,
        default=4.0,
        help="C1–C3 与 C2–C3 给出的 target 像素须一致的最大距离",
    )
    ap.add_argument("--depth-min-m", type=float, default=0.5)
    ap.add_argument("--depth-max-m", type=float, default=10.0)
    ap.add_argument("--min-pnp-points", type=int, default=6)
    ap.add_argument("--pnp-reproj-threshold", type=float, default=8.0)
    ap.add_argument("--pnp-confidence", type=float, default=0.999)
    ap.add_argument("--ba-prior-weight", type=float, default=10.0, help="参考帧 yaw 先验权重（legacy；未设 ba-prior-weight-yaw 时生效）")
    ap.add_argument("--ba-prior-weight-xy", type=float, default=500.0, help="参考帧 xy 强先验权重")
    ap.add_argument("--ba-prior-weight-yaw", type=float, default=None, help="参考帧 yaw 先验权重")
    ap.add_argument("--ba-prior-weight-attitude", type=float, default=10.0, help="参考帧 dz/roll/pitch 先验权重")
    ap.add_argument("--ba-ref-xy-max-m", type=float, default=0.05, help="参考帧 xy 硬边界 ±m")
    ap.add_argument("--ba-ref-yaw-max-deg", type=float, default=5.0, help="参考帧 yaw 硬边界 ±deg")
    ap.add_argument("--ba-ref-dz-max-m", type=float, default=0.1, help="参考帧 dz 硬边界 ±m")
    ap.add_argument("--ba-ref-angle-max-deg", type=float, default=1.0, help="参考帧 roll/pitch 硬边界 ±deg")
    ap.add_argument("--ba-sigma-xy-m", type=float, default=0.03, help="参考帧 xy 软先验 σ (m)")
    ap.add_argument("--ba-sigma-yaw-deg", type=float, default=1.5, help="参考帧 yaw 软先验 σ (deg)")
    ap.add_argument(
        "--no-ba-ref-extra-dofs",
        action="store_true",
        help="禁用参考帧 6DOF（dz/roll/pitch），回退纯平面 3DOF",
    )
    ap.add_argument(
        "--no-ba-target-extra-dofs",
        action="store_true",
        help="禁用目标帧 6DOF（dz/roll/pitch），仅优化平面 x,y,θ",
    )
    ap.add_argument("--ba-target-xy-max-m", type=float, default=0.15, help="目标帧 xy 硬边界 ±m")
    ap.add_argument("--ba-target-yaw-max-deg", type=float, default=15.0, help="目标帧 yaw 硬边界 ±deg")
    ap.add_argument("--ba-target-dz-max-m", type=float, default=0.1, help="目标帧 dz 硬边界 ±m")
    ap.add_argument("--ba-target-angle-max-deg", type=float, default=1.0, help="目标帧 roll/pitch 硬边界 ±deg")
    ap.add_argument("--ba-huber-px", type=float, default=2.0)
    ap.add_argument(
        "--ba-outlier-reproj-px",
        type=float,
        default=15.0,
        help="BA 后重投影超阈值 (px) 的 track/观测剔除并重跑 BA；<=0 或 --no-ba-outlier-filter 禁用",
    )
    ap.add_argument(
        "--no-ba-outlier-filter",
        action="store_true",
        help="禁用 BA 迭代外点过滤",
    )
    ap.add_argument(
        "--ba-outlier-max-passes",
        type=int,
        default=2,
        help="BA 最大轮数（含首轮）；每轮后可剔除外点再优化",
    )
    ap.add_argument(
        "--ba-outlier-mode",
        choices=["track", "observation", "target_observation"],
        default="track",
        help="外点过滤模式：整条 track / 任意观测 / 仅 target 观测",
    )
    ap.add_argument("--ba-optimize-points", action="store_true", help="BA 中同时优化 3D 点")
    ap.add_argument("--ba-max-points", type=int, default=150)
    ap.add_argument("--camera-model", default="fisheye")
    ap.add_argument("--pose-xy-scale", type=float, default=0.001)
    ap.add_argument("--theta-degrees", action="store_true")
    ap.add_argument("--body-z-world", type=float, default=0.0)
    ap.add_argument("--pose-x-key", default="value.pose.x")
    ap.add_argument("--pose-y-key", default="value.pose.y")
    ap.add_argument("--pose-theta-key", default="value.pose.theta")
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--ransac-threshold", type=float, default=0.5)
    ap.add_argument("--min-matches", type=int, default=10)
    ap.add_argument("--min-inliers", type=int, default=8)
    ap.add_argument(
        "--matcher",
        choices=["silk", "lightglue"],
        default="silk",
        help="图像对匹配器：silk=mutual NN；lightglue=LightGlue（需 SiLK 特征缓存或现场提取）",
    )
    ap.add_argument(
        "--lightglue-features",
        default="superpoint",
        help="LightGlue 特征头（SiLK 256 维描述子用 superpoint）",
    )
    ap.add_argument(
        "--lightglue-device",
        default=None,
        help="LightGlue 设备，默认 cuda/cpu 自动",
    )
    ap.add_argument(
        "--match-cache-dir",
        type=Path,
        default=None,
        help="匹配对缓存目录（保存极线 RANSAC 之前的 m0/m1；可复跑 RANSAC）",
    )
    ap.add_argument(
        "--no-epipolar-cache",
        action="store_true",
        help="禁用极线 inlier 缓存（默认随 --match-cache-dir 自动开启）",
    )
    ap.add_argument(
        "--feature-cache-dir",
        type=Path,
        default=None,
        help="SiLK 特征缓存根目录（含 features/*.npz）；有则 match 不再跑模型前向",
    )
    ap.add_argument(
        "--cache-only",
        action="store_true",
        help="仅使用特征缓存匹配，不加载 SiLK 模型（缺缓存则失败）",
    )
    ap.add_argument("--use-anms", action="store_true", help="匹配前对关键点做 ANMS 空间均匀化")
    ap.add_argument("--anms-top-k", type=int, default=2000, help="ANMS 保留的最大关键点数")
    ap.add_argument("--anms-min-radius", type=float, default=8.0, help="ANMS 最小 suppression radius（像素）")
    ap.add_argument("--no-viz-matches", action="store_true", help="不生成匹配/BA HTML 可视化")
    ap.add_argument(
        "--viz-max-detail",
        type=int,
        default=80,
        help="最多为多少张成功图像生成详情 HTML（按轨迹数降序）",
    )
    return ap


def _apply_derived_args(args: argparse.Namespace) -> None:
    args.multiview = not getattr(args, "no_multiview", False)
    args.viz_matches = not getattr(args, "no_viz_matches", False)


def main(argv: Optional[List[str]] = None) -> int:
    ap = build_arg_parser()
    args = ap.parse_args(argv)
    _apply_derived_args(args)

    if args.summarize_root is not None:
        if args.out_dir is None:
            print("错误：--summarize-root 需 --out-dir 指定汇总 HTML", file=sys.stderr)
            return 1
        return _summarize_multi(args.summarize_root, args.out_dir)

    missing = [k for k, v in [
        ("seq_id", args.seq_id),
        ("data_dir", args.data_dir),
        ("camera_json", args.camera_json),
        ("extrinsic_json", args.extrinsic_json),
        ("out_dir", args.out_dir),
    ] if not v]
    if missing:
        print(f"错误：缺少参数 {missing}", file=sys.stderr)
        return 1

    try:
        result = run_sequence(args)
    except (FileNotFoundError, ValueError) as e:
        print(f"错误: {e}", file=sys.stderr)
        return 1

    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def run_from_config(cfg) -> Any:
    """供 lib.cli.covis_absolute_pose 调用。"""
    m = cfg.mode
    argv = [
        "--seq-id", str(m.seq_id),
        "--data-dir", str(m.data_dir),
        "--camera-json", str(m.camera_json),
        "--extrinsic-json", str(m.extrinsic_json),
        "--out-dir", str(m.out_dir),
        "--min-covis-neighbors", str(int(m.min_covis_neighbors)),
        "--min-baseline-m", str(float(m.min_baseline_m)),
        "--associate-px", str(float(m.associate_px)),
        "--c23-agree-px", str(float(getattr(m, "c23_agree_px", 4.0))),
        "--depth-min-m", str(float(m.depth_min_m)),
        "--depth-max-m", str(float(m.depth_max_m)),
        "--min-pnp-points", str(int(m.min_pnp_points)),
        "--pnp-reproj-threshold", str(float(m.pnp_reproj_threshold)),
        "--ba-prior-weight", str(float(m.ba_prior_weight)),
        "--ba-huber-px", str(float(m.ba_huber_px)),
        "--ba-max-points", str(int(m.ba_max_points)),
        "--camera-model", str(m.camera_model),
        "--pose-xy-scale", str(float(m.pose_xy_scale)),
        "--ransac-threshold", str(float(m.ransac_threshold)),
        "--min-matches", str(int(m.min_matches)),
        "--min-inliers", str(int(m.min_inliers)),
    ]
    if getattr(m, "max_ref_cameras", None) is not None:
        argv.extend(["--max-ref-cameras", str(int(m.max_ref_cameras))])
    if getattr(m, "no_multiview", False):
        argv.append("--no-multiview")
    if getattr(m, "use_anms", False):
        argv.append("--use-anms")
    if getattr(m, "anms_top_k", None) is not None:
        argv.extend(["--anms-top-k", str(int(m.anms_top_k))])
    if getattr(m, "anms_min_radius", None) is not None:
        argv.extend(["--anms-min-radius", str(float(m.anms_min_radius))])
    if getattr(m, "no_viz_matches", False):
        argv.append("--no-viz-matches")
    if getattr(m, "covis_adj_json", None):
        argv.extend(["--covis-adj-json", str(m.covis_adj_json)])
    if getattr(m, "target_image", None):
        argv.extend(["--target-image", str(m.target_image)])
    if getattr(m, "theta_degrees", False):
        argv.append("--theta-degrees")
    if getattr(m, "checkpoint", None):
        argv.extend(["--checkpoint", str(m.checkpoint)])
    if getattr(m, "feature_cache_dir", None):
        argv.extend(["--feature-cache-dir", str(m.feature_cache_dir)])
    if getattr(m, "matcher", None):
        argv.extend(["--matcher", str(m.matcher)])
    if getattr(m, "lightglue_features", None):
        argv.extend(["--lightglue-features", str(m.lightglue_features)])
    if getattr(m, "match_cache_dir", None):
        argv.extend(["--match-cache-dir", str(m.match_cache_dir)])
    if getattr(m, "ba_optimize_points", False):
        argv.append("--ba-optimize-points")
    return main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
