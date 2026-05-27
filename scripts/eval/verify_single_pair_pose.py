#!/usr/bin/env python3
"""
单对图像 + 单对 NPZ（仅 matches_im0 / matches_im1，形状 (N,2)）验证位姿估计与真值的角度偏差。

真值与估计流程与 eval_pairs_pose_auc.py 一致：
  - 真值：两帧 *_extra.json 平面位姿 + Tvc -> T_cam0_to_cam1
  - 估计：去畸变 + findEssentialMat + recoverPose（见 eval 内 _estimate_pose_recover_pose）
  - 误差：旋转角误差、平移方向角误差；metric=pose_max 时为 max(二者)

用法示例：
  python3 scripts/eval/verify_single_pair_pose.py \\
    --image0 /path/a.jpg --image1 /path/b.jpg \\
    --matches-npz /path/matches.npz \\
    --camera-json /path/camera.json --extrinsic-json /path/extrinsic.json \\
    --theta-degrees --viz-out /tmp/pair_viz.png \\
    --exclude-top-fraction 0.12 --viz-inliers-only
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

_EVAL_PATH = Path(__file__).resolve().parent / "eval_pairs_pose_auc.py"
_spec = importlib.util.spec_from_file_location("eval_pairs_pose_auc", _EVAL_PATH)
_ep = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_ep)

_VIZ_PATH = Path(__file__).resolve().parent / "viz_pairs_matches.py"
_vz_spec = importlib.util.spec_from_file_location("viz_pairs_matches", _VIZ_PATH)
_vz = importlib.util.module_from_spec(_vz_spec)
assert _vz_spec.loader is not None
_vz_spec.loader.exec_module(_vz)


def _load_matches_npz(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    d = np.load(str(path), allow_pickle=True)
    if "matches_im0" not in d or "matches_im1" not in d:
        raise KeyError("NPZ 需包含键 matches_im0、matches_im1，形状均为 (N,2)")
    m0 = np.asarray(d["matches_im0"], dtype=np.float64).reshape(-1, 2)
    m1 = np.asarray(d["matches_im1"], dtype=np.float64).reshape(-1, 2)
    if m0.shape[0] != m1.shape[0]:
        raise ValueError(f"matches 行数不一致: {m0.shape[0]} vs {m1.shape[0]}")
    return m0, m1


def main() -> int:
    ap = argparse.ArgumentParser(description="单对图像 + 单对 NPZ 位姿偏差（复用 eval_pairs_pose_auc 逻辑）")
    ap.add_argument("--image0", type=Path, required=True)
    ap.add_argument("--image1", type=Path, required=True)
    ap.add_argument("--matches-npz", type=Path, required=True, help="含 matches_im0 / matches_im1 的 npz")
    ap.add_argument("--camera-json", type=Path, required=True)
    ap.add_argument("--extrinsic-json", type=Path, required=True)
    ap.add_argument(
        "--extra-json0",
        type=Path,
        default=None,
        help="第一帧位姿 JSON；默认 image0 同目录 stem_extra.json",
    )
    ap.add_argument(
        "--extra-json1",
        type=Path,
        default=None,
        help="第二帧位姿 JSON；默认 image1 同目录 stem_extra.json",
    )
    ap.add_argument("--pose-x-key", default="value.pose.x")
    ap.add_argument("--pose-y-key", default="value.pose.y")
    ap.add_argument("--pose-theta-key", default="value.pose.theta")
    ap.add_argument("--theta-degrees", action="store_true")
    ap.add_argument("--pose-xy-scale", type=float, default=0.001)
    ap.add_argument("--body-z-world", type=float, default=0.0)
    ap.add_argument(
        "--camera-model",
        choices=("fisheye", "pinhole_plumb", "pinhole_rational8"),
        default="fisheye",
    )
    ap.add_argument("--ransac-threshold", type=float, default=3.0)
    ap.add_argument("--min-matches", type=int, default=8)
    ap.add_argument("--min-inliers", type=int, default=6)
    ap.add_argument(
        "--metric",
        choices=("pose_max", "rotation"),
        default="pose_max",
        help="pose_max: max(旋转误差°, 平移方向角误差°)",
    )
    ap.add_argument("--fail-error-deg", type=float, default=180.0)
    ap.add_argument("--viz-out", type=Path, default=None, help="可选：保存左右拼图+连线 PNG")
    ap.add_argument("--viz-max-edge", type=int, default=1400)
    ap.add_argument("--viz-max-draw", type=int, default=500)
    ap.add_argument("--viz-gap", type=int, default=16)
    ap.add_argument(
        "--exclude-top-fraction",
        type=float,
        default=0.0,
        help="估计位姿与可视化前，去掉两图各自最上方该比例高度内的匹配（粗去天空）；0=关闭",
    )
    ap.add_argument(
        "--viz-inliers-only",
        action="store_true",
        help="与 --viz-out 联用：只绘制极几何内点（单色），不画外点",
    )
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    img0 = args.image0.expanduser().resolve()
    img1 = args.image1.expanduser().resolve()
    ex0 = (args.extra_json0 or _ep._extra_json_for_image(img0)).expanduser().resolve()
    ex1 = (args.extra_json1 or _ep._extra_json_for_image(img1)).expanduser().resolve()

    def auc_scalar(rot_e: float, tdir_e: float) -> float:
        if args.metric == "rotation":
            return rot_e
        return max(rot_e, tdir_e)

    fail_deg = float(args.fail_error_deg)
    out: Dict[str, Any] = {
        "image0": str(img0),
        "image1": str(img1),
        "matches_npz": str(args.matches_npz.expanduser().resolve()),
        "extra_json0": str(ex0),
        "extra_json1": str(ex1),
    }

    try:
        m0, m1 = _load_matches_npz(args.matches_npz.expanduser().resolve())
    except (OSError, KeyError, ValueError) as e:
        out["status"] = f"npz_error:{e}"
        print(json.dumps(out, indent=2, ensure_ascii=False))
        return 2

    n_matches_raw = int(m0.shape[0])
    out["n_matches"] = n_matches_raw

    need_images = float(args.exclude_top_fraction) > 0.0 or args.viz_out is not None
    im0: Optional[np.ndarray] = None
    im1: Optional[np.ndarray] = None
    if need_images:
        im0 = cv2.imread(str(img0), cv2.IMREAD_UNCHANGED)
        im1 = cv2.imread(str(img1), cv2.IMREAD_UNCHANGED)
        if im0 is None or im1 is None:
            out["status"] = "imread_failed"
            out["error_rot_deg"] = fail_deg
            out["error_trans_dir_deg"] = fail_deg
            out["error_pose_deg"] = auc_scalar(fail_deg, fail_deg)
            print(json.dumps(out, indent=2, ensure_ascii=False))
            return 1
        if im0.ndim == 3 and im0.shape[2] == 4:
            im0 = cv2.cvtColor(im0, cv2.COLOR_BGRA2BGR)
        if im1.ndim == 3 and im1.shape[2] == 4:
            im1 = cv2.cvtColor(im1, cv2.COLOR_BGRA2BGR)

    if float(args.exclude_top_fraction) > 0.0:
        if im0 is None or im1 is None:
            out["status"] = "internal_error_images"
            print(json.dumps(out, indent=2, ensure_ascii=False))
            return 2
        h0, h1 = int(im0.shape[0]), int(im1.shape[0])
        m0, m1 = _vz.filter_matches_exclude_image_top(
            m0, m1, h0, h1, float(args.exclude_top_fraction)
        )
        out["exclude_top_fraction"] = float(args.exclude_top_fraction)
        out["n_matches_after_top_filter"] = int(m0.shape[0])

    n_m = int(m0.shape[0])

    if not ex0.is_file() or not ex1.is_file():
        out["status"] = "missing_extra_json"
        out["error_rot_deg"] = fail_deg
        out["error_trans_dir_deg"] = fail_deg
        out["error_pose_deg"] = auc_scalar(fail_deg, fail_deg)
        print(json.dumps(out, indent=2, ensure_ascii=False))
        return 1

    if n_m < args.min_matches:
        out["status"] = "too_few_matches"
        out["error_rot_deg"] = fail_deg
        out["error_trans_dir_deg"] = fail_deg
        out["error_pose_deg"] = auc_scalar(fail_deg, fail_deg)
        print(json.dumps(out, indent=2, ensure_ascii=False))
        return 1

    try:
        x0, y0, th0 = _ep._load_pose_from_extra(
            ex0,
            args.pose_x_key,
            args.pose_y_key,
            args.pose_theta_key,
            args.theta_degrees,
            args.pose_xy_scale,
        )
        x1, y1, th1 = _ep._load_pose_from_extra(
            ex1,
            args.pose_x_key,
            args.pose_y_key,
            args.pose_theta_key,
            args.theta_degrees,
            args.pose_xy_scale,
        )
    except (KeyError, OSError, json.JSONDecodeError, TypeError, ValueError) as e:
        out["status"] = f"pose_load_error:{e}"
        out["error_rot_deg"] = fail_deg
        out["error_trans_dir_deg"] = fail_deg
        out["error_pose_deg"] = auc_scalar(fail_deg, fail_deg)
        print(json.dumps(out, indent=2, ensure_ascii=False))
        return 1

    with args.extrinsic_json.open("r", encoding="utf-8") as f:
        extrinsic_root = json.load(f)
    T_vc = _ep._load_Tvc_extrinsic(extrinsic_root, args.camera_model)
    K, D = _ep._load_camera_json(args.camera_json)

    T_WB0 = _ep._T_world_body_from_planar_pose(x0, y0, th0, args.body_z_world)
    T_WB1 = _ep._T_world_body_from_planar_pose(x1, y1, th1, args.body_z_world)
    T_WC0 = T_WB0 @ T_vc
    T_WC1 = T_WB1 @ T_vc
    T_c0_c1 = np.linalg.inv(T_WC0) @ T_WC1
    R_gt, t_gt = _ep._gt_Rt_from_T_cam0_cam1(T_c0_c1)

    est = _ep._estimate_pose_recover_pose(
        m0,
        m1,
        K,
        D,
        args.camera_model,
        args.ransac_threshold,
        args.min_inliers,
    )
    if est is None:
        out["status"] = "estimate_failed"
        out["error_rot_deg"] = fail_deg
        out["error_trans_dir_deg"] = fail_deg
        out["error_pose_deg"] = auc_scalar(fail_deg, fail_deg)
        print(json.dumps(out, indent=2, ensure_ascii=False))
        return 1

    R_est, t_est, n_epi, n_rec, rec_inlier_mask = est
    rot_e = _ep._rotation_error_deg(R_est, R_gt)
    tdir_e = _ep._translation_error_deg(t_est, t_gt)
    pose_e = max(rot_e, tdir_e)
    auc_e = auc_scalar(rot_e, tdir_e)

    out["status"] = "ok"
    out["metric"] = args.metric
    out["camera_model"] = args.camera_model
    out["n_epipolar_inliers"] = int(n_epi)
    out["recover_pose_inliers"] = int(n_rec)
    out["error_rot_deg"] = float(rot_e)
    out["error_trans_dir_deg"] = float(tdir_e)
    out["error_pose_deg"] = float(pose_e)
    out["error_auc_scalar_deg"] = float(auc_e)

    if args.viz_out is not None:
        if im0 is None or im1 is None:
            im0 = cv2.imread(str(img0), cv2.IMREAD_UNCHANGED)
            im1 = cv2.imread(str(img1), cv2.IMREAD_UNCHANGED)
            if im0 is not None and im1 is not None:
                if im0.ndim == 3 and im0.shape[2] == 4:
                    im0 = cv2.cvtColor(im0, cv2.COLOR_BGRA2BGR)
                if im1.ndim == 3 and im1.shape[2] == 4:
                    im1 = cv2.cvtColor(im1, cv2.COLOR_BGRA2BGR)
        uniform: Optional[Tuple[int, int, int]] = None
        m0v = m0.copy()
        m1v = m1.copy()
        mask_v: Optional[np.ndarray] = None
        if args.viz_inliers_only:
            if rec_inlier_mask is not None and int(np.count_nonzero(rec_inlier_mask)) > 0:
                m0v, m1v = m0v[rec_inlier_mask], m1v[rec_inlier_mask]
                mask_v = None
                uniform = (0, 220, 100)
            elif m0v.shape[0] >= 5:
                mask_v = _vz._epipolar_mask(m0v, m1v, K, D, args.camera_model, args.ransac_threshold)
                if mask_v is not None and int(np.count_nonzero(mask_v)) > 0:
                    m0v, m1v = m0v[mask_v], m1v[mask_v]
                    mask_v = None
                    uniform = (0, 220, 100)
            else:
                mask_v = None
        else:
            if m0v.shape[0] >= 5:
                mask_v = _vz._epipolar_mask(m0v, m1v, K, D, args.camera_model, args.ransac_threshold)

        if im0 is not None and im1 is not None:
            max_edge = int(args.viz_max_edge)
            if max_edge > 0:
                im0, im1, m0v, m1v = _vz._resize_pair_and_matches(im0, im1, m0v, m1v, max_edge)
            panel = _vz._draw_pair_panel(
                im0,
                im1,
                m0v,
                m1v,
                gap=args.viz_gap,
                max_draw=args.viz_max_draw,
                mask=mask_v,
                seed=0,
                uniform_bgr=uniform,
            )
            cap_lines = [
                f"rot_err={rot_e:.2f}deg trans_dir_err={tdir_e:.2f}deg pose_max={pose_e:.2f}deg",
                f"n_epi={n_epi} n_rec={n_rec} n_matches_used={n_m} (raw={n_matches_raw})",
            ]
            if float(args.exclude_top_fraction) > 0.0:
                cap_lines.append(f"exclude_top={float(args.exclude_top_fraction):g}")
        if args.viz_inliers_only:
            cap_lines.append("viz_inliers_only (recoverPose mask)")
            cap = _vz._caption(panel, cap_lines)
            outp = args.viz_out.expanduser().resolve()
            outp.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(outp), cap)
            out["viz_out"] = str(outp)
        else:
            out["viz_out"] = None
            out["viz_error"] = "imread_failed"

    print(json.dumps(out, indent=2, ensure_ascii=False))
    if args.json_out is not None:
        with args.json_out.open("w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"已写入: {args.json_out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
