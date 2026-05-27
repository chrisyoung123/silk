#!/usr/bin/env python3
"""
根据 pairs JSON + 合并的 matches NPZ，用 _extra.json 中的 body 位姿 + 相机外参评估位姿 AUC。

Body 位姿（_extra.json 中的 x, y, theta）按「平面 SE(2)」理解：
  - 位置在世界水平面内 (x, y)，读入后乘以 --pose-xy-scale（默认 0.001，即毫米→米）；
  - 竖直方向高度为常数 --body-z-world（默认 0）；
  - 朝向仅为绕世界竖直轴的 yaw（theta），横滚/俯仰为 0。
  即 body 系相对世界只有 3 自由度，适合地面车辆等；不是一般 6 自由度 IMU 姿态。

相机外参 JSON（见 --extrinsic-json）：
  - 鱼眼内参流程（--camera-model fisheye）：读取 data["Tvc_fish"]["Tvc"] 为 4×4；
    若无嵌套 "Tvc"，则尝试 data["Tvc_fish"] 本身为 4×4。
  - 针孔流程：读取 data["Tvc"] 为 4×4。

约定：Tvc 为齐次 4×4，满足 p_body = Tvc @ p_cam（4 元齐次坐标）。
则 p_world = T_world_body @ p_body = T_world_body @ Tvc @ p_cam，即 T_world_cam = T_world_body @ Tvc。
相机相对真值：T_cam0_to_cam1 = inv(T_world_cam0) @ T_world_cam1，从中取 R_gt 与平移方向 t_gt。

匹配点先去畸变（内参 JSON），再 findEssentialMat + recoverPose；误差定义同 megadepth_pose_tests。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    import cv2
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "需要 OpenCV Python（cv2）。例如: pip install opencv-python-headless"
    ) from e

from lib.geometry.epipolar import estimate_pose_recover_pose, undistort_normalized


def _get_nested(obj: Any, dotted: str) -> Any:
    cur = obj
    for part in dotted.split("."):
        if cur is None or not isinstance(cur, dict):
            raise KeyError(dotted)
        cur = cur[part]
    return cur


def _extra_json_for_image(image_path: Path) -> Path:
    return image_path.parent / f"{image_path.stem}_extra.json"


def _T_world_body_from_planar_pose(
    x: float, y: float, theta_rad: float, z_world: float
) -> np.ndarray:
    """
    Body 在世界系：原点高度 z_world，水平位置 (x,y)，仅 yaw=theta（横滚/俯仰为 0）。
    世界系 z 竖直向上；body 与 world 在 yaw=0 时轴向对齐（body x 沿世界 x）。
    p_world = T_world_body @ p_body。
    """
    c, s = math.cos(theta_rad), math.sin(theta_rad)
    T = np.eye(4, dtype=np.float64)
    T[0, 0], T[0, 1] = c, -s
    T[1, 0], T[1, 1] = s, c
    T[0, 3], T[1, 3], T[2, 3] = x, y, z_world
    return T


def _as_homogeneous_4x4(obj: Any, name: str) -> np.ndarray:
    T = np.asarray(obj, dtype=np.float64).reshape((4, 4))
    return T


def _load_Tvc_extrinsic(data: Dict[str, Any], camera_model: str) -> np.ndarray:
    """
    从外参 JSON 根对象读取 Tvc。
    鱼眼：优先 data['Tvc_fish']['Tvc']，否则 data['Tvc_fish'] 为 4×4。
    针孔：data['Tvc'] 为 4×4（若为 dict 且含 'Tvc' 则再下一层）。
    """
    if camera_model == "fisheye":
        node = data.get("Tvc_fish")
        if node is None:
            raise KeyError("鱼眼模式外参 JSON 缺少顶键 Tvc_fish")
        if isinstance(node, dict) and "Tvc" in node:
            return _as_homogeneous_4x4(node["Tvc"], "Tvc_fish.Tvc")
        return _as_homogeneous_4x4(node, "Tvc_fish")

    node = data.get("Tvc")
    if node is None:
        raise KeyError("针孔模式外参 JSON 缺少顶键 Tvc")
    if isinstance(node, dict) and "Tvc" in node:
        return _as_homogeneous_4x4(node["Tvc"], "Tvc.Tvc")
    return _as_homogeneous_4x4(node, "Tvc")


def _gt_Rt_from_T_cam0_cam1(T_c0_c1: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """T_c0_c1: p_cam0 = T_c0_c1 @ p_cam1（4×4）。取旋转与平移方向。"""
    R_gt = np.asarray(T_c0_c1[:3, :3], dtype=np.float64)
    t_gt = np.asarray(T_c0_c1[:3, 3], dtype=np.float64)
    return R_gt, t_gt


def _rotation_error_deg(R_est: np.ndarray, R_gt: np.ndarray) -> float:
    trace = float(np.trace(R_est @ R_gt.T))
    trace = float(np.clip((trace - 1.0) / 2.0, -1.0, 1.0))
    return float(np.rad2deg(np.arccos(trace)))


def _translation_error_deg(t_est: np.ndarray, t_gt: np.ndarray) -> float:
    t_est = np.asarray(t_est, dtype=np.float64).reshape(3)
    t_gt = np.asarray(t_gt, dtype=np.float64).reshape(3)
    denom = float(np.linalg.norm(t_est) * np.linalg.norm(t_gt))
    if denom < 1e-9:
        return 180.0
    cos_val = float(np.clip(np.dot(t_est, t_gt) / denom, -1.0, 1.0))
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

    auc: Dict[str, float] = {}
    trapz = getattr(np, "trapezoid", None)
    if trapz is None:
        trapz = np.trapz  # type: ignore[attr-defined]
    for t in thresholds:
        last_index = int(np.searchsorted(errors, t))
        rec = np.r_[recall[:last_index], recall[last_index - 1]]
        err = np.r_[errors[:last_index], t]
        auc[f"auc@{int(t)}"] = float(trapz(rec, x=err) / t)
    return auc


def _load_pose_from_extra(
    path: Path,
    pose_x_key: str,
    pose_y_key: str,
    pose_theta_key: str,
    theta_degrees: bool,
    pose_xy_scale: float,
) -> Tuple[float, float, float]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    x = float(_get_nested(data, pose_x_key)) * float(pose_xy_scale)
    y = float(_get_nested(data, pose_y_key)) * float(pose_xy_scale)
    th = float(_get_nested(data, pose_theta_key))
    if theta_degrees:
        th = math.radians(th)
    return x, y, th


def _load_camera_json(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """读取内参 JSON，返回 (K, D)；D 为鱼眼四维列向量 [k1,k2,k3,k4]^T。"""
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    def req(key: str) -> float:
        if key not in raw:
            raise KeyError(f"内参 JSON 缺少键: {key}")
        return float(raw[key])

    fx = req("Camera.fx")
    fy = req("Camera.fy")
    cx = req("Camera.cx")
    cy = req("Camera.cy")
    k1 = req("Camera.k1")
    k2 = req("Camera.k2")
    k3 = req("Camera.k3")
    k4 = req("Camera.k4")

    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    # 鱼眼 4 系数；针孔模型在 _undistort_normalized 里另行组装
    D_fisheye = np.array([[k1], [k2], [k3], [k4]], dtype=np.float64)
    return K, D_fisheye


def _undistort_normalized(
    pts_dist: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
    camera_model: str,
) -> np.ndarray:
    return undistort_normalized(pts_dist, K, D, camera_model)


def _estimate_pose_recover_pose(
    pts0_dist: np.ndarray,
    pts1_dist: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
    camera_model: str,
    pixel_threshold: float,
    min_inliers: int,
) -> Optional[Tuple[np.ndarray, np.ndarray, int, int, np.ndarray]]:
    return estimate_pose_recover_pose(
        pts0_dist,
        pts1_dist,
        K,
        D,
        camera_model,
        pixel_threshold,
        min_inliers,
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description="pairs JSON + merged NPZ + 相机内参 的位姿 AUC（recoverPose）"
    )
    ap.add_argument("--pairs-json", type=Path, required=True, help="含 pairs 列表的 JSON")
    ap.add_argument("--matches-npz", type=Path, required=True, help="all_pairs_merged.npz")
    ap.add_argument(
        "--camera-json",
        type=Path,
        required=True,
        help='内参 JSON（含 "Camera.fx" 等键）',
    )
    ap.add_argument(
        "--extrinsic-json",
        type=Path,
        required=True,
        help='相机相对 body 的外参 JSON：鱼眼读 Tvc_fish.Tvc，针孔读 Tvc（均为 4×4，p_body=Tvc@p_cam）',
    )
    ap.add_argument(
        "--body-z-world",
        type=float,
        default=0.0,
        help="body 原点在世界系竖直轴高度 z（米）；x,y 仍在水平面",
    )
    ap.add_argument(
        "--camera-model",
        choices=("fisheye", "pinhole_plumb", "pinhole_rational8"),
        default="fisheye",
        help=(
            "fisheye: cv2.fisheye.undistortPoints，使用 k1–k4；"
            "pinhole_plumb: cv2.undistortPoints，D=(k1,k2,0,0,k3)，忽略 k4；"
            "pinhole_rational8: D=(k1,k2,0,0,k3,k4,0,0)。"
        ),
    )
    ap.add_argument(
        "--pose-x-key", default="value.pose.x", help="extra JSON 中 x 的点分路径"
    )
    ap.add_argument(
        "--pose-y-key", default="value.pose.y", help="extra JSON 中 y 的点分路径"
    )
    ap.add_argument(
        "--pose-theta-key",
        default="value.pose.theta",
        help="extra JSON 中 theta 的点分路径",
    )
    ap.add_argument(
        "--theta-degrees",
        action="store_true",
        help="JSON 中 theta 为度；否则为弧度",
    )
    ap.add_argument(
        "--pose-xy-scale",
        type=float,
        default=0.001,
        help="对 extra.json 中读出的 x、y 乘以该系数（默认 0.001=毫米→米；若已是米则传 1）",
    )
    ap.add_argument(
        "--ransac-threshold",
        type=float,
        default=3.0,
        help="findEssentialMat RANSAC 阈值（像素）；内部会除以平均焦距换算到归一化平面",
    )
    ap.add_argument(
        "--min-matches",
        type=int,
        default=8,
        help="少于此数量的匹配不参与估计",
    )
    ap.add_argument(
        "--min-inliers",
        type=int,
        default=6,
        help="findEssentialMat RANSAC 极几何内点少于此则视为该对失败（非 recoverPose 的三角化内点）",
    )
    ap.add_argument(
        "--auc-thresholds-deg",
        type=float,
        nargs="+",
        default=[5.0, 10.0, 15.0, 20.0],
        help="AUC 积分上界（度）",
    )
    ap.add_argument(
        "--fail-error-deg",
        type=float,
        default=180.0,
        help="估计失败或文件缺失时记入的误差（度），默认 180",
    )
    ap.add_argument(
        "--metric",
        choices=("pose_max", "rotation"),
        default="pose_max",
        help="pose_max: max(旋转误差°, 平移方向角误差°)；rotation: 仅用旋转误差。",
    )
    ap.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="将指标写入该 JSON 文件",
    )
    args = ap.parse_args()

    def auc_scalar(rot_e: float, tdir_e: float) -> float:
        if args.metric == "rotation":
            return rot_e
        return max(rot_e, tdir_e)

    K, D_fisheye = _load_camera_json(args.camera_json)
    with args.extrinsic_json.open("r", encoding="utf-8") as f:
        extrinsic_root = json.load(f)
    T_vc = _load_Tvc_extrinsic(extrinsic_root, args.camera_model)

    with args.pairs_json.open("r", encoding="utf-8") as f:
        jsondata = json.load(f)
    pairs: List[dict] = jsondata["pairs"]
    d = np.load(str(args.matches_npz), allow_pickle=True)
    offsets = np.asarray(d["match_offsets"]).reshape(-1)
    n_pairs_npz = int(offsets.shape[0] - 1)

    max_index = max(int(p["index"]) for p in pairs)
    if max_index >= n_pairs_npz:
        print(
            f"错误：JSON 中最大 index={max_index}，但 NPZ 仅含 {n_pairs_npz} 对 "
            f"(match_offsets 长度应为 n+1={n_pairs_npz + 1})",
            file=sys.stderr,
        )
        return 2

    pose_errors: List[float] = []
    rot_errors: List[float] = []
    trans_dir_errors: List[float] = []
    n_inliers_list: List[int] = []
    n_matches_list: List[int] = []
    valid_flags: List[bool] = []
    per_pair: List[Dict[str, Any]] = []

    for p in pairs:
        idx = int(p["index"])
        path0 = Path(p["path0"])
        path1 = Path(p["path1"])
        ex0 = _extra_json_for_image(path0)
        ex1 = _extra_json_for_image(path1)

        s, e = int(offsets[idx]), int(offsets[idx + 1])
        m0 = np.asarray(d["matches_im0"][s:e], dtype=np.float64)
        m1 = np.asarray(d["matches_im1"][s:e], dtype=np.float64)
        n_m = int(m0.shape[0])
        n_matches_list.append(n_m)

        rec: Dict[str, Any] = {
            "index": idx,
            "path0": str(path0),
            "path1": str(path1),
            "n_matches": n_m,
        }

        fail_deg = float(args.fail_error_deg)

        if not ex0.is_file() or not ex1.is_file():
            rec["status"] = "missing_extra_json"
            pose_errors.append(auc_scalar(fail_deg, fail_deg))
            rot_errors.append(fail_deg)
            trans_dir_errors.append(fail_deg)
            n_inliers_list.append(0)
            valid_flags.append(False)
            per_pair.append(rec)
            continue

        if n_m < args.min_matches:
            rec["status"] = "too_few_matches"
            pose_errors.append(auc_scalar(fail_deg, fail_deg))
            rot_errors.append(fail_deg)
            trans_dir_errors.append(fail_deg)
            n_inliers_list.append(0)
            valid_flags.append(False)
            per_pair.append(rec)
            continue

        try:
            x0, y0, th0 = _load_pose_from_extra(
                ex0,
                args.pose_x_key,
                args.pose_y_key,
                args.pose_theta_key,
                args.theta_degrees,
                args.pose_xy_scale,
            )
            x1, y1, th1 = _load_pose_from_extra(
                ex1,
                args.pose_x_key,
                args.pose_y_key,
                args.pose_theta_key,
                args.theta_degrees,
                args.pose_xy_scale,
            )
        except (KeyError, OSError, json.JSONDecodeError, TypeError, ValueError) as e:
            rec["status"] = f"pose_load_error:{e}"
            pose_errors.append(auc_scalar(fail_deg, fail_deg))
            rot_errors.append(fail_deg)
            trans_dir_errors.append(fail_deg)
            n_inliers_list.append(0)
            valid_flags.append(False)
            per_pair.append(rec)
            continue

        T_WB0 = _T_world_body_from_planar_pose(x0, y0, th0, args.body_z_world)
        T_WB1 = _T_world_body_from_planar_pose(x1, y1, th1, args.body_z_world)
        T_WC0 = T_WB0 @ T_vc
        T_WC1 = T_WB1 @ T_vc
        T_c0_c1 = np.linalg.inv(T_WC0) @ T_WC1
        R_gt, t_gt = _gt_Rt_from_T_cam0_cam1(T_c0_c1)

        est = _estimate_pose_recover_pose(
            m0,
            m1,
            K,
            D_fisheye,
            args.camera_model,
            args.ransac_threshold,
            args.min_inliers,
        )
        if est is None:
            rec["status"] = "estimate_failed"
            pose_errors.append(auc_scalar(fail_deg, fail_deg))
            rot_errors.append(fail_deg)
            trans_dir_errors.append(fail_deg)
            n_inliers_list.append(0)
            valid_flags.append(False)
            per_pair.append(rec)
            continue

        R_est, t_est, n_epi, n_rec, _rec_mask = est
        rot_e = _rotation_error_deg(R_est, R_gt)
        tdir_e = _translation_error_deg(t_est, t_gt)
        pose_e = max(rot_e, tdir_e)
        auc_e = auc_scalar(rot_e, tdir_e)

        rec["status"] = "ok"
        rec["n_inliers"] = n_epi
        rec["recover_pose_inliers"] = n_rec
        rec["error_rot_deg"] = rot_e
        rec["error_trans_dir_deg"] = tdir_e
        rec["error_pose_deg"] = pose_e
        rec["error_auc_deg"] = auc_e

        pose_errors.append(auc_e)
        rot_errors.append(rot_e)
        trans_dir_errors.append(tdir_e)
        n_inliers_list.append(n_epi)
        valid_flags.append(True)
        per_pair.append(rec)

    auc = _compute_auc(pose_errors, args.auc_thresholds_deg)
    n_total = len(pairs)
    n_valid = int(sum(1 for v in valid_flags if v))

    metrics = {
        **auc,
        "metric": args.metric,
        "camera_model": args.camera_model,
        "body_z_world": args.body_z_world,
        "pose_xy_scale": args.pose_xy_scale,
        "n_pairs_total": n_total,
        "n_pairs_valid_estimate": n_valid,
        "valid_rate": float(n_valid / n_total) if n_total else 0.0,
        "mean_pose_error_deg": float(np.mean(pose_errors)) if pose_errors else float("nan"),
        "mean_rot_error_deg": float(np.mean(rot_errors)) if rot_errors else float("nan"),
        "mean_trans_dir_error_deg": float(np.mean(trans_dir_errors))
        if trans_dir_errors
        else float("nan"),
        "mean_n_matches": float(np.mean(n_matches_list)) if n_matches_list else 0.0,
        "mean_n_inliers": float(np.mean(n_inliers_list)) if n_inliers_list else 0.0,
    }

    print(json.dumps({"metrics": metrics}, indent=2, ensure_ascii=False))
    if args.json_out is not None:
        out = {"metrics": metrics, "per_pair": per_pair, "args": vars(args)}
        out["args"]["pairs_json"] = str(args.pairs_json)
        out["args"]["matches_npz"] = str(args.matches_npz)
        out["args"]["camera_json"] = str(args.camera_json)
        out["args"]["extrinsic_json"] = str(args.extrinsic_json)
        out["args"].pop("json_out", None)
        with args.json_out.open("w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"已写入: {args.json_out}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
