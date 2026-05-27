#!/usr/bin/env python3
"""
两张图 + 相机内参 + 真值：用 SiLK 匹配 + recoverPose 估计相对位姿，
输出与真值之间的旋转角偏差（度）；在提供 --extrinsic-json（可从平面位姿推出 t_gt）时，
另输出平移方向角偏差（度，与 eval_pairs_pose_auc 中 _translation_error_deg 一致）。

可选 --viz-out：保存左右拼图 + 匹配连线，仅绘制 recoverPose 内点（单色）。

真值来源（三选一，按优先级）：
  1) 提供 --extrinsic-json 时：从两张图对应的 *_extra.json（或 --extra0/--extra1）
     读取 body 平面位姿 value.pose.{x,y,theta}，经外参 Tvc 链式得到相机相对旋转 R_gt，
     与 eval_pairs_pose_auc.py 约定一致（p_body = Tvc @ p_cam）。
  2) --gt-rel-yaw-deg：直接给定相对 yaw（度），R_gt = Rz(...)。
  3) --gt-rel-r-json：直接给定 3×3 相对旋转矩阵。

权重默认优先使用仓库根下 checkpoints/*.ckpt（按名字排序取第一个），否则使用
  assets/models/silk/coco-rgb-aug.ckpt
可用 --checkpoint 显式指定。

用法（真值在 extra.json，与图同目录、文件名为 <图主名>_extra.json）：
  python scripts/examples/silk_pair_rotation_error.py \\
    --img0 a.jpg --img1 b.jpg \\
    --camera-json cam.json --extrinsic-json ext.json \\
    --camera-model fisheye --theta-degrees --pose-xy-scale 0.001 \\
    --viz-out /tmp/pair_inliers.png

环境变量 SILK_DEVICE 可覆盖运算设备（默认 cuda:0 或 cpu）。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import importlib
import math
import sys
from pathlib import Path
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np

# 与 silk-inference 相同：以本脚本目录为基准找到 common
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from common import (  # noqa: E402
    SILK_MATCHER,
    get_model,
    load_images,
)
from sky_seg_onnx_infer import SkySegOnnxSession  # noqa: E402
from silk.backbones.silk.silk import from_feature_coords_to_image_coords  # noqa: E402
from lib.inference.keypoint_filters import (  # noqa: E402
    erode_sky_binary as _erode_sky_binary,
    exclude_mask_for_hw as _exclude_mask_for_hw,
    load_exclude_mask_grayscale as _load_exclude_mask_grayscale,
    non_exclude_mask_point_mask as _non_exclude_mask_point_mask,
    non_sky_point_mask as _non_sky_point_mask,
    sky_binary_u8 as _sky_binary_u8,
)


def _resolve_checkpoint(repo: Path, user: Optional[Path]) -> Path:
    if user is not None:
        return user.resolve()
    ck_dir = repo / "checkpoints"
    if ck_dir.is_dir():
        cks = sorted(ck_dir.glob("*.ckpt"))
        if cks:
            return cks[0].resolve()
    fallback = repo / "assets/models/silk/coco-rgb-aug.ckpt"
    return fallback.resolve()


def _load_viz_pairs_module(repo: Path):
    path = repo / "scripts/eval/viz_pairs_matches.py"
    if not path.is_file():
        raise FileNotFoundError(f"缺少可视化模块: {path}")
    spec = importlib.util.spec_from_file_location("viz_pairs_matches", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("无法加载 viz_pairs_matches 模块")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_pose_auc_module(repo: Path):
    path = repo / "scripts/eval/eval_pairs_pose_auc.py"
    if not path.is_file():
        raise FileNotFoundError(f"缺少评估模块: {path}")
    spec = importlib.util.spec_from_file_location("eval_pairs_pose_auc", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("无法加载 eval_pairs_pose_auc 模块")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _extra_json_for_image(image_path: Path) -> Path:
    return image_path.parent / f"{image_path.stem}_extra.json"


def _Rz_deg(deg: float) -> np.ndarray:
    r = math.radians(deg)
    c, s = math.cos(r), math.sin(r)
    R = np.eye(3, dtype=np.float64)
    R[0, 0], R[0, 1] = c, -s
    R[1, 0], R[1, 1] = s, c
    return R


def _silk_positions_to_xy(pts: np.ndarray) -> np.ndarray:
    """SiLK sparse_positions 经 from_feature_coords_to_image_coords 后为 (row, col)=(y, x)，转为 OpenCV (x, y)。"""
    pts = np.asarray(pts, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] < 2:
        raise ValueError(f"期望关键点 shape 为 Nx(>=2)，当前为 {pts.shape}")
    # 只取前两列坐标，忽略第三列概率，避免 reshape 把概率列掺入坐标。
    pts_yx = pts[:, :2]
    return pts_yx[:, [1, 0]]


def _wrap_deg(deg: float) -> float:
    return float((deg + 180.0) % 360.0 - 180.0)


def _relative_world_yaw_deg_from_thetas(th0_rad: float, th1_rad: float) -> float:
    """extra 平面位姿 theta 为世界系 body yaw（弧度），相对 yaw = th1 - th0。"""
    return _wrap_deg(math.degrees(th1_rad - th0_rad))


def _T_from_Rt(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(R, dtype=np.float64).reshape(3, 3)
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


def _relative_world_yaw_deg_from_cam_est(
    R_est: np.ndarray,
    t_est: np.ndarray,
    T_WB0: np.ndarray,
    T_vc: np.ndarray,
) -> float:
    """recoverPose 的 cam0←cam1 经 T_WC=T_WB@T_vc 链到 world body 相对 yaw（度）。"""
    T_c0_c1 = _T_from_Rt(R_est, t_est)
    T_WC0 = T_WB0 @ T_vc
    T_WB1_est = T_WC0 @ T_c0_c1 @ np.linalg.inv(T_vc)
    T_b0_b1 = np.linalg.inv(T_WB0) @ T_WB1_est
    R_b0_b1 = T_b0_b1[:3, :3]
    return _wrap_deg(float(np.rad2deg(np.arctan2(R_b0_b1[1, 0], R_b0_b1[0, 0]))))


def _log_pose_angles(
    *,
    n_matches: int,
    n_epi: int,
    n_rec: int,
    rotation_error_deg: float,
    est_yaw_world_deg: Optional[float],
    gt_yaw_world_deg: Optional[float],
    world_yaw_error_deg: Optional[float],
    translation_direction_error_deg: Optional[float],
    est_t: np.ndarray,
    gt_t: Optional[np.ndarray],
    body_yaw0_deg: Optional[float] = None,
    body_yaw1_deg: Optional[float] = None,
) -> None:
    print(
        f"[pose] 匹配数={n_matches} 极几何内点={n_epi} recoverPose内点={n_rec}",
        file=sys.stderr,
    )
    print(
        f"[pose] 相机相对旋转误差={rotation_error_deg:.2f}°（R_est vs R_gt，recoverPose 坐标系）",
        file=sys.stderr,
    )
    if body_yaw0_deg is not None and body_yaw1_deg is not None:
        print(
            f"[pose] body 世界 yaw: frame0={body_yaw0_deg:.2f}° frame1={body_yaw1_deg:.2f}°",
            file=sys.stderr,
        )
    if est_yaw_world_deg is not None and gt_yaw_world_deg is not None:
        print(
            f"[pose] 世界系相对 yaw: 估计={est_yaw_world_deg:.2f}° 真值={gt_yaw_world_deg:.2f}° "
            f"误差={world_yaw_error_deg:.2f}°",
            file=sys.stderr,
        )
    elif gt_yaw_world_deg is not None:
        print(
            f"[pose] 世界系相对 yaw 真值={gt_yaw_world_deg:.2f}°（估计需 --extrinsic-json + extra 位姿）",
            file=sys.stderr,
        )
    if translation_direction_error_deg is not None and gt_t is not None:
        t_est_u = np.asarray(est_t, dtype=np.float64).reshape(3)
        t_gt_u = np.asarray(gt_t, dtype=np.float64).reshape(3)
        ne = float(np.linalg.norm(t_est_u))
        ng = float(np.linalg.norm(t_gt_u))
        if ne > 1e-12:
            t_est_u = t_est_u / ne
        if ng > 1e-12:
            t_gt_u = t_gt_u / ng
        print(
            f"[pose] 平移方向误差={translation_direction_error_deg:.2f}° "
            f"(估计 t̂={t_est_u.round(4).tolist()} 真值 t̂={t_gt_u.round(4).tolist()})",
            file=sys.stderr,
        )


def _select_draw_subset(
    m0: np.ndarray,
    m1: np.ndarray,
    *,
    max_draw: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """复现 _draw_pair_panel 的抽样逻辑，得到真正会被绘制的点子集。"""
    n = int(m0.shape[0])
    if max_draw > 0 and n > max_draw:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=max_draw, replace=False)
    else:
        idx = np.arange(n, dtype=np.int64)
    return idx, m0[idx], m1[idx]


def _log_viz_points(
    *,
    m0_draw: np.ndarray,
    m1_draw: np.ndarray,
    selected_idx: np.ndarray,
    max_print: int,
) -> None:
    n = int(m0_draw.shape[0])
    print(f"[viz] 实际绘制点数={n}", file=sys.stderr)
    if n == 0:
        return
    print(
        "[viz] img0(x,y)范围="
        f"([{m0_draw[:, 0].min():.2f}, {m0_draw[:, 0].max():.2f}], "
        f"[{m0_draw[:, 1].min():.2f}, {m0_draw[:, 1].max():.2f}])",
        file=sys.stderr,
    )
    print(
        "[viz] img1(x,y)范围="
        f"([{m1_draw[:, 0].min():.2f}, {m1_draw[:, 0].max():.2f}], "
        f"[{m1_draw[:, 1].min():.2f}, {m1_draw[:, 1].max():.2f}])",
        file=sys.stderr,
    )
    show_n = min(max(0, int(max_print)), n)
    if show_n <= 0:
        return
    order = np.argsort(m0_draw[:, 1], kind="stable")
    print(f"[viz] 打印最靠上 {show_n} 个绘制点（按 img0.y 升序）:", file=sys.stderr)
    for rank, pos in enumerate(order[:show_n]):
        p0 = m0_draw[pos]
        p1 = m1_draw[pos]
        src_idx = int(selected_idx[pos]) if selected_idx.shape[0] == n else -1
        print(
            f"  [{rank:02d}] idx={src_idx} "
            f"p0=({p0[0]:.2f},{p0[1]:.2f}) "
            f"p1=({p1[0]:.2f},{p1[1]:.2f})",
            file=sys.stderr,
        )


def _draw_keypoints_inplace(
    image: np.ndarray,
    points_xy: np.ndarray,
    *,
    color_bgr: Tuple[int, int, int],
    radius: int = 1,
    max_draw: int = 0,
    seed: int = 1,
) -> int:
    """在单张图上绘制关键点；返回实际绘制数量。"""
    pts = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    n = int(pts.shape[0])
    if n <= 0:
        return 0

    if max_draw > 0 and n > max_draw:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=max_draw, replace=False)
        pts = pts[idx]

    h, w = image.shape[:2]
    draw_n = 0
    rr = max(1, int(radius))
    for p in pts:
        x, y = int(round(float(p[0]))), int(round(float(p[1])))
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(image, (x, y), rr, color_bgr, -1, lineType=cv2.LINE_AA)
            draw_n += 1
    return draw_n


def _keypoint_mark_red(
    x: int,
    y: int,
    h: int,
    w: int,
    *,
    sky_u8: Optional[np.ndarray] = None,
    exclude_u8: Optional[np.ndarray] = None,
    exclude_value: int = 255,
) -> bool:
    """可视化用：天空点标红；有 exclude mask 时，mask 以外（非排除区）的点也标红。"""
    is_sky = False
    if sky_u8 is not None and 0 <= y < h and 0 <= x < w:
        is_sky = int(sky_u8[y, x]) > 0
    if is_sky:
        return True
    if exclude_u8 is None:
        return False
    if not (0 <= y < h and 0 <= x < w):
        return False
    return int(exclude_u8[y, x]) != int(exclude_value)


def _draw_keypoints_classified_inplace(
    image: np.ndarray,
    points_xy: np.ndarray,
    *,
    sky_u8: Optional[np.ndarray] = None,
    exclude_u8: Optional[np.ndarray] = None,
    exclude_value: int = 255,
    x_offset: float = 0.0,
    radius: int = 1,
    max_draw: int = 0,
    seed: int = 1,
    color_red_bgr: Tuple[int, int, int] = (0, 0, 255),
    color_other_bgr: Tuple[int, int, int] = (255, 180, 0),
) -> Tuple[int, int]:
    """按天空 / exclude mask 分类绘制关键点；返回 (红色数, 其他色数)。

    x_offset：绘制在拼图 panel 上时，右图需减去 (w0+gap) 再与 sky/mask 对齐。
    """
    pts = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    n = int(pts.shape[0])
    if n <= 0:
        return 0, 0
    if max_draw > 0 and n > max_draw:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=max_draw, replace=False)
        pts = pts[idx]
    canvas_h, canvas_w = int(image.shape[0]), int(image.shape[1])
    if sky_u8 is not None:
        mask_h, mask_w = int(sky_u8.shape[0]), int(sky_u8.shape[1])
    elif exclude_u8 is not None:
        mask_h, mask_w = int(exclude_u8.shape[0]), int(exclude_u8.shape[1])
    else:
        mask_h, mask_w = canvas_h, canvas_w
    x_off = int(round(float(x_offset)))
    rr = max(1, int(radius))
    n_red = n_other = 0
    for p in pts:
        x_d = int(round(float(p[0])))
        y_d = int(round(float(p[1])))
        if not (0 <= x_d < canvas_w and 0 <= y_d < canvas_h):
            continue
        x_m = x_d - x_off
        y_m = y_d
        if _keypoint_mark_red(
            x_m,
            y_m,
            mask_h,
            mask_w,
            sky_u8=sky_u8,
            exclude_u8=exclude_u8,
            exclude_value=exclude_value,
        ):
            col = color_red_bgr
            n_red += 1
        else:
            col = color_other_bgr
            n_other += 1
        cv2.circle(image, (x_d, y_d), rr, col, -1, lineType=cv2.LINE_AA)
    return n_red, n_other


VIZ_JPEG_QUALITY_DEFAULT = 88


def _viz_save_path(path: Path, prefer_jpg: bool = True) -> Path:
    """批量默认可视化用 .jpg；用户显式指定 .png 时保留。"""
    p = path.expanduser().resolve()
    if prefer_jpg and p.suffix.lower() not in (".jpg", ".jpeg", ".png"):
        return p.with_suffix(".jpg")
    return p


def _imwrite_viz_bgr(path: Path, image_bgr: np.ndarray, *, jpeg_quality: int = VIZ_JPEG_QUALITY_DEFAULT) -> bool:
    p = _viz_save_path(path, prefer_jpg=True)
    ext = p.suffix.lower()
    if ext in (".jpg", ".jpeg"):
        return bool(
            cv2.imwrite(
                str(p),
                image_bgr,
                [int(cv2.IMWRITE_JPEG_QUALITY), int(np.clip(jpeg_quality, 1, 100))],
            )
        )
    return bool(cv2.imwrite(str(p), image_bgr))


def _apply_sky_viz_to_image(
    image_bgr: np.ndarray,
    cls_map: np.ndarray,
    sky_class_id: int,
    style: str,
    mask_alpha: float,
    *,
    saliency_threshold: Optional[int] = None,
    sky_erode_px: int = 0,
) -> None:
    """在 BGR 图上叠加强调天空区域：半透明着色 + 外轮廓（就地修改 image_bgr）。"""
    if style == "none":
        return
    h, w = image_bgr.shape[:2]
    sky = _sky_binary_u8(
        cls_map, h, w, sky_class_id, saliency_threshold=saliency_threshold
    )
    sky = _erode_sky_binary(sky, sky_erode_px)
    if not np.any(sky):
        return
    if style in ("mask", "both"):
        a = float(np.clip(mask_alpha, 0.0, 1.0))
        if a > 0.0:
            color = np.array([255, 90, 90], dtype=np.float64)
            bgr = image_bgr.astype(np.float64)
            m = sky.astype(bool)
            bgr[m] = bgr[m] * (1.0 - a) + color * a
            image_bgr[:, :, :] = np.clip(bgr, 0, 255).astype(np.uint8)
    if style in ("contour", "both"):
        sky_u8 = (sky * 255).astype(np.uint8)
        cnts, _h = cv2.findContours(sky_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image_bgr, cnts, -1, (0, 255, 255), 2, lineType=cv2.LINE_AA)


def _load_R_gt_from_json(path: Path) -> np.ndarray:
    with path.open("r", encoding="utf-8") as f:
        data: Any = json.load(f)
    if isinstance(data, dict):
        if "R" in data:
            data = data["R"]
        elif "r" in data:
            data = data["r"]
    R = np.asarray(data, dtype=np.float64)
    if R.shape != (3, 3):
        raise ValueError(f"真值旋转须为 3×3，当前 shape={R.shape}")
    return R


def main() -> int:
    ap = argparse.ArgumentParser(description="SiLK 双图相对位姿误差（旋转角；外参真值时含平移方向角）")
    ap.add_argument("--img0", type=Path, required=True)
    ap.add_argument("--img1", type=Path, required=True)
    ap.add_argument(
        "--extra0",
        type=Path,
        default=None,
        help="图0 的 extra.json；默认与图0 同目录 <stem>_extra.json",
    )
    ap.add_argument(
        "--extra1",
        type=Path,
        default=None,
        help="图1 的 extra.json；默认与图1 同目录 <stem>_extra.json",
    )
    ap.add_argument(
        "--extrinsic-json",
        type=Path,
        default=None,
        help="与 eval 相同的外参 JSON（Tvc_fish.Tvc / Tvc）；提供时从 extra 计算 R_gt",
    )
    ap.add_argument(
        "--pose-x-key",
        default="value.pose.x",
        help="extra 中 x 的点分路径",
    )
    ap.add_argument(
        "--pose-y-key",
        default="value.pose.y",
        help="extra 中 y 的点分路径",
    )
    ap.add_argument(
        "--pose-theta-key",
        default="value.pose.theta",
        help="extra 中 theta 的点分路径",
    )
    ap.add_argument(
        "--theta-degrees",
        action="store_true",
        help="extra 中 theta 为度",
    )
    ap.add_argument(
        "--pose-xy-scale",
        type=float,
        default=0.001,
        help="extra 中 x,y 乘以该系数（默认 0.001 毫米→米；已为米时用 1）",
    )
    ap.add_argument(
        "--body-z-world",
        type=float,
        default=0.0,
        help="body 原点在世界 z（米），与 eval 一致",
    )
    ap.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="SiLK .ckpt；默认 checkpoints 下首个 .ckpt 或 assets/models/silk/coco-rgb-aug.ckpt",
    )
    ap.add_argument(
        "--camera-json",
        type=Path,
        required=True,
        help='内参 JSON（Camera.fx / fy / cx / cy / k1–k4）',
    )
    ap.add_argument(
        "--camera-model",
        choices=("fisheye", "pinhole_plumb", "pinhole_rational8"),
        default="fisheye",
    )
    ap.add_argument("--gt-rel-yaw-deg", type=float, default=None, help="手动相对 yaw 真值（度）")
    ap.add_argument(
        "--gt-rel-r-json",
        type=Path,
        default=None,
        help="手动 3×3 相对旋转真值 JSON",
    )
    ap.add_argument("--ransac-threshold", type=float, default=0.5)
    ap.add_argument("--min-matches", type=int, default=10)
    ap.add_argument("--min-inliers", type=int, default=8)
    ap.add_argument(
        "--sky-seg-onnx",
        type=Path,
        default=None,
        help="可选：天空分割 ONNX 模型路径；启用后会过滤天空点",
    )
    ap.add_argument(
        "--sky-class-id",
        type=int,
        default=1,
        help="多类分割（--sky-seg-post argmax）时天空类别 id；显著图模式（minmax）下忽略，改用 --sky-saliency-threshold",
    )
    ap.add_argument(
        "--sky-seg-norm",
        choices=("mmseg", "imagenet"),
        default="mmseg",
        help="天空 ONNX 前处理：mmseg=SegDataPreProcessor；imagenet=(x/255-mean)/std（如 skyseg.onnx）",
    )
    ap.add_argument(
        "--sky-seg-post",
        choices=("argmax", "minmax"),
        default="argmax",
        help="ONNX 输出解析：argmax=多类 logits；minmax=全局归一化 uint8 显著图（如 U2-Net/skyseg）",
    )
    ap.add_argument(
        "--sky-saliency-threshold",
        type=int,
        default=128,
        help="仅 --sky-seg-post minmax：显著图 > 该值(0~255) 视为天空，用于过滤与可视化",
    )
    ap.add_argument(
        "--sky-erode-px",
        type=int,
        default=0,
        help="天空二值 mask 腐蚀半径（像素，椭圆核）；>0 时内缩天空区，交界不确定带当作非天空保留点",
    )
    ap.add_argument(
        "--exclude-mask-png",
        type=Path,
        default=None,
        help="可选：灰度 mask.png；像素值==255 处的关键点丢弃（与图像尺寸不一致时最近邻缩放到原图）",
    )
    ap.add_argument(
        "--exclude-mask-value",
        type=int,
        default=255,
        help="与 --exclude-mask-png 联用：该灰度值视为排除区域",
    )
    ap.add_argument(
        "--viz-out",
        type=Path,
        default=None,
        help="可选：保存左右拼图 + 连线 PNG，仅绘制 recoverPose 内点",
    )
    ap.add_argument("--viz-max-edge", type=int, default=1400, help="长边超过则缩小再绘制，0=不缩放")
    ap.add_argument("--viz-max-draw", type=int, default=500, help="最多绘制的连线数（0=不限制）")
    ap.add_argument("--viz-gap", type=int, default=16, help="左右图间隙（像素）")
    ap.add_argument(
        "--viz-no-keypoints",
        action="store_true",
        help="不绘制原始检测关键点（默认绘制）",
    )
    ap.add_argument(
        "--viz-keypoint-max-draw",
        type=int,
        default=0,
        help="每张图最多绘制的原始关键点数量（0=全部）",
    )
    ap.add_argument(
        "--viz-keypoint-radius",
        type=int,
        default=1,
        help="原始关键点圆点半径（像素）",
    )
    ap.add_argument(
        "--viz-debug-points",
        type=int,
        default=0,
        help="打印实际绘制点（按 y 从上到下），值为打印条数；0=不打印",
    )
    ap.add_argument(
        "--viz-sky-style",
        choices=("none", "mask", "contour", "both"),
        default="both",
        help="若启用 --sky-seg-onnx：在可视化图上叠加强调天空（半透明 / 轮廓 / 二者）",
    )
    ap.add_argument(
        "--viz-sky-mask-alpha",
        type=float,
        default=0.28,
        help="天空区域半透明叠加强度（0~1，仅 mask/both 时生效）",
    )
    args = ap.parse_args()

    n_gt_modes = sum(
        [
            args.extrinsic_json is not None,
            args.gt_rel_yaw_deg is not None,
            args.gt_rel_r_json is not None,
        ]
    )
    if n_gt_modes != 1:
        print(
            "错误：真值来源请三选一且仅选其一：\n"
            "  --extrinsic-json（从两张图对应的 *_extra.json 读位姿并算 R_gt）\n"
            "  或 --gt-rel-yaw-deg\n"
            "  或 --gt-rel-r-json",
            file=sys.stderr,
        )
        return 2

    ckpt = _resolve_checkpoint(_REPO_ROOT, args.checkpoint)
    if not ckpt.is_file():
        print(f"错误：找不到权重文件: {ckpt}", file=sys.stderr)
        return 2

    pose_auc = _load_pose_auc_module(_REPO_ROOT)
    K, D = pose_auc._load_camera_json(args.camera_json)
    t_gt: Optional[np.ndarray] = None
    T_WB0: Optional[np.ndarray] = None
    T_vc: Optional[np.ndarray] = None
    th0 = th1 = None
    body_yaw0_deg = body_yaw1_deg = None

    if args.extrinsic_json is not None:
        with args.extrinsic_json.open("r", encoding="utf-8") as f:
            ext_root = json.load(f)
        T_vc = pose_auc._load_Tvc_extrinsic(ext_root, args.camera_model)
        ex0 = args.extra0 if args.extra0 is not None else _extra_json_for_image(args.img0)
        ex1 = args.extra1 if args.extra1 is not None else _extra_json_for_image(args.img1)
        if not ex0.is_file():
            print(f"错误：找不到 extra.json: {ex0}", file=sys.stderr)
            return 2
        if not ex1.is_file():
            print(f"错误：找不到 extra.json: {ex1}", file=sys.stderr)
            return 2
        try:
            x0, y0, th0 = pose_auc._load_pose_from_extra(
                ex0,
                args.pose_x_key,
                args.pose_y_key,
                args.pose_theta_key,
                args.theta_degrees,
                args.pose_xy_scale,
            )
            x1, y1, th1 = pose_auc._load_pose_from_extra(
                ex1,
                args.pose_x_key,
                args.pose_y_key,
                args.pose_theta_key,
                args.theta_degrees,
                args.pose_xy_scale,
            )
        except (KeyError, OSError, json.JSONDecodeError, TypeError, ValueError) as e:
            print(f"错误：读取 extra 位姿失败: {e}", file=sys.stderr)
            return 2
        T_WB0 = pose_auc._T_world_body_from_planar_pose(x0, y0, th0, args.body_z_world)
        T_WB1 = pose_auc._T_world_body_from_planar_pose(x1, y1, th1, args.body_z_world)
        T_WC0 = T_WB0 @ T_vc
        T_WC1 = T_WB1 @ T_vc
        T_c0_c1 = np.linalg.inv(T_WC0) @ T_WC1
        R_gt, t_gt = pose_auc._gt_Rt_from_T_cam0_cam1(T_c0_c1)
        body_yaw0_deg = math.degrees(th0)
        body_yaw1_deg = math.degrees(th1)
        gt_mode = "extra_json"
    elif args.gt_rel_yaw_deg is not None:
        R_gt = _Rz_deg(float(args.gt_rel_yaw_deg))
        gt_mode = "yaw_deg"
        ex0 = ex1 = None
    else:
        R_gt = _load_R_gt_from_json(args.gt_rel_r_json)
        gt_mode = "r_matrix_json"
        ex0 = ex1 = None

    model = get_model(checkpoint=str(ckpt), default_outputs=("sparse_positions", "sparse_descriptors"))
    img0 = load_images(str(args.img0))
    img1 = load_images(str(args.img1))

    sp0, d0 = model(img0)
    sp1, d1 = model(img1)
    sp0 = from_feature_coords_to_image_coords(model, sp0)
    sp1 = from_feature_coords_to_image_coords(model, sp1)
    kp0_all = _silk_positions_to_xy(sp0[0].detach().cpu().numpy())
    kp1_all = _silk_positions_to_xy(sp1[0].detach().cpu().numpy())

    sp0_used = sp0[0]
    sp1_used = sp1[0]
    d0_used = d0[0]
    d1_used = d1[0]
    sky_kept0 = sky_kept1 = None
    sky_removed0 = sky_removed1 = None
    sky_cls0: Optional[np.ndarray] = None
    sky_cls1: Optional[np.ndarray] = None
    if args.sky_seg_onnx is not None:
        seg_path = args.sky_seg_onnx.expanduser().resolve()
        if not seg_path.is_file():
            print(f"错误：找不到天空分割 ONNX: {seg_path}", file=sys.stderr)
            return 2
        im0_bgr = cv2.imread(str(args.img0.expanduser().resolve()), cv2.IMREAD_COLOR)
        im1_bgr = cv2.imread(str(args.img1.expanduser().resolve()), cv2.IMREAD_COLOR)
        if im0_bgr is None or im1_bgr is None:
            print("错误：天空分割读取图像失败", file=sys.stderr)
            return 2
        norm_style = "imagenet" if args.sky_seg_norm == "imagenet" else "mmseg"
        output_style = "minmax_u8" if args.sky_seg_post == "minmax" else "argmax"
        sky_seg = SkySegOnnxSession(
            seg_path,
            norm_style=norm_style,
            output_style=output_style,
        )
        print(f"[sky] providers: {sky_seg.providers}", file=sys.stderr)
        cls0 = sky_seg.predict(im0_bgr)
        cls1 = sky_seg.predict(im1_bgr)
        sky_cls0, sky_cls1 = cls0, cls1
        sal_thr: Optional[int] = int(args.sky_saliency_threshold) if args.sky_seg_post == "minmax" else None
        keep0 = _non_sky_point_mask(
            kp0_all,
            cls0,
            args.sky_class_id,
            saliency_threshold=sal_thr,
            sky_erode_px=args.sky_erode_px,
        )
        keep1 = _non_sky_point_mask(
            kp1_all,
            cls1,
            args.sky_class_id,
            saliency_threshold=sal_thr,
            sky_erode_px=args.sky_erode_px,
        )
        idx0 = np.nonzero(keep0)[0]
        idx1 = np.nonzero(keep1)[0]
        if idx0.size == 0 or idx1.size == 0:
            print("错误：天空过滤后某一帧无可用关键点", file=sys.stderr)
            return 1
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
        erode_note = f" erode_px={int(args.sky_erode_px)}" if int(args.sky_erode_px) > 0 else ""
        print(
            f"[sky] 过滤后关键点{erode_note}: img0={sky_kept0} (去除 {sky_removed0}) "
            f"img1={sky_kept1} (去除 {sky_removed1})",
            file=sys.stderr,
        )

    mask_removed0 = mask_removed1 = None
    if args.exclude_mask_png is not None:
        mask_path = args.exclude_mask_png.expanduser().resolve()
        if mask_path.is_file():
            try:
                mask_base = _load_exclude_mask_grayscale(mask_path)
            except FileNotFoundError as e:
                print(f"错误：{e}", file=sys.stderr)
                return 2
            im0_bgr = cv2.imread(str(args.img0.expanduser().resolve()), cv2.IMREAD_COLOR)
            im1_bgr = cv2.imread(str(args.img1.expanduser().resolve()), cv2.IMREAD_COLOR)
            if im0_bgr is None or im1_bgr is None:
                print("错误：exclude mask 读取图像失败", file=sys.stderr)
                return 2
            h0, w0 = im0_bgr.shape[:2]
            h1, w1 = im1_bgr.shape[:2]
            m0 = _exclude_mask_for_hw(mask_base, h0, w0)
            m1 = _exclude_mask_for_hw(mask_base, h1, w1)
            kp0_xy = _silk_positions_to_xy(sp0_used.detach().cpu().numpy())
            kp1_xy = _silk_positions_to_xy(sp1_used.detach().cpu().numpy())
            keep0 = _non_exclude_mask_point_mask(
                kp0_xy, m0, exclude_value=args.exclude_mask_value
            )
            keep1 = _non_exclude_mask_point_mask(
                kp1_xy, m1, exclude_value=args.exclude_mask_value
            )
            idx0 = np.nonzero(keep0)[0]
            idx1 = np.nonzero(keep1)[0]
            if idx0.size == 0 or idx1.size == 0:
                print("错误：exclude mask 过滤后某一帧无可用关键点", file=sys.stderr)
                return 1
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
            print(
                f"[mask] exclude={mask_path.name} val={args.exclude_mask_value}: "
                f"img0 去除 {mask_removed0}, img1 去除 {mask_removed1}",
                file=sys.stderr,
            )

    matches = SILK_MATCHER(d0_used, d1_used)
    if matches.shape[0] < args.min_matches:
        print(
            json.dumps(
                {
                    "ok": False,
                    "reason": "too_few_matches",
                    "n_matches": int(matches.shape[0]),
                    "checkpoint": str(ckpt),
                    "gt_mode": gt_mode,
                },
                ensure_ascii=False,
            )
        )
        return 1

    m0 = _silk_positions_to_xy(sp0_used[matches[:, 0]].detach().cpu().numpy())
    m1 = _silk_positions_to_xy(sp1_used[matches[:, 1]].detach().cpu().numpy())

    est = pose_auc._estimate_pose_recover_pose(
        m0,
        m1,
        K,
        D,
        args.camera_model,
        args.ransac_threshold,
        args.min_inliers,
    )
    if est is None:
        print(
            json.dumps(
                {
                    "ok": False,
                    "reason": "recover_pose_failed",
                    "n_matches": int(matches.shape[0]),
                    "checkpoint": str(ckpt),
                    "gt_mode": gt_mode,
                },
                ensure_ascii=False,
            )
        )
        return 1

    R_est, t_est, n_epi, n_rec, rec_inlier_mask = est
    R_est_np = np.asarray(R_est, dtype=np.float64).reshape(3, 3)
    err_deg = float(pose_auc._rotation_error_deg(R_est_np, R_gt))
    t_est_np = np.asarray(t_est, dtype=np.float64).reshape(3)
    if t_gt is not None:
        trans_err_deg = float(pose_auc._translation_error_deg(t_est_np, t_gt))
    else:
        trans_err_deg = None

    if gt_mode == "extra_json" and th0 is not None and th1 is not None:
        gt_yaw_world_deg = _relative_world_yaw_deg_from_thetas(th0, th1)
    elif gt_mode == "yaw_deg":
        gt_yaw_world_deg = _wrap_deg(float(args.gt_rel_yaw_deg))
    else:
        gt_yaw_world_deg = None

    if T_WB0 is not None and T_vc is not None:
        est_yaw_world_deg = _relative_world_yaw_deg_from_cam_est(
            R_est_np, t_est_np, T_WB0, T_vc
        )
    else:
        est_yaw_world_deg = None

    world_yaw_error_deg: Optional[float] = None
    if est_yaw_world_deg is not None and gt_yaw_world_deg is not None:
        world_yaw_error_deg = _wrap_deg(est_yaw_world_deg - gt_yaw_world_deg)

    _log_pose_angles(
        n_matches=int(matches.shape[0]),
        n_epi=int(n_epi),
        n_rec=int(n_rec),
        rotation_error_deg=err_deg,
        est_yaw_world_deg=est_yaw_world_deg,
        gt_yaw_world_deg=gt_yaw_world_deg,
        world_yaw_error_deg=world_yaw_error_deg,
        translation_direction_error_deg=trans_err_deg,
        est_t=t_est_np,
        gt_t=t_gt,
        body_yaw0_deg=body_yaw0_deg,
        body_yaw1_deg=body_yaw1_deg,
    )

    out: dict = {
        "ok": True,
        "gt_mode": gt_mode,
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
        "checkpoint": str(ckpt),
        "img0": str(args.img0),
        "img1": str(args.img1),
    }
    if args.extrinsic_json is not None:
        out["extra0"] = str(ex0)
        out["extra1"] = str(ex1)
        out["extrinsic_json"] = str(args.extrinsic_json)
    if args.sky_seg_onnx is not None:
        out["sky_seg_onnx"] = str(args.sky_seg_onnx.expanduser().resolve())
        out["sky_seg_norm"] = str(args.sky_seg_norm)
        out["sky_seg_post"] = str(args.sky_seg_post)
        out["sky_class_id"] = int(args.sky_class_id)
        if args.sky_seg_post == "minmax":
            out["sky_saliency_threshold"] = int(args.sky_saliency_threshold)
        out["sky_erode_px"] = int(args.sky_erode_px)
        out["n_keypoints_after_sky_img0"] = sky_kept0
        out["n_keypoints_after_sky_img1"] = sky_kept1
        out["n_keypoints_removed_sky_img0"] = sky_removed0
        out["n_keypoints_removed_sky_img1"] = sky_removed1
    if args.exclude_mask_png is not None and args.exclude_mask_png.expanduser().resolve().is_file():
        out["exclude_mask_png"] = str(args.exclude_mask_png.expanduser().resolve())
        out["exclude_mask_value"] = int(args.exclude_mask_value)
        if mask_removed0 is not None:
            out["n_keypoints_removed_exclude_mask_img0"] = mask_removed0
            out["n_keypoints_removed_exclude_mask_img1"] = mask_removed1

    if args.viz_out is not None:
        try:
            vz = _load_viz_pairs_module(_REPO_ROOT)
        except (FileNotFoundError, RuntimeError) as e:
            out["viz_error"] = f"load_viz_module:{e}"
            print(f"警告：{out['viz_error']}", file=sys.stderr)
        else:
            im0 = cv2.imread(str(args.img0.expanduser().resolve()), cv2.IMREAD_UNCHANGED)
            im1 = cv2.imread(str(args.img1.expanduser().resolve()), cv2.IMREAD_UNCHANGED)
            if im0 is None or im1 is None:
                out["viz_error"] = "imread_failed"
                print("警告：可视化失败（无法读取图像）", file=sys.stderr)
            else:
                if im0.ndim == 3 and im0.shape[2] == 4:
                    im0 = cv2.cvtColor(im0, cv2.COLOR_BGRA2BGR)
                if im1.ndim == 3 and im1.shape[2] == 4:
                    im1 = cv2.cvtColor(im1, cv2.COLOR_BGRA2BGR)

                m0v = m0.copy()
                m1v = m1.copy()
                if rec_inlier_mask is not None and int(np.count_nonzero(rec_inlier_mask)) > 0:
                    m0v, m1v = m0v[rec_inlier_mask], m1v[rec_inlier_mask]
                uniform: Optional[Tuple[int, int, int]] = (0, 220, 100)
                max_edge = int(args.viz_max_edge)
                h0_orig, w0_orig = int(im0.shape[0]), int(im0.shape[1])
                h1_orig, w1_orig = int(im1.shape[0]), int(im1.shape[1])
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
                    _apply_sky_viz_to_image(
                        im0_draw,
                        sky_cls0,
                        args.sky_class_id,
                        args.viz_sky_style,
                        args.viz_sky_mask_alpha,
                        saliency_threshold=sal_viz,
                        sky_erode_px=args.sky_erode_px,
                    )
                    _apply_sky_viz_to_image(
                        im1_draw,
                        sky_cls1,
                        args.sky_class_id,
                        args.viz_sky_style,
                        args.viz_sky_mask_alpha,
                        saliency_threshold=sal_viz,
                        sky_erode_px=args.sky_erode_px,
                    )
                draw_seed = 0
                draw_max = int(args.viz_max_draw)
                sel_idx, m0_draw, m1_draw = _select_draw_subset(
                    m0v, m1v, max_draw=draw_max, seed=draw_seed
                )
                if int(args.viz_debug_points) > 0:
                    _log_viz_points(
                        m0_draw=m0_draw,
                        m1_draw=m1_draw,
                        selected_idx=sel_idx,
                        max_print=int(args.viz_debug_points),
                    )
                panel = vz._draw_pair_panel(
                    im0_draw,
                    im1_draw,
                    m0v,
                    m1v,
                    gap=int(args.viz_gap),
                    max_draw=draw_max,
                    mask=None,
                    seed=draw_seed,
                    uniform_bgr=uniform,
                )
                if not args.viz_no_keypoints:
                    kp0v = kp0_all.copy() * scale_for_viz
                    kp1v = kp1_all.copy() * scale_for_viz
                    gap = int(args.viz_gap)
                    w0 = int(im0.shape[1])
                    h0, w0_im = int(im0.shape[0]), int(im0.shape[1])
                    h1, w1_im = int(im1.shape[0]), int(im1.shape[1])
                    sky0_v = sky1_v = None
                    if sky_cls0 is not None and sky_cls1 is not None:
                        sal_v = (
                            int(args.sky_saliency_threshold)
                            if args.sky_seg_post == "minmax"
                            else None
                        )
                        sky0_v = _erode_sky_binary(
                            _sky_binary_u8(
                                sky_cls0,
                                h0,
                                w0_im,
                                args.sky_class_id,
                                saliency_threshold=sal_v,
                            ),
                            int(args.sky_erode_px),
                        )
                        sky1_v = _erode_sky_binary(
                            _sky_binary_u8(
                                sky_cls1,
                                h1,
                                w1_im,
                                args.sky_class_id,
                                saliency_threshold=sal_v,
                            ),
                            int(args.sky_erode_px),
                        )
                    ex0_v = ex1_v = None
                    ex_val = int(args.exclude_mask_value)
                    if args.exclude_mask_png is not None and args.exclude_mask_png.is_file():
                        ex_base = _load_exclude_mask_grayscale(args.exclude_mask_png)
                        ex0_v = _exclude_mask_for_hw(ex_base, h0, w0_im)
                        ex1_v = _exclude_mask_for_hw(ex_base, h1, w1_im)
                    r0, o0 = _draw_keypoints_classified_inplace(
                        panel,
                        kp0v,
                        sky_u8=sky0_v,
                        exclude_u8=ex0_v,
                        exclude_value=ex_val,
                        radius=int(args.viz_keypoint_radius),
                        max_draw=int(args.viz_keypoint_max_draw),
                        seed=17,
                    )
                    kp1_on_panel = kp1v.copy()
                    kp1_on_panel[:, 0] += float(w0 + gap)
                    r1, o1 = _draw_keypoints_classified_inplace(
                        panel,
                        kp1_on_panel,
                        sky_u8=sky1_v,
                        exclude_u8=ex1_v,
                        exclude_value=ex_val,
                        x_offset=float(w0 + gap),
                        radius=int(args.viz_keypoint_radius),
                        max_draw=int(args.viz_keypoint_max_draw),
                        seed=29,
                    )
                    print(
                        f"[viz] 关键点 img0 红={r0} 橙={o0}, img1 红={r1} 橙={o1} "
                        f"(红=天空或 mask 外)",
                        file=sys.stderr,
                    )
                cap_lines: List[str] = [
                    f"cam_rot_err={err_deg:.2f}deg",
                    f"n_epi={n_epi} n_rec={n_rec} n_draw_inliers={int(m0v.shape[0])} n_matches={int(matches.shape[0])}",
                ]
                if world_yaw_error_deg is not None:
                    cap_lines[0] += f" world_yaw_err={world_yaw_error_deg:.2f}deg"
                if trans_err_deg is not None:
                    cap_lines[0] += f" trans_dir_err={trans_err_deg:.2f}deg"
                cap_lines.append("recoverPose inliers only")
                if not args.viz_no_keypoints:
                    cap_lines.append("orange dots: detected keypoints")
                if (
                    sky_cls0 is not None
                    and sky_cls1 is not None
                    and args.viz_sky_style != "none"
                ):
                    if args.sky_seg_post == "minmax":
                        cap_lines.append(
                            f"sky viz: style={args.viz_sky_style} "
                            f"norm={args.sky_seg_norm} post=minmax thr={args.sky_saliency_threshold}"
                        )
                    else:
                        cap_lines.append(
                            f"sky viz: style={args.viz_sky_style} "
                            f"norm={args.sky_seg_norm} post=argmax class={args.sky_class_id}"
                        )
                cap = vz._caption(panel, cap_lines)
                outp = _viz_save_path(args.viz_out, prefer_jpg=False)
                outp.parent.mkdir(parents=True, exist_ok=True)
                if _imwrite_viz_bgr(outp, cap):
                    out["viz_out"] = str(outp)
                else:
                    out["viz_error"] = "imwrite_failed"
                    print(f"警告：无法写入 {outp}", file=sys.stderr)

    print(json.dumps(out, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
