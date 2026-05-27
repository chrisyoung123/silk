#!/usr/bin/env python3
"""
根据 pairs JSON + all_pairs_merged.npz，为每个匹配对导出一张左右拼图 + 连线可视化。

与 eval_pairs_pose_auc.py 使用相同的 NPZ 键（match_offsets, matches_im0, matches_im1）与 pairs 项（path0, path1, index）。
可选：极几何 RANSAC 内点/外点着色；--inliers-only 只画内点；--exclude-top-fraction 去掉画面最上方条带内的点（粗去天空）。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np

from lib.geometry.epipolar import epipolar_essential_mask

# 复用 eval_pairs_pose_auc 中的内参读取
_EVAL_PATH = Path(__file__).resolve().parent / "eval_pairs_pose_auc.py"
_spec = importlib.util.spec_from_file_location("eval_pairs_pose_auc", _EVAL_PATH)
_ep = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_ep)


def _epipolar_mask(
    m0: np.ndarray,
    m1: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
    camera_model: str,
    ransac_threshold_px: float,
) -> Optional[np.ndarray]:
    """极线 RANSAC 内点 mask（essential 模式，供可视化着色）。"""
    return epipolar_essential_mask(m0, m1, K, D, camera_model, ransac_threshold_px)


def filter_matches_exclude_image_top(
    m0: np.ndarray,
    m1: np.ndarray,
    h0: int,
    h1: int,
    top_fraction: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    去掉两幅图各自「最靠上」高度比例 top_fraction 条带内的匹配点。
    像素坐标 y 向下增大，画面上方 y 小；天空常在顶部，可用此粗过滤。
    top_fraction=0.15 表示丢掉 y < 0.15*h 的点（两图都要满足才保留）。
    """
    if top_fraction <= 0.0:
        return m0.copy(), m1.copy()
    t0 = float(top_fraction) * float(h0)
    t1 = float(top_fraction) * float(h1)
    keep = (m0[:, 1] >= t0) & (m1[:, 1] >= t1)
    return m0[keep].copy(), m1[keep].copy()


def _load_pairs_and_offsets(
    pairs_json: Path, matches_npz: Path
) -> Tuple[List[dict], np.ndarray, Any]:
    with pairs_json.open("r", encoding="utf-8") as f:
        data = json.load(f)
    pairs: List[dict] = data["pairs"]
    d = np.load(str(matches_npz), allow_pickle=True)
    offsets = np.asarray(d["match_offsets"]).reshape(-1)
    return pairs, offsets, d


def _resize_pair_and_matches(
    img0: np.ndarray,
    img1: np.ndarray,
    m0: np.ndarray,
    m1: np.ndarray,
    max_edge: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    h0, w0 = img0.shape[:2]
    h1, w1 = img1.shape[:2]
    long0, long1 = max(h0, w0), max(h1, w1)
    scale = min(1.0, float(max_edge) / max(long0, long1, 1))
    if scale >= 1.0 - 1e-6:
        return img0, img1, m0.copy(), m1.copy()
    nw0, nh0 = max(1, int(round(w0 * scale))), max(1, int(round(h0 * scale)))
    nw1, nh1 = max(1, int(round(w1 * scale))), max(1, int(round(h1 * scale)))
    img0s = cv2.resize(img0, (nw0, nh0), interpolation=cv2.INTER_AREA)
    img1s = cv2.resize(img1, (nw1, nh1), interpolation=cv2.INTER_AREA)
    m0s = m0 * scale
    m1s = m1 * scale
    return img0s, img1s, m0s, m1s


def _draw_pair_panel(
    img0: np.ndarray,
    img1: np.ndarray,
    m0: np.ndarray,
    m1: np.ndarray,
    gap: int,
    max_draw: int,
    mask: Optional[np.ndarray],
    seed: int,
    uniform_bgr: Optional[Tuple[int, int, int]] = None,
) -> np.ndarray:
    h0, w0 = img0.shape[:2]
    h1, w1 = img1.shape[:2]
    H = max(h0, h1)
    W = w0 + gap + w1
    if img0.ndim == 2:
        img0 = cv2.cvtColor(img0, cv2.COLOR_GRAY2BGR)
    if img1.ndim == 2:
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    canvas[:h0, :w0] = img0
    canvas[:h1, w0 + gap : w0 + gap + w1] = img1

    n = int(m0.shape[0])
    if n == 0:
        return canvas
    rng = np.random.default_rng(seed)
    if max_draw > 0 and n > max_draw:
        idx = rng.choice(n, size=max_draw, replace=False)
        m0d, m1d = m0[idx], m1[idx]
        maskd = mask[idx] if mask is not None and mask.shape[0] == n else None
    else:
        m0d, m1d = m0, m1
        maskd = mask

    for k in range(m0d.shape[0]):
        x0, y0 = int(round(m0d[k, 0])), int(round(m0d[k, 1]))
        x1, y1 = int(round(m1d[k, 0] + w0 + gap)), int(round(m1d[k, 1]))
        if uniform_bgr is not None:
            col = uniform_bgr
        elif maskd is not None:
            col = (0, 200, 0) if bool(maskd[k]) else (0, 0, 255)
        else:
            col = (255, 128, 0)
        cv2.line(canvas, (x0, y0), (x1, y1), col, 1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, (x0, y0), 2, col, -1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, (x1, y1), 2, col, -1, lineType=cv2.LINE_AA)
    return canvas


def _caption(canvas: np.ndarray, lines: List[str]) -> np.ndarray:
    bar_h = 22 * len(lines) + 16
    bar = np.full((bar_h, canvas.shape[1], 3), 40, dtype=np.uint8)
    y = 18
    for ln in lines:
        cv2.putText(bar, ln[:120], (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1, cv2.LINE_AA)
        y += 22
    return np.vstack([bar, canvas])


def main() -> int:
    ap = argparse.ArgumentParser(description="逐对可视化 matches（pairs JSON + merged NPZ）")
    ap.add_argument("--pairs-json", type=Path, required=True)
    ap.add_argument("--matches-npz", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True, help="输出目录，每对一个 PNG")
    ap.add_argument(
        "--camera-json",
        type=Path,
        default=None,
        help="若设则与 --color-epipolar-inliers 联用，按 eval 相同 RANSAC 染色",
    )
    ap.add_argument(
        "--camera-model",
        choices=("fisheye", "pinhole_plumb", "pinhole_rational8"),
        default="fisheye",
    )
    ap.add_argument(
        "--ransac-threshold",
        type=float,
        default=3.0,
        help="与 eval_pairs_pose_auc 默认一致的像素阈值（仅染色时）",
    )
    ap.add_argument(
        "--color-epipolar-inliers",
        action="store_true",
        help="绿=极几何 RANSAC 内点，红=外点（需 --camera-json）；与 --inliers-only 互斥时以内点为准",
    )
    ap.add_argument(
        "--inliers-only",
        action="store_true",
        help="只绘制极几何内点（需 --camera-json）；连线单色，不画外点",
    )
    ap.add_argument(
        "--exclude-top-fraction",
        type=float,
        default=0.0,
        help="去掉两图各自最上方该比例高度内的匹配（0=关闭）。用于粗去天空等；鱼眼构图若天不在顶可调小或关",
    )
    ap.add_argument("--gap", type=int, default=16, help="左右图间隙宽度（像素）")
    ap.add_argument(
        "--max-draw-matches",
        type=int,
        default=500,
        help="每张图最多绘制的连线数（0=不限制；过多会很密）",
    )
    ap.add_argument(
        "--max-edge",
        type=int,
        default=1400,
        help="长边超过则整体缩小再画线（0=不缩放）",
    )
    ap.add_argument("--pair-indices", type=str, default=None, help='只画这些 index，逗号分隔，如 "0,3,10"')
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    pairs, offsets, d = _load_pairs_and_offsets(args.pairs_json, args.matches_npz)
    n_pairs_npz = int(offsets.shape[0] - 1)
    max_index = max(int(p["index"]) for p in pairs)
    if max_index >= n_pairs_npz:
        print(
            f"错误：JSON 最大 index={max_index}，NPZ 仅 {n_pairs_npz} 对",
            file=sys.stderr,
        )
        return 2

    filter_idx: Optional[set] = None
    if args.pair_indices:
        filter_idx = {int(x.strip()) for x in args.pair_indices.split(",") if x.strip()}

    K = D = None
    if args.camera_json is not None:
        K, D = _ep._load_camera_json(args.camera_json)

    if args.inliers_only or args.color_epipolar_inliers:
        if args.camera_json is None:
            print("错误：--inliers-only / --color-epipolar-inliers 需要 --camera-json", file=sys.stderr)
            return 2

    if args.inliers_only and args.color_epipolar_inliers:
        print("提示：已同时指定 --inliers-only 与 --color-epipolar-inliers，按仅内点绘制。", file=sys.stderr)

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    n_ok = 0
    for p in pairs:
        idx = int(p["index"])
        if filter_idx is not None and idx not in filter_idx:
            continue
        path0 = Path(p["path0"])
        path1 = Path(p["path1"])
        s, e = int(offsets[idx]), int(offsets[idx + 1])
        m0 = np.asarray(d["matches_im0"][s:e], dtype=np.float64)
        m1 = np.asarray(d["matches_im1"][s:e], dtype=np.float64)
        n_matches_raw = int(e - s)

        if not path0.is_file() or not path1.is_file():
            print(f"跳过 index={idx}：图像不存在", file=sys.stderr)
            continue

        img0 = cv2.imread(str(path0), cv2.IMREAD_UNCHANGED)
        img1 = cv2.imread(str(path1), cv2.IMREAD_UNCHANGED)
        if img0 is None or img1 is None:
            print(f"跳过 index={idx}：无法解码图像", file=sys.stderr)
            continue
        if img0.ndim == 3 and img0.shape[2] == 4:
            img0 = cv2.cvtColor(img0, cv2.COLOR_BGRA2BGR)
        if img1.ndim == 3 and img1.shape[2] == 4:
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGRA2BGR)

        h0, w0 = img0.shape[:2]
        h1, w1 = img1.shape[:2]
        if args.exclude_top_fraction > 0:
            m0, m1 = filter_matches_exclude_image_top(m0, m1, h0, h1, args.exclude_top_fraction)

        mask: Optional[np.ndarray] = None
        uniform: Optional[Tuple[int, int, int]] = None
        if args.inliers_only:
            if m0.shape[0] < 5:
                print(f"跳过 index={idx}：去顶后匹配不足 5 条", file=sys.stderr)
                continue
            mask = _epipolar_mask(m0, m1, K, D, args.camera_model, args.ransac_threshold)
            if mask is None or int(np.count_nonzero(mask)) < 1:
                print(f"跳过 index={idx}：无法得到极几何内点", file=sys.stderr)
                continue
            m0, m1 = m0[mask], m1[mask]
            mask = None
            uniform = (0, 220, 100)
        elif args.color_epipolar_inliers and K is not None and m0.shape[0] >= 5:
            mask = _epipolar_mask(m0, m1, K, D, args.camera_model, args.ransac_threshold)

        n_draw = int(m0.shape[0])
        max_edge = int(args.max_edge)
        if max_edge > 0:
            img0, img1, m0, m1 = _resize_pair_and_matches(img0, img1, m0, m1, max_edge)

        panel = _draw_pair_panel(
            img0,
            img1,
            m0,
            m1,
            gap=args.gap,
            max_draw=args.max_draw_matches,
            mask=mask,
            seed=args.seed + idx,
            uniform_bgr=uniform,
        )
        cap_lines = [
            f"index={idx}  n_draw={n_draw} (raw_slice={n_matches_raw})",
            f"{path0.name}  |  {path1.name}",
        ]
        if args.exclude_top_fraction > 0:
            cap_lines.append(f"exclude_top={args.exclude_top_fraction:g}")
        if args.inliers_only:
            cap_lines.append("inliers_only")
        elif args.color_epipolar_inliers and mask is not None:
            cap_lines.append(f"epipolar_inliers={int(np.count_nonzero(mask))} / {mask.size}")
        elif args.color_epipolar_inliers:
            cap_lines.append("epipolar_mask unavailable")
        panel = _caption(panel, cap_lines)

        out_path = out_dir / f"pair_{idx:05d}.png"
        if not cv2.imwrite(str(out_path), panel):
            print(f"写入失败: {out_path}", file=sys.stderr)
            continue
        n_ok += 1

    print(f"已写入 {n_ok} 张图到 {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
