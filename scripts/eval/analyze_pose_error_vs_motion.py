#!/usr/bin/env python3
"""
从 batch 输出的 per_pair.jsonl 分析：相对位姿（前后/左右位移）与旋转/平移方向误差的关系，并保存直方图。

用法:
  python3 scripts/eval/analyze_pose_error_vs_motion.py \\
    --jsonl outputs/silk_pair_pose/E01734789H09J45Y0601/per_pair.jsonl \\
    --out-dir outputs/silk_pair_pose/E01734789H09J45Y0601/motion_analysis

  # 多序列（每个含 per_pair.jsonl 的子目录）:
  python3 scripts/eval/analyze_pose_error_vs_motion.py \\
    --summarize-root outputs/silk_pair_pose \\
    --out-dir outputs/silk_pair_pose/motion_analysis_all
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EP_PATH = _REPO_ROOT / "scripts/eval/eval_pairs_pose_auc.py"


def _load_ep():
    spec = importlib.util.spec_from_file_location("eval_pairs_pose_auc", _EP_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 {_EP_PATH}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["eval_pairs_pose_auc"] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _motion_body_frame(
    x0: float,
    y0: float,
    th0: float,
    x1: float,
    y1: float,
    th1: float,
) -> Dict[str, float]:
    """相对位移在 frame0 车体坐标系：forward≈前后，lateral≈左右（米）。"""
    dx = float(x1 - x0)
    dy = float(y1 - y0)
    c, s = math.cos(th0), math.sin(th0)
    forward = c * dx + s * dy
    lateral = -s * dx + c * dy
    dist = math.hypot(dx, dy)
    dth_deg = math.degrees(math.atan2(math.sin(th1 - th0), math.cos(th1 - th0)))
    denom = abs(forward) + abs(lateral) + 1e-9
    forward_ratio = abs(forward) / denom
    lateral_ratio = abs(lateral) / denom
    if forward_ratio >= 0.65:
        motion_class = "fwd-dominant"
    elif lateral_ratio >= 0.65:
        motion_class = "lat-dominant"
    else:
        motion_class = "mixed"
    return {
        "dx_world_m": dx,
        "dy_world_m": dy,
        "forward_m": forward,
        "lateral_m": lateral,
        "distance_m": dist,
        "delta_yaw_deg": dth_deg,
        "abs_forward_m": abs(forward),
        "abs_lateral_m": abs(lateral),
        "forward_ratio": forward_ratio,
        "lateral_ratio": lateral_ratio,
        "motion_class": motion_class,
    }


def _extra_path_for_image(image_path: str) -> Path:
    p = Path(image_path).expanduser()
    return p.parent / f"{p.stem}_extra.json"


def _enrich_record(rec: Dict[str, Any], ep: Any, args: argparse.Namespace) -> Optional[Dict[str, Any]]:
    ex0 = rec.get("extra0")
    ex1 = rec.get("extra1")
    if not ex0 and rec.get("img0"):
        ex0 = str(_extra_path_for_image(str(rec["img0"])))
    if not ex1 and rec.get("img1"):
        ex1 = str(_extra_path_for_image(str(rec["img1"])))
    if not ex0 or not ex1:
        return None
    p0 = Path(ex0).expanduser()
    p1 = Path(ex1).expanduser()
    if not p0.is_file() or not p1.is_file():
        return None
    try:
        x0, y0, th0 = ep._load_pose_from_extra(
            p0,
            args.pose_x_key,
            args.pose_y_key,
            args.pose_theta_key,
            args.theta_degrees,
            args.pose_xy_scale,
        )
        x1, y1, th1 = ep._load_pose_from_extra(
            p1,
            args.pose_x_key,
            args.pose_y_key,
            args.pose_theta_key,
            args.theta_degrees,
            args.pose_xy_scale,
        )
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
        return None
    row = dict(rec)
    row.update(_motion_body_frame(x0, y0, th0, x1, y1, th1))
    row["ok"] = bool(rec.get("ok"))
    if rec.get("ok"):
        row["rotation_error_deg"] = float(rec["rotation_error_deg"])
        tdir = rec.get("translation_direction_error_deg")
        row["translation_direction_error_deg"] = (
            float(tdir) if tdir is not None else float("nan")
        )
        wy = rec.get("world_yaw_error_deg")
        row["abs_world_yaw_error_deg"] = abs(float(wy)) if wy is not None else float("nan")
    return row


def _collect_rows(
    jsonl_paths: List[Path], ep: Any, args: argparse.Namespace
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for jp in jsonl_paths:
        for rec in _load_jsonl(jp):
            row = _enrich_record(rec, ep, args)
            if row is not None:
                row["source_jsonl"] = str(jp)
                out.append(row)
    return out


def _plot_all(rows: List[Dict[str, Any]], out_dir: Path, title_prefix: str) -> None:
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    ok_rows = [r for r in rows if r.get("ok")]
    if not ok_rows:
        print("警告：无 ok 样本，仅输出位姿分布图", file=sys.stderr)
    classes = ["fwd-dominant", "lat-dominant", "mixed"]
    colors = {
        "fwd-dominant": "#2ecc71",
        "lat-dominant": "#3498db",
        "mixed": "#e67e22",
    }

    def _save(fig: Any, name: str) -> None:
        p = out_dir / name
        fig.savefig(p, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  写入 {p}")

    # --- 1. 相对位移分布 ---
    fwd = np.array([r["forward_m"] for r in rows], dtype=np.float64)
    lat = np.array([r["lateral_m"] for r in rows], dtype=np.float64)
    dist = np.array([r["distance_m"] for r in rows], dtype=np.float64)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    fig.suptitle(f"{title_prefix} - GT displacement", fontsize=11)
    axes[0].hist(fwd, bins=50, color="#27ae60", edgecolor="white", alpha=0.85)
    axes[0].set_xlabel("forward (m) in body0")
    axes[0].set_ylabel("count")
    axes[1].hist(lat, bins=50, color="#2980b9", edgecolor="white", alpha=0.85)
    axes[1].set_xlabel("lateral (m) in body0")
    axes[2].hist(dist, bins=50, color="#8e44ad", edgecolor="white", alpha=0.85)
    axes[2].set_xlabel("|Δp| (m)")
    _save(fig, "hist_gt_displacement.png")

    # --- 2. 运动类型占比 ---
    fig, ax = plt.subplots(figsize=(5, 4))
    counts = [sum(1 for r in rows if r["motion_class"] == c) for c in classes]
    ax.bar(classes, counts, color=[colors[c] for c in classes], edgecolor="white")
    ax.set_title(f"{title_prefix} - motion class (GT)")
    ax.set_ylabel("pairs")
    _save(fig, "hist_motion_class_counts.png")

    if not ok_rows:
        return

    rot = np.array([r["rotation_error_deg"] for r in ok_rows], dtype=np.float64)
    tdir = np.array(
        [r["translation_direction_error_deg"] for r in ok_rows], dtype=np.float64
    )
    wyaw = np.array([r["abs_world_yaw_error_deg"] for r in ok_rows], dtype=np.float64)
    ok_fwd = np.array([r["forward_m"] for r in ok_rows], dtype=np.float64)
    ok_lat = np.array([r["lateral_m"] for r in ok_rows], dtype=np.float64)

    # --- 3. 误差总体直方图 ---
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    fig.suptitle(f"{title_prefix} - errors (ok only)", fontsize=11)
    axes[0].hist(rot, bins=50, color="#c0392b", edgecolor="white", alpha=0.85)
    axes[0].set_xlabel("rotation_error (deg)")
    axes[1].hist(tdir[np.isfinite(tdir)], bins=50, color="#d35400", edgecolor="white", alpha=0.85)
    axes[1].set_xlabel("translation_direction_error (deg)")
    axes[2].hist(wyaw[np.isfinite(wyaw)], bins=50, color="#16a085", edgecolor="white", alpha=0.85)
    axes[2].set_xlabel("|world_yaw_error| (deg)")
    _save(fig, "hist_errors_overall.png")

    # --- 4. 按运动类型分组的误差直方图 ---
    for err_name, err_vals, fname in (
        ("rotation_error (deg)", rot, "hist_rotation_by_motion_class.png"),
        (
            "translation_direction_error (deg)",
            tdir,
            "hist_translation_by_motion_class.png",
        ),
        ("|world_yaw_error| (deg)", wyaw, "hist_world_yaw_by_motion_class.png"),
    ):
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.2), sharey=True)
        fig.suptitle(f"{title_prefix} - {err_name} by motion class", fontsize=11)
        for ax, cls in zip(axes, classes):
            mask = np.array([r["motion_class"] == cls for r in ok_rows], dtype=bool)
            v = err_vals[mask]
            v = v[np.isfinite(v)]
            ax.hist(
                v,
                bins=40,
                color=colors[cls],
                edgecolor="white",
                alpha=0.85,
                range=(0, min(180, np.percentile(v, 99) + 5) if v.size else 180),
            )
            ax.set_title(f"{cls} (n={v.size})")
            ax.set_xlabel(err_name)
        axes[0].set_ylabel("count")
        _save(fig, fname)

    # --- 5. 位移分桶 vs 平均误差 ---
    abs_fwd = np.abs(ok_fwd)
    fwd_bins = np.percentile(abs_fwd, [0, 20, 40, 60, 80, 100])
    fwd_bins = np.unique(fwd_bins)
    if fwd_bins.size < 3:
        fwd_bins = np.linspace(0, max(abs_fwd.max(), 0.01), 6)

    bin_labels: List[str] = []
    bin_rot: List[float] = []
    bin_tdir: List[float] = []
    for i in range(len(fwd_bins) - 1):
        lo, hi = fwd_bins[i], fwd_bins[i + 1]
        m = (abs_fwd >= lo) & (abs_fwd < hi if i < len(fwd_bins) - 2 else abs_fwd <= hi + 1e-9)
        if i == len(fwd_bins) - 2:
            m = abs_fwd >= lo
        if not np.any(m):
            continue
        bin_labels.append(f"|fwd|∈[{lo:.2f},{hi:.2f})")
        bin_rot.append(float(np.mean(rot[m])))
        bin_tdir.append(float(np.nanmean(tdir[m])))

    if bin_labels:
        x = np.arange(len(bin_labels))
        w = 0.35
        fig, ax = plt.subplots(figsize=(max(8, len(bin_labels) * 0.5), 4))
        ax.bar(x - w / 2, bin_rot, width=w, label="mean rotation err", color="#c0392b")
        ax.bar(x + w / 2, bin_tdir, width=w, label="mean trans dir err", color="#d35400")
        ax.set_xticks(x)
        ax.set_xticklabels(bin_labels, rotation=25, ha="right")
        ax.set_ylabel("deg")
        ax.legend()
        ax.set_title(f"{title_prefix} - mean error vs |forward| bins")
        _save(fig, "hist_mean_error_vs_forward_bin.png")

    # --- 6. 散点：位移 vs 误差 ---
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle(f"{title_prefix} - displacement vs errors (ok)", fontsize=11)
    sc_kw = dict(s=6, alpha=0.25, linewidths=0)
    axes[0, 0].scatter(ok_fwd, rot, c=rot, cmap="hot", **sc_kw)
    axes[0, 0].set_xlabel("forward (m)")
    axes[0, 0].set_ylabel("rotation_error (deg)")
    axes[0, 1].scatter(ok_lat, rot, c=rot, cmap="hot", **sc_kw)
    axes[0, 1].set_xlabel("lateral (m)")
    axes[0, 1].set_ylabel("rotation_error (deg)")
    axes[1, 0].scatter(ok_fwd, tdir, c=tdir, cmap="plasma", **sc_kw)
    axes[1, 0].set_xlabel("forward (m)")
    axes[1, 0].set_ylabel("translation_direction_error (deg)")
    axes[1, 1].scatter(ok_lat, tdir, c=tdir, cmap="plasma", **sc_kw)
    axes[1, 1].set_xlabel("lateral (m)")
    axes[1, 1].set_ylabel("translation_direction_error (deg)")
    _save(fig, "scatter_motion_vs_errors.png")

    # --- 7. 2D 直方图 |fwd| vs |lat| 着色 mean trans err（分桶）---
    af = np.abs(ok_fwd)
    al = np.abs(ok_lat)
    fig, ax = plt.subplots(figsize=(6, 5))
    h = ax.hist2d(
        af,
        al,
        bins=(30, 30),
        cmap="YlOrRd",
        weights=tdir,
    )
    plt.colorbar(h[3], ax=ax, label="sum trans_dir_err (weighted count)")
    ax.set_xlabel("|forward| (m)")
    ax.set_ylabel("|lateral| (m)")
    ax.set_title(f"{title_prefix} - |fwd| vs |lat| weighted trans err")
    _save(fig, "hist2d_abs_motion_weighted_trans_err.png")

    # 文本摘要
    lines = [
        f"# {title_prefix} 运动-误差摘要",
        f"总对数（有位姿）: {len(rows)}",
        f"ok: {len(ok_rows)}",
        "",
        "## 按运动类型（ok）",
    ]
    for cls in classes:
        sub = [r for r in ok_rows if r["motion_class"] == cls]
        if not sub:
            continue
        r_arr = np.array([x["rotation_error_deg"] for x in sub])
        t_arr = np.array([x["translation_direction_error_deg"] for x in sub])
        label = {
            "fwd-dominant": "前后主导",
            "lat-dominant": "左右主导",
            "mixed": "斜向/混合",
        }.get(cls, cls)
        lines.append(
            f"- {label}: n={len(sub)}, "
            f"rot mean={r_arr.mean():.2f}° med={np.median(r_arr):.2f}°, "
            f"trans_dir mean={np.nanmean(t_arr):.2f}° med={np.nanmedian(t_arr):.2f}°"
        )
    lines.append("")
    lines.append("## 相关性（ok, Pearson）")
    for ename, ev in (
        ("rotation", rot),
        ("trans_dir", tdir),
        ("|world_yaw|", wyaw),
    ):
        for mname, mv in (("forward", ok_fwd), ("lateral", ok_lat), ("distance", np.hypot(ok_fwd, ok_lat))):
            mask = np.isfinite(ev) & np.isfinite(mv)
            if mask.sum() < 10:
                continue
            c = float(np.corrcoef(ev[mask], mv[mask])[0, 1])
            lines.append(f"- {ename} vs {mname}: r={c:.3f}")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  写入 {out_dir / 'summary.md'}")


def main() -> int:
    ap = argparse.ArgumentParser(description="相对位移与位姿误差分析 + 直方图")
    ap.add_argument("--jsonl", type=Path, default=None, help="单序列 per_pair.jsonl")
    ap.add_argument(
        "--summarize-root",
        type=Path,
        default=None,
        help="多序列根目录（子目录含 per_pair.jsonl）",
    )
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--pose-x-key", default="value.pose.x")
    ap.add_argument("--pose-y-key", default="value.pose.y")
    ap.add_argument("--pose-theta-key", default="value.pose.theta")
    ap.add_argument("--theta-degrees", action="store_true", help="extra 中 theta 为度")
    ap.add_argument("--pose-xy-scale", type=float, default=0.001)
    args = ap.parse_args()

    paths: List[Path] = []
    if args.jsonl is not None:
        paths.append(args.jsonl.expanduser().resolve())
    elif args.summarize_root is not None:
        root = args.summarize_root.expanduser().resolve()
        for d in sorted(root.iterdir()):
            if d.is_dir():
                jp = d / "per_pair.jsonl"
                if jp.is_file():
                    paths.append(jp)
    else:
        print("请指定 --jsonl 或 --summarize-root", file=sys.stderr)
        return 1

    if not paths:
        print("未找到 per_pair.jsonl", file=sys.stderr)
        return 1

    ep = _load_ep()
    rows = _collect_rows(paths, ep, args)
    if not rows:
        print("无有效记录（需 extra0/extra1）", file=sys.stderr)
        return 1

    title = paths[0].parent.name if len(paths) == 1 else f"{len(paths)} sequences"
    out_dir = args.out_dir.expanduser().resolve()
    print(f"分析 {len(rows)} 条记录 → {out_dir}")
    _plot_all(rows, out_dir, title)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
