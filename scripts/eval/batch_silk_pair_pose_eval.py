#!/usr/bin/env python3
"""
批量 SiLK 双图位姿验证：单进程一次加载模型，循环推理；汇总 AUC 并生成 HTML。

由 batch_silk_pair_pose_one_seq.sh / batch_silk_pair_pose_multi_seq.sh 调用。
"""

from __future__ import annotations

import argparse
import html
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EVAL_DIR = _REPO_ROOT / "scripts" / "eval"
if str(_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_EVAL_DIR))

import silk_pair_runtime as spr  # noqa: E402


def _load_eval_pairs_pose_auc():
    path = _REPO_ROOT / "scripts/eval/eval_pairs_pose_auc.py"
    name = "eval_pairs_pose_auc"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _parse_pairs_txt(path: Path, sep: Optional[str], max_pairs: int) -> List[Tuple[str, str]]:
    lines: List[Tuple[str, str]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if sep is not None:
            parts = line.split(sep)
        else:
            parts = line.split()
        if len(parts) < 2:
            continue
        lines.append((parts[0].strip(), parts[1].strip()))
        if max_pairs >= 0 and len(lines) >= max_pairs:
            break
    return lines


def _world_rot_error_deg(rec: Dict[str, Any]) -> Optional[float]:
    if rec.get("world_yaw_error_deg") is not None:
        return abs(float(rec["world_yaw_error_deg"]))
    if rec.get("rotation_error_deg") is not None:
        return abs(float(rec["rotation_error_deg"]))
    return None


def _auc_scalar(rec: Dict[str, Any], metric: str, fail_deg: float) -> float:
    if not rec.get("ok"):
        return fail_deg
    rot = rec.get("rotation_error_deg")
    tdir = rec.get("translation_direction_error_deg")
    if rot is None:
        return fail_deg
    rot_f = float(rot)
    if metric == "rotation":
        return rot_f
    if tdir is None:
        return rot_f
    return max(rot_f, float(tdir))


def _compute_metrics(
    records: List[Dict[str, Any]],
    auc_thresholds: Sequence[float],
    metric: str,
    fail_deg: float,
) -> Dict[str, Any]:
    ep = _load_eval_pairs_pose_auc()
    pose_errors: List[float] = []
    world_errors: List[float] = []
    rot_errors: List[float] = []
    trans_errors: List[float] = []
    n_ok = 0

    for rec in records:
        pose_errors.append(_auc_scalar(rec, metric, fail_deg))
        w = _world_rot_error_deg(rec)
        if w is not None:
            world_errors.append(w)
        if rec.get("ok"):
            n_ok += 1
            if rec.get("rotation_error_deg") is not None:
                rot_errors.append(float(rec["rotation_error_deg"]))
            if rec.get("translation_direction_error_deg") is not None:
                trans_errors.append(float(rec["translation_direction_error_deg"]))

    auc = ep._compute_auc(pose_errors, auc_thresholds)
    n_total = len(records)
    return {
        **{k: float(v) for k, v in auc.items()},
        "metric": metric,
        "n_pairs_total": n_total,
        "n_pairs_ok": n_ok,
        "valid_rate": float(n_ok / n_total) if n_total else 0.0,
        "mean_pose_error_deg": float(np.mean(pose_errors)) if pose_errors else float("nan"),
        "mean_rotation_error_deg": float(np.mean(rot_errors)) if rot_errors else float("nan"),
        "mean_translation_direction_error_deg": float(np.mean(trans_errors))
        if trans_errors
        else float("nan"),
        "mean_world_yaw_error_deg": float(np.mean(world_errors)) if world_errors else float("nan"),
        "auc_thresholds_deg": list(auc_thresholds),
    }


def _rank_extremes(
    records: List[Dict[str, Any]], k: int
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    scored: List[Tuple[float, Dict[str, Any]]] = []
    for rec in records:
        if not rec.get("ok"):
            continue
        w = _world_rot_error_deg(rec)
        if w is None:
            continue
        scored.append((w, rec))
    if not scored:
        return [], []
    scored.sort(key=lambda x: x[0])
    worst = [r for _, r in reversed(scored[-k:])]
    best = [r for _, r in scored[:k]]
    return worst, best


def _rel_viz_path(viz: Optional[str], out_dir: Path) -> Optional[str]:
    if not viz:
        return None
    p = Path(viz)
    try:
        return html.escape(str(p.relative_to(out_dir)))
    except ValueError:
        return html.escape(str(p))


def _write_html_report(
    out_path: Path,
    seq_id: str,
    metrics: Dict[str, Any],
    records: List[Dict[str, Any]],
    worst: List[Dict[str, Any]],
    best: List[Dict[str, Any]],
    pairs_txt: Path,
) -> None:
    out_dir = out_path.parent

    def _metric_rows() -> str:
        rows = []
        for key in sorted(metrics.keys()):
            if key.startswith("auc@"):
                rows.append(f"<tr><td>{html.escape(key)}</td><td>{metrics[key]:.6f}</td></tr>")
        extra = [
            ("n_pairs_total", metrics.get("n_pairs_total")),
            ("n_pairs_ok", metrics.get("n_pairs_ok")),
            ("valid_rate", metrics.get("valid_rate")),
            ("mean_pose_error_deg", metrics.get("mean_pose_error_deg")),
            ("mean_world_yaw_error_deg", metrics.get("mean_world_yaw_error_deg")),
            ("mean_rotation_error_deg", metrics.get("mean_rotation_error_deg")),
            ("mean_translation_direction_error_deg", metrics.get("mean_translation_direction_error_deg")),
        ]
        for name, val in extra:
            if val is None:
                continue
            if isinstance(val, float):
                rows.append(f"<tr><td>{html.escape(name)}</td><td>{val:.4f}</td></tr>")
            else:
                rows.append(f"<tr><td>{html.escape(name)}</td><td>{html.escape(str(val))}</td></tr>")
        return "\n".join(rows)

    def _pair_cards(title: str, items: List[Dict[str, Any]]) -> str:
        if not items:
            return f"<h3>{html.escape(title)}</h3><p>无可用样本。</p>"
        blocks = [f"<h3>{html.escape(title)}</h3>"]
        for rec in items:
            w = _world_rot_error_deg(rec)
            rot = rec.get("rotation_error_deg")
            tdir = rec.get("translation_direction_error_deg")
            viz_rel = _rel_viz_path(rec.get("viz_out"), out_dir)
            img0 = html.escape(str(rec.get("img0", "")))
            img1 = html.escape(str(rec.get("img1", "")))
            ok = rec.get("ok", True)
            cap = f"pair #{rec.get('pair_idx', '?')}"
            if not ok:
                cap += f" | FAIL {rec.get('reason', '?')}"
            elif w is not None:
                cap += f" | |world_yaw_err|={w:.3f}°"
            if rot is not None:
                cap += f" rot={float(rot):.3f}°"
            if tdir is not None:
                cap += f" trans_dir={float(tdir):.3f}°"
            reason = rec.get("reason")
            fail_note = (
                f'<p class="fail"><em>{html.escape(str(reason))}</em></p>'
                if not rec.get("ok", True) and reason
                else ""
            )
            img_block = (
                f'<img src="{viz_rel}" alt="viz" style="max-width:100%;border:1px solid #ccc"/>'
                if viz_rel
                else "<p><em>无可视化</em></p>"
            )
            img_block = fail_note + img_block
            blocks.append(
                f'<section class="pair"><p class="cap">{html.escape(cap)}</p>'
                f'<p class="paths"><code>{img0}</code><br/><code>{img1}</code></p>{img_block}</section>'
            )
        return "\n".join(blocks)

    doc = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<title>SiLK 位姿验证 — {html.escape(seq_id)}</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 20px; max-width: 1200px; }}
table {{ border-collapse: collapse; margin: 12px 0; }}
td, th {{ border: 1px solid #ccc; padding: 6px 10px; }}
section.pair {{ border: 1px solid #ddd; border-radius: 8px; padding: 12px; margin: 16px 0; background: #fafafa; }}
.cap {{ font-weight: 600; }}
.paths {{ font-size: 11px; word-break: break-all; color: #444; }}
</style>
</head>
<body>
<h1>序列 {html.escape(seq_id)}</h1>
<p>图像对列表: <code>{html.escape(str(pairs_txt))}</code> · 共 {len(records)} 对</p>
<h2>AUC 与汇总</h2>
<table><tr><th>指标</th><th>值</th></tr>
{_metric_rows()}
</table>
{_pair_cards(f"全部 {len(records)} 对（含失败样本可视化）", records)}
{_pair_cards(f"世界航向误差最大 {len(worst)} 对", worst)}
{_pair_cards(f"世界航向误差最小 {len(best)} 对", best)}
</body>
</html>
"""
    out_path.write_text(doc, encoding="utf-8")


def _get_metric_auc(metrics: Dict[str, Any], deg: int) -> float:
    return float(metrics.get(f"auc@{deg}", metrics.get(f"auc@{deg}.0", float("nan"))))


def _success_rate(metrics: Dict[str, Any]) -> float:
    if metrics.get("valid_rate") is not None:
        return float(metrics["valid_rate"])
    total = int(metrics.get("n_pairs_total", 0))
    ok = int(metrics.get("n_pairs_ok", 0))
    return float(ok / total) if total else 0.0


def _write_multi_summary(
    out_path: Path,
    seq_results: List[Tuple[str, Path, Dict[str, Any]]],
) -> None:
    rows = []
    total_ok = 0
    total_pairs = 0
    auc_sums = {5: 0.0, 10: 0.0, 20: 0.0}
    auc_weights = 0

    for seq_id, report_dir, metrics in seq_results:
        report_rel = html.escape(str((report_dir / "report.html").relative_to(out_path.parent)))
        n_ok = int(metrics.get("n_pairs_ok", 0))
        n_total = int(metrics.get("n_pairs_total", 0))
        rate = _success_rate(metrics)
        auc5 = _get_metric_auc(metrics, 5)
        auc10 = _get_metric_auc(metrics, 10)
        auc20 = _get_metric_auc(metrics, 20)

        total_ok += n_ok
        total_pairs += n_total
        if n_total > 0 and not any(np.isnan(v) for v in (auc5, auc10, auc20)):
            auc_weights += n_total
            auc_sums[5] += auc5 * n_total
            auc_sums[10] += auc10 * n_total
            auc_sums[20] += auc20 * n_total

        rows.append(
            f"<tr><td>{html.escape(seq_id)}</td>"
            f"<td>{n_ok}/{n_total}</td>"
            f"<td>{rate * 100:.1f}%</td>"
            f"<td>{auc5:.4f}</td>"
            f"<td>{auc10:.4f}</td>"
            f"<td>{auc20:.4f}</td>"
            f'<td><a href="{report_rel}">report</a></td></tr>'
        )

    overall_rate = float(total_ok / total_pairs) if total_pairs else 0.0
    if auc_weights > 0:
        mean_auc5 = auc_sums[5] / auc_weights
        mean_auc10 = auc_sums[10] / auc_weights
        mean_auc20 = auc_sums[20] / auc_weights
    else:
        mean_auc5 = mean_auc10 = mean_auc20 = float("nan")

    summary_row = (
        f'<tr style="font-weight:600;background:#f0f0f0">'
        f"<td>合计（{len(seq_results)} 序列）</td>"
        f"<td>{total_ok}/{total_pairs}</td>"
        f"<td>{overall_rate * 100:.1f}%</td>"
        f"<td>{mean_auc5:.4f}</td>"
        f"<td>{mean_auc10:.4f}</td>"
        f"<td>{mean_auc20:.4f}</td>"
        f"<td>—</td></tr>"
    )

    doc = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8"/>
<title>多序列 SiLK 位姿验证汇总</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 20px; }}
table {{ border-collapse: collapse; margin: 12px 0; }}
td, th {{ border: 1px solid #ccc; padding: 6px 10px; text-align: right; }}
td:first-child, th:first-child {{ text-align: left; }}
</style>
</head><body>
<h1>多序列汇总</h1>
<p>成功率 = n_pairs_ok / n_pairs_total；AUC 列为各序列 auc@5/10/20，合计行为按图像对数加权平均。</p>
<table>
<tr><th>序列</th><th>成功/总数</th><th>成功率</th><th>auc@5</th><th>auc@10</th><th>auc@20</th><th>报告</th></tr>
{"".join(rows)}
{summary_row}
</table></body></html>"""
    out_path.write_text(doc, encoding="utf-8")


def _summarize_multi(root: Path, summary_html: Path) -> int:
    root = root.expanduser().resolve()
    summary_html = summary_html.expanduser().resolve()
    seq_results: List[Tuple[str, Path, Dict[str, Any]]] = []
    for metrics_path in sorted(root.glob("*/metrics.json")):
        seq_id = metrics_path.parent.name
        with metrics_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        metrics = data.get("metrics", data)
        seq_results.append((seq_id, metrics_path.parent, metrics))
    if not seq_results:
        print(f"警告：{root} 下无 */metrics.json", file=sys.stderr)
        return 1
    summary_html.parent.mkdir(parents=True, exist_ok=True)
    _write_multi_summary(summary_html, seq_results)
    print(f"写入多序列汇总: {summary_html} ({len(seq_results)} 个序列)", file=sys.stderr)
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--summarize-root",
        type=Path,
        default=None,
        help="仅汇总多序列：扫描 root/*/metrics.json 写 summary.html（需 --out-dir 作输出 HTML 路径）",
    )
    ap.add_argument("--seq-id", default=None, help="序列名（用于输出目录与报告标题）")
    ap.add_argument("--pairs-txt", type=Path, default=None)
    ap.add_argument("--camera-json", type=Path, default=None)
    ap.add_argument("--extrinsic-json", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--pairs-separator", default=None, help="covis_paths 列分隔符，默认空白")
    ap.add_argument("--max-pairs", type=int, default=-1)
    ap.add_argument("--top-k", type=int, default=5, help="最大/最小 world 误差各取 k 对")
    ap.add_argument("--no-viz", action="store_true", help="不生成逐对可视化")
    ap.add_argument(
        "--viz-jpeg-quality",
        type=int,
        default=88,
        help="可视化存 JPG 时的质量 1–100（默认 88，体积远小于 PNG）",
    )
    ap.add_argument("--fail-error-deg", type=float, default=180.0)
    ap.add_argument(
        "--metric",
        choices=("pose_max", "rotation"),
        default="pose_max",
    )
    ap.add_argument(
        "--auc-thresholds-deg",
        type=float,
        nargs="+",
        default=[5.0, 10.0, 20.0],
    )
    # 推理参数（单进程一次加载）
    ap.add_argument("--camera-model", default="fisheye")
    ap.add_argument("--pose-xy-scale", type=float, default=0.001)
    ap.add_argument("--theta-degrees", action="store_true", help="extra 中 theta 为度")
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--sky-seg-onnx", type=Path, default=None)
    ap.add_argument("--sky-seg-norm", default="imagenet")
    ap.add_argument("--sky-seg-post", default="minmax")
    ap.add_argument("--sky-saliency-threshold", type=int, default=128)
    ap.add_argument(
        "--sky-erode-px",
        type=int,
        default=0,
        help="天空 mask 腐蚀半径（像素）；内缩天空区，交界带保留关键点",
    )
    ap.add_argument("--viz-sky-mask-alpha", type=float, default=0.28)
    ap.add_argument("--ransac-threshold", type=float, default=0.5)
    ap.add_argument("--min-matches", type=int, default=10)
    ap.add_argument("--min-inliers", type=int, default=8)
    ap.add_argument("--sky-class-id", type=int, default=1)
    ap.add_argument("--body-z-world", type=float, default=0.0)
    ap.add_argument("--viz-max-edge", type=int, default=1400)
    ap.add_argument("--viz-max-draw", type=int, default=500)
    ap.add_argument("--viz-gap", type=int, default=16)
    ap.add_argument("--viz-sky-style", default="both", choices=("none", "mask", "contour", "both"))
    ap.add_argument(
        "--exclude-mask-png",
        type=Path,
        default=None,
        help="可选：灰度 mask；像素==255 处丢弃关键点（文件不存在则忽略）",
    )
    ap.add_argument("--exclude-mask-value", type=int, default=255)
    args = ap.parse_args(argv)

    if args.summarize_root is not None:
        if args.out_dir is None:
            print("错误：--summarize-root 需同时指定 --out-dir 为汇总 HTML 路径", file=sys.stderr)
            return 1
        return _summarize_multi(args.summarize_root, args.out_dir)

    if not args.seq_id:
        print("错误：请指定 --seq-id", file=sys.stderr)
        return 1
    if args.pairs_txt is None:
        print("错误：请指定 --pairs-txt", file=sys.stderr)
        return 1
    if args.out_dir is None:
        print("错误：请指定 --out-dir", file=sys.stderr)
        return 1
    if args.camera_json is None or args.extrinsic_json is None:
        print("错误：请指定 --camera-json 与 --extrinsic-json", file=sys.stderr)
        return 1

    pairs_txt = args.pairs_txt.expanduser().resolve()
    if not pairs_txt.is_file():
        print(f"错误：找不到 pairs 文件: {pairs_txt}", file=sys.stderr)
        return 1

    pairs = _parse_pairs_txt(pairs_txt, args.pairs_separator, args.max_pairs)
    if not pairs:
        print(f"错误：{pairs_txt} 中无有效图像对", file=sys.stderr)
        return 1

    out_dir = args.out_dir.expanduser().resolve()
    viz_dir = out_dir / "viz"
    out_dir.mkdir(parents=True, exist_ok=True)

    sky_path = None
    if args.sky_seg_onnx is not None and Path(args.sky_seg_onnx).is_file():
        sky_path = Path(args.sky_seg_onnx).expanduser().resolve()

    ns = spr.namespace_for_batch(
        camera_json=args.camera_json.expanduser().resolve(),
        extrinsic_json=args.extrinsic_json.expanduser().resolve(),
        camera_model=args.camera_model,
        pose_xy_scale=args.pose_xy_scale,
        theta_degrees=args.theta_degrees,
        checkpoint=args.checkpoint.expanduser().resolve() if args.checkpoint else None,
        sky_seg_onnx=sky_path,
        sky_seg_norm=args.sky_seg_norm,
        sky_seg_post=args.sky_seg_post,
        sky_saliency_threshold=args.sky_saliency_threshold,
        sky_erode_px=args.sky_erode_px,
        sky_class_id=args.sky_class_id,
        ransac_threshold=args.ransac_threshold,
        min_matches=args.min_matches,
        min_inliers=args.min_inliers,
        body_z_world=args.body_z_world,
        viz_max_edge=args.viz_max_edge,
        viz_max_draw=args.viz_max_draw,
        viz_gap=args.viz_gap,
        viz_sky_style=args.viz_sky_style,
        viz_sky_mask_alpha=args.viz_sky_mask_alpha,
        exclude_mask_png=(
            args.exclude_mask_png.expanduser().resolve()
            if args.exclude_mask_png is not None
            else None
        ),
        exclude_mask_value=args.exclude_mask_value,
        viz_jpeg_quality=args.viz_jpeg_quality,
    )

    try:
        runtime = spr.build_pair_runtime(ns)
    except (FileNotFoundError, ValueError) as e:
        print(f"错误：初始化运行时失败: {e}", file=sys.stderr)
        return 1

    records: List[Dict[str, Any]] = []
    for idx, (img0, img1) in enumerate(pairs):
        viz_path = None if args.no_viz else viz_dir / f"pair_{idx:05d}.jpg"
        print(f"[{idx + 1}/{len(pairs)}] {Path(img0).name} <-> {Path(img1).name}", file=sys.stderr)
        rec = spr.evaluate_pair(runtime, Path(img0), Path(img1), viz_path)
        rec["pair_idx"] = idx
        records.append(rec)

    metrics = _compute_metrics(
        records, args.auc_thresholds_deg, args.metric, args.fail_error_deg
    )
    metrics["seq_id"] = args.seq_id
    metrics["pairs_txt"] = str(pairs_txt)

    worst, best = _rank_extremes(records, args.top_k)

    jsonl_path = out_dir / "per_pair.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    metrics_path = out_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "worst_world_rot": worst, "best_world_rot": best}, f, indent=2, ensure_ascii=False)

    report_path = out_dir / "report.html"
    _write_html_report(report_path, args.seq_id, metrics, records, worst, best, pairs_txt)

    print(json.dumps({"metrics": metrics, "out_dir": str(out_dir), "report_html": str(report_path)}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
