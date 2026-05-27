#!/usr/bin/env python3
"""
批量提取共视图中所有唯一图像的 SiLK 关键点 + 描述子（不做 match）。

可选在缓存阶段做天空 / exclude mask 点过滤，写入已过滤的 positions+descriptors；
绝对位姿阶段读缓存时不再重复 sky ONNX 与 mask 过滤。

输出目录结构:
  <out-dir>/features/<hash>.npz
  <out-dir>/index.jsonl
  <out-dir>/meta.json

由 batch_cache_covis_silk_features.sh 调用；绝对位姿阶段通过 --feature-cache-dir 复用。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EVAL_DIR = _REPO_ROOT / "scripts" / "eval"
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_EVAL_DIR))

import silk_pair_runtime as spr  # noqa: E402
import silk_feature_cache as sfc  # noqa: E402
from lib.geometry.absolute_pose_from_covis import load_covis_adjacency  # noqa: E402


def _filter_config(args: argparse.Namespace) -> Dict[str, Any]:
    sky_path = None
    if args.sky_seg_onnx is not None and Path(args.sky_seg_onnx).is_file():
        sky_path = str(Path(args.sky_seg_onnx).expanduser().resolve())
    mask_path = None
    if args.exclude_mask_png is not None and Path(args.exclude_mask_png).is_file():
        mask_path = str(Path(args.exclude_mask_png).expanduser().resolve())
    return {
        "points_filtered": bool(sky_path or mask_path),
        "sky_seg_onnx": sky_path,
        "sky_seg_device": getattr(args, "sky_seg_device", None),
        "sky_seg_norm": args.sky_seg_norm,
        "sky_seg_post": args.sky_seg_post,
        "sky_saliency_threshold": args.sky_saliency_threshold,
        "sky_erode_px": args.sky_erode_px,
        "sky_class_id": args.sky_class_id,
        "exclude_mask_png": mask_path,
        "exclude_mask_value": args.exclude_mask_value,
    }


def run_cache(args: argparse.Namespace) -> Dict[str, Any]:
    covis_path = (
        args.covis_adj_json.expanduser().resolve()
        if args.covis_adj_json is not None
        else (args.data_dir / "covis_output" / "covis_adj.json").expanduser().resolve()
    )
    if not covis_path.is_file():
        raise FileNotFoundError(f"找不到共视 JSON: {covis_path}")

    _, nodes = load_covis_adjacency(covis_path)
    images = sfc.unique_images_from_covis_nodes(nodes)
    if not images:
        raise RuntimeError("共视图中无图像节点")

    out_dir = args.out_dir.expanduser().resolve()
    store = sfc.FeatureCacheStore(out_dir)
    filt = _filter_config(args)

    sky_path = filt["sky_seg_onnx"]
    mask_path = filt["exclude_mask_png"]
    ns = spr.namespace_for_feature_extract(
        checkpoint=args.checkpoint.expanduser().resolve() if args.checkpoint else None,
        sky_seg_onnx=Path(sky_path) if sky_path else None,
        sky_seg_device=getattr(args, "sky_seg_device", None),
        sky_seg_norm=args.sky_seg_norm,
        sky_seg_post=args.sky_seg_post,
        sky_saliency_threshold=args.sky_saliency_threshold,
        sky_erode_px=args.sky_erode_px,
        sky_class_id=args.sky_class_id,
        exclude_mask_png=Path(mask_path) if mask_path else None,
        exclude_mask_value=args.exclude_mask_value,
        nms_dist=int(getattr(args, "nms_dist", 9)),
        border_dist=int(getattr(args, "border_dist", 20)),
        detection_top_k=int(getattr(args, "detection_top_k", 20000)),
        detection_threshold=float(getattr(args, "detection_threshold", 1.0)),
        use_anms=bool(getattr(args, "use_anms", False)),
        anms_top_k=int(getattr(args, "anms_top_k", 2000)),
        anms_min_radius=float(getattr(args, "anms_min_radius", 8.0)),
    )
    runtime = spr.build_feature_extract_runtime(ns)

    require_filtered = bool(filt["points_filtered"])
    if require_filtered:
        print(
            "缓存阶段做点过滤: "
            f"sky={sky_path or '无'} mask={mask_path or '无'}",
            file=sys.stderr,
        )
    else:
        print("缓存阶段不做 sky/mask 点过滤（未指定 sky ONNX 与 exclude mask）", file=sys.stderr)

    n_ok = n_skip = n_fail = 0
    failed: List[str] = []

    for idx, img in enumerate(images):
        print(f"[{idx + 1}/{len(images)}] {img.name}", file=sys.stderr)
        if args.skip_existing and store.has(img, require_points_filtered=require_filtered):
            n_skip += 1
            continue
        if not img.is_file():
            print(f"  跳过：文件不存在 {img}", file=sys.stderr)
            n_fail += 1
            failed.append(str(img))
            continue
        try:
            if require_filtered:
                feat = spr.extract_and_filter_image_features(runtime, img)
            else:
                feat = spr.extract_image_features(runtime, img)
                if getattr(args, "use_anms", False):
                    pos, desc = spr._maybe_anms_filter(
                        runtime, feat.positions_xy, feat.descriptors
                    )
                    feat = sfc.ImageFeatures(
                        image_path=feat.image_path,
                        positions_xy=pos,
                        descriptors=desc,
                        image_hw=feat.image_hw,
                        points_filtered=False,
                        n_keypoints_raw=int(feat.n_keypoints_raw),
                    )
            store.save(feat)
            n_ok += 1
        except (OSError, RuntimeError, FileNotFoundError, ValueError) as e:
            print(f"  失败: {e}", file=sys.stderr)
            n_fail += 1
            failed.append(str(img))

    summary = {
        "seq_id": args.seq_id,
        "covis_adj_json": str(covis_path),
        "out_dir": str(out_dir),
        "n_images_total": len(images),
        "n_cached_ok": n_ok,
        "n_skipped_existing": n_skip,
        "n_failed": n_fail,
        "checkpoint": str(runtime.ckpt),
        "failed_images": failed,
        **_filter_config(args),
    }
    store.write_meta(summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seq-id", required=True)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--covis-adj-json", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--sky-seg-onnx", type=Path, default=None)
    ap.add_argument("--sky-seg-device", default=None, help="cuda|cpu，默认 SKY_SEG_DEVICE 或 cuda")
    ap.add_argument("--sky-seg-norm", default="imagenet")
    ap.add_argument("--sky-seg-post", default="minmax")
    ap.add_argument("--sky-saliency-threshold", type=int, default=128)
    ap.add_argument("--sky-erode-px", type=int, default=4)
    ap.add_argument("--sky-class-id", type=int, default=1)
    ap.add_argument("--exclude-mask-png", type=Path, default=None)
    ap.add_argument("--exclude-mask-value", type=int, default=0)
    ap.add_argument("--nms-dist", type=int, default=9)
    ap.add_argument("--border-dist", type=int, default=20)
    ap.add_argument("--detection-top-k", type=int, default=20000)
    ap.add_argument("--detection-threshold", type=float, default=1.0)
    ap.add_argument("--use-anms", action="store_true")
    ap.add_argument("--anms-top-k", type=int, default=2000)
    ap.add_argument("--anms-min-radius", type=float, default=8.0)
    return ap


def run_from_config(cfg) -> Any:
    """供 lib.cli.feature_cache 调用。"""
    m = cfg.mode
    argv = [
        "--seq-id", str(m.seq_id),
        "--data-dir", str(m.data_dir),
        "--out-dir", str(m.out_dir),
        "--sky-seg-norm", str(getattr(m, "sky_seg_norm", "imagenet")),
        "--sky-seg-post", str(getattr(m, "sky_seg_post", "minmax")),
        "--sky-saliency-threshold", str(int(getattr(m, "sky_saliency_threshold", 128))),
        "--sky-erode-px", str(int(getattr(m, "sky_erode_px", 4))),
        "--sky-class-id", str(int(getattr(m, "sky_class_id", 1))),
        "--exclude-mask-value", str(int(getattr(m, "exclude_mask_value", 0))),
        "--nms-dist", str(int(getattr(m, "nms_dist", 9))),
        "--border-dist", str(int(getattr(m, "border_dist", 20))),
        "--detection-top-k", str(int(getattr(m, "detection_top_k", 20000))),
        "--detection-threshold", str(float(getattr(m, "detection_threshold", 1.0))),
        "--anms-top-k", str(int(getattr(m, "anms_top_k", 2000))),
        "--anms-min-radius", str(float(getattr(m, "anms_min_radius", 8.0))),
    ]
    if getattr(m, "covis_adj_json", None):
        argv.extend(["--covis-adj-json", str(m.covis_adj_json)])
    if getattr(m, "checkpoint", None):
        argv.extend(["--checkpoint", str(m.checkpoint)])
    if getattr(m, "skip_existing", False):
        argv.append("--skip-existing")
    if getattr(m, "sky_seg_onnx", None):
        argv.extend(["--sky-seg-onnx", str(m.sky_seg_onnx)])
    if getattr(m, "sky_seg_device", None):
        argv.extend(["--sky-seg-device", str(m.sky_seg_device)])
    if getattr(m, "exclude_mask_png", None):
        argv.extend(["--exclude-mask-png", str(m.exclude_mask_png)])
    if getattr(m, "use_anms", False):
        argv.append("--use-anms")
    return main(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        summary = run_cache(args)
    except (FileNotFoundError, RuntimeError) as e:
        print(f"错误: {e}", file=sys.stderr)
        return 1
    return 1 if summary.get("n_failed", 0) > 0 and summary.get("n_cached_ok", 0) == 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
