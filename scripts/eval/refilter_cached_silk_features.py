#!/usr/bin/env python3
"""
阶段 2：读 bin/silk-features 的 .pt，sky/mask/ANMS 过滤后写入 features/*.npz。

sky ONNX / exclude mask 在 batch 开始时各加载一次，循环内仅 predict / 点过滤。
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLES = _REPO_ROOT / "scripts" / "examples"
_EVAL_DIR = _REPO_ROOT / "scripts" / "eval"
for p in (_EXAMPLES, _EVAL_DIR, _REPO_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import silk_feature_cache as sfc  # noqa: E402
from lib.inference.feature_postprocess import (  # noqa: E402
    PointFilterConfig,
    apply_point_filters,
    maybe_anms_filter,
)
from lib.inference.keypoint_filters import load_exclude_mask_grayscale  # noqa: E402
from sky_seg_onnx_infer import SkySegOnnxSession, providers_for_device  # noqa: E402


@dataclass
class PointFilterContext:
    sky_seg: Optional[SkySegOnnxSession]
    exclude_mask: Optional[np.ndarray]
    config: PointFilterConfig


def _build_filter_ctx(args: argparse.Namespace) -> PointFilterContext:
    cfg = PointFilterConfig(
        sky_seg_post=str(args.sky_seg_post),
        sky_saliency_threshold=int(args.sky_saliency_threshold),
        sky_erode_px=int(args.sky_erode_px),
        sky_class_id=int(args.sky_class_id),
        exclude_mask_value=int(args.exclude_mask_value),
        use_anms=bool(getattr(args, "use_anms", False)),
        anms_top_k=int(getattr(args, "anms_top_k", 2000)),
        anms_min_radius=float(getattr(args, "anms_min_radius", 8.0)),
    )

    sky_seg: Optional[SkySegOnnxSession] = None
    if args.sky_seg_onnx is not None:
        seg_path = Path(args.sky_seg_onnx).expanduser().resolve()
        if not seg_path.is_file():
            raise FileNotFoundError(f"找不到天空分割 ONNX: {seg_path}")
        norm_style = "imagenet" if args.sky_seg_norm == "imagenet" else "mmseg"
        output_style = "minmax_u8" if args.sky_seg_post == "minmax" else "argmax"
        sky_device = getattr(args, "sky_seg_device", None)
        providers = providers_for_device(str(sky_device)) if sky_device else None
        print(f"[refilter] 加载 sky ONNX（一次）: {seg_path}", file=sys.stderr)
        sky_seg = SkySegOnnxSession(
            seg_path,
            norm_style=norm_style,
            output_style=output_style,
            providers=providers,
        )
        print(f"[refilter] sky providers: {sky_seg.providers}", file=sys.stderr)

    exclude_mask: Optional[np.ndarray] = None
    if args.exclude_mask_png is not None:
        mask_path = Path(args.exclude_mask_png).expanduser().resolve()
        if mask_path.is_file():
            print(f"[refilter] 加载 exclude mask（一次）: {mask_path}", file=sys.stderr)
            exclude_mask = load_exclude_mask_grayscale(mask_path)

    if sky_seg is None and exclude_mask is None and not cfg.use_anms:
        raise ValueError("请至少指定 --sky-seg-onnx、--exclude-mask-png 之一，或 --use-anms")

    return PointFilterContext(sky_seg=sky_seg, exclude_mask=exclude_mask, config=cfg)


def _load_scores_from_pt(pt_path: Path) -> Optional[np.ndarray]:
    import torch

    data = torch.load(str(pt_path.expanduser().resolve()), map_location="cpu")
    if "score" not in data:
        return None
    return np.asarray(data["score"], dtype=np.float64).reshape(-1)


def _postprocess_features(
    ctx: PointFilterContext,
    img_path: Path,
    positions_xy: np.ndarray,
    descriptors: np.ndarray,
    scores: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    pos = np.asarray(positions_xy, dtype=np.float64).reshape(-1, 2)
    desc = np.asarray(descriptors, dtype=np.float32)
    sc = scores

    if ctx.sky_seg is not None or ctx.exclude_mask is not None:
        im_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if im_bgr is None:
            raise FileNotFoundError(f"无法读取图像: {img_path}")
        filtered = apply_point_filters(
            pos,
            desc,
            img_path=img_path,
            im_bgr=im_bgr,
            sky_seg=ctx.sky_seg,
            exclude_mask=ctx.exclude_mask,
            config=ctx.config,
            scores=sc,
        )
        if sc is None:
            pos, desc = filtered
        else:
            pos, desc, sc = filtered

    result = maybe_anms_filter(pos, desc, config=ctx.config, scores=sc)
    if sc is None:
        return result
    pos, desc, _sc = result
    return pos, desc


def _filter_meta(args: argparse.Namespace) -> Dict[str, Any]:
    sky_path = None
    if args.sky_seg_onnx is not None and Path(args.sky_seg_onnx).is_file():
        sky_path = str(Path(args.sky_seg_onnx).expanduser().resolve())
    mask_path = None
    if args.exclude_mask_png is not None and Path(args.exclude_mask_png).is_file():
        mask_path = str(Path(args.exclude_mask_png).expanduser().resolve())
    return {
        "points_filtered": True,
        "refilter_from_pt": True,
        "sky_seg_onnx": sky_path,
        "sky_seg_device": getattr(args, "sky_seg_device", None),
        "sky_seg_norm": args.sky_seg_norm,
        "sky_seg_post": args.sky_seg_post,
        "sky_saliency_threshold": args.sky_saliency_threshold,
        "sky_erode_px": args.sky_erode_px,
        "sky_class_id": args.sky_class_id,
        "exclude_mask_png": mask_path,
        "exclude_mask_value": args.exclude_mask_value,
        "use_anms": bool(getattr(args, "use_anms", False)),
        "anms_top_k": int(getattr(args, "anms_top_k", 2000)),
        "anms_min_radius": float(getattr(args, "anms_min_radius", 8.0)),
    }


def _list_pt_files(pt_dir: Path) -> List[Path]:
    return sorted(pt_dir.glob("*.pt"))


def run_refilter(args: argparse.Namespace) -> Dict[str, Any]:
    cache_dir = args.cache_dir.expanduser().resolve()
    if not cache_dir.is_dir():
        raise FileNotFoundError(f"特征缓存目录不存在: {cache_dir}")

    pt_dir = (args.pt_dir or (cache_dir / "pt")).expanduser().resolve()
    if not pt_dir.is_dir():
        raise FileNotFoundError(f"找不到 .pt 目录: {pt_dir}，请先跑 batch_cache_covis_silk_features_filter.sh")

    pt_files = _list_pt_files(pt_dir)
    if not pt_files:
        raise FileNotFoundError(f"{pt_dir} 下无 .pt 文件")

    store = sfc.FeatureCacheStore(cache_dir)
    ctx = _build_filter_ctx(args)
    index_rows: List[Dict[str, Any]] = []
    n_ok = n_skip = n_fail = 0
    failed: List[str] = []

    for idx, pt_path in enumerate(pt_files):
        try:
            feat = sfc.load_from_silk_features_pt(pt_path)
        except (OSError, ValueError, KeyError, FileNotFoundError) as e:
            print(f"  跳过损坏 pt {pt_path.name}: {e}", file=sys.stderr)
            n_fail += 1
            failed.append(str(pt_path))
            continue

        img = Path(feat.image_path)
        print(f"[{idx + 1}/{len(pt_files)}] {img.name}", file=sys.stderr)

        if (
            args.skip_existing
            and store.has(img, require_points_filtered=True)
            and not args.force
        ):
            print("  npz 已是 points_filtered，跳过", file=sys.stderr)
            n_skip += 1
            npz = store.npz_path(img)
            index_rows.append(
                {
                    "image": str(img),
                    "npz": str(npz),
                    "pt": str(pt_path),
                    "n_keypoints": int(feat.positions_xy.shape[0]),
                    "n_keypoints_raw": int(feat.n_keypoints_raw),
                    "points_filtered": True,
                }
            )
            continue

        if not img.is_file():
            print(f"  失败：图像不存在 {img}", file=sys.stderr)
            n_fail += 1
            failed.append(str(img))
            continue

        try:
            n_raw = int(feat.positions_xy.shape[0])
            scores = _load_scores_from_pt(pt_path)
            pos_f, desc_f = _postprocess_features(
                ctx, img, feat.positions_xy, feat.descriptors, scores
            )
            out_feat = sfc.ImageFeatures(
                image_path=str(img),
                positions_xy=pos_f,
                descriptors=desc_f,
                image_hw=feat.image_hw,
                points_filtered=True,
                n_keypoints_raw=n_raw,
            )
            out_npz = store.save(out_feat, update_index=False)
            index_rows.append(
                {
                    "image": str(img),
                    "npz": str(out_npz),
                    "pt": str(pt_path),
                    "n_keypoints": int(pos_f.shape[0]),
                    "n_keypoints_raw": n_raw,
                    "points_filtered": True,
                }
            )
            print(
                f"  kp {n_raw} -> {pos_f.shape[0]} "
                f"({100.0 * pos_f.shape[0] / max(n_raw, 1):.1f}% kept)",
                file=sys.stderr,
            )
            n_ok += 1
        except (OSError, RuntimeError, FileNotFoundError, ValueError) as e:
            print(f"  失败: {e}", file=sys.stderr)
            n_fail += 1
            failed.append(str(img))

    store.rewrite_index(index_rows)
    prev_meta = store.read_meta()
    summary = {
        **prev_meta,
        **_filter_meta(args),
        "seq_id": args.seq_id or prev_meta.get("seq_id"),
        "cache_dir": str(cache_dir),
        "pt_dir": str(pt_dir),
        "n_pt_total": len(pt_files),
        "n_refiltered_ok": n_ok,
        "n_skipped": n_skip,
        "n_failed": n_fail,
        "failed_images": failed,
    }
    store.write_meta(summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seq-id", default=None)
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--pt-dir", type=Path, default=None, help="默认 <cache-dir>/pt")
    ap.add_argument("--sky-seg-onnx", type=Path, default=None)
    ap.add_argument("--sky-seg-device", default=None)
    ap.add_argument("--sky-seg-norm", default="imagenet")
    ap.add_argument("--sky-seg-post", default="minmax")
    ap.add_argument("--sky-saliency-threshold", type=int, default=128)
    ap.add_argument("--sky-erode-px", type=int, default=4)
    ap.add_argument("--sky-class-id", type=int, default=1)
    ap.add_argument("--exclude-mask-png", type=Path, default=None)
    ap.add_argument("--exclude-mask-value", type=int, default=0)
    ap.add_argument("--use-anms", action="store_true")
    ap.add_argument("--anms-top-k", type=int, default=2000)
    ap.add_argument("--anms-min-radius", type=float, default=8.0)
    ap.add_argument("--skip-existing", action="store_true", help="已有 points_filtered npz 则跳过")
    ap.add_argument(
        "--force",
        action="store_true",
        help="即使已有 points_filtered npz 也重新从 .pt 过滤",
    )
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        summary = run_refilter(args)
    except (FileNotFoundError, RuntimeError, ValueError) as e:
        print(f"错误: {e}", file=sys.stderr)
        return 1
    return 1 if summary.get("n_failed", 0) > 0 and summary.get("n_refiltered_ok", 0) == 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
