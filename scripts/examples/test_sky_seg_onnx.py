#!/usr/bin/env python3
"""
单独验证天空/语义分割 ONNX：加载 Session、跑一张图、打印输入输出与类别直方图，
可选保存叠加 mask/轮廓 的可视化图。不依赖 PyTorch / SiLK。

用法（在仓库根目录）:
  python scripts/examples/test_sky_seg_onnx.py ./checkpoints/seg_sky_grass.onnx --image /path/to/a.jpg
  python scripts/examples/test_sky_seg_onnx.py ./checkpoints/seg_sky_grass.onnx   # 无图时用合成图

  # 与常见 PyTorch ImageNet 前处理 + argmax 一致（多类分割）:
  python scripts/examples/test_sky_seg_onnx.py model.onnx --image a.jpg --norm imagenet --post argmax

  # 与参考脚本一致：ImageNet 前处理 + squeeze 后全局 min-max 可视化:
  python scripts/examples/test_sky_seg_onnx.py model.onnx --image a.jpg --norm imagenet --post minmax --out /tmp/sal.png
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from sky_seg_onnx_infer import SkySegOnnxSession, providers_for_device  # noqa: E402


def _synthetic_bgr(h: int, w: int) -> np.ndarray:
    """简单渐变图，用于无样例图时的冒烟测试。"""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    b = (255 * xx / max(w - 1, 1)).astype(np.uint8)
    g = (255 * yy / max(h - 1, 1)).astype(np.uint8)
    r = np.full((h, w), 128, dtype=np.uint8)
    return np.stack([b, g, r], axis=-1)


def _providers_for_device(device: str) -> List[str]:
    return providers_for_device(device)


def _class_histogram(cls: np.ndarray, max_classes: int = 32) -> Tuple[np.ndarray, np.ndarray]:
    flat = cls.reshape(-1)
    ids, cnt = np.unique(flat, return_counts=True)
    order = np.argsort(-cnt)
    ids, cnt = ids[order], cnt[order]
    if ids.size > max_classes:
        ids, cnt = ids[:max_classes], cnt[:max_classes]
    return ids, cnt


def _save_overlay(
    image_bgr: np.ndarray,
    cls_map: np.ndarray,
    sky_class_id: int,
    out_path: Path,
    *,
    mask_alpha: float = 0.28,
) -> None:
    vis = image_bgr.copy()
    sky = (cls_map == int(sky_class_id)).astype(np.uint8)
    if np.any(sky):
        a = float(np.clip(mask_alpha, 0.0, 1.0))
        if a > 0.0:
            color = np.array([255, 90, 90], dtype=np.float64)
            bgr = vis.astype(np.float64)
            m = sky.astype(bool)
            bgr[m] = bgr[m] * (1.0 - a) + color * a
            vis = np.clip(bgr, 0, 255).astype(np.uint8)
        sky_u8 = (sky * 255).astype(np.uint8)
        cnts, _ = cv2.findContours(sky_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cnts, -1, (0, 255, 255), 2, lineType=cv2.LINE_AA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), vis)


def _save_overlay_minmax(
    image_bgr: np.ndarray,
    sal_u8: np.ndarray,
    out_path: Path,
    *,
    mask_alpha: float = 0.28,
    threshold: int = 128,
) -> None:
    """min-max 后的 uint8 显著图：大于阈值的区域视为天空做半透明 + 轮廓。"""
    vis = image_bgr.copy()
    sky = (sal_u8.astype(np.int32) > int(threshold)).astype(np.uint8)
    if np.any(sky):
        a = float(np.clip(mask_alpha, 0.0, 1.0))
        if a > 0.0:
            color = np.array([255, 90, 90], dtype=np.float64)
            bgr = vis.astype(np.float64)
            m = sky.astype(bool)
            bgr[m] = bgr[m] * (1.0 - a) + color * a
            vis = np.clip(bgr, 0, 255).astype(np.uint8)
        sky_u8 = (sky * 255).astype(np.uint8)
        cnts, _ = cv2.findContours(sky_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cnts, -1, (0, 255, 255), 2, lineType=cv2.LINE_AA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), vis)


def main() -> int:
    ap = argparse.ArgumentParser(description="单独测试天空分割 ONNX 权重")
    ap.add_argument("onnx", type=Path, help="ONNX 模型路径")
    ap.add_argument(
        "--image",
        type=Path,
        default=None,
        help="输入 BGR 图像；省略则用合成图（仅验证能否跑通）",
    )
    ap.add_argument("--sky-class-id", type=int, default=1, help="天空类别 id（用于叠加与像素占比）")
    ap.add_argument(
        "--norm",
        choices=("mmseg", "imagenet"),
        default="mmseg",
        help="前处理：mmseg=SegDataPreProcessor(0-255 mean/std)；imagenet=Resize+BGR→RGB+(x/255-mean)/std",
    )
    ap.add_argument(
        "--post",
        choices=("argmax", "minmax"),
        default="argmax",
        help="后处理：argmax=多类分割类别图；minmax=参考脚本 squeeze 后全局 min-max 再 *255→uint8",
    )
    ap.add_argument(
        "--minmax-threshold",
        type=int,
        default=128,
        help="--post minmax 时，显著图大于该值(0-255)的像素视为天空用于叠加",
    )
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cpu", help="ExecutionProvider")
    ap.add_argument("--out", type=Path, default=None, help="保存叠加 sky mask+轮廓 的可视化路径")
    ap.add_argument("--mask-alpha", type=float, default=0.28, help="叠加透明度 0~1")
    ap.add_argument("--warmup", type=int, default=1, help="计时时预热次数")
    ap.add_argument("--repeat", type=int, default=1, help="计时时重复推理次数（同一 Session）")
    ap.add_argument("--synthetic-h", type=int, default=480)
    ap.add_argument("--synthetic-w", type=int, default=640)
    args = ap.parse_args()

    onnx_p = args.onnx.expanduser().resolve()
    if not onnx_p.is_file():
        print(f"错误：找不到 ONNX 文件: {onnx_p}", file=sys.stderr)
        return 2

    if args.image is not None:
        img_p = args.image.expanduser().resolve()
        if not img_p.is_file():
            print(f"错误：找不到图像: {img_p}", file=sys.stderr)
            return 2
        im = cv2.imread(str(img_p), cv2.IMREAD_COLOR)
        if im is None:
            print(f"错误：无法读取图像: {img_p}", file=sys.stderr)
            return 2
        src = str(img_p)
    else:
        im = _synthetic_bgr(int(args.synthetic_h), int(args.synthetic_w))
        src = f"<synthetic {im.shape[0]}x{im.shape[1]}>"

    prov = _providers_for_device(args.device)
    t0 = time.perf_counter()
    out_style = "minmax_u8" if args.post == "minmax" else "argmax"
    try:
        seg = SkySegOnnxSession(
            onnx_p,
            providers=prov,
            norm_style=args.norm,
            output_style=out_style,
        )
    except Exception as e:  # noqa: BLE001
        print(f"错误：加载 ONNX 失败: {e}", file=sys.stderr)
        return 2
    t_load = time.perf_counter() - t0

    print(f"onnx: {onnx_p}")
    print(f"image: {src} shape={im.shape} dtype={im.dtype}")
    print(f"providers: {prov} (load {t_load*1000:.1f} ms)")
    print(f"model input: name={seg.input_name!r} shape={seg.input_shape!r}")
    print(f"preprocess: norm={args.norm!r} post={args.post!r}")

    for i, (name, shape, typ) in enumerate(seg.output_infos):
        print(f"model output[{i}]: name={name!r} shape={shape!r} type={typ}")

    cls: Optional[np.ndarray] = None
    for _ in range(max(0, int(args.warmup))):
        cls = seg.predict(im)
    if cls is None:
        cls = seg.predict(im)

    t_inf0 = time.perf_counter()
    for _ in range(max(1, int(args.repeat))):
        cls = seg.predict(im)
    t_inf = time.perf_counter() - t_inf0

    print(f"result: shape={cls.shape} dtype={cls.dtype} min={cls.min()} max={cls.max()}")
    if cls.dtype == np.uint8 and args.post == "minmax":
        thr = int(args.minmax_threshold)
        sky_frac = float(np.mean(cls.astype(np.int32) > thr))
        print(f"minmax saliency: threshold={thr} -> sky-like pixel fraction: {100.0 * sky_frac:.2f}%")
    else:
        ids, cnt = _class_histogram(cls.astype(np.int32))
        total = int(cls.size)
        lines = [f"  class {int(cid)}: {int(n)} px ({100.0 * n / total:.2f}%)" for cid, n in zip(ids, cnt)]
        print("class histogram (top by count):")
        print("\n".join(lines))

        sky_id = int(args.sky_class_id)
        sky_frac = float(np.mean(cls.astype(np.int32) == sky_id))
        print(f"sky (class_id={sky_id}) pixel fraction: {100.0 * sky_frac:.2f}%")

    per = t_inf / max(1, int(args.repeat))
    print(f"inference: {args.repeat} run(s) in {t_inf*1000:.2f} ms ({per*1000:.2f} ms / run, after warmup)")

    if args.out is not None:
        outp = args.out.expanduser().resolve()
        if args.post == "minmax" and cls.dtype == np.uint8:
            _save_overlay_minmax(
                im,
                cls,
                outp,
                mask_alpha=float(args.mask_alpha),
                threshold=int(args.minmax_threshold),
            )
        else:
            _save_overlay(
                im,
                cls.astype(np.int32),
                int(args.sky_class_id),
                outp,
                mask_alpha=float(args.mask_alpha),
            )
        print(f"wrote: {outp.resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
