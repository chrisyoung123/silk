#!/usr/bin/env python3
"""
在 2D 平面上可视化多帧位姿 (x, y, theta)，鼠标悬停到某帧附近时在侧栏显示对应图片。

默认从 JSON 读取：
  jsondata["value"]["pose"]["x"]
  jsondata["value"]["pose"]["y"]
  jsondata["value"]["pose"]["theta"]

图片默认与 JSON 同目录、同主文件名，扩展名为 .jpg（可用参数改）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, List, Optional, Tuple

import imageio.v2 as imageio
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager


def _configure_matplotlib_cjk(font_family: Optional[str] = None) -> None:
    """避免中文标题/说明触发 DejaVu Sans 缺字警告；优先用系统已装的中日韩字体。"""
    matplotlib.rcParams["axes.unicode_minus"] = False

    if font_family:
        matplotlib.rcParams["font.sans-serif"] = [font_family, "DejaVu Sans", "sans-serif"]
        return

    preferred = (
        "Noto Sans CJK SC",
        "Noto Sans CJK JP",
        "Noto Sans CJK TC",
        "Noto Serif CJK SC",
        "Source Han Sans SC",
        "Source Han Sans CN",
        "WenQuanYi Micro Hei",
        "WenQuanYi Zen Hei",
        "AR PL UMing CN",
        "Droid Sans Fallback",
        "SimHei",
        "Microsoft YaHei",
        "PingFang SC",
        "Hiragino Sans GB",
    )
    try:
        avail = {getattr(f, "name", "") for f in font_manager.fontManager.ttflist}
    except Exception:
        avail = set()
    for name in preferred:
        if name in avail:
            matplotlib.rcParams["font.sans-serif"] = [name, "DejaVu Sans", "sans-serif"]
            return
    for f in font_manager.fontManager.ttflist:
        n = getattr(f, "name", "") or ""
        if "CJK" in n and "Noto" in n:
            matplotlib.rcParams["font.sans-serif"] = [n, "DejaVu Sans", "sans-serif"]
            return
    for f in font_manager.fontManager.ttflist:
        n = getattr(f, "name", "") or ""
        if "WenQuanYi" in n or "Source Han Sans" in n:
            matplotlib.rcParams["font.sans-serif"] = [n, "DejaVu Sans", "sans-serif"]
            return
    print(
        "提示：未找到常见中文字体，中文可能仍显示为方框。可安装 "
        "`fonts-noto-cjk`（Debian/Ubuntu）或使用 --font-family 指定本机字体名。",
        file=sys.stderr,
    )


def _get_nested(obj: Any, dotted: str) -> Any:
    cur = obj
    for part in dotted.split("."):
        if cur is None or not isinstance(cur, dict):
            raise KeyError(dotted)
        cur = cur[part]
    return cur


def _resolve_image_path(
    json_path: Path,
    image_dir: Path,
    image_ext: str,
    image_extra: str,
    image_json_key: Optional[str],
    data: dict,
) -> Path:
    if image_json_key:
        rel = _get_nested(data, image_json_key)
        if not isinstance(rel, str):
            raise TypeError(f"{image_json_key} must be a string path, got {type(rel)}")
        p = Path(rel)
        if p.is_absolute():
            return p
        return image_dir / p
    # 默认：图片主名 = JSON 主名；若约定「jpg 主名 + extra == json 主名」，用 --image-extra 还原图片主名
    jstem = json_path.stem
    if image_extra:
        if jstem.endswith(image_extra):
            img_stem = jstem[: -len(image_extra)]
        else:
            img_stem = jstem
        name = f"{img_stem}{image_ext}"
    else:
        name = f"{jstem}{image_ext}"
    return image_dir / name


def _collect_samples(
    root: Path,
    recursive: bool,
    json_glob: str,
    pose_x_key: str,
    pose_y_key: str,
    pose_theta_key: str,
    image_dir: Path,
    image_ext: str,
    image_extra: str,
    image_json_key: Optional[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Path], List[Path], List[Optional[np.ndarray]]]:
    if recursive:
        json_files = sorted(root.rglob(json_glob))
    else:
        json_files = sorted(root.glob(json_glob))

    xs: List[float] = []
    ys: List[float] = []
    thetas: List[float] = []
    json_paths: List[Path] = []
    image_paths: List[Path] = []
    thumbs: List[Optional[np.ndarray]] = []

    for jp in json_files:
        if not jp.is_file():
            continue
        try:
            with jp.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"跳过（无法读取 JSON）: {jp}: {e}", file=sys.stderr)
            continue

        try:
            x = float(_get_nested(data, pose_x_key))
            y = float(_get_nested(data, pose_y_key))
            theta = float(_get_nested(data, pose_theta_key))
        except (KeyError, TypeError, ValueError) as e:
            print(f"跳过（位姿字段缺失或非法）: {jp}: {e}", file=sys.stderr)
            continue

        img_path = _resolve_image_path(jp, image_dir, image_ext, image_extra, image_json_key, data)

        xs.append(x)
        ys.append(y)
        thetas.append(theta)
        json_paths.append(jp)
        image_paths.append(img_path)

        arr: Optional[np.ndarray] = None
        if img_path.is_file():
            try:
                arr = imageio.imread(img_path)
            except OSError as e:
                print(f"警告：无法解码图片 {img_path}: {e}", file=sys.stderr)
        else:
            print(f"警告：图片不存在 {img_path}", file=sys.stderr)
        thumbs.append(arr)

    if not xs:
        raise RuntimeError("未加载到任何有效样本，请检查目录与 JSON 字段路径。")

    return (
        np.asarray(xs, dtype=np.float64),
        np.asarray(ys, dtype=np.float64),
        np.asarray(thetas, dtype=np.float64),
        json_paths,
        image_paths,
        thumbs,
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description="2D 地图位姿 + 朝向可视化；悬停显示对应图片。"
    )
    ap.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="包含 JSON（及可选图片）的根目录",
    )
    ap.add_argument(
        "--recursive",
        action="store_true",
        help="递归扫描子目录中的 JSON",
    )
    ap.add_argument(
        "--json-glob",
        default="*.json",
        help="匹配 JSON 的 glob（相对 data-dir）",
    )
    ap.add_argument("--pose-x-key", default="value.pose.x", help="JSON 中 x 的点分路径")
    ap.add_argument("--pose-y-key", default="value.pose.y", help="JSON 中 y 的点分路径")
    ap.add_argument(
        "--pose-theta-key",
        default="value.pose.theta",
        help="JSON 中 theta 的点分路径（与 --theta-degrees 配合）",
    )
    ap.add_argument(
        "--theta-degrees",
        action="store_true",
        help="theta 为度；否则按弧度",
    )
    ap.add_argument(
        "--theta-offset-deg",
        type=float,
        default=0.0,
        help="在内部换算成弧度前给 theta 加的偏置（度）",
    )
    ap.add_argument(
        "--flip-y",
        action="store_true",
        help="绘图时 y 轴取反（图像坐标系常用）",
    )
    ap.add_argument(
        "--position-scale",
        type=float,
        default=1.0,
        help="对 JSON 读出的 x、y 乘以此系数后再绘图；单位为毫米、要按米显示时请设为 0.001（即除以 1000）",
    )
    ap.add_argument(
        "--image-dir",
        type=Path,
        default=None,
        help="图片目录（默认与 --data-dir 相同）",
    )
    ap.add_argument(
        "--image-ext",
        default=".jpg",
        help="当未使用 --image-json-key 时，与 JSON 同 stem 的图片扩展名",
    )
    ap.add_argument(
        "--image-extra",
        default="",
        help='若图片名为 stem+extra+ext，而 JSON 为 stem.json，则填 extra（例如 "_rgb"）',
    )
    ap.add_argument(
        "--image-json-key",
        default=None,
        help="若 JSON 内含图片相对/绝对路径字段，填点分键名（优先级高于 stem 规则）",
    )
    ap.add_argument(
        "--arrow-scale",
        type=float,
        default=0.025,
        help="箭头长度 = 该系数 × max(x跨度, y跨度)；仍嫌长可再改小，例如 0.015",
    )
    ap.add_argument(
        "--pick-radius",
        type=float,
        default=None,
        help="悬停选中半径（与 x,y 同单位）；默认约为坐标跨度的 2%%",
    )
    ap.add_argument(
        "--title",
        default="Pose map (hover to preview image)",
        help="窗口标题",
    )
    ap.add_argument(
        "--font-family",
        default=None,
        help="Matplotlib 无衬线字体族名（本机 fontconfig 已注册）；用于中文，不设则自动探测",
    )
    args = ap.parse_args()

    _configure_matplotlib_cjk(args.font_family)

    root = args.data_dir.expanduser().resolve()
    image_dir = (args.image_dir or root).expanduser().resolve()
    if not root.is_dir():
        print(f"错误：data-dir 不是目录: {root}", file=sys.stderr)
        return 1

    xs, ys, thetas, json_paths, image_paths, thumbs = _collect_samples(
        root=root,
        recursive=args.recursive,
        json_glob=args.json_glob,
        pose_x_key=args.pose_x_key,
        pose_y_key=args.pose_y_key,
        pose_theta_key=args.pose_theta_key,
        image_dir=image_dir,
        image_ext=args.image_ext,
        image_extra=args.image_extra,
        image_json_key=args.image_json_key,
    )

    xs = xs * args.position_scale
    ys = ys * args.position_scale

    th = np.asarray(thetas, dtype=np.float64)
    if args.theta_degrees:
        theta_rad = np.deg2rad(th + args.theta_offset_deg)
    else:
        theta_rad = th + np.deg2rad(args.theta_offset_deg)

    span_x = float(np.ptp(xs)) if xs.size else 1.0
    span_y = float(np.ptp(ys)) if ys.size else 1.0
    span = max(span_x, span_y, 1e-9)
    pick_r = args.pick_radius if args.pick_radius is not None else 0.02 * span
    arrow_len = args.arrow_scale * span

    u = np.cos(theta_rad) * arrow_len
    v = np.sin(theta_rad) * arrow_len

    plot_ys = -ys if args.flip_y else ys
    plot_v = -v if args.flip_y else v

    fig, (ax_map, ax_img) = plt.subplots(
        1,
        2,
        figsize=(14, 7),
        gridspec_kw={"width_ratios": [1.15, 1.0]},
    )
    fig.suptitle(args.title)

    ax_map.scatter(xs, plot_ys, s=36, c=np.arange(len(xs)), cmap="tab20")
    ax_map.quiver(xs, plot_ys, u, plot_v, angles="xy", scale_units="xy", scale=1.0, width=0.004, color="0.2")
    ax_map.set_aspect("equal", adjustable="box")
    if abs(args.position_scale - 0.001) < 1e-12:
        ax_map.set_xlabel("x (m)")
        ax_map.set_ylabel(
            "y (m" + (", flipped)" if args.flip_y else ")")
        )
    else:
        ax_map.set_xlabel("x" + (f" (×{args.position_scale})" if args.position_scale != 1.0 else ""))
        ax_map.set_ylabel("y" + (" (flipped)" if args.flip_y else ""))
    ax_map.grid(True, linestyle=":", alpha=0.5)
    ax_map.set_title("Positions + heading (theta) · 滚轮缩放 · 双击重置")

    # 初始视野在 tight_layout + draw 之后写入 map_view
    map_view: dict = {}

    ax_img.set_title("Hover preview")
    ax_img.axis("off")
    hint = ax_img.text(
        0.5,
        0.5,
        "将鼠标移到左侧轨迹点附近",
        ha="center",
        va="center",
        transform=ax_img.transAxes,
        fontsize=12,
    )
    state = {"idx": -1}

    def _show_index(i: int) -> None:
        nonlocal hint
        if i < 0:
            return
        if hint is not None:
            hint.remove()
            hint = None
        arr = thumbs[i]
        ax_img.clear()
        ax_img.axis("off")
        ax_img.set_title(image_paths[i].name)
        if arr is not None:
            ax_img.imshow(arr)
        else:
            ax_img.text(
                0.5,
                0.5,
                f"无图或无法加载\n{image_paths[i]}",
                ha="center",
                va="center",
                transform=ax_img.transAxes,
                fontsize=10,
                wrap=True,
            )
        fig.canvas.draw_idle()

    def on_motion(event: Any) -> None:
        if event.inaxes != ax_map or event.xdata is None or event.ydata is None:
            return
        dx = xs - event.xdata
        dy = plot_ys - event.ydata
        d = np.sqrt(dx * dx + dy * dy)
        j = int(np.argmin(d))
        if float(d[j]) <= pick_r:
            if state["idx"] != j:
                state["idx"] = j
                _show_index(j)
        else:
            if state["idx"] != -1:
                state["idx"] = -1
                ax_img.clear()
                ax_img.axis("off")
                ax_img.set_title("Hover preview")
                ax_img.text(
                    0.5,
                    0.5,
                    "将鼠标移到左侧轨迹点附近",
                    ha="center",
                    va="center",
                    transform=ax_img.transAxes,
                    fontsize=12,
                )
                fig.canvas.draw_idle()

    def on_scroll(event: Any) -> None:
        if event.inaxes != ax_map or event.xdata is None or event.ydata is None:
            return
        if not map_view:
            return
        step = float(getattr(event, "step", 0) or 0.0)
        if step == 0.0:
            return
        # 向上滚：放大（视野范围变小）
        base = 1.12
        scale = 1.0 / base if step > 0 else base

        x0, x1 = ax_map.get_xlim()
        y0, y1 = ax_map.get_ylim()
        cx, cy = float(event.xdata), float(event.ydata)
        w = (x1 - x0) * scale
        h = (y1 - y0) * scale
        if w < map_view["min_zoom_w"] or h < map_view["min_zoom_h"]:
            return
        if w > map_view["max_zoom_w"] or h > map_view["max_zoom_h"]:
            return

        rx = (cx - x0) / (x1 - x0) if abs(x1 - x0) > 1e-15 else 0.5
        ry = (cy - y0) / (y1 - y0) if abs(y1 - y0) > 1e-15 else 0.5
        nx0 = cx - rx * w
        nx1 = cx + (1.0 - rx) * w
        ny0 = cy - ry * h
        ny1 = cy + (1.0 - ry) * h
        ax_map.set_xlim(nx0, nx1)
        ax_map.set_ylim(ny0, ny1)
        ax_map.set_aspect("equal", adjustable="box")
        fig.canvas.draw_idle()

    def on_button(event: Any) -> None:
        if not getattr(event, "dblclick", False) or event.inaxes != ax_map or not map_view:
            return
        ax_map.set_xlim(*map_view["full_xlim"])
        ax_map.set_ylim(*map_view["full_ylim"])
        ax_map.set_aspect("equal", adjustable="box")
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", on_motion)
    fig.canvas.mpl_connect("scroll_event", on_scroll)
    fig.canvas.mpl_connect("button_press_event", on_button)
    plt.tight_layout()
    fig.canvas.draw()
    fx0, fx1 = ax_map.get_xlim()
    fy0, fy1 = ax_map.get_ylim()
    fw = max(fx1 - fx0, 1e-12)
    fh = max(fy1 - fy0, 1e-12)
    map_view["full_xlim"] = (fx0, fx1)
    map_view["full_ylim"] = (fy0, fy1)
    map_view["min_zoom_w"] = fw * 1e-4
    map_view["min_zoom_h"] = fh * 1e-4
    map_view["max_zoom_w"] = fw * 50.0
    map_view["max_zoom_h"] = fh * 50.0
    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
