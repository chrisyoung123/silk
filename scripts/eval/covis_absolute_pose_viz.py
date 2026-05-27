#!/usr/bin/env python3
"""共视绝对位姿：匹配对与 Local BA 的 HTML 可视化。"""

from __future__ import annotations

import base64
import html
import importlib.util
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
_VIZ_PAIRS = _REPO_ROOT / "scripts/eval/viz_pairs_matches.py"


def _load_viz_pairs():
    spec = importlib.util.spec_from_file_location("viz_pairs_matches", _VIZ_PAIRS)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 {_VIZ_PAIRS}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _norm_image_key(path) -> str:
    return str(Path(path).expanduser().resolve())


def _fmt_num(v: Any, ndigits: int = 3, empty: str = "—") -> str:
    """格式化数值；None / nan / inf 显示为 empty。"""
    if v is None or v == "":
        return empty
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if not np.isfinite(f):
        return empty
    return f"{f:.{ndigits}f}"


def _match_epi_summary(meta: Optional[Dict[str, Any]]) -> str:
    """极线过滤后的匹配摘要。"""
    if not meta:
        return "—"
    pre = meta.get("n_matches_pre_epi", meta.get("n_raw", ""))
    epi = meta.get("n_epipolar_inliers", "")
    used = meta.get("n_used", "")
    return f"raw→epi {pre}→{epi}→used {used}"


def _track_funnel_row(rd: Dict[str, Any]) -> str:
    """单条参考对的轨迹漏斗行。"""
    ts = rd.get("track_stats") or {}
    used = rd.get("used", False)
    reason = rd.get("reason", "")
    status = "✓" if used else html.escape(str(reason or "—"))
    mode = html.escape(str(rd.get("track_mode", "")))
    return (
        "<tr>"
        f"<td>{html.escape(Path(rd.get('ref0', '')).name)}</td>"
        f"<td>{html.escape(Path(rd.get('ref1', '')).name)}</td>"
        f"<td>{_fmt_num(rd.get('baseline_m'), 2)}</td>"
        f"<td>{status}</td>"
        f"<td>{mode}</td>"
        f"<td>{_match_epi_summary(rd.get('match_c12'))}</td>"
        f"<td>{_match_epi_summary(rd.get('match_c13'))}</td>"
        f"<td>{_match_epi_summary(rd.get('match_c23'))}</td>"
        f"<td>{ts.get('n_triangulated', '—')}</td>"
        f"<td>{ts.get('n_depth_ok', '—')}</td>"
        f"<td>{ts.get('n_c13_assoc', '—')}</td>"
        f"<td>{_fmt_num(ts.get('mean_c13_assoc_dist_px'))}</td>"
        f"<td>{ts.get('n_c23_assoc', '—')}</td>"
        f"<td>{_fmt_num(ts.get('mean_c23_assoc_dist_px'))}</td>"
        f"<td>{ts.get('n_c23_agree', '—')}</td>"
        f"<td>{_fmt_num(ts.get('mean_c23_agree_dist_px'))}</td>"
        f"<td>{ts.get('n_tracks', rd.get('n_tracks', '—'))}</td>"
        "</tr>"
    )


def track_cross_uvs_for_pair(
    tracks: Sequence[Any],
    key0,
    key1,
) -> Tuple[np.ndarray, np.ndarray]:
    """提取在两幅图上均有观测的 3D track 像素坐标（用于打叉标记）。"""
    k0, k1 = _norm_image_key(key0), _norm_image_key(key1)
    u0: List[np.ndarray] = []
    u1: List[np.ndarray] = []
    for tr in tracks:
        obs = getattr(tr, "observations", None) or {}
        if k0 not in obs or k1 not in obs:
            continue
        u0.append(np.asarray(obs[k0], dtype=np.float64).reshape(2))
        u1.append(np.asarray(obs[k1], dtype=np.float64).reshape(2))
    if not u0:
        return np.zeros((0, 2), dtype=np.float64), np.zeros((0, 2), dtype=np.float64)
    return np.stack(u0, axis=0), np.stack(u1, axis=0)


def _draw_cross_bgr(
    canvas: np.ndarray,
    x: int,
    y: int,
    *,
    size: int = 7,
    color: Tuple[int, int, int] = (0, 0, 255),
    thickness: int = 2,
) -> None:
    """在 canvas 上画叉号（BGR）。"""
    h, w = canvas.shape[:2]
    if x < 0 or y < 0 or x >= w or y >= h:
        return
    cv2.line(canvas, (x - size, y - size), (x + size, y + size), color, thickness, cv2.LINE_AA)
    cv2.line(canvas, (x - size, y + size), (x + size, y - size), color, thickness, cv2.LINE_AA)


def _overlay_track_crosses(
    panel: np.ndarray,
    cross_m0: np.ndarray,
    cross_m1: np.ndarray,
    w0: int,
    gap: int,
    *,
    scale: float = 1.0,
) -> np.ndarray:
    """在匹配拼图上叠加 3D track 叉号：左图 cross_m0，右图 cross_m1（原图像素坐标）。"""
    out = panel.copy()
    c0 = np.asarray(cross_m0, dtype=np.float64).reshape(-1, 2) * scale
    c1 = np.asarray(cross_m1, dtype=np.float64).reshape(-1, 2) * scale
    x_off = w0 + gap
    # 洋红叉号，与绿色匹配线区分
    col = (255, 0, 255)
    for i in range(c0.shape[0]):
        _draw_cross_bgr(out, int(round(c0[i, 0])), int(round(c0[i, 1])), color=col, size=8, thickness=2)
    for i in range(c1.shape[0]):
        _draw_cross_bgr(
            out,
            int(round(c1[i, 0] + x_off)),
            int(round(c1[i, 1])),
            color=col,
            size=8,
            thickness=2,
        )
    return out


def render_match_panel_jpeg(
    img0: Path,
    img1: Path,
    m0: np.ndarray,
    m1: np.ndarray,
    *,
    max_edge: int = 900,
    max_draw: int = 200,
    gap: int = 12,
    caption: str = "",
    cross_m0: Optional[np.ndarray] = None,
    cross_m1: Optional[np.ndarray] = None,
) -> Optional[bytes]:
    """左右拼图 + 匹配连线；可选叉号标记 3D track 观测点。"""
    vz = _load_viz_pairs()
    im0 = cv2.imread(str(img0), cv2.IMREAD_COLOR)
    im1 = cv2.imread(str(img1), cv2.IMREAD_COLOR)
    if im0 is None or im1 is None:
        return None
    m0v = np.asarray(m0, dtype=np.float64).reshape(-1, 2)
    m1v = np.asarray(m1, dtype=np.float64).reshape(-1, 2)
    if m0v.shape[0] == 0:
        return None
    h0, w0_orig = im0.shape[:2]
    h1, w1_orig = im1.shape[:2]
    long_max = max(h0, w0_orig, h1, w1_orig)
    scale = min(1.0, float(max_edge) / max(long_max, 1))
    im0s, im1s, m0s, m1s = vz._resize_pair_and_matches(im0, im1, m0v, m1v, max_edge)
    w0s = im0s.shape[1]
    panel = vz._draw_pair_panel(
        im0s,
        im1s,
        m0s,
        m1s,
        gap=gap,
        max_draw=max_draw,
        mask=None,
        seed=0,
        uniform_bgr=(0, 220, 100),
    )
    n_cross = 0
    if cross_m0 is not None and cross_m1 is not None:
        c0 = np.asarray(cross_m0, dtype=np.float64).reshape(-1, 2)
        c1 = np.asarray(cross_m1, dtype=np.float64).reshape(-1, 2)
        if c0.shape[0] > 0 and c0.shape == c1.shape:
            n_cross = int(c0.shape[0])
            panel = _overlay_track_crosses(panel, c0, c1, w0s, gap, scale=scale)
    cap_lines = [caption] if caption else []
    if n_cross > 0:
        cap_lines.append(f"洋红叉=3D轨迹点 n={n_cross}")
    if cap_lines:
        panel = vz._caption(panel, cap_lines)
    ok, buf = cv2.imencode(".jpg", panel, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    return buf.tobytes() if ok else None


def _b64_img(jpeg_bytes: Optional[bytes]) -> str:
    if not jpeg_bytes:
        return ""
    return base64.b64encode(jpeg_bytes).decode("ascii")


def _body_pose_on_map(body: Any) -> Optional[Dict[str, float]]:
    """2D 地图：车体 (x, y, θ)，箭头为 (cos θ, sin θ)。"""
    if body is None:
        return None
    if isinstance(body, dict):
        if body.get("x") is None or body.get("y") is None:
            return None
        x, y, th = float(body["x"]), float(body["y"]), float(body.get("theta", 0.0))
    else:
        x, y, th = float(body.x), float(body.y), float(body.theta)
    c, s = math.cos(th), math.sin(th)
    return {"x": x, "y": y, "z": 0.0, "fx": c, "fy": s, "theta": th}


def _body_pose_from_T_world_cam(
    T_world_cam: np.ndarray,
    Tvc: np.ndarray,
) -> Optional[Dict[str, float]]:
    """T_world_cam + Tvc → 平面 body 位姿（与解算链路一致）。"""
    from lib.geometry.local_ba_planar import planar_body_from_T_world_cam

    try:
        b = planar_body_from_T_world_cam(T_world_cam, Tvc)
    except (ValueError, np.linalg.LinAlgError):
        return None
    return _body_pose_on_map(b)


def _resolve_body_pose_for_map(
    body: Any,
    T_world_cam: Optional[np.ndarray],
    Tvc: Optional[np.ndarray],
) -> Optional[Dict[str, float]]:
    pose = _body_pose_on_map(body)
    if pose is not None:
        return pose
    if T_world_cam is not None and Tvc is not None:
        return _body_pose_from_T_world_cam(T_world_cam, Tvc)
    return None


def _resolve_detail_asset_rel(
    rel: str,
    stem: str,
    *,
    rel_viz_prefix: str = "",
) -> str:
    """
    详情页位于 viz/{stem}_detail.html，资源位于 viz/{stem}/xxx.jpg 或 viz/{stem}/map_xy.svg。
    返回相对详情页 HTML 的正确路径。
    """
    raw = str(rel).replace("\\", "/")
    if not raw:
        return ""
    p = Path(raw)
    if p.is_absolute():
        return f"{stem}/{p.name}"
    if "/" in raw:
        return raw
    if rel_viz_prefix:
        return f"{rel_viz_prefix}/{stem}/{p.name}"
    return f"{stem}/{p.name}"


def build_sequence_cam_positions(
    body_cache: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """序列全部帧的**车体** XY（与 pose_map_hover 一致），供地图背景展示。"""
    out: List[Dict[str, Any]] = []
    for key in sorted(body_cache.keys()):
        pose = _body_pose_on_map(body_cache.get(key))
        if pose is None:
            continue
        out.append(
            {
                "key": key,
                "stem": Path(key).stem,
                "x": pose["x"],
                "y": pose["y"],
            }
        )
    return out


def build_map_viz_data(
    *,
    tracks_X_world: Sequence[Sequence[float]],
    cam_keys: Sequence[str],
    ref_keys: Sequence[str],
    target_key: str,
    body_out: Dict[str, Dict[str, float]],
    body_gt: Optional[Dict[str, Dict[str, float]]] = None,
    T_world_cam_out: Optional[Dict[str, np.ndarray]] = None,
    T_world_cam_gt: Optional[Dict[str, np.ndarray]] = None,
    Tvc: Optional[np.ndarray] = None,
    sequence_cams: Optional[Sequence[Dict[str, Any]]] = None,
    involved_keys: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """构建 2D 地图数据：位姿均为 BA 优化后的 body (x,y,θ)；缺失时由 T_world_cam @ Tvc⁻¹ 转换。"""
    T_world_cam_out = T_world_cam_out or {}
    body_gt = body_gt or {}
    points = []
    for xyz in tracks_X_world:
        if len(xyz) >= 3:
            points.append({"x": float(xyz[0]), "y": float(xyz[1]), "z": float(xyz[2])})

    def _cam_entry(key: str, role: str) -> Dict[str, Any]:
        stem = Path(key).stem
        ent: Dict[str, Any] = {
            "key": key,
            "stem": stem,
            "role": role,
            "body_out": body_out.get(key),
        }
        p_out = _resolve_body_pose_for_map(
            body_out.get(key), T_world_cam_out.get(key), Tvc
        )
        p_gt = _resolve_body_pose_for_map(
            body_gt.get(key),
            T_world_cam_gt.get(key) if T_world_cam_gt else None,
            Tvc,
        )
        if p_out is not None:
            ent["body_pose_out"] = p_out
        if p_gt is not None:
            ent["body_pose_gt"] = p_gt
        return ent

    cameras = []
    for k in cam_keys:
        role = "target" if k == target_key else "ref"
        cameras.append(_cam_entry(k, role))

    involved = set(involved_keys or cam_keys)
    seq_other: List[Dict[str, Any]] = []
    if sequence_cams:
        for sc in sequence_cams:
            key = sc.get("key", "")
            seq_other.append(
                {
                    "stem": sc.get("stem", Path(str(key)).stem),
                    "x": float(sc["x"]),
                    "y": float(sc["y"]),
                    "involved": key in involved,
                }
            )

    return {
        "points_xy": points,
        "cameras": cameras,
        "sequence_cams": seq_other,
        "n_points": len(points),
        "n_sequence_cams": len(seq_other),
    }


def render_map_svg(
    map_data: Dict[str, Any],
    *,
    width: int = 820,
    height: int = 620,
    local_only: bool = False,
) -> str:
    """将 map_viz 数据渲染为 SVG 字符串（XY 俯视，Y 轴向上）。

    local_only=True：仅绘制参与本次解算的相机与 3D 点，不含序列其它帧。
    """
    pts = map_data.get("points_xy") or []
    cams = map_data.get("cameras") or []
    seq_cams = [] if local_only else (map_data.get("sequence_cams") or [])

    xs: List[float] = [p["x"] for p in pts]
    ys: List[float] = [p["y"] for p in pts]
    for sc in seq_cams:
        xs.append(float(sc["x"]))
        ys.append(float(sc["y"]))
    for c in cams:
        for prefix in ("body_pose_out", "body_pose_gt"):
            p = c.get(prefix)
            if p:
                xs.append(float(p["x"]))
                ys.append(float(p["y"]))

    if not xs:
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
            f'<text x="20" y="40" font-size="14">无地图数据</text></svg>'
        )

    pad = 0.5
    raw_xmin, raw_xmax = min(xs) - pad, max(xs) + pad
    raw_ymin, raw_ymax = min(ys) - pad, max(ys) + pad
    cx = (raw_xmin + raw_xmax) / 2.0
    cy = (raw_ymin + raw_ymax) / 2.0
    half = max(raw_xmax - raw_xmin, raw_ymax - raw_ymin) / 2.0
    if half < 1e-3:
        half = 0.5
    xmin, xmax = cx - half, cx + half
    ymin, ymax = cy - half, cy + half

    margin = 48
    plot_w = width - 2 * margin
    plot_h = height - 2 * margin
    world_span = 2.0 * half
    scale = min(plot_w, plot_h) / world_span
    pad_x = margin + (plot_w - world_span * scale) / 2.0
    pad_y = margin + (plot_h - world_span * scale) / 2.0

    def tx(x: float) -> float:
        return pad_x + (x - xmin) * scale

    def ty(y: float) -> float:
        return pad_y + (ymax - y) * scale

    arrow_world = 0.35

    def arrow_line(x: float, y: float, fx: float, fy: float, color: str, dash: str = "") -> str:
        x2 = x + fx * arrow_world
        y2 = y + fy * arrow_world
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        return (
            f'<line x1="{tx(x):.1f}" y1="{ty(y):.1f}" x2="{tx(x2):.1f}" y2="{ty(y2):.1f}" '
            f'stroke="{color}" stroke-width="2"{dash_attr}/>'
        )

    if local_only:
        title = "本次解算局部 · 等比例XY · 圆点+箭头=BA优化body位姿"
    else:
        title = "俯视 XY（等比例）· 紫点=序列车体 · 箭头=BA优化body"
    parts: List[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        f'<rect width="100%" height="100%" fill="#fafafa"/>',
        f'<text x="{margin}" y="22" font-size="13" fill="#333">{title}</text>',
    ]

    # 序列其它相机（全序列地图才显示）
    if not local_only:
        for sc in seq_cams:
            if sc.get("involved"):
                continue
            parts.append(
                f'<circle cx="{tx(sc["x"]):.1f}" cy="{ty(sc["y"]):.1f}" r="3.5" '
                f'fill="#8e44ad" fill-opacity="0.75" stroke="#6c3483" stroke-width="1"/>'
            )

    # 三角化 3D 点 XY 投影（深灰实心小点）
    for p in pts:
        parts.append(
            f'<circle cx="{tx(p["x"]):.1f}" cy="{ty(p["y"]):.1f}" r="2.2" '
            f'fill="#2d2d2d" fill-opacity="0.9"/>'
        )

    for c in cams:
        role = c.get("role", "ref")
        label = html.escape(str(c.get("stem", ""))[:28])
        if role == "target":
            colors = {"init": "#e67e22", "out": "#c0392b", "gt": "#27ae60"}
        else:
            colors = {"init": "#3498db", "out": "#2980b9", "gt": "#16a085"}

        for kind, dash in (("out", ""), ("gt", "2,2")):
            pk = f"body_pose_{kind}"
            p = c.get(pk)
            if not p:
                continue
            col = colors.get(kind, "#333")
            r = 5 if role == "target" else 4
            parts.append(
                f'<circle cx="{tx(p["x"]):.1f}" cy="{ty(p["y"]):.1f}" r="{r}" '
                f'fill="{col}" fill-opacity="0.85" stroke="#fff" stroke-width="1"/>'
            )
            parts.append(arrow_line(p["x"], p["y"], p["fx"], p["fy"], col, dash))

        cy = c.get("body_pose_out")
        if cy:
            parts.append(
                f'<text x="{tx(cy["x"])+6:.1f}" y="{ty(cy["y"])-6:.1f}" font-size="10" fill="#444">{label}</text>'
            )

    # 图例
    leg_y = height - 18
    leg_items: List[str] = [
        f'<circle cx="{margin}" cy="{leg_y}" r="4" fill="#2d2d2d"/>',
        f'<text x="{margin+10}" y="{leg_y+4}" font-size="11">3D点</text>',
    ]
    leg_x = margin + 60
    if not local_only:
        leg_items.extend(
            [
                f'<circle cx="{leg_x}" cy="{leg_y}" r="4" fill="#8e44ad" stroke="#6c3483"/>',
                f'<text x="{leg_x+10}" y="{leg_y+4}" font-size="11">序列其它帧</text>',
            ]
        )
        leg_x += 95
    leg_items.extend(
        [
            f'<circle cx="{leg_x}" cy="{leg_y}" r="4" fill="#2980b9"/>',
            f'<text x="{leg_x+10}" y="{leg_y+4}" font-size="11">参考</text>',
            f'<circle cx="{leg_x+70}" cy="{leg_y}" r="4" fill="#c0392b"/>',
            f'<text x="{leg_x+80}" y="{leg_y+4}" font-size="11">目标(估)</text>',
            f'<circle cx="{leg_x+145}" cy="{leg_y}" r="4" fill="#27ae60"/>',
            f'<text x="{leg_x+155}" y="{leg_y+4}" font-size="11">GT(里程计)</text>',
            "</svg>",
        ]
    )
    parts.extend(leg_items)
    return "".join(parts)


def write_map_viz_file(
    out_dir: Path,
    target_stem: str,
    map_data: Dict[str, Any],
    *,
    fname: str = "map_xy.svg",
    local_only: bool = False,
) -> str:
    """写入 SVG 并返回相对 viz 目录的文件名。"""
    out_dir = out_dir / target_stem
    out_dir.mkdir(parents=True, exist_ok=True)
    svg = render_map_svg(map_data, local_only=local_only)
    (out_dir / fname).write_text(svg, encoding="utf-8")
    return f"{target_stem}/{fname}"


def write_match_viz_files(
    out_dir: Path,
    target_stem: str,
    pair_panels: Sequence[Any],
) -> List[Dict[str, Any]]:
    """
    写入匹配可视化 JPEG，返回元数据列表。

    pair_panels 每项为 (tag, p0, p1, m0, m1, caption) 或
    (tag, p0, p1, m0, m1, caption, cross_m0, cross_m1)。
    """
    out_dir = out_dir / target_stem
    out_dir.mkdir(parents=True, exist_ok=True)
    meta: List[Dict[str, Any]] = []
    for item in pair_panels:
        tag, p0, p1, m0, m1, caption = item[:6]
        cross_m0 = item[6] if len(item) > 6 else None
        cross_m1 = item[7] if len(item) > 7 else None
        jpeg = render_match_panel_jpeg(
            p0, p1, m0, m1, caption=caption, cross_m0=cross_m0, cross_m1=cross_m1
        )
        if jpeg is None:
            continue
        fname = f"{tag}.jpg"
        (out_dir / fname).write_bytes(jpeg)
        n_cross = int(cross_m0.shape[0]) if cross_m0 is not None else 0
        meta.append(
            {
                "tag": tag,
                "path": str(out_dir / fname),
                "caption": caption,
                "n_matches": int(m0.shape[0]),
                "n_track_crosses": n_cross,
                "img0": p0.name,
                "img1": p1.name,
            }
        )
    return meta


def write_image_detail_html(
    path: Path,
    rec: Dict[str, Any],
    *,
    rel_viz_prefix: str = "viz",
) -> None:
    """单张目标图详情页：匹配对 + Local BA。"""
    target_name = Path(rec["image"]).name
    stem = Path(rec["image"]).stem
    pair_meta = rec.get("viz_pairs") or []
    ba = rec.get("ba_detail") or {}

    pair_sections: List[str] = []
    for pm in pair_meta:
        rel = pm.get("rel_path") or pm.get("path", "")
        rel_name = _resolve_detail_asset_rel(str(rel), stem, rel_viz_prefix=rel_viz_prefix)
        if not rel_name:
            continue
        pair_sections.append(
            f"<section><h3>{html.escape(pm.get('tag', ''))}</h3>"
            f"<p>{html.escape(pm.get('caption', ''))} · n={pm.get('n_matches', 0)}"
            f"{(' · 3D叉号 n=' + str(pm.get('n_track_crosses'))) if pm.get('n_track_crosses') else ''}</p>"
            f'<img src="{html.escape(rel_name)}" style="max-width:100%"/></section>'
        )

    ref_rows = ""
    funnel_rows = ""
    for rd in rec.get("ref_pair_details") or []:
        funnel_rows += _track_funnel_row(rd)
        if rd.get("used"):
            ref_rows += (
                f"<tr><td>{html.escape(Path(rd.get('ref0', '')).name)}</td>"
                f"<td>{html.escape(Path(rd.get('ref1', '')).name)}</td>"
                f"<td>{_fmt_num(rd.get('baseline_m'), 2)}</td>"
                f"<td>{rd.get('n_tracks', '')}</td>"
                f"<td>{html.escape(str(rd.get('track_mode', '')))}</td></tr>"
            )

    ba_rows = ""
    per_cam = ba.get("per_camera_reproj_rmse_px") or {}
    for cam, rmse in sorted(per_cam.items(), key=lambda x: x[1] if np.isfinite(x[1]) else 1e9):
        ba_rows += f"<tr><td>{html.escape(str(cam))}</td><td>{_fmt_num(rmse)}</td></tr>"

    body_init = ba.get("body_init") or {}
    body_out = ba.get("body_out") or ba.get("body") or {}
    ref_init = ba.get("ref_body_init") or {}
    ref_out = ba.get("ref_body_out") or ba.get("ref_body") or {}
    body_rows = ""
    for k in sorted(set(body_init) | set(body_out)):
        bi = body_init.get(k, {})
        bo = body_out.get(k, {})
        ri = ref_init.get(k, {})
        ro = ref_out.get(k, {})
        extra = ""
        if ro:
            extra = (
                f"<br/><small>6DOF: dz={_fmt_num(ro.get('dz'), 4)} roll={_fmt_num(ro.get('roll_deg'), 2)}° "
                f"pitch={_fmt_num(ro.get('pitch_deg'), 2)}°</small>"
            )
        body_rows += (
            f"<tr><td>{html.escape(Path(str(k)).name)}</td>"
            f"<td>({bi.get('x', '')}, {bi.get('y', '')}, {bi.get('theta', '')})</td>"
            f"<td>({bo.get('x', '')}, {bo.get('y', '')}, {bo.get('theta', '')}){extra}</td></tr>"
        )

    ref6_rows = ""
    for k in sorted(ref_out.keys()):
        ri = ref_init.get(k, {})
        ro = ref_out.get(k, {})
        ref6_rows += (
            f"<tr><td>{html.escape(Path(str(k)).name)}</td>"
            f"<td>{ri.get('dz', 0)}</td><td>{ro.get('dz', '')}</td>"
            f"<td>{ri.get('roll_deg', 0)}</td><td>{ro.get('roll_deg', '')}</td>"
            f"<td>{ri.get('pitch_deg', 0)}</td><td>{ro.get('pitch_deg', '')}</td></tr>"
        )

    outlier = ba.get("outlier_filter") or {}
    outlier_section = ""
    if outlier.get("enabled"):
        pass_rows = ""
        for ps in outlier.get("passes") or []:
            pass_rows += (
                f"<tr><td>{ps.get('pass', '')}</td>"
                f"<td>{ps.get('n_tracks_in', '')}</td>"
                f"<td>{ps.get('n_tracks_out', '')}</td>"
                f"<td>{ps.get('n_tracks_removed', '')}</td>"
                f"<td>{ps.get('n_obs_removed', '')}</td>"
                f"<td>{html.escape(str(ps.get('aborted', '')))}</td></tr>"
            )
        suspicious = outlier.get("target_suspicious")
        flag = (
            ' <span style="color:#c00">目标重投影仍异常</span>'
            if suspicious
            else ""
        )
        outlier_section = (
            f"<h3>外点过滤</h3>"
            f"<p>模式={html.escape(str(outlier.get('mode', '')))} · "
            f"阈值={outlier.get('max_reproj_px', '')} px · "
            f"轮数={outlier.get('n_filter_passes', 0)}/{outlier.get('max_passes', '')} · "
            f"最终 track 数={outlier.get('n_tracks_final', '')}{flag}</p>"
            f"<table><tr><th>轮</th><th>track in</th><th>track out</th>"
            f"<th>track 剔除</th><th>obs 剔除</th><th>中止</th></tr>{pass_rows}</table>"
        )

    map_rel = rec.get("map_viz_rel") or ""
    map_local_rel = rec.get("map_viz_local_rel") or ""
    map_local_section = ""
    if map_local_rel:
        map_local_src = _resolve_detail_asset_rel(map_local_rel, stem, rel_viz_prefix=rel_viz_prefix)
        n_cam = rec.get("n_covis_used", 0)
        n_pt = (rec.get("map_viz_local") or rec.get("map_viz") or {}).get("n_points", rec.get("n_tracks", ""))
        map_local_section = (
            f'<h2>2D 局部地图（仅本次解算）</h2>'
            f'<p>只显示参与本次解算的 <b>{n_cam}</b> 帧、'
            f'<b>{n_pt}</b> 个三角化 3D 点（XY 等比例，单位：米）。</p>'
            f'<p><b>图例：</b>'
            f'深灰小点=3D 点；'
            f'圆点+箭头=BA 优化后的 body 位姿 (x,y,θ)（蓝=参考，红=目标，绿=里程计 GT）。</p>'
            f'<object data="{html.escape(map_local_src)}" type="image/svg+xml" '
            f'style="max-width:100%;border:1px solid #ddd;background:#fafafa"></object>'
        )
    map_section = ""
    if map_rel:
        map_src = _resolve_detail_asset_rel(map_rel, stem, rel_viz_prefix=rel_viz_prefix)
        map_section = (
            f'<h2>2D 序列地图（含全序列上下文）</h2>'
            f'<p><b>图例：</b>'
            f'深灰小点=三角化 3D 点；'
            f'紫色圆点=序列其它帧 body 位置；'
            f'彩色圆点+箭头=参与解算帧的 BA 优化 body 位姿。</p>'
            f'<object data="{html.escape(map_src)}" type="image/svg+xml" '
            f'style="max-width:100%;border:1px solid #ddd;background:#fafafa"></object>'
        )

    doc = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8"/>
<title>{html.escape(target_name)} — 匹配与 BA</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 20px; max-width: 1200px; }}
table {{ border-collapse: collapse; font-size: 13px; margin: 8px 0; }}
td, th {{ border: 1px solid #ccc; padding: 4px 8px; }}
section {{ margin: 24px 0; }}
h2 {{ border-bottom: 1px solid #ddd; padding-bottom: 4px; }}
</style></head><body>
<h1>{html.escape(target_name)}</h1>
<ul>
<li>状态: {html.escape(str(rec.get('status', '')))}</li>
<li>共视邻帧: {rec.get('n_covis_neighbors', 0)} · 参与 BA 参考帧: {rec.get('n_covis_used', 0)}</li>
<li>轨迹数: {rec.get('n_tracks', 0)} · PnP 内点: {rec.get('n_pnp_inliers', 0)}</li>
<li>BA: {'是' if rec.get('ba_applied') else '否'} · 重投影 RMSE: {_fmt_num(rec.get('ba_vis_reproj_rmse_px'))} px</li>
<li>旋转误差: {_fmt_num(rec.get('rotation_error_deg'), 2)}° · XY误差: {_fmt_num(rec.get('position_xy_error_m'), 3)} m</li>
</ul>

<h2>参考对与轨迹漏斗</h2>
<p>各候选参考对的匹配→极线→三角化→关联→交叉验证漏斗；均值列为实际像素偏差（非阈值）。</p>
<table>
<tr><th>ref0</th><th>ref1</th><th>基线 m</th><th>状态</th><th>模式</th>
<th>C1–C2 匹配</th><th>C1–T 匹配</th><th>C2–T 匹配</th>
<th>三角化</th><th>深度 OK</th><th>C13 关联</th><th>C13 均值 px</th>
<th>C23 关联</th><th>C23 均值 px</th><th>C23 一致</th><th>一致均值 px</th><th>轨迹</th></tr>
{funnel_rows or '<tr><td colspan="17">无</td></tr>'}
</table>

<h2>选用的参考对</h2>
<table><tr><th>ref0</th><th>ref1</th><th>基线 m</th><th>轨迹</th><th>模式</th></tr>{ref_rows}</table>

<h2>匹配可视化（按质量优先）</h2>
<p>绿线=特征匹配；<b>洋红叉号</b>=最终参与三角化/BA 的 3D 轨迹点在该图上的观测。</p>
{"".join(pair_sections) if pair_sections else "<p>无匹配可视化</p>"}

<h2>Local BA</h2>
<table><tr><th>相机</th><th>重投影 RMSE (px)</th></tr>{ba_rows}</table>
<p>总 cost={_fmt_num(ba.get('cost'), 4)} · prior RMSE={_fmt_num(ba.get('prior_rmse'))} · iterations={ba.get('n_iterations', '')} · ref 6DOF={ba.get('ref_extra_dofs', '')} · target 6DOF={ba.get('target_extra_dofs', '')}</p>
{outlier_section}
<table><tr><th>车体</th><th>初始 (x,y,θ)</th><th>优化后</th></tr>{body_rows}</table>
{f'<h3>6DOF 姿态扰动 (dz / roll / pitch)</h3><table><tr><th>帧</th><th>dz init</th><th>dz out</th><th>roll° init</th><th>roll° out</th><th>pitch° init</th><th>pitch° out</th></tr>{ref6_rows}</table>' if ref6_rows else ''}
{map_local_section}
{map_section}
</body></html>"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(doc, encoding="utf-8")


def write_sequence_report_html(
    path: Path,
    seq_id: str,
    summary: Dict[str, Any],
    records: List[Dict[str, Any]],
    covis_path: Path,
) -> None:
    """序列级报告：按匹配质量（轨迹数）降序，链到详情页。"""
    ok_recs = [r for r in records if r.get("ok")]
    ok_recs.sort(key=lambda r: (r.get("n_tracks", 0), r.get("n_pnp_inliers", 0)), reverse=True)

    rows = []
    for rec in ok_recs:
        name = Path(rec["image"]).name
        stem = Path(rec["image"]).stem
        detail = f"viz/{stem}_detail.html"
        rows.append(
            "<tr>"
            f'<td><a href="{html.escape(detail)}">{html.escape(name)}</a></td>'
            f"<td>{rec.get('n_covis_neighbors', 0)}</td>"
            f"<td>{rec.get('n_covis_used', 0)}</td>"
            f"<td>{rec.get('n_tracks', 0)}</td>"
            f"<td>{rec.get('n_pnp_inliers', 0)}</td>"
            f"<td>{'Y' if rec.get('ba_applied') else 'N'}</td>"
            f"<td>{rec.get('ba_vis_reproj_rmse_px', '')}</td>"
            f"<td>{rec.get('rotation_error_deg', '')}</td>"
            f"<td>{rec.get('position_xy_error_m', '')}</td>"
            "</tr>"
        )

    for rec in records:
        if rec.get("ok"):
            continue
        name = Path(rec["image"]).name
        rows.append(
            "<tr>"
            f"<td>{html.escape(name)}</td>"
            f"<td>{rec.get('n_covis_neighbors', 0)}</td>"
            f"<td>{rec.get('n_covis_used', 0)}</td>"
            f"<td>{rec.get('n_tracks', 0)}</td>"
            f"<td>{rec.get('n_pnp_inliers', 0)}</td>"
            f"<td>-</td><td>-</td><td>-</td>"
            f"<td>{html.escape(str(rec.get('reason', rec.get('status', ''))))}</td>"
            "</tr>"
        )

    doc = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8"/>
<title>共视绝对位姿 — {html.escape(seq_id)}</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 20px; }}
table {{ border-collapse: collapse; font-size: 13px; }}
td, th {{ border: 1px solid #ccc; padding: 4px 8px; }}
tr:nth-child(even) {{ background: #f9f9f9; }}
</style></head><body>
<h1>序列 {html.escape(seq_id)} — 多相机 Local BA</h1>
<p>共视图: <code>{html.escape(str(covis_path))}</code></p>
<ul>
<li>总图像: {summary.get('n_images_total')}</li>
<li>跳过（共视&lt;{summary.get('min_covis_neighbors')}）: {summary.get('n_skipped_low_covis')}</li>
<li>求解成功: {summary.get('n_pose_ok')}</li>
<li>求解失败: {summary.get('n_pose_failed')}</li>
<li>平均旋转误差(°): {summary.get('mean_rotation_error_deg')}</li>
<li>平均平面位置误差(m): {summary.get('mean_position_xy_error_m')}</li>
<li>多相机最大参考帧: {summary.get('max_ref_cameras', '')}</li>
<li>ANMS: {summary.get('use_anms', False)}</li>
</ul>
<p>成功样本按轨迹数降序；点击文件名查看匹配对与 BA 详情。</p>
<table>
<tr><th>图像</th><th>共视</th><th>参与BA</th><th>轨迹</th><th>PnP内点</th><th>BA</th><th>BA RMSE</th><th>旋转°</th><th>XY m / 原因</th></tr>
{"".join(rows)}
</table></body></html>"""
    path.write_text(doc, encoding="utf-8")
