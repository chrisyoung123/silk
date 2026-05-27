#!/usr/bin/env python3
"""
基于平面位姿 (x, y, θ) 的共视筛选：距离门控 + 朝前射线相交 + 航向视差角。

默认与 pose_map_hover.py 相同的 JSON / 路径 / theta 约定。
输出 covis_edges.csv、covis_adj.json、默认 covis_paths.txt；可选 covis_components.txt、PNG、逐对 HTML、几何示意（--viz-sectors）。
"""

from __future__ import annotations

import argparse
import base64
import csv
import html
import io
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import imageio.v2 as imageio
import numpy as np
from skimage.transform import resize

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None


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


def _load_pose_records(
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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Path], List[Path]]:
    if recursive:
        json_files = sorted(root.rglob(json_glob))
    else:
        json_files = sorted(root.glob(json_glob))

    xs: List[float] = []
    ys: List[float] = []
    thetas: List[float] = []
    json_paths: List[Path] = []
    image_paths: List[Path] = []

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

    if not xs:
        raise RuntimeError("未加载到任何有效样本，请检查目录与 JSON 字段路径。")

    return (
        np.asarray(xs, dtype=np.float64),
        np.asarray(ys, dtype=np.float64),
        np.asarray(thetas, dtype=np.float64),
        json_paths,
        image_paths,
    )


def _theta_to_rad(thetas: np.ndarray, theta_degrees: bool, theta_offset_deg: float) -> np.ndarray:
    th = np.asarray(thetas, dtype=np.float64)
    if theta_degrees:
        return np.deg2rad(th + theta_offset_deg)
    return th + np.deg2rad(theta_offset_deg)


def _cross2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """2D 叉积 z 分量；a,b 末维为 2。"""
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _forward_dirs(theta_rad: np.ndarray) -> np.ndarray:
    """朝前单位方向 (N, 2)。"""
    return np.stack([np.cos(theta_rad), np.sin(theta_rad)], axis=1)


def _heading_parallax_rad(theta_i: float, theta_j: float) -> float:
    """两帧朝前射线方向夹角（视差角），范围 [0, π]。"""
    d = abs(float(theta_i) - float(theta_j))
    d = min(d, 2.0 * np.pi - d)
    return d


def _rays_intersect_forward(
    pi: np.ndarray,
    di: np.ndarray,
    pj: np.ndarray,
    dj: np.ndarray,
    eps: float = 1e-9,
) -> Tuple[bool, float, float]:
    """
    射线 pi + t*di、pj + s*dj（t,s≥0 为朝前）是否相交。
    返回 (ok, t, s)；平行不相交、共线但交点在后方则 ok=False。
    """
    v = pj - pi
    denom = float(_cross2(di, dj))
    if abs(denom) > eps:
        t = float(_cross2(v, dj) / denom)
        s = float(_cross2(v, di) / denom)
        ok = t > eps and s > eps
        return ok, t, s
    if abs(float(_cross2(v, di))) > eps:
        return False, float("nan"), float("nan")
    t = float(np.dot(v, di))
    s = float(np.dot(-v, dj))
    ok = t > eps and s > eps
    return ok, t, s


def _ray_intersection_point(
    pi: np.ndarray,
    di: np.ndarray,
    pj: np.ndarray,
    dj: np.ndarray,
) -> Optional[np.ndarray]:
    ok, t, _ = _rays_intersect_forward(pi, di, pj, dj)
    if not ok:
        return None
    return pi + t * di


def _pairwise_covis_metrics(
    xy: np.ndarray,
    theta_rad: np.ndarray,
    r_min: float,
    r_max: float,
    ray_eps: float = 1e-9,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    xy: (N, 2), theta_rad: (N,)
    Returns upper-triangle i<j: i_idx, j_idx, dist, parallax_rad, rays_intersect
    """
    n = xy.shape[0]
    diff = xy[None, :, :] - xy[:, None, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.nan)

    dtheta = np.abs(theta_rad[:, None] - theta_rad[None, :])
    parallax = np.minimum(dtheta, 2.0 * np.pi - dtheta)
    np.fill_diagonal(parallax, np.nan)

    d = _forward_dirs(theta_rad)
    d_i = d[:, None, :]
    d_j = d[None, :, :]
    denom = _cross2(d_i, d_j)
    with np.errstate(divide="ignore", invalid="ignore"):
        t_ray = _cross2(diff, d_j) / denom
        s_ray = _cross2(diff, d_i) / denom
    parallel = np.abs(denom) <= ray_eps
    collinear = parallel & (np.abs(_cross2(diff, d_i)) <= ray_eps)
    t_col = np.sum(diff * d_i, axis=-1)
    s_col = np.sum(-diff * d_j, axis=-1)
    rays_ok = (~parallel & (t_ray > ray_eps) & (s_ray > ray_eps)) | (
        collinear & (t_col > ray_eps) & (s_col > ray_eps)
    )
    np.fill_diagonal(rays_ok, False)

    ii, jj = np.triu_indices(n, k=1)
    d_out = dist[ii, jj]
    p = parallax[ii, jj]
    r_ok = rays_ok[ii, jj]

    valid = np.isfinite(d_out) & (d_out >= r_min) & (d_out <= r_max)
    return ii[valid], jj[valid], d_out[valid], p[valid], r_ok[valid]


def _write_csv(
    path: Path,
    rows: List[Tuple[int, int, float, float, int, int]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id_i", "id_j", "dist", "parallax_deg", "rays_intersect", "covis"])
        for i, j, dist, parallax_deg, rays_intersect, covis in rows:
            w.writerow([i, j, f"{dist:.8g}", f"{parallax_deg:.6g}", rays_intersect, covis])


def _plot_covis(
    xy: np.ndarray,
    edges: List[Tuple[int, int]],
    out_path: Path,
    max_edges: int,
    title: str,
) -> None:
    if plt is None:
        print("未安装 matplotlib，跳过绘图。", file=sys.stderr)
        return
    rng = np.random.default_rng(0)
    e = list(edges)
    if len(e) > max_edges:
        idx = rng.choice(len(e), size=max_edges, replace=False)
        e = [e[int(k)] for k in idx]

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(xy[:, 0], xy[:, 1], s=8, c=np.arange(len(xy)), cmap="tab20", zorder=2)
    for a, b in e:
        ax.plot([xy[a, 0], xy[b, 0]], [xy[a, 1], xy[b, 1]], color="0.5", alpha=0.25, linewidth=0.6, zorder=1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _pair_geometry_png_bytes(
    xy: np.ndarray,
    theta_rad: np.ndarray,
    max_parallax_deg: float,
    i: int,
    j: int,
    figsize: Tuple[float, float] = (5.5, 4.5),
    dpi: int = 110,
) -> Optional[bytes]:
    """单对 (i,j) 平面位置、基线、朝前箭头与航向视差角示意。"""
    if plt is None:
        return None

    pi = np.array([xy[i, 0], xy[i, 1]], dtype=np.float64)
    pj = np.array([xy[j, 0], xy[j, 1]], dtype=np.float64)
    xi, yi = float(pi[0]), float(pi[1])
    xj, yj = float(pj[0]), float(pj[1])
    vij = pj - pi
    fi = np.array([np.cos(theta_rad[i]), np.sin(theta_rad[i])], dtype=np.float64)
    fj = np.array([np.cos(theta_rad[j]), np.sin(theta_rad[j])], dtype=np.float64)
    parallax_deg = float(np.rad2deg(_heading_parallax_rad(theta_rad[i], theta_rad[j])))
    rays_ok, t_hit, s_hit = _rays_intersect_forward(pi, fi, pj, fj)
    parallax_ok = parallax_deg <= max_parallax_deg + 1e-6
    covis = rays_ok and parallax_ok
    hit_pt = _ray_intersection_point(pi, fi, pj, fj) if rays_ok else None

    dist = float(np.linalg.norm(vij))
    span = max(float(np.ptp(xy[:, 0])), float(np.ptp(xy[:, 1])), 1e-9)

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(xy[:, 0], xy[:, 1], s=10, c="0.88", zorder=1, label="poses")
    ax.scatter([xi, xj], [yi, yj], s=90, c=["tab:green", "tab:orange"], zorder=5, edgecolors="k")

    ax.plot([xi, xj], [yi, yj], "k--", lw=1.8, zorder=3, label="baseline")
    L = max(dist * 0.4, span * 0.03)
    ax.quiver(
        [xi],
        [yi],
        [fi[0] * L],
        [fi[1] * L],
        angles="xy",
        scale_units="xy",
        scale=1,
        color="darkgreen",
        width=0.02,
        zorder=6,
        label=f"id={i} fwd",
    )
    ax.quiver(
        [xj],
        [yj],
        [fj[0] * L],
        [fj[1] * L],
        angles="xy",
        scale_units="xy",
        scale=1,
        color="darkorange",
        width=0.02,
        zorder=6,
        label=f"id={j} fwd",
    )
    if hit_pt is not None:
        hx, hy = float(hit_pt[0]), float(hit_pt[1])
        ax.scatter([hx], [hy], s=120, c="crimson", marker="x", zorder=7, label="ray hit")
        ax.plot([xi, hx], [yi, hy], color="darkgreen", lw=1.2, alpha=0.5, linestyle=":")
        ax.plot([xj, hx], [yj, hy], color="darkorange", lw=1.2, alpha=0.5, linestyle=":")

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle=":", alpha=0.45)
    ax.set_xlabel("x (scaled)")
    ax.set_ylabel("y")
    note = (
        f"parallax={parallax_deg:.1f}° (≤{max_parallax_deg:g})  "
        f"rays={int(rays_ok)} (t={t_hit:.3g},s={s_hit:.3g})  covis={int(covis)}  dist={dist:.4g}"
    )
    ax.text(0.02, 0.98, note, transform=ax.transAxes, fontsize=9, va="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.88))
    ax.set_title(f"Plan view: {i} <-> {j}", fontsize=11)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def _component_group_geometry_png_bytes(
    xy: np.ndarray,
    theta_rad: np.ndarray,
    max_parallax_deg: float,
    group: Sequence[int],
    edges: Sequence[Tuple[int, int]],
    figsize: Tuple[float, float] = (6.0, 5.0),
    dpi: int = 100,
) -> Optional[bytes]:
    """连通组：组内位姿、共视边、朝前箭头。"""
    if plt is None:
        return None

    gset = set(int(x) for x in group)
    glist = sorted(gset)
    if len(glist) < 2:
        return None

    sub = xy[glist]
    span = max(float(np.ptp(sub[:, 0])), float(np.ptp(sub[:, 1])), 1e-9)
    arrow_len = span * 0.12

    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(xy[:, 0], xy[:, 1], s=8, c="0.9", zorder=1, label="other poses")
    for a, b in edges:
        a, b = int(a), int(b)
        if a not in gset or b not in gset:
            continue
        ax.plot([xy[a, 0], xy[b, 0]], [xy[a, 1], xy[b, 1]], "k-", lw=1.0, alpha=0.35, zorder=3)

    for k, nid in enumerate(glist):
        col = np.asarray(cmap(k % 10), dtype=float)
        rgb = (float(col[0]), float(col[1]), float(col[2]))
        cx, cy = float(xy[nid, 0]), float(xy[nid, 1])
        ax.scatter([cx], [cy], s=70, c=[rgb], zorder=5, edgecolors="k")
        f = np.array([np.cos(theta_rad[nid]), np.sin(theta_rad[nid])])
        ax.quiver(
            [cx],
            [cy],
            [f[0] * arrow_len],
            [f[1] * arrow_len],
            angles="xy",
            scale_units="xy",
            scale=1,
            color=rgb,
            width=0.018,
            zorder=6,
        )

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.set_xlabel("x (scaled)")
    ax.set_ylabel("y")
    ax.set_title(
        f"Group plan |max_parallax|={max_parallax_deg:g}deg |nodes|={len(glist)}",
        fontsize=11,
    )
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def _plot_geometry_demo(
    xy: np.ndarray,
    theta_rad: np.ndarray,
    max_parallax_deg: float,
    i: int,
    j: int,
    out_path: Path,
) -> None:
    """写出与 HTML 内嵌相同几何语义的平面示意 PNG（略大尺寸）。"""
    data = _pair_geometry_png_bytes(
        xy, theta_rad, max_parallax_deg, i, j, figsize=(10, 8), dpi=150
    )
    if data is None:
        print("未安装 matplotlib，跳过 --viz-sectors。", file=sys.stderr)
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(data)


def _connected_components(n: int, edges: Sequence[Tuple[int, int]]) -> List[List[int]]:
    """无向边并查集；仅返回大小 >=2 的连通分量（节点为整数 id）。"""
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for a, b in edges:
        union(int(a), int(b))

    buckets: Dict[int, List[int]] = {}
    for i in range(n):
        r = find(i)
        buckets.setdefault(r, []).append(i)
    comps = [sorted(g) for g in buckets.values() if len(g) >= 2]
    comps.sort(key=lambda g: (min(g), len(g)))
    return comps


def _paths_line(image_paths: List[Path], ids: Sequence[int], sep: str) -> str:
    parts = [str(image_paths[i].resolve()) for i in ids]
    return sep.join(parts)


def _write_covis_txt_pairs(
    path: Path,
    image_paths: List[Path],
    pair_metrics: Dict[Tuple[int, int], Tuple[float, float]],
    sep: str,
) -> int:
    """每行恰好一对：path_u sep path_v（键为 id_u<id_v），Unix 换行 \\n。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [_paths_line(image_paths, (u, v), sep) for (u, v) in sorted(pair_metrics.keys())]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return len(lines)


def _write_components_txt(
    path: Path,
    image_paths: List[Path],
    edges: Sequence[Tuple[int, int]],
    sep: str,
) -> int:
    """可选：每个共视连通分量一行（多路径）。"""
    n = len(image_paths)
    comps = _connected_components(n, edges)
    lines = [_paths_line(image_paths, group, sep) for group in comps]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return len(lines)


def _thumbnail_jpeg_b64(path: Path, max_edge: int, jpeg_quality: int) -> Optional[str]:
    if not path.is_file():
        return None
    try:
        arr = imageio.imread(path)
    except OSError:
        return None
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.shape[2] == 4:
        arr = arr[:, :, :3]
    h, w = arr.shape[:2]
    scale = max_edge / max(h, w, 1)
    nh = max(1, int(round(h * scale)))
    nw = max(1, int(round(w * scale)))
    small = (resize(arr, (nh, nw), preserve_range=True, anti_aliasing=True)).clip(0, 255).astype(np.uint8)
    buf = io.BytesIO()
    imageio.imwrite(buf, small, format="jpeg", quality=int(jpeg_quality))
    return base64.standard_b64encode(buf.getvalue()).decode("ascii")


def _write_covis_html_pairs(
    path: Path,
    image_paths: List[Path],
    pair_metrics: Dict[Tuple[int, int], Tuple[float, float]],
    xy: np.ndarray,
    theta_rad: np.ndarray,
    max_parallax_deg: float,
    thumb_max: int,
    jpeg_quality: int,
) -> None:
    """逐对可视化：左右两栏标明 id，附几何量；下方嵌入平面位置示意 PNG。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    blocks: List[str] = []
    title_esc = html.escape("Covisibility pairs (approx)")

    for idx, ((u, v), (dist, parallax_deg)) in enumerate(sorted(pair_metrics.items())):
        b64_u = _thumbnail_jpeg_b64(image_paths[u], thumb_max, jpeg_quality)
        b64_v = _thumbnail_jpeg_b64(image_paths[v], thumb_max, jpeg_quality)
        pu_raw, pv_raw = str(image_paths[u].resolve()), str(image_paths[v].resolve())
        pu, pv = html.escape(pu_raw), html.escape(pv_raw)
        img_u = (
            f'<img src="data:image/jpeg;base64,{b64_u}" alt="{pu}" title="{pu}"/>'
            if b64_u
            else f'<p class="missing">缺失图: {pu}</p>'
        )
        img_v = (
            f'<img src="data:image/jpeg;base64,{b64_v}" alt="{pv}" title="{pv}"/>'
            if b64_v
            else f'<p class="missing">缺失图: {pv}</p>'
        )
        geom_png = _pair_geometry_png_bytes(xy, theta_rad, max_parallax_deg, u, v)
        geom_block = ""
        if geom_png:
            g64 = base64.standard_b64encode(geom_png).decode("ascii")
            geom_block = f'<div class="geomMap"><div class="geomCap">平面相对位置（绿=id{u}，橙=id{v}，虚线=基线，箭头=朝前）</div><img class="geomPng" src="data:image/png;base64,{g64}" alt="plan geometry"/></div>'
        else:
            geom_block = '<p class="missing">几何图不可用（未安装 matplotlib）</p>'

        meta_rows = (
            f"<tr><td>id_u</td><td>{u}</td></tr>"
            f"<tr><td>id_v</td><td>{v}</td></tr>"
            f"<tr><td>dist</td><td>{dist:.6g}</td></tr>"
            f"<tr><td>parallax_deg</td><td>{parallax_deg:.4f}</td></tr>"
            f"<tr><td>max_parallax_deg</td><td>{max_parallax_deg:g}</td></tr>"
            f"<tr><td>rays_intersect</td><td>1（朝前射线相交）</td></tr>"
        )
        blocks.append(
            f"""
<section class="pair">
  <h3>共视对 #{idx} · 帧 <span class="idtag">{u}</span> ↔ 帧 <span class="idtag">{v}</span></h3>
  <table class="meta">{meta_rows}</table>
  {geom_block}
  <div class="pairRow">
    <div class="cell left">
      <div class="banner">第一帧 · id={u}</div>
      {img_u}
      <pre class="path">{pu}</pre>
    </div>
    <div class="mid">⇄<br/><span class="sub">匹配对</span></div>
    <div class="cell right">
      <div class="banner">第二帧 · id={v}</div>
      {img_v}
      <pre class="path">{pv}</pre>
    </div>
  </div>
</section>"""
        )

    body = "\n".join(blocks) if blocks else "<p>无共视边。</p>"
    doc = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<title>{title_esc}</title>
<style>
body {{ font-family: sans-serif; margin: 16px; max-width: 1400px; }}
section.pair {{ border: 1px solid #bbb; border-radius: 8px; margin-bottom: 28px; padding: 12px 16px; background: #fafafa; }}
h3 {{ margin-top: 0; font-size: 17px; }}
.idtag {{ color: #06c; font-weight: bold; }}
table.meta {{ border-collapse: collapse; margin: 8px 0 14px; font-size: 13px; }}
table.meta td {{ border: 1px solid #ddd; padding: 4px 10px; }}
table.meta td:first-child {{ background: #eee; font-weight: 600; }}
.pairRow {{ display: flex; align-items: flex-start; gap: 12px; flex-wrap: wrap; }}
.cell {{ flex: 1; min-width: 200px; border: 2px solid #333; border-radius: 6px; padding: 8px; background: #fff; }}
.cell.left {{ border-color: #1565c0; }}
.cell.right {{ border-color: #c62828; }}
.banner {{ font-weight: bold; margin-bottom: 6px; font-size: 14px; }}
.mid {{ align-self: center; text-align: center; font-size: 28px; color: #555; padding: 0 8px; }}
.mid .sub {{ font-size: 12px; display: block; }}
.cell img {{ max-width: 100%; height: auto; display: block; margin-top: 4px; }}
pre.path {{ font-size: 11px; white-space: pre-wrap; word-break: break-all; margin: 8px 0 0; color: #333; }}
.missing {{ color: #c00; }}
.geomMap {{ margin: 10px 0 16px; padding: 10px; background: #f0f4f8; border: 1px solid #90caf9; border-radius: 6px; }}
.geomCap {{ font-size: 13px; font-weight: 600; margin-bottom: 6px; color: #1565c0; }}
.geomPng {{ max-width: min(720px, 100%); height: auto; display: block; border: 1px solid #444; border-radius: 4px; }}
</style>
</head>
<body>
<h2>{title_esc}</h2>
<div class="note">
  <p><b>共视筛选</b>：平面距离 + 朝前射线相交 + 航向视差角，<b>不等于</b>三维严格共视。</p>
  <ul style="margin:8px 0;padding-left:1.2em;">
    <li><code>dist</code>：两帧在地图平面 (x,y) 上的欧氏距离，单位与 <code>--position-scale</code> 一致。</li>
    <li><code>rays_intersect</code>：从各位姿沿 <code>theta</code> 朝前发出的射线须在正前方相交（平行或交点在后方则否）。</li>
    <li><code>parallax_deg</code>：两朝前射线方向夹角（0°–180°）；另需 <code>parallax_deg ≤ --max-parallax-deg</code>。</li>
    <li><code>--r-max</code> / <code>--r-min</code>：只保留 <code>r_min ≤ dist ≤ r_max</code> 的候选对。</li>
    <li><strong>平面几何示意</strong>：绿/橙为当前对、虚线为基线、箭头为朝前；红叉为射线交点。可加 <code>--viz-sectors</code> 导出示意 PNG。</li>
    <li><strong>全局边拓扑</strong>：加 <code>--plot</code> 得 <code>covis_graph.png</code>（轨迹点 + 半透明共视边）。</li>
  </ul>
</div>
<p>缩略图最长边 ≤ {thumb_max}px；<code>covis_paths.txt</code> 与下表顺序一致（均按 id_u &lt; id_v 排序）。</p>
{body}
</body>
</html>
"""
    path.write_text(doc, encoding="utf-8")


def _write_covis_html_components(
    path: Path,
    image_paths: List[Path],
    edges: Sequence[Tuple[int, int]],
    xy: np.ndarray,
    theta_rad: np.ndarray,
    max_parallax_deg: float,
    thumb_max: int,
    jpeg_quality: int,
) -> None:
    """连通组视图：每节含组内平面拓扑、再跟多帧缩略图。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(image_paths)
    comps = _connected_components(n, edges)
    blocks: List[str] = []
    for gi, group in enumerate(comps):
        geom_png = _component_group_geometry_png_bytes(xy, theta_rad, max_parallax_deg, group, edges)
        geom_block = ""
        if geom_png:
            g64 = base64.standard_b64encode(geom_png).decode("ascii")
            geom_block = f'<div class="geomMap"><div class="geomCap">组内平面位置、共视边（灰线）与朝前箭头</div><img class="geomPng" src="data:image/png;base64,{g64}" alt="group geometry"/></div>'
        else:
            geom_block = '<p class="missing">几何图不可用</p>'
        cells: List[str] = []
        for nid in group:
            p = image_paths[nid]
            b64 = _thumbnail_jpeg_b64(p, thumb_max, jpeg_quality)
            pe = html.escape(str(p.resolve()))
            inner = (
                f'<img src="data:image/jpeg;base64,{b64}" alt="{pe}" title="{pe}"/>'
                if b64
                else f'<p class="missing">缺失 id{nid}</p>'
            )
            cells.append(
                f'<div class="cell"><div class="banner">id {nid}</div>{inner}<pre class="path">{pe}</pre></div>'
            )
        blocks.append(
            f'<section><h3>连通组 #{gi}（{len(group)} 帧）</h3>{geom_block}<div class="pairRow">{"".join(cells)}</div></section>'
        )
    title_esc = html.escape("Covisibility components (approx)")
    body = "\n".join(blocks) if blocks else "<p>无多帧连通组。</p>"
    doc = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head><meta charset="utf-8"/><title>{title_esc}</title>
<style>
body {{ font-family: sans-serif; margin: 16px; }}
section {{ border-bottom: 1px solid #ccc; margin-bottom: 20px; }}
.pairRow {{ display: flex; flex-wrap: wrap; gap: 10px; }}
.cell {{ border: 1px solid #888; padding: 8px; border-radius: 6px; }}
.banner {{ font-weight: bold; font-size: 13px; }}
pre.path {{ font-size: 10px; white-space: pre-wrap; word-break: break-all; }}
.note {{ background: #fff8e1; border-left: 4px solid #ff9800; padding: 10px 14px; margin-bottom: 16px; font-size: 13px; }}
.geomMap {{ margin: 8px 0 12px; padding: 10px; background: #f0f4f8; border: 1px solid #90caf9; border-radius: 6px; }}
.geomCap {{ font-size: 12px; font-weight: 600; margin-bottom: 6px; color: #1565c0; }}
.geomPng {{ max-width: min(720px, 100%); height: auto; display: block; border: 1px solid #444; border-radius: 4px; }}
</style></head><body><h2>{title_esc}</h2>
<div class="note">
  每节顶部为<strong>组内平面几何</strong>（浅灰=全部位姿，箭头=朝前；黑灰线段=组内共视边）。
  逐对 parallax/dist 仍以 <code>covis_images.html</code> / <code>covis_edges.csv</code> 为准。
</div>
{body}</body></html>"""
    path.write_text(doc, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="基于 2D 位姿的共视筛选：距离 + 朝前射线相交 + 航向视差角；导出 CSV/JSON、默认 covis_paths.txt；可选 PNG/HTML。"
    )
    ap.add_argument("--data-dir", type=Path, required=True, help="含 JSON 的根目录")
    ap.add_argument("--recursive", action="store_true", help="递归扫描 JSON")
    ap.add_argument("--json-glob", default="*.json", help="匹配 JSON 的 glob")
    ap.add_argument("--pose-x-key", default="value.pose.x")
    ap.add_argument("--pose-y-key", default="value.pose.y")
    ap.add_argument("--pose-theta-key", default="value.pose.theta")
    ap.add_argument("--theta-degrees", action="store_true", help="theta 为度")
    ap.add_argument("--theta-offset-deg", type=float, default=0.0)
    ap.add_argument(
        "--position-scale",
        type=float,
        default=1.0,
        help="对 x,y 乘以此系数（与 pose_map_hover 一致，如毫米→米 0.001）",
    )
    ap.add_argument("--image-dir", type=Path, default=None)
    ap.add_argument("--image-ext", default=".jpg")
    ap.add_argument("--image-extra", default="")
    ap.add_argument("--image-json-key", default=None)

    ap.add_argument(
        "--max-parallax-deg",
        type=float,
        default=30.0,
        help="最大航向视差角（度）：两朝前射线方向夹角 parallax_deg ≤ 该值（且射线须相交）",
    )
    ap.add_argument(
        "--fov-half-deg",
        type=float,
        default=None,
        help="已弃用，等同于 --max-parallax-deg（若未指定 max-parallax-deg）",
    )
    ap.add_argument(
        "--r-max",
        type=float,
        default=15.0,
        help="最大基线距离（与 position-scale 后坐标同单位，例如米）",
    )
    ap.add_argument(
        "--r-min",
        type=float,
        default=0.1,
        help="最小基线距离，小于此距离的帧对不计入",
    )
    ap.add_argument(
        "--export-candidates",
        action="store_true",
        help="CSV 列出所有满足距离门控的 i<j，并带 parallax_deg、rays_intersect、covis；邻接仍只保留共视对",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="输出目录（默认在 data-dir 下 covis_output）",
    )
    ap.add_argument(
        "--plot",
        action="store_true",
        help="保存轨迹+共视边 PNG 到 out-dir/covis_graph.png",
    )
    ap.add_argument(
        "--plot-png",
        type=Path,
        default=None,
        help="指定 PNG 路径（若设置则优先于 --plot）",
    )
    ap.add_argument("--plot-max-edges", type=int, default=8000, help="绘图时最多绘制的边数（随机抽样）")
    ap.add_argument(
        "--no-covis-txt",
        action="store_true",
        help="不写入 covis_paths.txt（默认会写）",
    )
    ap.add_argument(
        "--write-components-txt",
        action="store_true",
        help="额外写入 covis_components.txt：每行一个连通分量内全部路径（非逐对）",
    )
    ap.add_argument(
        "--components-txt-path",
        type=Path,
        default=None,
        help="连通分量 txt 路径（默认 out-dir/covis_components.txt）",
    )
    ap.add_argument(
        "--covis-txt-path",
        type=Path,
        default=None,
        help="txt 输出路径（默认 out-dir/covis_paths.txt）",
    )
    ap.add_argument(
        "--txt-separator",
        default=" ",
        help="txt 列分隔符（路径含空格时请用 \\t 等）",
    )
    ap.add_argument(
        "--viz-html",
        type=Path,
        default=None,
        help="写出带缩略图的 HTML 报告（若仅想默认路径可配合 --viz-images）",
    )
    ap.add_argument(
        "--viz-images",
        action="store_true",
        help="写出 out-dir/covis_images.html（逐对：左 id_u / 右 id_v + 几何量，与 covis_paths.txt 顺序一致）",
    )
    ap.add_argument(
        "--viz-components-html",
        action="store_true",
        help="额外写出 out-dir/covis_components.html（连通组多图，仅拓扑浏览）",
    )
    ap.add_argument(
        "--viz-sectors",
        action="store_true",
        help="写出平面几何示意 PNG（默认 out-dir/covis_sectors_demo.png）：基线、朝前箭头与视差角",
    )
    ap.add_argument(
        "--viz-sectors-pair",
        type=str,
        default=None,
        help='示意用的帧 id，格式 "i,j"（如 3,7）；默认取共视边列表首条，无边则用 0,1',
    )
    ap.add_argument(
        "--viz-sectors-png",
        type=Path,
        default=None,
        help="几何示意 PNG 路径（覆盖默认 out-dir/covis_sectors_demo.png）",
    )
    ap.add_argument("--viz-thumb-max", type=int, default=320, help="HTML 缩略图最长边（像素）")
    ap.add_argument("--viz-jpeg-quality", type=int, default=82, help="HTML 内嵌 JPEG 质量 1-100")

    args = ap.parse_args()

    root = args.data_dir.expanduser().resolve()
    image_dir = (args.image_dir or root).expanduser().resolve()
    out_dir = (args.out_dir or (root / "covis_output")).expanduser().resolve()

    if not root.is_dir():
        print(f"错误：data-dir 不是目录: {root}", file=sys.stderr)
        return 1

    xs, ys, thetas, json_paths, image_paths = _load_pose_records(
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

    xy = np.stack([xs * args.position_scale, ys * args.position_scale], axis=1)
    theta_rad = _theta_to_rad(thetas, args.theta_degrees, args.theta_offset_deg)

    max_parallax_deg = args.max_parallax_deg
    if args.fov_half_deg is not None:
        if max_parallax_deg != 30.0 and args.fov_half_deg != max_parallax_deg:
            print(
                "警告：同时指定 --max-parallax-deg 与 --fov-half-deg，使用 --max-parallax-deg。",
                file=sys.stderr,
            )
        elif max_parallax_deg == 30.0:
            max_parallax_deg = args.fov_half_deg
            print(
                f"提示：--fov-half-deg 已弃用，本次使用 max_parallax_deg={max_parallax_deg:g}。",
                file=sys.stderr,
            )
    max_parallax_rad = np.deg2rad(max_parallax_deg)

    ii, jj, dist, parallax_rad, rays_mask = _pairwise_covis_metrics(
        xy, theta_rad, args.r_min, args.r_max
    )
    parallax_ok = parallax_rad <= max_parallax_rad + 1e-9
    covis_mask = parallax_ok & rays_mask

    rows: List[Tuple[int, int, float, float, int, int]] = []
    pair_metrics: Dict[Tuple[int, int], Tuple[float, float]] = {}
    n = len(json_paths)

    for k in range(ii.size):
        i, j = int(ii[k]), int(jj[k])
        d = float(dist[k])
        parallax_deg = float(np.rad2deg(parallax_rad[k]))
        rays_intersect = bool(rays_mask[k])
        covis = bool(covis_mask[k])

        if args.export_candidates:
            rows.append((i, j, d, parallax_deg, int(rays_intersect), int(covis)))
        elif covis:
            rows.append((i, j, d, parallax_deg, 1, 1))

        if covis:
            u, v = (min(i, j), max(i, j))
            pair_metrics[(u, v)] = (d, parallax_deg)

    edges_plot = list(sorted(pair_metrics.keys()))

    adj: Dict[str, List[int]] = {str(i): [] for i in range(n)}
    for u, v in edges_plot:
        adj[str(u)].append(v)
        adj[str(v)].append(u)
    for k in adj:
        adj[k] = sorted(set(adj[k]))

    meta = {
        "n_poses": n,
        "max_parallax_deg": max_parallax_deg,
        "r_min": args.r_min,
        "r_max": args.r_max,
        "position_scale": args.position_scale,
        "export_candidates": args.export_candidates,
        "theta_degrees": args.theta_degrees,
        "theta_offset_deg": args.theta_offset_deg,
        "covis_paths_txt": "one_pair_per_line; columns: image_id_u image_id_v (sorted u<v); LF newline",
    }
    nodes = [
        {
            "id": i,
            "json": str(json_paths[i]),
            "image": str(image_paths[i]),
        }
        for i in range(n)
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "covis_edges.csv"
    json_path = out_dir / "covis_adj.json"

    _write_csv(csv_path, rows)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump({"meta": meta, "adjacency": adj, "nodes": nodes}, f, indent=2, ensure_ascii=False)

    print(f"写入 {csv_path}（{len(rows)} 行）")
    print(f"写入 {json_path}")

    if not args.no_covis_txt:
        txt_path = (args.covis_txt_path or (out_dir / "covis_paths.txt")).expanduser().resolve()
        sep = args.txt_separator
        n_lines = _write_covis_txt_pairs(txt_path, image_paths, pair_metrics, sep)
        print(f"写入 {txt_path}（{n_lines} 行；每行一对路径，id_u<id_v，换行符为 LF）")
        bad = [str(p.resolve()) for p in image_paths if sep in str(p.resolve())]
        if bad:
            print(
                f"警告：{len(bad)} 条路径含分隔符 {sep!r}，按空白切分脚本可能误解析；可改用 --txt-separator $'\\t'。",
                file=sys.stderr,
            )

    if args.write_components_txt:
        cpath = (args.components_txt_path or (out_dir / "covis_components.txt")).expanduser().resolve()
        nc = _write_components_txt(cpath, image_paths, edges_plot, args.txt_separator)
        print(f"写入 {cpath}（{nc} 行连通组，可选）")

    html_path: Optional[Path] = None
    if args.viz_html is not None:
        html_path = args.viz_html.expanduser().resolve()
    elif args.viz_images:
        html_path = out_dir / "covis_images.html"

    if html_path is not None:
        _write_covis_html_pairs(
            html_path,
            image_paths,
            pair_metrics,
            xy,
            theta_rad,
            max_parallax_deg,
            thumb_max=args.viz_thumb_max,
            jpeg_quality=int(np.clip(args.viz_jpeg_quality, 1, 100)),
        )
        print(f"写入 {html_path}")

    if args.viz_components_html:
        comp_html = out_dir / "covis_components.html"
        _write_covis_html_components(
            comp_html,
            image_paths,
            edges_plot,
            xy,
            theta_rad,
            max_parallax_deg,
            thumb_max=args.viz_thumb_max,
            jpeg_quality=int(np.clip(args.viz_jpeg_quality, 1, 100)),
        )
        print(f"写入 {comp_html}")

    plot_path: Optional[Path] = None
    if args.plot_png is not None:
        plot_path = args.plot_png.expanduser().resolve()
    elif args.plot:
        plot_path = out_dir / "covis_graph.png"

    if plot_path is not None:
        _plot_covis(
            xy,
            edges_plot,
            plot_path,
            max_edges=args.plot_max_edges,
            title=f"Covis (approx) edges={len(edges_plot)}",
        )
        print(f"写入 {plot_path}")

    if args.viz_sectors:
        si, sj = 0, min(1, n - 1)
        if edges_plot:
            si, sj = edges_plot[0]
        if args.viz_sectors_pair is not None:
            parts = args.viz_sectors_pair.replace(" ", "").split(",")
            if len(parts) != 2:
                print('错误：--viz-sectors-pair 需为 "i,j" 两个整数', file=sys.stderr)
                return 2
            si, sj = int(parts[0]), int(parts[1])
        if not (0 <= si < n and 0 <= sj < n) or si == sj:
            print(f"错误：非法帧对 ({si},{sj})，n={n}", file=sys.stderr)
            return 2
        sec_path = (args.viz_sectors_png or (out_dir / "covis_sectors_demo.png")).expanduser().resolve()
        _plot_geometry_demo(xy, theta_rad, max_parallax_deg, si, sj, sec_path)
        print(
            f"写入 {sec_path}（几何示意：帧 {si} ↔ {sj}；max_parallax={max_parallax_deg:g}°）"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
