#!/usr/bin/env python3
"""
按 JPG 文件名（不含扩展名）以下划线分割后的第一段作为类别，将 JPG 与同名的 JSON 成对整理到子目录。

JSON 文件名规则：与 JPG 相同的「主文件名」+ extra + .json
例如 extra 默认为空：foo_bar.jpg 对应 foo_bar.json；
若 --extra _ann：foo_bar.jpg 对应 foo_bar_ann.json
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".JPG", ".JPEG"}


def iter_images(root: Path, recursive: bool) -> list[Path]:
    if recursive:
        out: list[Path] = []
        for p in root.rglob("*"):
            if p.is_file() and p.suffix in IMAGE_SUFFIXES:
                out.append(p)
        return sorted(out)
    return sorted(
        p for p in root.iterdir() if p.is_file() and p.suffix in IMAGE_SUFFIXES
    )


def category_from_stem(stem: str) -> str:
    part = stem.split("_", 1)[0] if "_" in stem else stem
    return part or "unknown"


def main() -> int:
    ap = argparse.ArgumentParser(
        description="按 JPG 主文件名下划线首段分类，复制或移动 JPG+JSON 对。"
    )
    ap.add_argument(
        "--src",
        type=Path,
        required=True,
        help="包含 JPG（及对应 JSON）的源目录",
    )
    ap.add_argument(
        "--dst",
        type=Path,
        required=True,
        help="输出根目录；会在其下创建各类别子目录",
    )
    ap.add_argument(
        "--extra",
        default="",
        help='插在 JPG 主文件名与 ".json" 之间的字符串（默认空，即 stem.json）',
    )
    ap.add_argument(
        "--recursive",
        action="store_true",
        help="递归扫描源目录下的 JPG（默认只扫描 --src 一层）",
    )
    ap.add_argument(
        "--move",
        action="store_true",
        help="移动文件；默认为复制，源目录保留",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="目标已存在同名文件时覆盖；默认跳过并告警",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印将要执行的操作，不写磁盘",
    )
    args = ap.parse_args()

    src: Path = args.src.expanduser().resolve()
    dst_root: Path = args.dst.expanduser().resolve()
    extra: str = args.extra

    if not src.is_dir():
        print(f"错误：源目录不存在或不是目录：{src}", file=sys.stderr)
        return 1

    images = iter_images(src, args.recursive)
    if not images:
        print(f"在 {src} 未找到 JPG/JPEG 文件。", file=sys.stderr)
        return 1

    verb = "移动" if args.move else "复制"
    missing_json = 0
    skipped = 0
    done = 0

    for jpg in images:
        stem = jpg.stem
        json_name = f"{stem}{extra}.json"
        json_path = jpg.with_name(json_name)
        if not json_path.is_file():
            print(f"警告：未找到配对 JSON，跳过 JPG：{jpg}（期望：{json_path}）")
            missing_json += 1
            continue

        cat = category_from_stem(stem)
        out_dir = dst_root / cat
        dest_jpg = out_dir / jpg.name
        dest_json = out_dir / json_path.name

        for dest in (dest_jpg, dest_json):
            if dest.exists() and not args.overwrite:
                print(f"警告：目标已存在，跳过：{dest}")
                skipped += 1
                break
        else:
            if args.dry_run:
                print(f"[dry-run] {verb}: {jpg.name} + {json_path.name} -> {out_dir}/")
            else:
                out_dir.mkdir(parents=True, exist_ok=True)
                if args.move:
                    shutil.move(str(jpg), str(dest_jpg))
                    shutil.move(str(json_path), str(dest_json))
                else:
                    shutil.copy2(jpg, dest_jpg)
                    shutil.copy2(json_path, dest_json)
            done += 1

    print(
        f"完成：成功处理 {done} 对；缺少 JSON {missing_json}；"
        f"因目标已存在跳过 {skipped}。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
