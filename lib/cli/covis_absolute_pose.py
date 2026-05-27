# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""CLI：基于共视图的绝对位姿 PnP 解算。"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from silk.logger import LOG


def _load_covis_absolute_pose_main():
    root = Path(__file__).resolve().parents[2]
    path = root / "scripts/eval/covis_absolute_pose.py"
    spec = importlib.util.spec_from_file_location("covis_absolute_pose", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["covis_absolute_pose"] = mod
    spec.loader.exec_module(mod)
    return mod


def main(cfg: DictConfig) -> Any:
    mod = _load_covis_absolute_pose_main()
    LOG.info("covis_absolute_pose: 开始序列解算")
    result = mod.run_from_config(cfg)
    if isinstance(result, int) and result != 0:
        raise RuntimeError(f"covis_absolute_pose 退出码 {result}")
    return result
