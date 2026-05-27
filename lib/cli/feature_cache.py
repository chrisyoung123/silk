# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""CLI：批量缓存共视图 SiLK 特征（可选 sky/mask/ANMS 后处理）。"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from silk.logger import LOG


def _load_feature_cache_main():
    root = Path(__file__).resolve().parents[2]
    path = root / "scripts/eval/cache_covis_silk_features.py"
    spec = importlib.util.spec_from_file_location("cache_covis_silk_features", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["cache_covis_silk_features"] = mod
    spec.loader.exec_module(mod)
    return mod


def main(cfg: DictConfig) -> Any:
    mod = _load_feature_cache_main()
    LOG.info("feature_cache: 开始批量提取 SiLK 特征")
    result = mod.run_from_config(cfg)
    if isinstance(result, int) and result != 0:
        raise RuntimeError(f"feature_cache 退出码 {result}")
    return result
