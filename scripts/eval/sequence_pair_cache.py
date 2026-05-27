#!/usr/bin/env python3
"""序列级共视边缓存：整个序列所有匹配 + 极线结果存于单个 npz。"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Set

import numpy as np

from lib.matching.cache import pair_cache_key

ARCHIVE_NAME = "covis_edges.npz"

_PAIR_SCALAR_FIELDS = frozenset(
    {
        "img0",
        "img1",
        "matcher",
        "camera_model",
        "reason",
    }
)
_PAIR_INT_FIELDS = frozenset(
    {
        "n_keypoints0",
        "n_keypoints1",
        "n_matches_raw",
        "n_matches",
        "n_matches_pre_epi",
        "n_epipolar_inliers",
        "n_used",
        "min_inliers",
        "ok",
    }
)
_PAIR_FLOAT_FIELDS = frozenset({"ransac_threshold"})
_PAIR_ARRAY_FIELDS = frozenset(
    {
        "m0_raw",
        "m1_raw",
        "m0_epi",
        "m1_epi",
        "m0",
        "m1",
        "idx0",
        "idx1",
    }
)


class SequencePairCacheStore:
    """<root>/covis_edges.npz：序列内全部共视边（匹配 + 极线 inlier）。

    运行时只在内存更新，默认结束时一次性落盘，避免每条边重写整个文件。
    """

    def __init__(
        self,
        root: Path,
        *,
        camera_model: str,
        ransac_threshold: float,
        min_inliers: int,
        flush_every: Optional[int] = None,
    ) -> None:
        self.root = root.expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.archive_path = self.root / ARCHIVE_NAME
        self._camera_model = str(camera_model)
        self._ransac_threshold = float(ransac_threshold)
        self._min_inliers = int(min_inliers)
        self._legacy_pairs_dir = self.root / "pairs"
        self._legacy_epi_dir = self.root / "epipolar" / "pairs"
        self._loaded = False
        self._dirty = False
        self._pairs: Dict[str, Dict[str, Any]] = {}
        self._n_updates = 0
        if flush_every is None:
            flush_every = int(os.environ.get("CACHE_FLUSH_EVERY", "0"))
        self._flush_every = max(0, int(flush_every))

    def npz_path(self, img0: Path, img1: Path, *, matcher: str) -> Path:
        return self.archive_path

    def _pair_id(self, img0: Path, img1: Path, *, matcher: str) -> str:
        return pair_cache_key(img0, img1, matcher=matcher)

    def _scalar(self, v: Any) -> Any:
        if isinstance(v, np.ndarray):
            if v.shape == ():
                return v.item()
            if v.dtype.kind in ("U", "S"):
                return str(v)
        return v

    def _bundle_from_legacy_npz(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k, v in raw.items():
            if isinstance(v, np.ndarray) and v.dtype.kind not in ("U", "S") and v.ndim > 0:
                out[k] = np.asarray(v)
            else:
                out[k] = self._scalar(v)
        return out

    def _read_npz_file(self, path: Path) -> Dict[str, Any]:
        d = np.load(path, allow_pickle=False)
        return {k: d[k] for k in d.files}

    def _split_archive_key(self, name: str) -> Optional[tuple[str, str]]:
        if name.startswith("__") or len(name) <= 17 or name[16] != "_":
            return None
        return name[:16], name[17:]

    def _archive_to_pairs(self, raw: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        pair_keys: Set[str] = set()
        if "__pair_keys" in raw:
            for pk in np.asarray(raw["__pair_keys"]).tolist():
                pair_keys.add(str(pk))
        for name in raw:
            split = self._split_archive_key(name)
            if split is not None:
                pair_keys.add(split[0])

        pairs: Dict[str, Dict[str, Any]] = {}
        for pk in sorted(pair_keys):
            bundle: Dict[str, Any] = {}
            prefix = f"{pk}_"
            for name, value in raw.items():
                if not name.startswith(prefix):
                    continue
                field = name[len(prefix) :]
                if field in _PAIR_ARRAY_FIELDS:
                    bundle[field] = np.asarray(value)
                elif field in _PAIR_SCALAR_FIELDS:
                    bundle[field] = self._scalar(value)
                elif field in _PAIR_INT_FIELDS:
                    bundle[field] = int(self._scalar(value))
                elif field in _PAIR_FLOAT_FIELDS:
                    bundle[field] = float(self._scalar(value))
                else:
                    bundle[field] = self._scalar(value)
            if bundle:
                pairs[pk] = bundle
        return pairs

    def _pairs_to_archive(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "__meta_camera_model": np.array(self._camera_model),
            "__meta_ransac_threshold": np.float64(self._ransac_threshold),
            "__meta_min_inliers": np.int32(self._min_inliers),
            "__pair_keys": np.array(sorted(self._pairs.keys()), dtype="U32"),
        }
        for pk, bundle in self._pairs.items():
            for field, value in bundle.items():
                key = f"{pk}_{field}"
                if field in _PAIR_ARRAY_FIELDS:
                    out[key] = np.asarray(value)
                elif field in _PAIR_SCALAR_FIELDS:
                    out[key] = np.array(str(value))
                elif field in _PAIR_INT_FIELDS:
                    out[key] = np.int32(int(value))
                elif field in _PAIR_FLOAT_FIELDS:
                    out[key] = np.float64(float(value))
                else:
                    out[key] = np.array(str(value))
        return out

    def _import_legacy_dir(self, directory: Path) -> None:
        if not directory.is_dir():
            return
        for path in sorted(directory.glob("*.npz")):
            raw = self._read_npz_file(path)
            bundle = self._bundle_from_legacy_npz(raw)
            img0 = bundle.get("img0")
            img1 = bundle.get("img1")
            matcher = str(bundle.get("matcher", "silk"))
            if not img0 or not img1:
                pk = path.stem
            else:
                pk = self._pair_id(Path(str(img0)), Path(str(img1)), matcher=matcher)
            existing = self._pairs.get(pk, {})
            existing.update(bundle)
            self._pairs[pk] = existing
            self._dirty = True

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if self.archive_path.is_file():
            self._pairs = self._archive_to_pairs(self._read_npz_file(self.archive_path))
            return
        self._import_legacy_dir(self._legacy_pairs_dir)
        self._import_legacy_dir(self._legacy_epi_dir)

    def flush(self, *, force: bool = False) -> None:
        """将内存缓存写入 covis_edges.npz（默认仅在 dirty 时写入）。"""
        self._ensure_loaded()
        if not self._dirty and not force:
            return
        staging = self.archive_path.with_name("_" + self.archive_path.stem)
        np.savez_compressed(str(staging), **self._pairs_to_archive())
        written = Path(str(staging) + ".npz")
        written.replace(self.archive_path)
        self._dirty = False

    def _maybe_periodic_flush(self) -> None:
        if self._flush_every > 0 and self._n_updates % self._flush_every == 0:
            self.flush()

    def _get_bundle(self, img0: Path, img1: Path, *, matcher: str) -> Optional[Dict[str, Any]]:
        self._ensure_loaded()
        pk = self._pair_id(img0, img1, matcher=matcher)
        bundle = self._pairs.get(pk)
        if bundle is None:
            return None
        return dict(bundle)

    def _put_fields(self, img0: Path, img1: Path, *, matcher: str, fields: Dict[str, Any]) -> Path:
        self._ensure_loaded()
        img0 = img0.expanduser().resolve()
        img1 = img1.expanduser().resolve()
        pk = self._pair_id(img0, img1, matcher=matcher)
        bundle = dict(self._pairs.get(pk, {}))
        bundle.update(fields)
        bundle["img0"] = str(img0)
        bundle["img1"] = str(img1)
        bundle["matcher"] = str(matcher)
        self._pairs[pk] = bundle
        self._dirty = True
        self._n_updates += 1
        self._maybe_periodic_flush()
        return self.archive_path

    def _epi_params_match(self, bundle: Dict[str, Any]) -> bool:
        cm = str(bundle.get("camera_model", self._camera_model))
        rt = float(bundle.get("ransac_threshold", -1))
        mi = int(bundle.get("min_inliers", -1))
        return (
            cm == self._camera_model
            and abs(rt - self._ransac_threshold) < 1e-9
            and mi == self._min_inliers
        )

    def load_match(
        self,
        img0: Path,
        img1: Path,
        *,
        matcher: str,
    ) -> Optional[Dict[str, Any]]:
        bundle = self._get_bundle(img0, img1, matcher=matcher)
        if bundle is None:
            return None
        cache_path = str(self.archive_path)
        if "m0_raw" in bundle:
            return {
                "img0": str(bundle["img0"]),
                "img1": str(bundle["img1"]),
                "matcher": str(bundle["matcher"]),
                "m0": np.asarray(bundle["m0_raw"], dtype=np.float64),
                "m1": np.asarray(bundle["m1_raw"], dtype=np.float64),
                "idx0": np.asarray(bundle["idx0"], dtype=np.int64),
                "idx1": np.asarray(bundle["idx1"], dtype=np.int64),
                "n_keypoints0": int(bundle["n_keypoints0"]),
                "n_keypoints1": int(bundle["n_keypoints1"]),
                "n_matches": int(bundle["n_matches_raw"]),
                "cache_path": cache_path,
            }
        if "m0" in bundle and "idx0" in bundle:
            return {
                "img0": str(bundle.get("img0", img0)),
                "img1": str(bundle.get("img1", img1)),
                "matcher": str(bundle.get("matcher", matcher)),
                "m0": np.asarray(bundle["m0"], dtype=np.float64),
                "m1": np.asarray(bundle["m1"], dtype=np.float64),
                "idx0": np.asarray(bundle["idx0"], dtype=np.int64),
                "idx1": np.asarray(bundle["idx1"], dtype=np.int64),
                "n_keypoints0": int(bundle["n_keypoints0"]),
                "n_keypoints1": int(bundle["n_keypoints1"]),
                "n_matches": int(bundle.get("n_matches", bundle.get("n_matches_raw", 0))),
                "cache_path": cache_path,
            }
        return None

    def load_epipolar(
        self,
        img0: Path,
        img1: Path,
        *,
        matcher: str,
        camera_model: str,
        ransac_threshold: float,
        min_inliers: int,
    ) -> Optional[Dict[str, Any]]:
        bundle = self._get_bundle(img0, img1, matcher=matcher)
        if bundle is None or not self._epi_params_match(bundle):
            return None
        cache_path = str(self.archive_path)
        if "m0_epi" in bundle:
            reason = str(bundle["reason"]) if bundle.get("reason") else None
            return {
                "img0": str(bundle["img0"]),
                "img1": str(bundle["img1"]),
                "matcher": str(bundle["matcher"]),
                "camera_model": str(bundle["camera_model"]),
                "ransac_threshold": float(bundle["ransac_threshold"]),
                "min_inliers": int(bundle["min_inliers"]),
                "m0": np.asarray(bundle["m0_epi"], dtype=np.float64),
                "m1": np.asarray(bundle["m1_epi"], dtype=np.float64),
                "n_matches_pre_epi": int(bundle["n_matches_pre_epi"]),
                "n_epipolar_inliers": int(bundle["n_epipolar_inliers"]),
                "n_used": int(bundle["n_used"]),
                "n_keypoints0": int(bundle["n_keypoints0"]),
                "n_keypoints1": int(bundle["n_keypoints1"]),
                "ok": bool(int(bundle["ok"])),
                "reason": reason,
                "cache_path": cache_path,
            }
        if "m0" in bundle and "n_used" in bundle:
            reason = str(bundle["reason"]) if bundle.get("reason") else None
            return {
                "img0": str(bundle.get("img0", img0)),
                "img1": str(bundle.get("img1", img1)),
                "matcher": str(bundle.get("matcher", matcher)),
                "camera_model": str(bundle.get("camera_model", camera_model)),
                "ransac_threshold": float(bundle.get("ransac_threshold", ransac_threshold)),
                "min_inliers": int(bundle.get("min_inliers", min_inliers)),
                "m0": np.asarray(bundle["m0"], dtype=np.float64),
                "m1": np.asarray(bundle["m1"], dtype=np.float64),
                "n_matches_pre_epi": int(bundle.get("n_matches_pre_epi", 0)),
                "n_epipolar_inliers": int(bundle.get("n_epipolar_inliers", 0)),
                "n_used": int(bundle.get("n_used", 0)),
                "n_keypoints0": int(bundle.get("n_keypoints0", 0)),
                "n_keypoints1": int(bundle.get("n_keypoints1", 0)),
                "ok": bool(int(bundle.get("ok", 0))),
                "reason": reason,
                "cache_path": cache_path,
            }
        return None

    def save_match(
        self,
        img0: Path,
        img1: Path,
        *,
        matcher: str,
        m0: np.ndarray,
        m1: np.ndarray,
        idx0: np.ndarray,
        idx1: np.ndarray,
        n_keypoints0: int,
        n_keypoints1: int,
    ) -> Path:
        return self._put_fields(
            img0,
            img1,
            matcher=matcher,
            fields={
                "m0_raw": np.asarray(m0, dtype=np.float64),
                "m1_raw": np.asarray(m1, dtype=np.float64),
                "idx0": np.asarray(idx0, dtype=np.int64),
                "idx1": np.asarray(idx1, dtype=np.int64),
                "n_keypoints0": int(n_keypoints0),
                "n_keypoints1": int(n_keypoints1),
                "n_matches_raw": int(m0.shape[0]),
            },
        )

    def save_epipolar(
        self,
        img0: Path,
        img1: Path,
        *,
        matcher: str,
        camera_model: str,
        ransac_threshold: float,
        min_inliers: int,
        m0: np.ndarray,
        m1: np.ndarray,
        n_matches_pre_epi: int,
        n_epipolar_inliers: int,
        n_used: int,
        n_keypoints0: int,
        n_keypoints1: int,
        ok: bool,
        reason: Optional[str] = None,
    ) -> Path:
        return self._put_fields(
            img0,
            img1,
            matcher=matcher,
            fields={
                "camera_model": str(camera_model),
                "ransac_threshold": float(ransac_threshold),
                "min_inliers": int(min_inliers),
                "m0_epi": np.asarray(m0, dtype=np.float64),
                "m1_epi": np.asarray(m1, dtype=np.float64),
                "n_matches_pre_epi": int(n_matches_pre_epi),
                "n_epipolar_inliers": int(n_epipolar_inliers),
                "n_used": int(n_used),
                "n_keypoints0": int(n_keypoints0),
                "n_keypoints1": int(n_keypoints1),
                "ok": int(1 if ok else 0),
                "reason": str(reason or ""),
            },
        )

    def save_pair(
        self,
        img0: Path,
        img1: Path,
        *,
        matcher: str,
        camera_model: str,
        ransac_threshold: float,
        min_inliers: int,
        m0_raw: Optional[np.ndarray] = None,
        m1_raw: Optional[np.ndarray] = None,
        idx0: Optional[np.ndarray] = None,
        idx1: Optional[np.ndarray] = None,
        n_keypoints0: int = 0,
        n_keypoints1: int = 0,
        m0_epi: np.ndarray,
        m1_epi: np.ndarray,
        n_matches_pre_epi: int,
        n_epipolar_inliers: int,
        n_used: int,
        ok: bool,
        reason: Optional[str] = None,
    ) -> Path:
        """一次写入匹配 + 极线结果（单次内存更新）。"""
        fields: Dict[str, Any] = {
            "camera_model": str(camera_model),
            "ransac_threshold": float(ransac_threshold),
            "min_inliers": int(min_inliers),
            "m0_epi": np.asarray(m0_epi, dtype=np.float64),
            "m1_epi": np.asarray(m1_epi, dtype=np.float64),
            "n_matches_pre_epi": int(n_matches_pre_epi),
            "n_epipolar_inliers": int(n_epipolar_inliers),
            "n_used": int(n_used),
            "n_keypoints0": int(n_keypoints0),
            "n_keypoints1": int(n_keypoints1),
            "ok": int(1 if ok else 0),
            "reason": str(reason or ""),
        }
        if m0_raw is not None and m1_raw is not None:
            fields.update(
                {
                    "m0_raw": np.asarray(m0_raw, dtype=np.float64),
                    "m1_raw": np.asarray(m1_raw, dtype=np.float64),
                    "idx0": np.asarray(idx0 if idx0 is not None else np.zeros((0,), np.int64)),
                    "idx1": np.asarray(idx1 if idx1 is not None else np.zeros((0,), np.int64)),
                    "n_matches_raw": int(np.asarray(m0_raw).shape[0]),
                }
            )
        return self._put_fields(img0, img1, matcher=matcher, fields=fields)
