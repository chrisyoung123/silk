# Copyright (c) Meta Platforms, Inc.

import numpy as np

from lib.matching.cache import EpipolarInlierCacheStore, MatchPairCacheStore
from scripts.eval import pair_matcher as pm


def test_silk_mutual_nn():
    rng = np.random.default_rng(0)
    pos0 = rng.random((50, 2)).astype(np.float64) * 100
    pos1 = pos0 + rng.normal(0, 0.5, size=pos0.shape)
    desc0 = rng.random((50, 256), dtype=np.float32)
    desc1 = desc0 + rng.normal(0, 0.01, size=desc0.shape).astype(np.float32)
    m0, m1, idx0, idx1 = pm.match_silk_mutual_nn(pos0, desc0, pos1, desc1)
    assert m0.shape[0] > 10
    assert m0.shape == m1.shape


def test_match_pair_cache_roundtrip(tmp_path):
    img0 = tmp_path / "a.jpg"
    img1 = tmp_path / "b.jpg"
    img0.write_bytes(b"x")
    img1.write_bytes(b"y")
    store = MatchPairCacheStore(tmp_path / "cache")
    m0 = np.array([[1.0, 2.0], [3.0, 4.0]])
    m1 = np.array([[5.0, 6.0], [7.0, 8.0]])
    idx0 = np.array([0, 2], dtype=np.int64)
    idx1 = np.array([1, 3], dtype=np.int64)
    store.save(
        img0,
        img1,
        matcher="silk",
        m0=m0,
        m1=m1,
        idx0=idx0,
        idx1=idx1,
        n_keypoints0=50,
        n_keypoints1=60,
    )
    loaded = store.load(img0, img1, matcher="silk")
    assert loaded is not None
    assert loaded["n_matches"] == 2
    np.testing.assert_allclose(loaded["m0"], m0)


def test_epipolar_inlier_cache_roundtrip(tmp_path):
    img0 = tmp_path / "a.jpg"
    img1 = tmp_path / "b.jpg"
    img0.write_bytes(b"x")
    img1.write_bytes(b"y")
    store = EpipolarInlierCacheStore(tmp_path / "epi")
    m0 = np.array([[1.0, 2.0], [3.0, 4.0]])
    m1 = np.array([[5.0, 6.0], [7.0, 8.0]])
    store.save(
        img0,
        img1,
        matcher="silk",
        camera_model="fisheye",
        ransac_threshold=0.5,
        min_inliers=8,
        m0=m0,
        m1=m1,
        n_matches_pre_epi=10,
        n_epipolar_inliers=2,
        n_used=2,
        n_keypoints0=50,
        n_keypoints1=60,
        ok=True,
    )
    loaded = store.load(
        img0,
        img1,
        matcher="silk",
        camera_model="fisheye",
        ransac_threshold=0.5,
        min_inliers=8,
    )
    assert loaded is not None
    assert loaded["n_used"] == 2
    assert loaded["ok"] is True
    np.testing.assert_allclose(loaded["m0"], m0)
    assert store.load(
        img0,
        img1,
        matcher="silk",
        camera_model="fisheye",
        ransac_threshold=0.6,
        min_inliers=8,
    ) is None
