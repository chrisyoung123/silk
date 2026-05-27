#!/usr/bin/env python3

import numpy as np

from sequence_pair_cache import ARCHIVE_NAME, SequencePairCacheStore


def test_sequence_pair_cache_single_archive(tmp_path):
    root = tmp_path / "cache"
    store = SequencePairCacheStore(
        root,
        camera_model="fisheye",
        ransac_threshold=0.5,
        min_inliers=5,
    )
    img0 = tmp_path / "a.jpg"
    img1 = tmp_path / "b.jpg"
    img2 = tmp_path / "c.jpg"
    for p in (img0, img1, img2):
        p.write_bytes(p.name.encode())

    m0 = np.array([[1.0, 2.0], [3.0, 4.0]])
    m1 = np.array([[1.1, 2.1], [3.1, 4.1]])
    store.save_pair(
        img0,
        img1,
        matcher="silk",
        camera_model="fisheye",
        ransac_threshold=0.5,
        min_inliers=5,
        m0_raw=m0,
        m1_raw=m1,
        idx0=np.array([0, 1]),
        idx1=np.array([0, 1]),
        n_keypoints0=10,
        n_keypoints1=12,
        m0_epi=m0[:1],
        m1_epi=m1[:1],
        n_matches_pre_epi=2,
        n_epipolar_inliers=1,
        n_used=1,
        ok=True,
    )
    store.save_pair(
        img0,
        img2,
        matcher="silk",
        camera_model="fisheye",
        ransac_threshold=0.5,
        min_inliers=5,
        m0_raw=m0,
        m1_raw=m1,
        idx0=np.array([0, 1]),
        idx1=np.array([0, 1]),
        n_keypoints0=10,
        n_keypoints1=11,
        m0_epi=m0[:1],
        m1_epi=m1[:1],
        n_matches_pre_epi=2,
        n_epipolar_inliers=1,
        n_used=1,
        ok=True,
    )
    store.flush()

    archive = root / ARCHIVE_NAME
    assert archive.is_file()
    assert not (root / "pairs").exists() or list((root / "pairs").glob("*.npz")) == []

    store2 = SequencePairCacheStore(
        root,
        camera_model="fisheye",
        ransac_threshold=0.5,
        min_inliers=5,
    )
    epi = store2.load_epipolar(
        img0,
        img1,
        matcher="silk",
        camera_model="fisheye",
        ransac_threshold=0.5,
        min_inliers=5,
    )
    assert epi is not None
    assert epi["n_used"] == 1

    match = store2.load_match(img0, img2, matcher="silk")
    assert match is not None
    assert match["n_matches"] == 2

    raw = np.load(archive)
    assert "__pair_keys" in raw.files
    assert len(raw["__pair_keys"]) == 2
