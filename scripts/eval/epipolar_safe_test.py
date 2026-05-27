#!/usr/bin/env python3

import numpy as np

from epipolar_safe import _essential_matrix_candidates, estimate_pose_recover_pose


def test_essential_candidates_3x9():
    E = np.arange(27, dtype=np.float64).reshape(3, 9)
    cands = _essential_matrix_candidates(E)
    assert len(cands) == 3
    for c in cands:
        assert c.shape == (3, 3)


def test_essential_candidates_9x3():
    E = np.arange(27, dtype=np.float64).reshape(9, 3)
    cands = _essential_matrix_candidates(E)
    assert len(cands) == 3
    for c in cands:
        assert c.shape == (3, 3)


def test_recover_pose_stacked_E_no_crash():
    K = np.array([[500.0, 0, 320], [0, 500.0, 240], [0, 0, 1.0]])
    D = np.zeros((4, 1))
    rng = np.random.default_rng(1)
    n = 80
    pts1 = rng.uniform([50, 50], [590, 430], size=(n, 2))
    R = np.array([[0.99, -0.1, 0.0], [0.1, 0.99, 0.0], [0.0, 0.0, 1.0]])
    t = np.array([0.05, 0.0, 0.0])
    pts0_h = (R @ np.vstack([pts1.T, np.ones(n)])).T
    pts0 = (pts0_h[:, :2] / pts0_h[:, 2:3]) * 500 + np.array([320, 240])
    pts0 += rng.normal(0, 0.3, pts0.shape)
    est = estimate_pose_recover_pose(
        pts0, pts1, K, D, "fisheye", pixel_threshold=1.0, min_inliers=20
    )
    assert est is not None
