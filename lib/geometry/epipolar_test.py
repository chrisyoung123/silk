# Copyright (c) Meta Platforms, Inc. and affiliates.

import numpy as np

from lib.geometry.epipolar import (
    epipolar_essential_mask,
    estimate_pose_recover_pose,
    filter_matches_epipolar,
)


def test_filter_matches_epipolar_empty():
    K = np.eye(3)
    D = np.zeros((4, 1))
    m0 = np.zeros((0, 2))
    m1 = np.zeros((0, 2))
    out0, out1, meta = filter_matches_epipolar(
        m0,
        m1,
        K,
        D,
        camera_model="fisheye",
        ransac_threshold=0.5,
        min_inliers=8,
        min_matches=10,
    )
    assert out0.shape[0] == 0
    assert meta["ok"] is False


def test_epipolar_essential_mask_too_few():
    K = np.eye(3)
    D = np.zeros((4, 1))
    m0 = np.random.rand(3, 2)
    m1 = np.random.rand(3, 2)
    assert epipolar_essential_mask(m0, m1, K, D, "fisheye", 0.5) is None


def test_estimate_pose_recover_pose_synthetic():
    K = np.array([[500.0, 0, 320], [0, 500.0, 240], [0, 0, 1.0]])
    D = np.zeros((4, 1))
    rng = np.random.default_rng(0)
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
    _, _, n_epi, _, mask = est
    assert n_epi >= 20
    assert mask.sum() >= 20
