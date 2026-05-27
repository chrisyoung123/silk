# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import torch

from silk.geometry.pose_correspondence import (
    compute_pair_correspondences,
    planar_body_poses_to_T_cam0_cam1,
)


def test_planar_pose_T():
    Tvc = torch.eye(4)
    T = planar_body_poses_to_T_cam0_cam1(0.0, 0.0, 0.0, 1.0, 0.0, 0.1, Tvc)
    assert T.shape == (4, 4)


def test_constant_depth_correspondence():
    B, L = 1, 16
    xy = torch.stack(
        [
            torch.linspace(8, 40, L),
            torch.linspace(8, 40, L),
        ],
        dim=-1,
    ).unsqueeze(0)
    K = torch.eye(3).unsqueeze(0)
    T = torch.eye(4).unsqueeze(0)
    T[0, 0, 3] = 0.5
    cf, cb = compute_pair_correspondences(
        xy,
        8,
        8,
        T_0to1=T,
        K0=K,
        K1=K,
        default_depth=2.0,
        image_shape=(64, 64),
    )
    assert cf.shape == (B, L)
    assert cb.shape == (B, L)
