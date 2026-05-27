# Copyright (c) Meta Platforms, Inc. and affiliates.

import numpy as np

from lib.geometry.local_ba_planar import VisualTrack, filter_tracks_by_reprojection


def test_filter_tracks_by_reprojection_track_mode():
    T = np.eye(4, dtype=np.float64)
    K = np.eye(3, dtype=np.float64)
    K[0, 0] = K[1, 1] = 500.0
    K[0, 2], K[1, 2] = 320.0, 240.0
    D = np.zeros((4, 1), dtype=np.float64)
    tracks = [
        VisualTrack(
            X_world=np.array([0.0, 0.0, 5.0]),
            observations={"cam0": np.array([320.0, 240.0])},
        ),
        VisualTrack(
            X_world=np.array([0.0, 0.0, 5.0]),
            observations={"cam0": np.array([420.0, 340.0])},
        ),
    ]
    filtered, stats = filter_tracks_by_reprojection(
        tracks,
        ["cam0"],
        {"cam0": T},
        K,
        D,
        "fisheye",
        max_reproj_px=50.0,
        mode="track",
    )
    assert stats["n_tracks_removed"] == 1
    assert len(filtered) == 1
    assert np.allclose(filtered[0].observations["cam0"], [320.0, 240.0])


def test_filter_tracks_target_observation_mode():
    T = np.eye(4, dtype=np.float64)
    K = np.eye(3, dtype=np.float64)
    K[0, 0] = K[1, 1] = 500.0
    K[0, 2], K[1, 2] = 320.0, 240.0
    D = np.zeros((4, 1), dtype=np.float64)
    tracks = [
        VisualTrack(
            X_world=np.array([0.0, 0.0, 5.0]),
            observations={
                "ref0": np.array([320.0, 240.0]),
                "target": np.array([420.0, 340.0]),
            },
        ),
    ]
    filtered, stats = filter_tracks_by_reprojection(
        tracks,
        ["ref0", "target"],
        {"ref0": T, "target": T},
        K,
        D,
        "fisheye",
        max_reproj_px=50.0,
        mode="target_observation",
        target_key="target",
    )
    assert stats["n_obs_removed"] == 1
    assert len(filtered) == 1
    assert "target" not in filtered[0].observations
    assert "ref0" in filtered[0].observations
