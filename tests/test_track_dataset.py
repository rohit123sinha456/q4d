from pathlib import Path

import numpy as np
import pytest
import torch

from q4d_wam.data import (
    NormalizationStats,
    SplitManifest,
    TrackDataset,
    compute_normalization,
    discover_training_files,
    trajectory_group_id,
    validate_training_file,
)


def _write_fragment(path: Path, offset: float) -> None:
    points = np.array(
        [[offset, 0, 0], [offset + 1, 0, 0], [offset + 2, 0, 0], [offset + 3, 0, 0]],
        dtype=np.float32,
    )
    targets = np.stack((points + [0.1, 0, 0], points + [0.2, 0, 0]), axis=1)
    np.savez_compressed(
        path,
        rgb=np.zeros((4, 4, 3), dtype=np.uint8),
        depth_mm=np.ones((4, 4, 1), dtype=np.int16),
        intrinsic_cv=np.eye(3, dtype=np.float32),
        extrinsic_cv=np.eye(4, dtype=np.float32)[:3],
        cam2world_gl=np.eye(4, dtype=np.float32),
        actions=np.array([[offset, 0], [offset + 1, 0]], dtype=np.float32),
        point_pixels_uv=np.zeros((4, 2), dtype=np.int64),
        xyz0_world_m=points,
        point_rgb=np.zeros((4, 3), dtype=np.uint8),
        target_tracks_world_m=targets,
    )


def _make_dataset(root: Path, count: int = 10) -> list[Path]:
    for index in range(count):
        _write_fragment(root / f"episode_{index:06d}.train.npz", float(index))
    np.savez_compressed(root / "episode_000000.audit.npz", body_indices=np.array([0]))
    return discover_training_files(root)


def test_splits_are_deterministic_disjoint_and_ignore_audits(tmp_path: Path) -> None:
    files = _make_dataset(tmp_path)

    first = SplitManifest.create(
        files,
        seed=7,
        train_fraction=0.8,
        validation_fraction=0.1,
        test_fraction=0.1,
    )
    second = SplitManifest.create(
        files,
        seed=7,
        train_fraction=0.8,
        validation_fraction=0.1,
        test_fraction=0.1,
    )

    assert first == second
    assert (len(first.train), len(first.validation), len(first.test)) == (8, 1, 1)
    assert len(set(first.all_files)) == 10
    assert all("audit" not in name for name in first.all_files)


def test_counterfactual_siblings_stay_in_the_same_split(tmp_path: Path) -> None:
    files = []
    for state_index in range(10):
        for branch in ("success", "perturbed", "no_op", "failure"):
            path = tmp_path / f"state_{state_index:06d}__{branch}.train.npz"
            _write_fragment(path, float(state_index))
            files.append(path)

    split = SplitManifest.create(
        files,
        seed=7,
        train_fraction=0.8,
        validation_fraction=0.1,
        test_fraction=0.1,
    )

    assert (len(split.train), len(split.validation), len(split.test)) == (32, 4, 4)
    split_groups = [
        {trajectory_group_id(name) for name in names}
        for names in (split.train, split.validation, split.test)
    ]
    assert not (split_groups[0] & split_groups[1])
    assert not (split_groups[0] & split_groups[2])
    assert not (split_groups[1] & split_groups[2])


def test_normalization_uses_only_explicit_training_files(tmp_path: Path) -> None:
    files = _make_dataset(tmp_path)
    training_files = files[:8]

    stats = compute_normalization(training_files)

    expected_points = torch.cat(
        [torch.from_numpy(np.load(path)["xyz0_world_m"]) for path in training_files]
    )
    torch.testing.assert_close(stats.xyz_mean_m, expected_points.mean(dim=0))
    assert stats.source_files == tuple(path.name for path in training_files)
    assert 1 in stats.constant_action_channels


def test_dataset_shapes_are_fixed_and_queries_are_deterministic(tmp_path: Path) -> None:
    files = _make_dataset(tmp_path)
    stats = compute_normalization(files[:8])
    dataset = TrackDataset(files, stats, num_queries=2, cache_size=2)

    first = dataset[0]
    second = dataset[0]

    assert first["scene_xyz"].shape == (4, 3)
    assert first["scene_rgb"].shape == (4, 3)
    assert first["actions"].shape == (2, 2)
    assert first["query_xyz"].shape == (2, 3)
    assert first["target_displacement"].shape == (2, 2, 3)
    torch.testing.assert_close(first["query_indices"], second["query_indices"])
    assert first is second
    assert not (set(first) & {"body_indices", "point_segmentation_ids", "local_xyz_m"})
    assert validate_training_file(files[0])["points"] == 4


def test_dataset_can_slice_a_stored_trajectory_horizon(tmp_path: Path) -> None:
    files = _make_dataset(tmp_path)
    stats = compute_normalization(files[:8])
    dataset = TrackDataset(files, stats, num_queries=2, horizon=1)

    sample = dataset[0]

    assert sample["actions"].shape == (1, 2)
    assert sample["target_displacement"].shape == (2, 1, 3)
    assert sample["target_world_m"].shape == (2, 1, 3)


def test_dataset_rejects_horizon_beyond_stored_trajectory(tmp_path: Path) -> None:
    files = _make_dataset(tmp_path)
    stats = compute_normalization(files[:8])
    dataset = TrackDataset(files, stats, num_queries=2, horizon=3)

    with pytest.raises(ValueError, match="exceeds stored horizon"):
        dataset[0]


def test_normalization_round_trip(tmp_path: Path) -> None:
    files = _make_dataset(tmp_path)
    stats = compute_normalization(files[:8])
    path = tmp_path / "normalization.json"

    stats.save(path)
    loaded = NormalizationStats.load(path)

    torch.testing.assert_close(loaded.xyz_mean_m, stats.xyz_mean_m)
    torch.testing.assert_close(loaded.action_scale, stats.action_scale)
    assert loaded.source_files == stats.source_files
