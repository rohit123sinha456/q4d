import torch

from q4d_wam.labels import (
    CATEGORY_OBJECT,
    CATEGORY_ROBOT,
    attach_points_to_bodies,
    farthest_point_indices,
    reconstruct_rigid_tracks,
    stratified_point_indices,
)


def _translation(x: float, y: float, z: float) -> torch.Tensor:
    result = torch.eye(4)
    result[:3, 3] = torch.tensor([x, y, z])
    return result


def test_attachment_and_future_reconstruction() -> None:
    body_ids = torch.tensor([10, 20])
    initial_poses = torch.stack((_translation(1, 0, 0), _translation(0, 2, 0)))
    points_world = torch.tensor([[1.5, 0.0, 0.0], [0.0, 2.0, 1.0], [9.0, 9.0, 9.0]])
    point_ids = torch.tensor([10, 20, 99])

    attached = attach_points_to_bodies(points_world, point_ids, body_ids, initial_poses)
    pose_sequence = torch.stack(
        (
            initial_poses,
            torch.stack((_translation(2, 0, 0), _translation(0, 4, 0))),
        )
    )
    tracks = reconstruct_rigid_tracks(
        attached.local_xyz_m, attached.body_indices, pose_sequence
    )

    torch.testing.assert_close(attached.local_xyz_m[0], torch.tensor([0.5, 0.0, 0.0]))
    torch.testing.assert_close(attached.local_xyz_m[1], torch.tensor([0.0, 0.0, 1.0]))
    assert attached.body_indices.tolist() == [0, 1, -1]
    torch.testing.assert_close(tracks[:, 0], points_world)
    torch.testing.assert_close(tracks[0, 1], torch.tensor([2.5, 0.0, 0.0]))
    torch.testing.assert_close(tracks[1, 1], torch.tensor([0.0, 4.0, 1.0]))
    torch.testing.assert_close(tracks[2, 1], points_world[2])


def test_farthest_point_sampling_is_deterministic_and_unique() -> None:
    points = torch.tensor([[0.0, 0, 0], [1.0, 0, 0], [2.0, 0, 0], [3.0, 0, 0]])

    first = farthest_point_indices(points, 3)
    second = farthest_point_indices(points, 3)

    assert first.tolist() == second.tolist()
    assert len(torch.unique(first)) == 3


def test_farthest_sampling_supports_visible_feature_vectors() -> None:
    features = torch.tensor(
        [[0.0, 0, 0, 0, 0, 0], [1.0, 0, 0, 0, 0, 0], [0.0, 0, 0, 10.0, 0, 0]]
    )

    selected = farthest_point_indices(features, 2)

    assert 2 in selected.tolist()


def test_stratified_sampling_preserves_rare_object_points() -> None:
    points = torch.stack(
        [torch.arange(10, dtype=torch.float32), torch.zeros(10), torch.zeros(10)], dim=-1
    )
    categories = torch.tensor([CATEGORY_OBJECT] * 2 + [CATEGORY_ROBOT] * 8)

    selected = stratified_point_indices(
        points,
        categories,
        total_count=5,
        quotas={CATEGORY_OBJECT: 2, CATEGORY_ROBOT: 2},
    )

    assert int((categories[selected] == CATEGORY_OBJECT).sum()) == 2
    assert len(selected) == 5
