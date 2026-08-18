import pytest
import torch

from q4d_wam.geometry import (
    backproject_depth_cv,
    camera_cv_to_gl,
    camera_cv_to_world,
    camera_gl_to_world,
    invert_rigid_transform,
    to_homogeneous_transform,
    transform_points,
)


def test_backprojection_uses_metric_depth_and_opencv_axes() -> None:
    depth_mm = torch.tensor([[[1000], [2000]], [[0], [1000]]], dtype=torch.int16)
    intrinsics = torch.tensor([[2.0, 0.0, 0.5], [0.0, 2.0, 0.5], [0.0, 0.0, 1.0]])

    points, valid = backproject_depth_cv(depth_mm, intrinsics)

    expected = torch.tensor(
        [
            [[0.0, 0.0, 1.0], [1.0, 0.0, 2.0]],
            [[0.0, 0.0, 0.0], [0.5, 0.5, 1.0]],
        ]
    )
    torch.testing.assert_close(points, expected)
    assert valid.tolist() == [[True, True], [False, True]]


def test_world_camera_rigid_transform_round_trip() -> None:
    world_to_camera = torch.tensor(
        [
            [0.0, -1.0, 0.0, 2.0],
            [1.0, 0.0, 0.0, -1.0],
            [0.0, 0.0, 1.0, 0.5],
        ]
    )
    world_points = torch.tensor([[1.0, 2.0, 3.0], [-1.0, 0.5, 0.0]])

    camera_points = transform_points(world_points, world_to_camera)
    recovered_world = camera_cv_to_world(camera_points, world_to_camera)

    torch.testing.assert_close(recovered_world, world_points)
    identity = to_homogeneous_transform(world_to_camera) @ invert_rigid_transform(
        world_to_camera
    )
    torch.testing.assert_close(identity, torch.eye(4))


def test_opencv_and_opengl_routes_reach_same_world_points() -> None:
    points_cv = torch.tensor([[0.2, -0.1, 1.5], [-0.3, 0.4, 2.0]])
    cv_to_world = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.3],
            [0.0, 1.0, 0.0, -0.2],
            [0.0, 0.0, 1.0, 0.7],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    world_to_cv = invert_rigid_transform(cv_to_world)
    cv_to_gl = torch.diag(torch.tensor([1.0, -1.0, -1.0, 1.0]))
    gl_to_world = cv_to_world @ cv_to_gl

    world_from_cv = camera_cv_to_world(points_cv, world_to_cv)
    world_from_gl = camera_gl_to_world(camera_cv_to_gl(points_cv), gl_to_world)

    torch.testing.assert_close(world_from_cv, world_from_gl)


def test_bad_intrinsics_are_rejected() -> None:
    with pytest.raises(ValueError, match="focal lengths"):
        backproject_depth_cv(torch.ones(2, 2), torch.zeros(3, 3))
