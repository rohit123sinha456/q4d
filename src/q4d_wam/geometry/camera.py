"""Pinhole-camera geometry using ManiSkill's metric and frame conventions."""

from __future__ import annotations

import torch
from torch import Tensor


def _as_float_tensor(value: Tensor) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"expected torch.Tensor, got {type(value).__name__}")
    if value.dtype in (torch.float32, torch.float64):
        return value
    return value.to(torch.float32)


def _validate_intrinsics(intrinsics: Tensor) -> None:
    if intrinsics.shape[-2:] != (3, 3):
        raise ValueError(f"intrinsics must end in [3, 3], got {tuple(intrinsics.shape)}")
    if torch.any(intrinsics[..., 0, 0] <= 0) or torch.any(intrinsics[..., 1, 1] <= 0):
        raise ValueError("camera focal lengths must be positive")


def backproject_depth_cv(
    depth_mm: Tensor,
    intrinsics_cv: Tensor,
    *,
    min_depth_m: float = 0.0,
    max_depth_m: float | None = None,
) -> tuple[Tensor, Tensor]:
    """Backproject axial depth into metric OpenCV-camera coordinates.

    ManiSkill RGB-D depth is stored in millimetres. OpenCV camera coordinates use
    +x right, +y down, and +z forward. Invalid output locations are zeroed and returned
    separately in the boolean mask.

    Args:
        depth_mm: Tensor shaped ``[..., H, W]`` or ``[..., H, W, 1]``.
        intrinsics_cv: Tensor shaped ``[..., 3, 3]`` and broadcastable to the depth batch.
        min_depth_m: Strict lower validity bound in metres.
        max_depth_m: Optional inclusive upper validity bound in metres.

    Returns:
        ``(points_cv_m, valid_mask)`` with shapes ``[..., H, W, 3]`` and ``[..., H, W]``.
    """
    if depth_mm.ndim < 2:
        raise ValueError("depth must contain height and width dimensions")
    if depth_mm.shape[-1:] == (1,):
        if depth_mm.ndim < 3:
            raise ValueError("singleton-channel depth must contain height and width")
        depth_mm = depth_mm.squeeze(-1)

    depth_m = _as_float_tensor(depth_mm) / 1000.0
    intrinsics_cv = _as_float_tensor(intrinsics_cv).to(depth_m.device)
    _validate_intrinsics(intrinsics_cv)

    height, width = depth_m.shape[-2:]
    rows, columns = torch.meshgrid(
        torch.arange(height, device=depth_m.device, dtype=depth_m.dtype),
        torch.arange(width, device=depth_m.device, dtype=depth_m.dtype),
        indexing="ij",
    )

    fx = intrinsics_cv[..., 0, 0, None, None]
    fy = intrinsics_cv[..., 1, 1, None, None]
    cx = intrinsics_cv[..., 0, 2, None, None]
    cy = intrinsics_cv[..., 1, 2, None, None]

    # SAPIEN evaluates camera rays at pixel centres rather than integer pixel corners.
    x = (columns + 0.5 - cx) * depth_m / fx
    y = (rows + 0.5 - cy) * depth_m / fy
    points = torch.stack((x, y, depth_m), dim=-1)

    valid = torch.isfinite(depth_m) & (depth_m > min_depth_m)
    if max_depth_m is not None:
        valid &= depth_m <= max_depth_m
    points = torch.where(valid[..., None], points, torch.zeros_like(points))
    return points, valid


def to_homogeneous_transform(transform: Tensor) -> Tensor:
    """Convert a ``[..., 3, 4]`` or validate a ``[..., 4, 4]`` transform."""
    transform = _as_float_tensor(transform)
    if transform.shape[-2:] == (4, 4):
        return transform
    if transform.shape[-2:] != (3, 4):
        raise ValueError(f"transform must end in [3, 4] or [4, 4], got {transform.shape}")

    bottom = torch.zeros((*transform.shape[:-2], 1, 4), device=transform.device)
    bottom[..., 0, 3] = 1
    return torch.cat((transform, bottom), dim=-2)


def invert_rigid_transform(transform: Tensor) -> Tensor:
    """Invert a batched rigid transform without a general matrix inverse."""
    transform = to_homogeneous_transform(transform)
    rotation = transform[..., :3, :3]
    translation = transform[..., :3, 3]
    rotation_inverse = rotation.transpose(-1, -2)
    translation_inverse = -(rotation_inverse @ translation[..., None]).squeeze(-1)

    result = torch.zeros_like(transform)
    result[..., :3, :3] = rotation_inverse
    result[..., :3, 3] = translation_inverse
    result[..., 3, 3] = 1
    return result


def transform_points(points: Tensor, transform: Tensor) -> Tensor:
    """Apply a batched rigid transform to points with arbitrary spatial dimensions."""
    points = _as_float_tensor(points)
    if points.shape[-1] != 3:
        raise ValueError(f"points must end in 3 coordinates, got {tuple(points.shape)}")
    transform = to_homogeneous_transform(transform).to(points.device)

    rotation = transform[..., :3, :3]
    translation = transform[..., :3, 3]
    spatial_dimensions = points.ndim - rotation.ndim + 1
    if spatial_dimensions < 0:
        raise ValueError("transform has more batch dimensions than points")
    for _ in range(spatial_dimensions):
        rotation = rotation.unsqueeze(-3)
        translation = translation.unsqueeze(-2)
    return torch.matmul(rotation, points.unsqueeze(-1)).squeeze(-1) + translation


def camera_cv_to_gl(points_cv: Tensor) -> Tensor:
    """Convert OpenCV (+x right, +y down, +z forward) to OpenGL camera axes."""
    points_cv = _as_float_tensor(points_cv)
    if points_cv.shape[-1] != 3:
        raise ValueError("camera points must end in three coordinates")
    signs = points_cv.new_tensor((1.0, -1.0, -1.0))
    return points_cv * signs


def camera_gl_to_world(points_gl: Tensor, cam2world_gl: Tensor) -> Tensor:
    """Transform OpenGL-camera points into the simulator world frame."""
    return transform_points(points_gl, cam2world_gl)


def camera_cv_to_world(points_cv: Tensor, extrinsic_cv: Tensor) -> Tensor:
    """Transform OpenCV-camera points to world coordinates.

    ManiSkill's ``extrinsic_cv`` is the world-to-camera transform, so this operation
    explicitly inverts it.
    """
    return transform_points(points_cv, invert_rigid_transform(extrinsic_cv))
