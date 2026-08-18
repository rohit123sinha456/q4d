"""Attach visible points to rigid simulator bodies and reconstruct persistent tracks."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from q4d_wam.geometry import invert_rigid_transform, transform_points

CATEGORY_STATIC = 0
CATEGORY_ROBOT = 1
CATEGORY_OBJECT = 2
CATEGORY_GOAL = 3
CATEGORY_UNKNOWN = 4


@dataclass(frozen=True)
class AttachmentBatch:
    """Privileged attachment data used only to construct trajectory targets."""

    body_indices: Tensor
    local_xyz_m: Tensor
    known_body: Tensor


def attach_points_to_bodies(
    points_world_m: Tensor,
    point_segmentation_ids: Tensor,
    body_segmentation_ids: Tensor,
    body_poses_world: Tensor,
) -> AttachmentBatch:
    """Attach world points to bodies via the renderer's actor segmentation IDs.

    Unknown IDs are assigned body index ``-1`` and treated as world-fixed points.
    """
    if points_world_m.ndim != 2 or points_world_m.shape[-1] != 3:
        raise ValueError("points_world_m must have shape [N, 3]")
    if point_segmentation_ids.shape != (len(points_world_m),):
        raise ValueError("point_segmentation_ids must have shape [N]")
    if body_poses_world.shape != (len(body_segmentation_ids), 4, 4):
        raise ValueError("body_poses_world must have shape [B, 4, 4]")
    if len(body_segmentation_ids) == 0:
        return AttachmentBatch(
            body_indices=torch.full_like(point_segmentation_ids, -1, dtype=torch.long),
            local_xyz_m=points_world_m.clone(),
            known_body=torch.zeros_like(point_segmentation_ids, dtype=torch.bool),
        )

    matches = point_segmentation_ids[:, None] == body_segmentation_ids[None, :]
    known_body = matches.any(dim=1)
    body_indices = matches.to(torch.int64).argmax(dim=1)
    selected_poses = body_poses_world[body_indices]
    local_xyz = transform_points(points_world_m, invert_rigid_transform(selected_poses))
    local_xyz = torch.where(known_body[:, None], local_xyz, points_world_m)
    body_indices = torch.where(known_body, body_indices, -torch.ones_like(body_indices))
    return AttachmentBatch(body_indices, local_xyz, known_body)


def reconstruct_rigid_tracks(
    local_xyz_m: Tensor,
    body_indices: Tensor,
    body_pose_sequence_world: Tensor,
) -> Tensor:
    """Transform body-local points through a body-pose sequence.

    Args:
        local_xyz_m: ``[N, 3]`` point coordinates in their attached body frames.
        body_indices: ``[N]`` indices into the body dimension, or ``-1`` for world-fixed.
        body_pose_sequence_world: ``[T, B, 4, 4]`` body-to-world transforms.

    Returns:
        Persistent world trajectories shaped ``[N, T, 3]``.
    """
    if local_xyz_m.ndim != 2 or local_xyz_m.shape[-1] != 3:
        raise ValueError("local_xyz_m must have shape [N, 3]")
    if body_indices.shape != (len(local_xyz_m),):
        raise ValueError("body_indices must have shape [N]")
    if body_pose_sequence_world.ndim != 4 or body_pose_sequence_world.shape[-2:] != (
        4,
        4,
    ):
        raise ValueError("body_pose_sequence_world must have shape [T, B, 4, 4]")
    if body_pose_sequence_world.shape[1] == 0 and torch.any(body_indices >= 0):
        raise ValueError("tracks reference bodies but the pose sequence has no bodies")
    if torch.any(body_indices >= body_pose_sequence_world.shape[1]):
        raise IndexError("body index exceeds the pose sequence body dimension")

    unknown = body_indices < 0
    safe_indices = body_indices.clamp_min(0)
    if body_pose_sequence_world.shape[1] == 0:
        time_steps = body_pose_sequence_world.shape[0]
        return local_xyz_m[:, None, :].expand(-1, time_steps, -1).clone()

    selected = body_pose_sequence_world[:, safe_indices]
    rotation = selected[..., :3, :3]
    translation = selected[..., :3, 3]
    tracks_time_first = torch.einsum("tnij,nj->tni", rotation, local_xyz_m) + translation
    tracks = tracks_time_first.transpose(0, 1)
    tracks[unknown] = local_xyz_m[unknown, None, :]
    return tracks


def farthest_point_indices(points: Tensor, count: int) -> Tensor:
    """Deterministic farthest-feature sampling, starting farthest from the centroid."""
    if points.ndim != 2 or points.shape[-1] == 0:
        raise ValueError("points must have shape [N, D] with D greater than zero")
    if count < 0:
        raise ValueError("count must be non-negative")
    if count == 0 or len(points) == 0:
        return torch.empty(0, dtype=torch.long, device=points.device)
    if count >= len(points):
        return torch.arange(len(points), device=points.device)

    selected = torch.empty(count, dtype=torch.long, device=points.device)
    centroid = points.mean(dim=0)
    selected[0] = torch.linalg.vector_norm(points - centroid, dim=-1).argmax()
    minimum_distance = torch.full((len(points),), torch.inf, device=points.device)
    for index in range(1, count):
        latest = points[selected[index - 1]]
        distance = torch.sum((points - latest) ** 2, dim=-1)
        minimum_distance = torch.minimum(minimum_distance, distance)
        selected[index] = minimum_distance.argmax()
    return selected


def stratified_point_indices(
    points: Tensor,
    categories: Tensor,
    total_count: int,
    quotas: dict[int, int],
) -> Tensor:
    """Sample category quotas first, then fill unused capacity from remaining points."""
    if categories.shape != (len(points),):
        raise ValueError("categories must have shape [N]")
    if total_count <= 0:
        raise ValueError("total_count must be positive")
    total_count = min(total_count, len(points))
    chosen_parts: list[Tensor] = []
    chosen_mask = torch.zeros(len(points), dtype=torch.bool, device=points.device)

    for category, quota in quotas.items():
        candidates = torch.nonzero(categories == category, as_tuple=False).squeeze(-1)
        local_indices = farthest_point_indices(points[candidates], min(quota, len(candidates)))
        chosen = candidates[local_indices]
        chosen_parts.append(chosen)
        chosen_mask[chosen] = True

    chosen_count = sum(len(part) for part in chosen_parts)
    remaining_count = total_count - chosen_count
    if remaining_count > 0:
        remaining = torch.nonzero(~chosen_mask, as_tuple=False).squeeze(-1)
        local_indices = farthest_point_indices(
            points[remaining], min(remaining_count, len(remaining))
        )
        chosen_parts.append(remaining[local_indices])

    result = torch.cat(chosen_parts) if chosen_parts else torch.empty(0, dtype=torch.long)
    return result[:total_count]
