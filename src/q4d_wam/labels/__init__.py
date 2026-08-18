"""Privileged simulator-label construction utilities."""

from q4d_wam.labels.rigid_tracks import (
    CATEGORY_GOAL,
    CATEGORY_OBJECT,
    CATEGORY_ROBOT,
    CATEGORY_STATIC,
    CATEGORY_UNKNOWN,
    AttachmentBatch,
    attach_points_to_bodies,
    farthest_point_indices,
    reconstruct_rigid_tracks,
    stratified_point_indices,
)

__all__ = [
    "CATEGORY_GOAL",
    "CATEGORY_OBJECT",
    "CATEGORY_ROBOT",
    "CATEGORY_STATIC",
    "CATEGORY_UNKNOWN",
    "AttachmentBatch",
    "attach_points_to_bodies",
    "farthest_point_indices",
    "reconstruct_rigid_tracks",
    "stratified_point_indices",
]

