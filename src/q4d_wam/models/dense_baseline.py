"""Parameter-matched dense trajectory baseline."""

from __future__ import annotations

import torch
from torch import Tensor

from q4d_wam.models.micro_q4d import MicroQ4D, QueryCache, SceneCache


def dense_query_set_is_complete(scene_xyz: Tensor, query_indices: Tensor) -> bool:
    """Return whether query indices cover every scene point exactly once."""
    if scene_xyz.ndim != 2 or scene_xyz.shape[-1] != 3 or query_indices.ndim != 1:
        return False
    scene_points = scene_xyz.shape[0]
    return len(query_indices) == scene_points and bool(
        query_indices.sort().values.equal(
            query_indices.new_tensor(range(scene_points))
        )
    )


class DensePointFutureModel(MicroQ4D):
    """Predict a future trajectory for every visible scene point.

    This is intentionally the same network as :class:`MicroQ4D`. The controlled
    difference is the output protocol: dense inference always uses all scene points as
    queries, while micro-Q4D permits an arbitrary sparse query set.
    """

    def encode_dense_queries(self, scene: SceneCache) -> QueryCache:
        """Encode every visible point from an already cached scene."""
        batch_size, scene_points, _ = scene.scene_xyz.shape
        indices = torch.arange(scene_points, device=scene.scene_xyz.device)[None].expand(
            batch_size, -1
        )
        return self.encode_query_indices(scene, indices)

    def forward(self, scene_xyz: Tensor, scene_rgb: Tensor, actions: Tensor) -> Tensor:
        """Return normalized displacements shaped ``[B, N, H, 3]``."""
        scene = self.encode_scene(scene_xyz, scene_rgb)
        queries = self.encode_dense_queries(scene)
        return self.decode(queries, actions)
