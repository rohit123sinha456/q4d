"""A small action-conditioned, cacheable query trajectory model."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class SceneCache:
    """Action-independent features computed once for an observed scene."""

    scene_xyz: Tensor
    point_features: Tensor
    global_context: Tensor


@dataclass(frozen=True)
class QueryCache:
    """Action-independent features for a chosen set of scene queries."""

    query_features: Tensor


class MicroQ4D(nn.Module):
    """Predict future query displacements conditioned on executable action chunks.

    The scene and query paths mirror ``NoActionTrajectoryModel``. A GRU encodes the
    normalized action prefix at each future step, and a small decoder fuses it with each
    cached query. Candidate actions can therefore share both scene and query encoding.
    """

    def __init__(self, *, action_dimensions: int, horizon: int, width: int = 128):
        super().__init__()
        if action_dimensions <= 0 or horizon <= 0 or width <= 0:
            raise ValueError("action dimensions, horizon, and width must be positive")
        self.action_dimensions = action_dimensions
        self.horizon = horizon
        self.width = width
        self.scene_encoder = nn.Sequential(
            nn.Linear(6, width),
            nn.GELU(),
            nn.Linear(width, width),
            nn.GELU(),
        )
        self.query_encoder = nn.Sequential(
            nn.Linear(width * 3 + 3, width),
            nn.GELU(),
            nn.Linear(width, width),
            nn.GELU(),
        )
        self.action_input = nn.Sequential(nn.Linear(action_dimensions, width), nn.GELU())
        self.action_encoder = nn.GRU(width, width, batch_first=True)
        self.time_embedding = nn.Parameter(torch.zeros(horizon, width))
        nn.init.normal_(self.time_embedding, std=0.02)
        self.trajectory_decoder = nn.Sequential(
            nn.Linear(width * 3, width),
            nn.GELU(),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, 3),
        )

    def encode_scene(self, scene_xyz: Tensor, scene_rgb: Tensor) -> SceneCache:
        if scene_xyz.ndim != 3 or scene_xyz.shape[-1] != 3:
            raise ValueError("scene_xyz must have shape [B, N, 3]")
        if scene_rgb.shape != scene_xyz.shape:
            raise ValueError("scene_rgb must match scene_xyz")
        point_features = self.scene_encoder(torch.cat((scene_xyz, scene_rgb), dim=-1))
        global_context = torch.cat(
            (point_features.mean(dim=1), point_features.amax(dim=1)), dim=-1
        )
        return SceneCache(scene_xyz, point_features, global_context)

    def encode_queries(self, scene: SceneCache, query_xyz: Tensor) -> QueryCache:
        if (
            query_xyz.ndim != 3
            or query_xyz.shape[0] != scene.scene_xyz.shape[0]
            or query_xyz.shape[-1] != 3
        ):
            raise ValueError("query_xyz must have shape [B, Q, 3]")
        nearest = torch.cdist(query_xyz, scene.scene_xyz).argmin(dim=-1)
        return self._encode_selected_queries(scene, query_xyz, nearest)

    def _encode_selected_queries(
        self, scene: SceneCache, query_xyz: Tensor, point_indices: Tensor
    ) -> QueryCache:
        local_features = torch.gather(
            scene.point_features,
            dim=1,
            index=point_indices[..., None].expand(-1, -1, self.width),
        )
        context = scene.global_context[:, None, :].expand(-1, query_xyz.shape[1], -1)
        query_features = self.query_encoder(
            torch.cat((local_features, context, query_xyz), dim=-1)
        )
        return QueryCache(query_features)

    def encode_query_indices(
        self, scene: SceneCache, query_indices: Tensor
    ) -> QueryCache:
        """Encode queries known to be exact indices into the cached scene."""
        if (
            query_indices.ndim != 2
            or query_indices.shape[0] != scene.scene_xyz.shape[0]
            or query_indices.numel() == 0
        ):
            raise ValueError("query_indices must have non-empty shape [B, Q]")
        if query_indices.dtype not in (torch.int32, torch.int64):
            raise ValueError("query_indices must have an integer dtype")
        if (
            int(query_indices.min()) < 0
            or int(query_indices.max()) >= scene.scene_xyz.shape[1]
        ):
            raise IndexError("query index is outside the cached scene")
        query_xyz = torch.gather(
            scene.scene_xyz,
            dim=1,
            index=query_indices[..., None].expand(-1, -1, 3),
        )
        return self._encode_selected_queries(
            scene, query_xyz, query_indices.to(torch.long)
        )

    def encode_actions(self, actions: Tensor) -> Tensor:
        if (
            actions.ndim != 3
            or actions.shape[1] != self.horizon
            or actions.shape[2] != self.action_dimensions
        ):
            raise ValueError(
                f"actions must have shape [B, {self.horizon}, {self.action_dimensions}]"
            )
        action_features, _ = self.action_encoder(self.action_input(actions))
        return action_features + self.time_embedding[None]

    def decode(self, queries: QueryCache, actions: Tensor) -> Tensor:
        action_features = self.encode_actions(actions)
        if queries.query_features.shape[0] != action_features.shape[0]:
            raise ValueError("query and action batch dimensions must match")
        query_features = queries.query_features[:, :, None, :].expand(
            -1, -1, self.horizon, -1
        )
        action_features = action_features[:, None, :, :].expand(
            -1, queries.query_features.shape[1], -1, -1
        )
        fused = torch.cat(
            (query_features, action_features, query_features * action_features), dim=-1
        )
        return self.trajectory_decoder(fused)

    def predict_candidates(self, queries: QueryCache, candidate_actions: Tensor) -> Tensor:
        """Decode ``K`` action branches while reusing one query cache.

        Args:
            queries: Cached query features shaped internally as ``[B, Q, C]``.
            candidate_actions: Normalized actions with shape ``[B, K, H, A]``.

        Returns:
            Normalized displacements with shape ``[B, K, Q, H, 3]``.
        """
        if candidate_actions.ndim != 4:
            raise ValueError("candidate_actions must have shape [B, K, H, A]")
        batch_size, candidates, horizon, action_dimensions = candidate_actions.shape
        if batch_size != queries.query_features.shape[0]:
            raise ValueError("query and candidate batch dimensions must match")
        expanded_queries = QueryCache(
            queries.query_features[:, None]
            .expand(-1, candidates, -1, -1)
            .reshape(batch_size * candidates, queries.query_features.shape[1], self.width)
        )
        flat_actions = candidate_actions.reshape(
            batch_size * candidates, horizon, action_dimensions
        )
        prediction = self.decode(expanded_queries, flat_actions)
        return prediction.reshape(
            batch_size, candidates, queries.query_features.shape[1], self.horizon, 3
        )

    def forward(
        self, scene_xyz: Tensor, scene_rgb: Tensor, actions: Tensor, query_xyz: Tensor
    ) -> Tensor:
        scene = self.encode_scene(scene_xyz, scene_rgb)
        queries = self.encode_queries(scene, query_xyz)
        return self.decode(queries, actions)
