"""An intentionally action-free scene-to-query trajectory predictor."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from q4d_wam.models.micro_q4d import QueryCache, SceneCache


class NoActionTrajectoryModel(nn.Module):
    """Predict normalized query displacements without accepting robot actions.

    Visible XYZ+RGB points are encoded independently, pooled into a global scene token,
    and gathered locally at the scene point nearest each query. The decoder sees the
    scene context, local visible feature, and query XYZ—but has no action input.
    """

    def __init__(self, *, horizon: int, width: int = 128):
        super().__init__()
        if horizon <= 0 or width <= 0:
            raise ValueError("horizon and width must be positive")
        self.horizon = horizon
        self.width = width
        self.scene_encoder = nn.Sequential(
            nn.Linear(6, width),
            nn.GELU(),
            nn.Linear(width, width),
            nn.GELU(),
        )
        self.query_decoder = nn.Sequential(
            nn.Linear(width * 3 + 3, width),
            nn.GELU(),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, horizon * 3),
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

    def _encode_selected_queries(
        self, scene: SceneCache, query_xyz: Tensor, point_indices: Tensor
    ) -> QueryCache:
        local_features = torch.gather(
            scene.point_features,
            dim=1,
            index=point_indices[..., None].expand(-1, -1, self.width),
        )
        context = scene.global_context[:, None, :].expand(-1, query_xyz.shape[1], -1)
        return QueryCache(torch.cat((local_features, context, query_xyz), dim=-1))

    def encode_queries(self, scene: SceneCache, query_xyz: Tensor) -> QueryCache:
        if (
            query_xyz.ndim != 3
            or query_xyz.shape[0] != scene.scene_xyz.shape[0]
            or query_xyz.shape[-1] != 3
        ):
            raise ValueError("query_xyz must have shape [B, Q, 3]")
        nearest = torch.cdist(query_xyz, scene.scene_xyz).argmin(dim=-1)
        return self._encode_selected_queries(scene, query_xyz, nearest)

    def encode_query_indices(
        self, scene: SceneCache, query_indices: Tensor
    ) -> QueryCache:
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

    def decode_queries(self, queries: QueryCache) -> Tensor:
        decoded = self.query_decoder(queries.query_features)
        return decoded.reshape(
            queries.query_features.shape[0],
            queries.query_features.shape[1],
            self.horizon,
            3,
        )

    def predict_candidates(
        self, queries: QueryCache, candidate_actions: Tensor
    ) -> Tensor:
        """Repeat one action-free prediction for every candidate action branch."""
        if candidate_actions.ndim != 4:
            raise ValueError("candidate_actions must have shape [B, K, H, A]")
        if candidate_actions.shape[0] != queries.query_features.shape[0]:
            raise ValueError("query and candidate batch dimensions must match")
        prediction = self.decode_queries(queries)
        return prediction[:, None].expand(-1, candidate_actions.shape[1], -1, -1, -1)

    def forward(self, scene_xyz: Tensor, scene_rgb: Tensor, query_xyz: Tensor) -> Tensor:
        scene = self.encode_scene(scene_xyz, scene_rgb)
        queries = self.encode_queries(scene, query_xyz)
        return self.decode_queries(queries)
