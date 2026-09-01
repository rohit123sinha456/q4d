"""Cached model adapter for a task-adapter-selected object-centroid cost."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from q4d_wam.data import NormalizationStats
from q4d_wam.models import (
    DensePointFutureModel,
    MicroQ4D,
    NoActionTrajectoryModel,
    QueryCache,
)


@dataclass(frozen=True)
class _PreparedCost:
    query_cache: QueryCache
    initial_world_m: Tensor
    score_indices: Tensor
    goal_world_m: Tensor


def object_goal_and_stability_cost(
    prediction_world_m: Tensor,
    score_indices: Tensor,
    goal_world_m: Tensor,
    *,
    settling_steps: int,
) -> tuple[Tensor, Tensor]:
    """Return final centroid distance and late-horizon centroid motion per candidate."""
    if prediction_world_m.ndim != 4 or prediction_world_m.shape[-1] != 3:
        raise ValueError("prediction_world_m must have shape [C, Q, H, 3]")
    if score_indices.ndim != 1 or score_indices.numel() == 0:
        raise ValueError("score_indices must select at least one predicted point")
    if goal_world_m.shape != (3,):
        raise ValueError("goal_world_m must have shape [3]")
    if settling_steps < 0:
        raise ValueError("settling_steps cannot be negative")
    object_centroid = prediction_world_m[:, score_indices].mean(dim=1)
    goal_cost = torch.linalg.vector_norm(
        object_centroid[:, -1] - goal_world_m[None], dim=-1
    )
    usable_steps = min(settling_steps, prediction_world_m.shape[2] - 1)
    if usable_steps == 0:
        stability_cost = torch.zeros_like(goal_cost)
    else:
        final_motion = torch.diff(
            object_centroid[:, -(usable_steps + 1) :], dim=1
        )
        stability_cost = torch.linalg.vector_norm(final_motion, dim=-1).mean(dim=-1)
    return goal_cost, stability_cost


class CachedTaskCost:
    """Encode a scene once and score candidates for a selected object and task goal."""

    def __init__(
        self,
        model: MicroQ4D | NoActionTrajectoryModel,
        normalization: NormalizationStats,
        *,
        dense_output: bool,
        action_penalty: float = 1e-4,
        settling_penalty: float = 0.0,
        settling_steps: int = 2,
        use_amp: bool = True,
    ):
        if action_penalty < 0:
            raise ValueError("action_penalty cannot be negative")
        if settling_penalty < 0:
            raise ValueError("settling_penalty cannot be negative")
        if settling_steps < 0:
            raise ValueError("settling_steps cannot be negative")
        self.model = model
        self.normalization = normalization
        self.dense_output = dense_output
        self.action_penalty = action_penalty
        self.settling_penalty = settling_penalty
        self.settling_steps = settling_steps
        self.use_amp = use_amp
        self.prepared: _PreparedCost | None = None
        self.scene_encode_count = 0

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @torch.no_grad()
    def prepare(
        self,
        scene_xyz_world_m: Tensor,
        scene_rgb: Tensor,
        object_indices: Tensor,
        goal_world_m: Tensor,
    ) -> None:
        if scene_xyz_world_m.ndim != 2 or scene_xyz_world_m.shape[-1] != 3:
            raise ValueError("scene_xyz_world_m must have shape [N, 3]")
        if scene_rgb.shape != scene_xyz_world_m.shape:
            raise ValueError("scene_rgb must match scene XYZ")
        if object_indices.ndim != 1 or object_indices.numel() == 0:
            raise ValueError("object_indices must select at least one scene point")
        if goal_world_m.shape != (3,):
            raise ValueError("goal_world_m must have shape [3]")
        device = self.device
        stats = self.normalization
        scene_world = scene_xyz_world_m.to(device)
        normalized_scene = (
            scene_world - stats.xyz_mean_m.to(device)
        ) / stats.xyz_scale_m.to(device)
        rgb = scene_rgb.to(device)
        indices = object_indices.to(device=device, dtype=torch.long)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=self.use_amp and device.type == "cuda",
        ):
            scene_cache = self.model.encode_scene(
                normalized_scene[None], rgb[None]
            )
            if self.dense_output:
                if not isinstance(self.model, DensePointFutureModel):
                    raise TypeError("dense output requires DensePointFutureModel")
                query_cache = self.model.encode_dense_queries(scene_cache)
                initial_world = scene_world
                score_indices = indices
            else:
                query_cache = self.model.encode_query_indices(
                    scene_cache, indices[None]
                )
                initial_world = scene_world[indices]
                score_indices = torch.arange(len(indices), device=device)
        self.prepared = _PreparedCost(
            query_cache,
            initial_world,
            score_indices,
            goal_world_m.to(device),
        )
        self.scene_encode_count += 1

    @torch.no_grad()
    def __call__(self, candidate_actions: Tensor) -> Tensor:
        if self.prepared is None:
            raise RuntimeError("prepare must be called once before candidate evaluation")
        device = self.device
        actions = candidate_actions.to(device)
        stats = self.normalization
        normalized_actions = (
            actions - stats.action_mean.to(device)
        ) / stats.action_scale.to(device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=self.use_amp and device.type == "cuda",
        ):
            normalized_displacement = self.model.predict_candidates(
                self.prepared.query_cache, normalized_actions[None]
            )[0]
        displacement = (
            normalized_displacement.float()
            * stats.displacement_scale_m.to(device)
            + stats.displacement_mean_m.to(device)
        )
        prediction_world = (
            self.prepared.initial_world_m[None, :, None, :] + displacement
        )
        goal_cost, stability_cost = object_goal_and_stability_cost(
            prediction_world,
            self.prepared.score_indices,
            self.prepared.goal_world_m,
            settling_steps=self.settling_steps,
        )
        action_cost = actions[..., :3].square().mean(dim=(1, 2))
        return (
            goal_cost
            + self.settling_penalty * stability_cost
            + self.action_penalty * action_cost
        )


# Backward-compatible public name used by the original PushCube reports and configs.
CachedCubeCost = CachedTaskCost
