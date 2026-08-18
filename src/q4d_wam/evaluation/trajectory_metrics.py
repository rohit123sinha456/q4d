"""Aggregate 3D trajectory accuracy and temporal-consistency metrics."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import torch
from torch import Tensor


def _validate_trajectories(prediction: Tensor, target: Tensor, initial: Tensor) -> None:
    if prediction.shape != target.shape or prediction.ndim != 4 or prediction.shape[-1] != 3:
        raise ValueError("prediction and target must have matching shape [B, Q, H, 3]")
    if initial.shape != prediction.shape[:2] + (3,):
        raise ValueError("initial must have shape [B, Q, 3]")
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise ValueError("trajectory tensors must be finite")


@dataclass
class _GroupTotals:
    ade_sum: float = 0.0
    ade_count: int = 0
    fde_sum: float = 0.0
    fde_count: int = 0

    def update(self, errors: Tensor, mask: Tensor) -> None:
        selected = errors[mask]
        final_selected = errors[..., -1][mask]
        self.ade_sum += float(selected.sum())
        self.ade_count += selected.numel()
        self.fde_sum += float(final_selected.sum())
        self.fde_count += final_selected.numel()

    def report(self) -> dict[str, float | int | None]:
        return {
            "points": self.fde_count,
            "ade_m": self.ade_sum / self.ade_count if self.ade_count else None,
            "fde_m": self.fde_sum / self.fde_count if self.fde_count else None,
        }


@dataclass
class TrajectoryMetricAccumulator:
    """Accumulate unbiased dataset-level metrics over batches."""

    horizon: int
    moving_threshold_m: float = 0.001
    compute_geometry_metrics: bool = True
    _groups: dict[str, _GroupTotals] = field(default_factory=lambda: defaultdict(_GroupTotals))
    _horizon_sum: Tensor = field(init=False)
    _horizon_count: int = 0
    _all_errors: list[Tensor] = field(default_factory=list)
    _acceleration_sum: float = 0.0
    _acceleration_count: int = 0
    _pairwise_sum: float = 0.0
    _pairwise_count: int = 0
    _same_body_sum: float = 0.0
    _same_body_count: int = 0

    def __post_init__(self) -> None:
        if self.horizon <= 0:
            raise ValueError("horizon must be positive")
        self._horizon_sum = torch.zeros(self.horizon, dtype=torch.float64)

    def update(
        self,
        prediction: Tensor,
        target: Tensor,
        initial: Tensor,
        *,
        point_groups: dict[str, Tensor] | None = None,
        body_indices: Tensor | None = None,
    ) -> None:
        prediction = prediction.detach().cpu().to(torch.float64)
        target = target.detach().cpu().to(torch.float64)
        initial = initial.detach().cpu().to(torch.float64)
        _validate_trajectories(prediction, target, initial)
        if prediction.shape[2] != self.horizon:
            raise ValueError("trajectory horizon does not match the accumulator")

        errors = torch.linalg.vector_norm(prediction - target, dim=-1)
        all_points = torch.ones(errors.shape[:2], dtype=torch.bool)
        self._groups["all"].update(errors, all_points)
        motion = torch.linalg.vector_norm(target - initial[:, :, None, :], dim=-1).amax(dim=-1)
        self._groups["moving"].update(errors, motion > self.moving_threshold_m)
        for name, mask in (point_groups or {}).items():
            mask = mask.detach().cpu().to(torch.bool)
            if mask.shape != errors.shape[:2]:
                raise ValueError(f"point group {name!r} must have shape [B, Q]")
            self._groups[name].update(errors, mask)

        self._horizon_sum += errors.sum(dim=(0, 1))
        self._horizon_count += errors.shape[0] * errors.shape[1]
        self._all_errors.append(errors.flatten().to(torch.float32))

        predicted_sequence = torch.cat((initial[:, :, None, :], prediction), dim=2)
        target_sequence = torch.cat((initial[:, :, None, :], target), dim=2)
        if predicted_sequence.shape[2] >= 3:
            acceleration_error = torch.linalg.vector_norm(
                torch.diff(predicted_sequence, n=2, dim=2)
                - torch.diff(target_sequence, n=2, dim=2),
                dim=-1,
            )
            self._acceleration_sum += float(acceleration_error.sum())
            self._acceleration_count += acceleration_error.numel()

        if self.compute_geometry_metrics:
            for batch_index in range(len(prediction)):
                self._update_pairwise(prediction[batch_index], target[batch_index])
                if body_indices is not None:
                    sample_bodies = body_indices[batch_index].detach().cpu()
                    if sample_bodies.shape != (prediction.shape[1],):
                        raise ValueError("body_indices must have shape [B, Q]")
                    for body_index in torch.unique(sample_bodies):
                        if int(body_index) < 0:
                            continue
                        body_mask = sample_bodies == body_index
                        if int(body_mask.sum()) >= 2:
                            self._update_same_body(
                                prediction[batch_index, body_mask],
                                target[batch_index, body_mask],
                            )

    def _update_pairwise(self, prediction: Tensor, target: Tensor) -> None:
        if prediction.shape[0] < 2:
            return
        for time_index in range(self.horizon):
            error = torch.abs(
                torch.pdist(prediction[:, time_index]) - torch.pdist(target[:, time_index])
            )
            self._pairwise_sum += float(error.sum())
            self._pairwise_count += error.numel()

    def _update_same_body(self, prediction: Tensor, target: Tensor) -> None:
        for time_index in range(self.horizon):
            error = torch.abs(
                torch.pdist(prediction[:, time_index]) - torch.pdist(target[:, time_index])
            )
            self._same_body_sum += float(error.sum())
            self._same_body_count += error.numel()

    def report(self) -> dict[str, object]:
        all_errors = torch.cat(self._all_errors) if self._all_errors else torch.empty(0)
        group_order = ["all", "moving", "contact", "static", "robot", "object", "goal"]
        group_order.extend(sorted(set(self._groups) - set(group_order)))
        groups = {
            name: self._groups[name].report()
            for name in group_order
            if name in self._groups
        }
        return {
            "groups": groups,
            "per_horizon_ade_m": (
                (self._horizon_sum / self._horizon_count).tolist()
                if self._horizon_count
                else []
            ),
            "p95_point_time_error_m": (
                float(torch.quantile(all_errors, 0.95)) if len(all_errors) else None
            ),
            "acceleration_error_m_per_step2": (
                self._acceleration_sum / self._acceleration_count
                if self._acceleration_count
                else None
            ),
            "pairwise_distance_error_m": (
                self._pairwise_sum / self._pairwise_count if self._pairwise_count else None
            ),
            "same_body_pairwise_distance_error_m": (
                self._same_body_sum / self._same_body_count
                if self._same_body_count
                else None
            ),
        }
