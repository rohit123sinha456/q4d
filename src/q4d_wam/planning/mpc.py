"""Batched random-shooting and cross-entropy MPC planners."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor

CandidateCost = Callable[[Tensor], Tensor]


@dataclass(frozen=True)
class PlannerConfig:
    horizon: int
    action_dimensions: int = 7
    candidates_per_batch: int = 64
    elite_fraction: float = 0.1
    initial_std_xy: float = 0.8
    initial_std_z: float = 0.2
    minimum_std: float = 0.05
    maximum_batches: int | None = None

    def __post_init__(self) -> None:
        if self.horizon <= 0 or self.action_dimensions < 4:
            raise ValueError("horizon must be positive and actions need translation/gripper")
        if self.candidates_per_batch <= 0:
            raise ValueError("candidates_per_batch must be positive")
        if not 0 < self.elite_fraction <= 1:
            raise ValueError("elite_fraction must be in (0, 1]")
        if min(self.initial_std_xy, self.initial_std_z, self.minimum_std) <= 0:
            raise ValueError("action standard deviations must be positive")


@dataclass(frozen=True)
class PlanResult:
    action_sequence: Tensor
    predicted_cost: float
    candidates_evaluated: int
    batches_evaluated: int
    elapsed_ms: float
    budget_ms: float
    method: str

    @property
    def first_action(self) -> Tensor:
        return self.action_sequence[0]

    @property
    def budget_overrun_ms(self) -> float:
        return max(0.0, self.elapsed_ms - self.budget_ms)


def _generator(device: torch.device, seed: int) -> torch.Generator:
    return torch.Generator(device=device).manual_seed(seed)


def _actions_from_translation(translation: Tensor, action_dimensions: int) -> Tensor:
    actions = torch.zeros(
        *translation.shape[:-1],
        action_dimensions,
        device=translation.device,
        dtype=translation.dtype,
    )
    actions[..., :3] = translation.clamp(-1.0, 1.0)
    actions[..., -1] = -1.0
    return actions


def sample_random_action_sequences(
    config: PlannerConfig,
    *,
    device: torch.device,
    generator: torch.Generator,
) -> Tensor:
    """Sample smooth, bounded translation sequences in the executable action space."""
    count = config.candidates_per_batch
    base_scale = torch.tensor(
        [config.initial_std_xy, config.initial_std_xy, config.initial_std_z],
        device=device,
    )
    noise_scale = torch.tensor([0.12, 0.12, 0.04], device=device)
    base = torch.randn(count, 1, 3, device=device, generator=generator) * base_scale
    temporal_noise = (
        torch.randn(count, config.horizon, 3, device=device, generator=generator)
        * noise_scale
    )
    return _actions_from_translation(base + temporal_noise, config.action_dimensions)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _validate_costs(costs: Tensor, expected: int) -> None:
    if costs.shape != (expected,) or not torch.isfinite(costs).all():
        raise ValueError("candidate evaluator must return one finite cost per sequence")


def random_shooting(
    evaluate: CandidateCost,
    config: PlannerConfig,
    *,
    budget_ms: float,
    device: torch.device,
    seed: int,
    started_at_s: float | None = None,
) -> PlanResult:
    """Evaluate random action batches until the wall-clock deadline."""
    if budget_ms <= 0:
        raise ValueError("budget_ms must be positive")
    start = time.perf_counter() if started_at_s is None else started_at_s
    deadline = start + budget_ms / 1000.0
    generator = _generator(device, seed)
    best_sequence = None
    best_cost = torch.inf
    batches = 0
    candidates = 0
    while batches == 0 or time.perf_counter() < deadline:
        actions = sample_random_action_sequences(
            config, device=device, generator=generator
        )
        costs = evaluate(actions)
        _synchronize(device)
        _validate_costs(costs, len(actions))
        batch_cost, batch_index = costs.min(dim=0)
        if batch_cost < best_cost:
            best_cost = batch_cost
            best_sequence = actions[int(batch_index)].detach().clone()
        batches += 1
        candidates += len(actions)
        if config.maximum_batches is not None and batches >= config.maximum_batches:
            break
    assert best_sequence is not None
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return PlanResult(
        best_sequence,
        float(best_cost),
        candidates,
        batches,
        elapsed_ms,
        budget_ms,
        "random_shooting",
    )


def cem(
    evaluate: CandidateCost,
    config: PlannerConfig,
    *,
    budget_ms: float,
    device: torch.device,
    seed: int,
    started_at_s: float | None = None,
) -> PlanResult:
    """Refit a diagonal Gaussian to elite action sequences until the deadline."""
    if budget_ms <= 0:
        raise ValueError("budget_ms must be positive")
    start = time.perf_counter() if started_at_s is None else started_at_s
    deadline = start + budget_ms / 1000.0
    generator = _generator(device, seed)
    mean = torch.zeros(config.horizon, 3, device=device)
    std = torch.empty_like(mean)
    std[..., :2] = config.initial_std_xy
    std[..., 2] = config.initial_std_z
    best_sequence = None
    best_cost = torch.inf
    batches = 0
    candidates = 0
    elite_count = min(
        config.candidates_per_batch,
        max(2, round(config.candidates_per_batch * config.elite_fraction)),
    )
    while batches == 0 or time.perf_counter() < deadline:
        translations = mean[None] + std[None] * torch.randn(
            config.candidates_per_batch,
            config.horizon,
            3,
            device=device,
            generator=generator,
        )
        actions = _actions_from_translation(translations, config.action_dimensions)
        costs = evaluate(actions)
        _synchronize(device)
        _validate_costs(costs, len(actions))
        elite_indices = costs.topk(elite_count, largest=False).indices
        elite = actions[elite_indices, :, :3]
        mean = elite.mean(dim=0)
        std = elite.std(dim=0, unbiased=False).clamp_min(config.minimum_std)
        batch_cost, batch_index = costs.min(dim=0)
        if batch_cost < best_cost:
            best_cost = batch_cost
            best_sequence = actions[int(batch_index)].detach().clone()
        batches += 1
        candidates += len(actions)
        if config.maximum_batches is not None and batches >= config.maximum_batches:
            break
    assert best_sequence is not None
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return PlanResult(
        best_sequence,
        float(best_cost),
        candidates,
        batches,
        elapsed_ms,
        budget_ms,
        "cem",
    )
