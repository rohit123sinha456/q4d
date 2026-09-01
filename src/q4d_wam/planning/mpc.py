"""Batched random-shooting and cross-entropy MPC planners."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor

CandidateCost = Callable[[Tensor], Tensor]

GRIPPER_CLOSED = -1.0
GRIPPER_OPEN = 1.0
DEFAULT_GRIPPER_SCHEDULES = (
    "hold_closed",
    "hold_open",
    "closed_to_open_halfway",
    "closed_to_open_final_quarter",
    "open_to_closed_halfway",
)
ACTION_SPACES = ("translation_only", "gripper_schedules")


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
    action_space: str = "translation_only"
    gripper_schedules: tuple[str, ...] = DEFAULT_GRIPPER_SCHEDULES
    minimum_schedule_probability: float = 0.05

    def __post_init__(self) -> None:
        if self.horizon <= 0 or self.action_dimensions != 7:
            raise ValueError("horizon must be positive and executable actions must be 7D")
        if self.candidates_per_batch <= 0:
            raise ValueError("candidates_per_batch must be positive")
        if not 0 < self.elite_fraction <= 1:
            raise ValueError("elite_fraction must be in (0, 1]")
        if min(self.initial_std_xy, self.initial_std_z, self.minimum_std) <= 0:
            raise ValueError("action standard deviations must be positive")
        if self.action_space not in ACTION_SPACES:
            raise ValueError(f"action_space must be one of {ACTION_SPACES}")
        if not 0 <= self.minimum_schedule_probability < 1:
            raise ValueError("minimum_schedule_probability must be in [0, 1)")
        if self.action_space == "gripper_schedules":
            if not self.gripper_schedules:
                raise ValueError("gripper-aware planning needs at least one schedule")
            if len(set(self.gripper_schedules)) != len(self.gripper_schedules):
                raise ValueError("gripper schedule names must be unique")
            unknown = set(self.gripper_schedules) - set(DEFAULT_GRIPPER_SCHEDULES)
            if unknown:
                raise ValueError(f"unknown gripper schedules: {sorted(unknown)}")
            if self.candidates_per_batch < len(self.gripper_schedules):
                raise ValueError(
                    "candidates_per_batch must cover every configured gripper schedule"
                )
            maximum_floor = 1.0 / len(self.gripper_schedules)
            if self.minimum_schedule_probability > maximum_floor:
                raise ValueError(
                    "minimum_schedule_probability cannot exceed the uniform probability"
                )


@dataclass(frozen=True)
class PlanResult:
    action_sequence: Tensor
    predicted_cost: float
    candidates_evaluated: int
    batches_evaluated: int
    elapsed_ms: float
    budget_ms: float
    method: str
    gripper_schedule: str

    @property
    def first_action(self) -> Tensor:
        return self.action_sequence[0]

    @property
    def budget_overrun_ms(self) -> float:
        return max(0.0, self.elapsed_ms - self.budget_ms)


def _generator(device: torch.device, seed: int) -> torch.Generator:
    return torch.Generator(device=device).manual_seed(seed)


def build_gripper_schedule_library(
    horizon: int,
    schedule_names: tuple[str, ...] = DEFAULT_GRIPPER_SCHEDULES,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Build named, executable discrete gripper commands with shape [S, H]."""
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    unknown = set(schedule_names) - set(DEFAULT_GRIPPER_SCHEDULES)
    if unknown:
        raise ValueError(f"unknown gripper schedules: {sorted(unknown)}")
    schedules = []
    halfway = horizon // 2
    final_quarter = max(1, (horizon + 3) // 4)
    for name in schedule_names:
        schedule = torch.full(
            (horizon,), GRIPPER_CLOSED, device=device, dtype=dtype
        )
        if name == "hold_open":
            schedule.fill_(GRIPPER_OPEN)
        elif name == "closed_to_open_halfway":
            schedule[halfway:] = GRIPPER_OPEN
        elif name == "closed_to_open_final_quarter":
            schedule[horizon - final_quarter :] = GRIPPER_OPEN
        elif name == "open_to_closed_halfway":
            schedule[:halfway] = GRIPPER_OPEN
        schedules.append(schedule)
    if not schedules:
        return torch.empty(0, horizon, device=device, dtype=dtype)
    return torch.stack(schedules)


def _sample_schedule_indices(
    count: int,
    schedule_count: int,
    *,
    device: torch.device,
    generator: torch.Generator,
    probabilities: Tensor | None = None,
) -> Tensor:
    """Sample a balanced categorical batch while guaranteeing full library coverage."""
    if count < schedule_count:
        raise ValueError("candidate count cannot cover the schedule library")
    covered = torch.arange(schedule_count, device=device)
    remainder_count = count - schedule_count
    if remainder_count:
        if probabilities is None:
            remainder = torch.randint(
                schedule_count,
                (remainder_count,),
                device=device,
                generator=generator,
            )
        else:
            remainder = torch.multinomial(
                probabilities,
                remainder_count,
                replacement=True,
                generator=generator,
            )
        indices = torch.cat((covered, remainder))
    else:
        indices = covered
    permutation = torch.randperm(count, device=device, generator=generator)
    return indices[permutation]


def _actions_from_translation(
    translation: Tensor,
    config: PlannerConfig,
    *,
    generator: torch.Generator,
    schedule_probabilities: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    actions = torch.zeros(
        *translation.shape[:-1],
        config.action_dimensions,
        device=translation.device,
        dtype=translation.dtype,
    )
    actions[..., :3] = translation.clamp(-1.0, 1.0)
    if config.action_space == "translation_only":
        actions[..., -1] = GRIPPER_CLOSED
        schedule_indices = torch.zeros(
            translation.shape[0], device=translation.device, dtype=torch.long
        )
    else:
        library = build_gripper_schedule_library(
            config.horizon,
            config.gripper_schedules,
            device=translation.device,
            dtype=translation.dtype,
        )
        schedule_indices = _sample_schedule_indices(
            translation.shape[0],
            len(library),
            device=translation.device,
            generator=generator,
            probabilities=schedule_probabilities,
        )
        actions[..., -1] = library[schedule_indices]
    validate_action_sequences(actions, config)
    return actions, schedule_indices


def validate_action_sequences(actions: Tensor, config: PlannerConfig) -> None:
    """Validate finite, bounded 7D delta-pose actions and discrete gripper schedules."""
    expected = (config.candidates_per_batch, config.horizon, 7)
    if actions.shape != expected:
        raise ValueError(f"candidate actions must have shape {expected}")
    if not torch.isfinite(actions).all():
        raise ValueError("candidate actions must be finite")
    if torch.any(actions < -1.0) or torch.any(actions > 1.0):
        raise ValueError("candidate actions must be within the executable [-1, 1] bounds")
    if torch.count_nonzero(actions[..., 3:6]):
        raise ValueError("the sampler must leave delta-rotation channels at zero")
    gripper = actions[..., -1]
    if config.action_space == "translation_only":
        if not torch.all(gripper == GRIPPER_CLOSED):
            raise ValueError("translation-only actions must hold the gripper closed")
        return
    library = build_gripper_schedule_library(
        config.horizon,
        config.gripper_schedules,
        device=actions.device,
        dtype=actions.dtype,
    )
    valid = (gripper[:, None, :] == library[None, :, :]).all(dim=-1).any(dim=-1)
    if not torch.all(valid):
        raise ValueError("candidate gripper commands must match a configured schedule")


def identify_gripper_schedule(actions: Tensor, config: PlannerConfig) -> str:
    """Return the configured name for one action sequence's gripper trajectory."""
    if actions.shape != (config.horizon, 7):
        raise ValueError("one action sequence must have shape [H, 7]")
    if config.action_space == "translation_only":
        return "translation_only_hold_closed"
    library = build_gripper_schedule_library(
        config.horizon,
        config.gripper_schedules,
        device=actions.device,
        dtype=actions.dtype,
    )
    matches = (actions[:, -1][None, :] == library).all(dim=-1)
    match = torch.nonzero(matches, as_tuple=False).flatten()
    if len(match) != 1:
        raise ValueError("action sequence does not match exactly one gripper schedule")
    return config.gripper_schedules[int(match[0])]


def _sample_random_candidates(
    config: PlannerConfig,
    *,
    device: torch.device,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor]:
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
    return _actions_from_translation(
        base + temporal_noise, config, generator=generator
    )


def sample_random_action_sequences(
    config: PlannerConfig,
    *,
    device: torch.device,
    generator: torch.Generator,
) -> Tensor:
    """Jointly sample smooth translations and a balanced discrete gripper schedule."""
    actions, _ = _sample_random_candidates(
        config, device=device, generator=generator
    )
    return actions


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
    best_schedule = None
    best_cost = torch.inf
    batches = 0
    candidates = 0
    while batches == 0 or time.perf_counter() < deadline:
        actions, schedule_indices = _sample_random_candidates(
            config, device=device, generator=generator
        )
        costs = evaluate(actions)
        _synchronize(device)
        _validate_costs(costs, len(actions))
        batch_cost, batch_index = costs.min(dim=0)
        if batch_cost < best_cost:
            best_cost = batch_cost
            best_sequence = actions[int(batch_index)].detach().clone()
            if config.action_space == "translation_only":
                best_schedule = "translation_only_hold_closed"
            else:
                best_schedule = config.gripper_schedules[
                    int(schedule_indices[int(batch_index)])
                ]
        batches += 1
        candidates += len(actions)
        if config.maximum_batches is not None and batches >= config.maximum_batches:
            break
    assert best_sequence is not None
    assert best_schedule is not None
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return PlanResult(
        best_sequence,
        float(best_cost),
        candidates,
        batches,
        elapsed_ms,
        budget_ms,
        "random_shooting",
        best_schedule,
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
    best_schedule = None
    best_cost = torch.inf
    batches = 0
    candidates = 0
    elite_count = min(
        config.candidates_per_batch,
        max(2, round(config.candidates_per_batch * config.elite_fraction)),
    )
    schedule_probabilities = None
    if config.action_space == "gripper_schedules":
        schedule_probabilities = torch.full(
            (len(config.gripper_schedules),),
            1.0 / len(config.gripper_schedules),
            device=device,
        )
    while batches == 0 or time.perf_counter() < deadline:
        translations = mean[None] + std[None] * torch.randn(
            config.candidates_per_batch,
            config.horizon,
            3,
            device=device,
            generator=generator,
        )
        actions, schedule_indices = _actions_from_translation(
            translations,
            config,
            generator=generator,
            schedule_probabilities=schedule_probabilities,
        )
        costs = evaluate(actions)
        _synchronize(device)
        _validate_costs(costs, len(actions))
        elite_indices = costs.topk(elite_count, largest=False).indices
        elite = actions[elite_indices, :, :3]
        mean = elite.mean(dim=0)
        std = elite.std(dim=0, unbiased=False).clamp_min(config.minimum_std)
        if schedule_probabilities is not None:
            counts = torch.bincount(
                schedule_indices[elite_indices],
                minlength=len(config.gripper_schedules),
            ).to(torch.float32)
            empirical = counts / counts.sum()
            floor = config.minimum_schedule_probability
            schedule_probabilities = empirical * (
                1.0 - floor * len(config.gripper_schedules)
            ) + floor
        batch_cost, batch_index = costs.min(dim=0)
        if batch_cost < best_cost:
            best_cost = batch_cost
            best_sequence = actions[int(batch_index)].detach().clone()
            if config.action_space == "translation_only":
                best_schedule = "translation_only_hold_closed"
            else:
                best_schedule = config.gripper_schedules[
                    int(schedule_indices[int(batch_index)])
                ]
        batches += 1
        candidates += len(actions)
        if config.maximum_batches is not None and batches >= config.maximum_batches:
            break
    assert best_sequence is not None
    assert best_schedule is not None
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return PlanResult(
        best_sequence,
        float(best_cost),
        candidates,
        batches,
        elapsed_ms,
        budget_ms,
        "cem",
        best_schedule,
    )
