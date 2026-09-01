import torch

from q4d_wam.data import NormalizationStats
from q4d_wam.models import DensePointFutureModel, MicroQ4D
from q4d_wam.planning import (
    DEFAULT_GRIPPER_SCHEDULES,
    CachedCubeCost,
    build_gripper_schedule_library,
    object_goal_and_stability_cost,
    sample_random_action_sequences,
    validate_action_sequences,
)
from q4d_wam.planning.mpc import PlannerConfig, cem, random_shooting


def _identity_stats(action_dimensions: int = 7) -> NormalizationStats:
    return NormalizationStats(
        xyz_mean_m=torch.zeros(3),
        xyz_scale_m=torch.ones(3),
        action_mean=torch.zeros(action_dimensions),
        action_scale=torch.ones(action_dimensions),
        displacement_mean_m=torch.zeros(3),
        displacement_scale_m=torch.ones(3),
        constant_action_channels=(),
        source_files=(),
        epsilon=1e-6,
    )


def _quadratic_cost(actions: torch.Tensor) -> torch.Tensor:
    target = torch.tensor([0.7, -0.25, 0.0])
    return (actions[..., :3] - target).square().mean(dim=(1, 2))


def test_random_shooting_batches_bounded_executable_actions() -> None:
    config = PlannerConfig(horizon=4, candidates_per_batch=16, maximum_batches=3)

    result = random_shooting(
        _quadratic_cost,
        config,
        budget_ms=1000,
        device=torch.device("cpu"),
        seed=3,
    )

    assert result.candidates_evaluated == 48
    assert result.batches_evaluated == 3
    assert result.first_action.shape == (7,)
    assert torch.all(result.action_sequence[..., :3].abs() <= 1)
    assert torch.count_nonzero(result.action_sequence[..., 3:6]) == 0
    assert torch.all(result.action_sequence[..., -1] == -1)
    assert result.gripper_schedule == "translation_only_hold_closed"


def test_h8_gripper_schedule_library_has_declared_semantics() -> None:
    library = build_gripper_schedule_library(8)

    torch.testing.assert_close(
        library,
        torch.tensor(
            [
                [-1, -1, -1, -1, -1, -1, -1, -1],
                [1, 1, 1, 1, 1, 1, 1, 1],
                [-1, -1, -1, -1, 1, 1, 1, 1],
                [-1, -1, -1, -1, -1, -1, 1, 1],
                [1, 1, 1, 1, -1, -1, -1, -1],
            ],
            dtype=torch.float32,
        ),
    )


def test_gripper_aware_sampler_is_valid_balanced_and_fixed_seed_reproducible() -> None:
    config = PlannerConfig(
        horizon=8,
        candidates_per_batch=16,
        action_space="gripper_schedules",
    )

    first = sample_random_action_sequences(
        config,
        device=torch.device("cpu"),
        generator=torch.Generator().manual_seed(91),
    )
    repeated = sample_random_action_sequences(
        config,
        device=torch.device("cpu"),
        generator=torch.Generator().manual_seed(91),
    )
    different = sample_random_action_sequences(
        config,
        device=torch.device("cpu"),
        generator=torch.Generator().manual_seed(92),
    )

    validate_action_sequences(first, config)
    torch.testing.assert_close(first, repeated, rtol=0, atol=0)
    assert not torch.equal(first, different)
    sampled_schedules = {tuple(row.tolist()) for row in first[..., -1]}
    expected_schedules = {
        tuple(row.tolist()) for row in build_gripper_schedule_library(8)
    }
    assert sampled_schedules == expected_schedules
    assert first.shape == (16, 8, 7)


def test_translation_only_sampler_matches_legacy_generation_exactly() -> None:
    config = PlannerConfig(horizon=4, candidates_per_batch=6)
    actual = sample_random_action_sequences(
        config,
        device=torch.device("cpu"),
        generator=torch.Generator().manual_seed(7),
    )
    legacy_generator = torch.Generator().manual_seed(7)
    base = torch.randn(6, 1, 3, generator=legacy_generator) * torch.tensor(
        [0.8, 0.8, 0.2]
    )
    noise = torch.randn(6, 4, 3, generator=legacy_generator) * torch.tensor(
        [0.12, 0.12, 0.04]
    )
    expected = torch.zeros(6, 4, 7)
    expected[..., :3] = (base + noise).clamp(-1.0, 1.0)
    expected[..., -1] = -1.0

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_gripper_aware_random_shooting_reproduces_selected_candidate() -> None:
    config = PlannerConfig(
        horizon=8,
        candidates_per_batch=20,
        maximum_batches=2,
        action_space="gripper_schedules",
    )
    first = random_shooting(
        _quadratic_cost,
        config,
        budget_ms=1000,
        device=torch.device("cpu"),
        seed=17,
    )
    repeated = random_shooting(
        _quadratic_cost,
        config,
        budget_ms=1000,
        device=torch.device("cpu"),
        seed=17,
    )

    torch.testing.assert_close(first.action_sequence, repeated.action_sequence)
    assert first.predicted_cost == repeated.predicted_cost
    assert first.gripper_schedule == repeated.gripper_schedule
    assert first.gripper_schedule in DEFAULT_GRIPPER_SCHEDULES


def test_cem_refinement_does_not_worsen_best_sample() -> None:
    first_batch = PlannerConfig(horizon=4, candidates_per_batch=32, maximum_batches=1)
    refined = PlannerConfig(horizon=4, candidates_per_batch=32, maximum_batches=6)

    initial = cem(
        _quadratic_cost,
        first_batch,
        budget_ms=1000,
        device=torch.device("cpu"),
        seed=4,
    )
    final = cem(
        _quadratic_cost,
        refined,
        budget_ms=1000,
        device=torch.device("cpu"),
        seed=4,
    )

    assert final.predicted_cost <= initial.predicted_cost
    assert final.candidates_evaluated == 192


def test_gripper_aware_cem_returns_a_library_schedule() -> None:
    config = PlannerConfig(
        horizon=4,
        candidates_per_batch=20,
        maximum_batches=3,
        action_space="gripper_schedules",
    )

    result = cem(
        _quadratic_cost,
        config,
        budget_ms=1000,
        device=torch.device("cpu"),
        seed=23,
    )

    assert result.gripper_schedule in DEFAULT_GRIPPER_SCHEDULES
    expected = build_gripper_schedule_library(
        config.horizon, config.gripper_schedules
    )
    assert any(torch.equal(result.action_sequence[:, -1], row) for row in expected)


def test_final_state_stability_cost_penalizes_unsettled_placement() -> None:
    predictions = torch.zeros(2, 1, 4, 3)
    predictions[1, 0, -3:, 0] = torch.tensor([-0.2, -0.1, 0.0])

    goal, stability = object_goal_and_stability_cost(
        predictions,
        torch.tensor([0]),
        torch.zeros(3),
        settling_steps=2,
    )

    torch.testing.assert_close(goal, torch.zeros(2))
    assert stability[0] == 0
    assert stability[1] == torch.tensor(0.1)


def test_sparse_and_dense_adapters_share_cube_centroid_cost() -> None:
    torch.manual_seed(5)
    sparse_model = MicroQ4D(action_dimensions=7, horizon=2, width=16).eval()
    dense_model = DensePointFutureModel(
        action_dimensions=7, horizon=2, width=16
    ).eval()
    dense_model.load_state_dict(sparse_model.state_dict())
    stats = _identity_stats()
    scene_xyz = torch.randn(10, 3)
    scene_rgb = torch.rand(10, 3)
    object_indices = torch.tensor([1, 4, 7])
    goal = torch.tensor([0.2, -0.1, 0.0])
    actions = torch.randn(6, 2, 7).clamp(-1, 1)
    sparse = CachedCubeCost(
        sparse_model,
        stats,
        dense_output=False,
        settling_penalty=1.0,
        settling_steps=1,
        use_amp=False,
    )
    dense = CachedCubeCost(
        dense_model,
        stats,
        dense_output=True,
        settling_penalty=1.0,
        settling_steps=1,
        use_amp=False,
    )

    sparse.prepare(scene_xyz, scene_rgb, object_indices, goal)
    dense.prepare(scene_xyz, scene_rgb, object_indices, goal)
    sparse_cost = sparse(actions)
    dense_cost = dense(actions)
    sparse(actions)

    assert sparse.scene_encode_count == 1
    assert dense.scene_encode_count == 1
    torch.testing.assert_close(sparse_cost, dense_cost)
