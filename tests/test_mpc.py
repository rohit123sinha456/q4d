import torch

from q4d_wam.data import NormalizationStats
from q4d_wam.models import DensePointFutureModel, MicroQ4D
from q4d_wam.planning import CachedCubeCost
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
        sparse_model, stats, dense_output=False, use_amp=False
    )
    dense = CachedCubeCost(dense_model, stats, dense_output=True, use_amp=False)

    sparse.prepare(scene_xyz, scene_rgb, object_indices, goal)
    dense.prepare(scene_xyz, scene_rgb, object_indices, goal)
    sparse_cost = sparse(actions)
    dense_cost = dense(actions)
    sparse(actions)

    assert sparse.scene_encode_count == 1
    assert dense.scene_encode_count == 1
    torch.testing.assert_close(sparse_cost, dense_cost)
