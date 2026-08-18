import pytest
import torch

from q4d_wam.evaluation import TrajectoryMetricAccumulator


def test_ade_fde_groups_and_horizon_metrics() -> None:
    initial = torch.zeros(1, 2, 3)
    target = torch.zeros(1, 2, 2, 3)
    target[0, 0, :, 0] = torch.tensor([1.0, 2.0])
    prediction = torch.zeros_like(target)
    metrics = TrajectoryMetricAccumulator(horizon=2, moving_threshold_m=0.1)

    metrics.update(
        prediction,
        target,
        initial,
        point_groups={"contact": torch.tensor([[True, False]])},
        body_indices=torch.tensor([[0, 0]]),
    )
    report = metrics.report()

    assert report["groups"]["all"]["ade_m"] == pytest.approx(0.75)
    assert report["groups"]["all"]["fde_m"] == pytest.approx(1.0)
    assert report["groups"]["moving"]["ade_m"] == pytest.approx(1.5)
    assert report["groups"]["contact"]["fde_m"] == pytest.approx(2.0)
    assert report["per_horizon_ade_m"] == pytest.approx([0.5, 1.0])
    assert report["pairwise_distance_error_m"] == pytest.approx(1.5)
    assert report["same_body_pairwise_distance_error_m"] == pytest.approx(1.5)


def test_perfect_prediction_has_zero_temporal_errors() -> None:
    initial = torch.zeros(1, 2, 3)
    target = torch.randn(1, 2, 3, 3)
    metrics = TrajectoryMetricAccumulator(horizon=3)

    metrics.update(target, target, initial)
    report = metrics.report()

    assert report["groups"]["all"]["ade_m"] == 0.0
    assert report["acceleration_error_m_per_step2"] == 0.0
    assert report["pairwise_distance_error_m"] == 0.0


def test_quadratic_geometry_metrics_can_be_disabled_for_large_evaluations() -> None:
    initial = torch.zeros(1, 4, 3)
    target = torch.randn(1, 4, 2, 3)
    metrics = TrajectoryMetricAccumulator(horizon=2, compute_geometry_metrics=False)

    metrics.update(target, target, initial, body_indices=torch.zeros(1, 4, dtype=torch.long))
    report = metrics.report()

    assert report["groups"]["all"]["ade_m"] == 0.0
    assert report["pairwise_distance_error_m"] is None
    assert report["same_body_pairwise_distance_error_m"] is None
