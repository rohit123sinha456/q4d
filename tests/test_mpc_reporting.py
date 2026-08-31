import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from evaluate_mpc import _matched_comparisons, _summarize  # noqa: E402


def _record(
    model: str,
    seed: int,
    planning_ms: float,
    candidates: int,
    *,
    visible: bool = True,
) -> dict:
    return {
        "model": model,
        "method": "cem",
        "budget_ms": 50.0,
        "seed": seed,
        "success": seed % 2 == 0,
        "termination_reason": "control_cycle_limit" if visible else "object_not_visible",
        "final_cube_goal_distance_m": 0.1 + 0.01 * seed,
        "final_task_distance_m": 0.1 + 0.01 * seed,
        "control_cycles": 1,
        "cycles": [
            {
                "planning_ms": planning_ms,
                "budget_overrun_ms": max(0.0, planning_ms - 50.0),
                "candidates_evaluated": candidates,
            }
        ],
    }


def test_mpc_summary_reports_latency_throughput_overruns_and_visibility() -> None:
    records = [
        _record("q4d", 1, 40.0, 400),
        _record("q4d", 2, 60.0, 600, visible=False),
        _record("dense", 1, 80.0, 400),
        _record("dense", 2, 120.0, 600),
    ]

    summary = _summarize(records)
    indexed = {row["model"]: row for row in summary}
    q4d = indexed["q4d"]

    assert q4d["p50_planning_ms"] == pytest.approx(50.0)
    assert q4d["p95_planning_ms"] == pytest.approx(59.0)
    assert q4d["candidate_throughput_per_second"] == pytest.approx(10_000.0)
    assert q4d["budget_overrun_cycles"] == 1
    assert q4d["p95_budget_overrun_ms"] == pytest.approx(9.5)
    assert q4d["object_visibility_failures"] == 1
    assert q4d["object_visibility_failure_rate"] == pytest.approx(0.5)

    comparison = _matched_comparisons(summary)[0]
    assert comparison["q4d_over_dense_candidate_throughput"] == pytest.approx(2.0)
    assert comparison["q4d_minus_dense_p50_planning_ms"] == pytest.approx(-50.0)
