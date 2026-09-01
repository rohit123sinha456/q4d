import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from evaluate_mpc import (  # noqa: E402
    _matched_comparisons,
    _summarize,
    _valid_executed_actions,
    _write_contact_sheet,
)


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
                "selected_gripper_schedule": "closed_to_open_final_quarter",
                "executed_first_action": [0.1, -0.2, 0.0, 0.0, 0.0, 0.0, -1.0],
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
    assert q4d["selected_gripper_schedule_counts"] == {
        "closed_to_open_final_quarter": 2
    }
    assert _valid_executed_actions(records)

    comparison = _matched_comparisons(summary)[0]
    assert comparison["q4d_over_dense_candidate_throughput"] == pytest.approx(2.0)
    assert comparison["q4d_minus_dense_p50_planning_ms"] == pytest.approx(-50.0)


def test_mpc_action_validation_rejects_non_7d_record() -> None:
    record = _record("q4d", 1, 40.0, 400)
    record["cycles"][0]["executed_first_action"] = [0.0] * 6

    assert not _valid_executed_actions([record])


def test_trajectory_contact_sheet_is_written(tmp_path: Path) -> None:
    frames = [
        ("initial", np.zeros((8, 10, 3), dtype=np.uint8)),
        ("cycle 0", np.full((8, 10, 3), 255, dtype=np.uint8)),
    ]
    output = tmp_path / "trajectory.png"

    _write_contact_sheet(frames, output, columns=2)

    with Image.open(output) as image:
        assert image.size == (20, 36)
