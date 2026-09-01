from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from evaluate_gripper_pilot import evaluate_pilot_gate  # noqa: E402

TASKS = ("pull_cube", "pick_cube", "place_sphere", "stack_cube")
SCHEDULES = (
    "hold_closed",
    "hold_open",
    "closed_to_open_halfway",
    "closed_to_open_final_quarter",
    "open_to_closed_halfway",
)


def _raw() -> dict:
    return {
        "protocol": {
            "tasks": list(TASKS),
            "model": "q4d",
            "method": "random_shooting",
            "budget_ms": 100.0,
            "episodes_per_task": 10,
            "required_candidate_schedules": list(SCHEDULES),
        },
        "thresholds": {
            "minimum_genuine_successes_per_task": 1,
            "minimum_improved_episodes_per_task": 6,
            "minimum_mean_distance_improvement_m": 0.0,
            "minimum_median_distance_improvement_m": 0.0,
            "settled_object_speed_max_m_per_s": 0.05,
        },
    }


def _reports(tmp_path: Path) -> dict[str, dict]:
    reports = {}
    for task in TASKS:
        episodes = []
        for seed in range(10):
            image = tmp_path / f"{task}_{seed}.png"
            image.write_bytes(b"visualization-placeholder")
            episodes.append(
                {
                    "model": "q4d",
                    "method": "random_shooting",
                    "budget_ms": 100.0,
                    "seed": seed,
                    "success": True,
                    "termination_reason": "success",
                    "distance_improvement_m": 0.01,
                    "release_command_executed": task
                    in {"place_sphere", "stack_cube"},
                    "final_object_behavior": {
                        "speed_m_per_s": 0.01,
                        "is_grasped": task == "pick_cube",
                    },
                    "trajectory_visualization": str(image),
                }
            )
        reports[task] = {
            "passed": True,
            "protocol": {"candidate_gripper_schedules": list(SCHEDULES)},
            "checks": {"valid_executable_7d_actions": True},
            "episodes": episodes,
        }
    return reports


def test_pilot_gate_accepts_genuine_success_and_complete_visual_evidence(
    tmp_path: Path,
) -> None:
    result = evaluate_pilot_gate(_raw(), _reports(tmp_path))

    assert result["passed"]
    assert result["critical_zero_success_tasks"] == []
    assert all(task["genuine_successes"] == 10 for task in result["tasks"].values())


def test_pilot_gate_rejects_success_flag_that_disagrees_with_release_behavior(
    tmp_path: Path,
) -> None:
    reports = _reports(tmp_path)
    reports["place_sphere"]["episodes"][0]["release_command_executed"] = False

    result = evaluate_pilot_gate(_raw(), reports)

    assert not result["passed"]
    assert not result["tasks"]["place_sphere"]["checks"][
        "success_flags_agree_with_behavior"
    ]


def test_settling_speed_applies_only_to_placement_and_stacking(tmp_path: Path) -> None:
    reports = _reports(tmp_path)
    for episode in reports["pull_cube"]["episodes"]:
        episode["final_object_behavior"]["speed_m_per_s"] = 1.0
    for episode in reports["pick_cube"]["episodes"]:
        episode["final_object_behavior"]["speed_m_per_s"] = 1.0

    result = evaluate_pilot_gate(_raw(), reports)

    assert result["tasks"]["pull_cube"]["genuine_successes"] == 10
    assert result["tasks"]["pick_cube"]["genuine_successes"] == 10
