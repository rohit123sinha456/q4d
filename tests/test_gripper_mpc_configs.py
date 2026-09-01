from __future__ import annotations

import tomllib
from pathlib import Path

from q4d_wam.planning import DEFAULT_GRIPPER_SCHEDULES, PlannerConfig

ROOT = Path(__file__).parents[1]
TASKS = {
    "pull_cube": ("PullCube-v1", 13601, 0.0),
    "pick_cube": ("PickCube-v1", 14601, 0.0),
    "place_sphere": ("PlaceSphere-v1", 15601, 1.0),
    "stack_cube": ("StackCube-v1", 16601, 1.0),
}


def _load(path: Path) -> dict:
    with path.open("rb") as stream:
        return tomllib.load(stream)


def test_submission_gripper_configs_share_library_and_isolate_outputs() -> None:
    outputs = set()
    for task, (environment, seed, settling_penalty) in TASKS.items():
        raw = _load(
            ROOT / "configs" / "submission_v1" / f"mpc_{task}_gripper_pilot.toml"
        )
        model = raw["model"]
        planning = raw["planning"]
        assert raw["simulation"]["env_id"] == environment
        assert model["horizon"] == 8
        assert model["action_dimensions"] == 7
        assert planning["seed"] == seed
        assert planning["episodes"] == 10
        assert planning["models"] == ["q4d"]
        assert planning["methods"] == ["random_shooting"]
        assert planning["budgets_ms"] == [100.0]
        assert planning["action_space"] == "gripper_schedules"
        assert tuple(planning["gripper_schedules"]) == DEFAULT_GRIPPER_SCHEDULES
        assert planning["settling_penalty"] == settling_penalty
        assert planning["settling_steps"] == 2
        assert raw["visualization"]["save_episode_contact_sheets"] is True
        PlannerConfig(
            horizon=model["horizon"],
            action_dimensions=model["action_dimensions"],
            candidates_per_batch=planning["candidates_per_batch"],
            action_space=planning["action_space"],
            gripper_schedules=tuple(planning["gripper_schedules"]),
        )
        output = raw["paths"]["output"]
        assert output.startswith(
            "artifacts/submission_v1/planning/gripper_aware_pilot_v1/"
        )
        outputs.add(output)
        assert raw["ablation"] == {
            "translation_only_config": f"configs/mpc_{task}.toml",
            "translation_only_results": f"artifacts/planning/{task}_mpc_v1/report.json",
        }
    assert len(outputs) == len(TASKS)


def test_gripper_pilot_gate_is_frozen_before_execution() -> None:
    raw = _load(ROOT / "configs" / "submission_v1" / "gripper_pilot_gate.toml")

    assert raw["protocol"] == {
        "tasks": ["pull_cube", "pick_cube", "place_sphere", "stack_cube"],
        "model": "q4d",
        "method": "random_shooting",
        "budget_ms": 100.0,
        "episodes_per_task": 10,
        "required_candidate_schedules": list(DEFAULT_GRIPPER_SCHEDULES),
    }
    assert raw["thresholds"] == {
        "minimum_genuine_successes_per_task": 1,
        "minimum_improved_episodes_per_task": 6,
        "minimum_mean_distance_improvement_m": 0.0,
        "minimum_median_distance_improvement_m": 0.0,
        "settled_object_speed_max_m_per_s": 0.05,
    }
