#!/usr/bin/env python3
"""Evaluate the frozen implementation and physical-behavior gate for item 3."""

from __future__ import annotations

import argparse
import json
import statistics
import tomllib
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        return tomllib.load(stream)


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _genuine_success(
    task: str, episode: dict[str, Any], *, settled_speed_max: float
) -> bool:
    if not episode.get("success") or episode.get("termination_reason") != "success":
        return False
    improvement = float(episode["distance_improvement_m"])
    final_behavior = episode["final_object_behavior"]
    speed = final_behavior.get("speed_m_per_s")
    if improvement <= 0 or speed is None or float(speed) > settled_speed_max:
        return False
    if task == "pick_cube":
        return final_behavior.get("is_grasped") is True
    if task in {"place_sphere", "stack_cube"}:
        return bool(
            episode.get("release_command_executed")
            and final_behavior.get("is_grasped") is False
        )
    return True


def evaluate_pilot_gate(
    raw: dict[str, Any], reports: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    protocol = raw["protocol"]
    thresholds = raw["thresholds"]
    expected_tasks = list(protocol["tasks"])
    required_schedules = set(protocol["required_candidate_schedules"])
    task_reports: dict[str, Any] = {}
    all_checks: dict[str, bool] = {}

    for task in expected_tasks:
        report = reports[task]
        episodes = [
            episode
            for episode in report.get("episodes", [])
            if episode.get("model") == protocol["model"]
            and episode.get("method") == protocol["method"]
            and float(episode.get("budget_ms", -1)) == float(protocol["budget_ms"])
        ]
        improvements = [float(row["distance_improvement_m"]) for row in episodes]
        successes = [row for row in episodes if row.get("success")]
        genuine = [
            row
            for row in episodes
            if _genuine_success(
                task,
                row,
                settled_speed_max=float(
                    thresholds["settled_object_speed_max_m_per_s"]
                ),
            )
        ]
        visualizations = [
            _resolve(row["trajectory_visualization"])
            for row in episodes
            if row.get("trajectory_visualization")
        ]
        report_checks = report.get("checks", {})
        task_checks = {
            "implementation_checks_pass": bool(report.get("passed")),
            "expected_episode_count": len(episodes)
            == int(protocol["episodes_per_task"]),
            "valid_executable_actions": bool(
                report_checks.get("valid_executable_7d_actions")
            ),
            "candidate_library_complete": set(
                report.get("protocol", {}).get("candidate_gripper_schedules", [])
            )
            == required_schedules,
            "closed_and_release_candidates_present": bool(
                "hold_closed" in required_schedules
                and any("open" in name for name in required_schedules)
            ),
            "at_least_one_genuine_success": len(genuine)
            >= int(thresholds["minimum_genuine_successes_per_task"]),
            "success_flags_agree_with_behavior": len(successes) == len(genuine),
            "distance_generally_improves": bool(
                improvements
                and sum(value > 0 for value in improvements)
                >= int(thresholds["minimum_improved_episodes_per_task"])
                and statistics.mean(improvements)
                > float(thresholds["minimum_mean_distance_improvement_m"])
                and statistics.median(improvements)
                > float(thresholds["minimum_median_distance_improvement_m"])
            ),
            "trajectory_visualizations_complete": len(visualizations) == len(episodes)
            and all(path.is_file() for path in visualizations),
        }
        task_reports[task] = {
            "episodes": len(episodes),
            "successes": len(successes),
            "genuine_successes": len(genuine),
            "improved_episodes": sum(value > 0 for value in improvements),
            "mean_distance_improvement_m": (
                statistics.mean(improvements) if improvements else None
            ),
            "median_distance_improvement_m": (
                statistics.median(improvements) if improvements else None
            ),
            "success_seeds": [row["seed"] for row in successes],
            "genuine_success_seeds": [row["seed"] for row in genuine],
            "failure_seeds": [row["seed"] for row in episodes if not row.get("success")],
            "visualizations": [str(path) for path in visualizations],
            "checks": task_checks,
            "passed": all(task_checks.values()),
        }
        for name, passed in task_checks.items():
            all_checks[f"{task}_{name}"] = passed

    return {
        "protocol": protocol,
        "thresholds": thresholds,
        "tasks": task_reports,
        "checks": all_checks,
        "critical_zero_success_tasks": [
            task
            for task in ("place_sphere", "stack_cube")
            if task_reports[task]["genuine_successes"] == 0
        ],
        "passed": all(all_checks.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/submission_v1/gripper_pilot_gate.toml"),
    )
    args = parser.parse_args()
    raw = _read_toml(args.config)
    reports = {
        task: _read_json(_resolve(raw["paths"][f"{task}_report"]))
        for task in raw["protocol"]["tasks"]
    }
    result = evaluate_pilot_gate(raw, reports)
    output = _resolve(raw["paths"]["output"])
    _write_json_atomic(output, result)
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise RuntimeError("gripper-aware pilot failed its frozen gate")


if __name__ == "__main__":
    main()
