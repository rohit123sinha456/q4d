#!/usr/bin/env python3
"""Prepare and run the gated submission-v1 definitive MPC matrices."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PHASES = ("wall_clock", "fixed_count")


def _read_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        return tomllib.load(stream)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _quote(value: str | Path) -> str:
    return '"' + str(value).replace("\\", "/").replace('"', '\\"') + '"'


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_state() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"], check=True, capture_output=True, text=True
    ).stdout
    return {"commit": commit, "working_tree_clean": not bool(status.strip())}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def validate_protocol_contract(raw: dict[str, Any]) -> dict[str, bool]:
    protocol = _read_toml(Path(raw["experiment"]["protocol"]))
    scope = protocol["scope"]
    planning = protocol["planning"]
    experiment = raw["experiment"]
    tasks = list(experiment["tasks"])
    checks = {
        "protocol_locked": protocol["protocol"]["status"]
        == "locked_before_new_test_results",
        "tasks_match": tasks == list(scope["tasks"]),
        "models_match": set(experiment["models"]) == {"q4d", "dense", "no_action"},
        "horizon_matches": int(raw["model"]["horizon"])
        == int(scope["primary_horizon"]),
        "planner_matches": experiment["method"] == scope["primary_planner"],
        "budget_matches": float(experiment["budget_ms"])
        == float(scope["primary_budget_ms"]),
        "episodes_match": int(experiment["episodes_per_task_model"])
        == int(planning["definitive_episodes_per_task_model"]),
        "seed_count_matches": int(experiment["episodes_per_task_model"])
        == int(planning["definitive_seed_count"]),
        "seed_starts_match": all(
            int(raw["task"][task]["seed_start"])
            == int(planning[f"{task}_definitive_seed_start"])
            for task in tasks
        ),
        "fixed_count_enabled": bool(raw["fixed_count_control"]["enabled"])
        == bool(planning["fixed_candidate_count_control"]),
        "gripper_aware": raw["planning"]["action_space"] == "gripper_schedules",
        "planning_label_matches": experiment["planning_label"]
        == scope["planning_label"],
    }
    checks["passed"] = all(checks.values())
    return checks


def build_task_config(raw: dict[str, Any], task: str, phase: str) -> str:
    if phase not in PHASES:
        raise ValueError(f"unknown phase: {phase}")
    experiment = raw["experiment"]
    task_raw = raw["task"][task]
    model = raw["model"]
    simulation = raw["simulation"]
    planning = raw["planning"]
    output = Path(experiment["output_root"]) / phase / task / "report.json"
    lines = [
        "[paths]",
        f"normalization = {_quote(task_raw['normalization'])}",
        f"micro_q4d_checkpoint = {_quote(task_raw['micro_q4d_checkpoint'])}",
        f"dense_checkpoint = {_quote(task_raw['dense_checkpoint'])}",
        f"no_action_checkpoint = {_quote(task_raw['no_action_checkpoint'])}",
        f"output = {_quote(output)}",
        "",
        "[model]",
        f"width = {int(model['width'])}",
        f"horizon = {int(model['horizon'])}",
        f"action_dimensions = {int(model['action_dimensions'])}",
        f"scene_points = {int(model['scene_points'])}",
        f"object_query_limit = {int(model['object_query_limit'])}",
        "",
        "[simulation]",
        f"env_id = {_quote(task_raw['env_id'])}",
        f"control_mode = {_quote(simulation['control_mode'])}",
        f"approach_max_steps = {int(task_raw['approach_max_steps'])}",
        f"control_cycles = {int(simulation['control_cycles'])}",
        f"max_depth_m = {float(simulation['max_depth_m'])}",
        f"object_quota = {int(simulation['object_quota'])}",
        f"robot_quota = {int(simulation['robot_quota'])}",
        f"goal_quota = {int(simulation['goal_quota'])}",
        "",
        "[planning]",
        f"seed = {int(task_raw['seed_start'])}",
        f"episodes = {int(experiment['episodes_per_task_model'])}",
        'models = ["q4d", "dense", "no_action"]',
        f"methods = [{_quote(experiment['method'])}]",
        f"budgets_ms = [{float(experiment['budget_ms'])}]",
        f"candidates_per_batch = {int(planning['candidates_per_batch'])}",
        f"elite_fraction = {float(planning['elite_fraction'])}",
        f"initial_std_xy = {float(planning['initial_std_xy'])}",
        f"initial_std_z = {float(planning['initial_std_z'])}",
        f"minimum_std = {float(planning['minimum_std'])}",
        f"action_penalty = {float(planning['action_penalty'])}",
        f"settling_penalty = {float(task_raw['settling_penalty'])}",
        f"settling_steps = {int(planning['settling_steps'])}",
        f"action_space = {_quote(planning['action_space'])}",
        "gripper_schedules = [",
        *[f"  {_quote(name)}," for name in planning["gripper_schedules"]],
        "]",
        f"minimum_schedule_probability = {float(planning['minimum_schedule_probability'])}",
        f"amp = {str(bool(planning['amp'])).lower()}",
    ]
    if phase == "fixed_count":
        lines.append(
            f"maximum_batches = {int(raw['fixed_count_control']['maximum_batches'])}"
        )
    lines.extend(
        [
            "",
            "[visualization]",
            "save_episode_contact_sheets = false",
        ]
    )
    return "\n".join(lines) + "\n"


def prepare_configs(raw: dict[str, Any]) -> list[dict[str, Any]]:
    output_root = Path(raw["experiment"]["output_root"])
    config_root = output_root / "generated_configs"
    runs = []
    for phase in PHASES:
        for task in raw["experiment"]["tasks"]:
            config = config_root / phase / f"{task}.toml"
            config.parent.mkdir(parents=True, exist_ok=True)
            config.write_text(build_task_config(raw, task, phase), encoding="utf-8")
            output = output_root / phase / task / "report.json"
            runs.append(
                {
                    "phase": phase,
                    "task": task,
                    "config": str(config),
                    "config_sha256": _sha256(config),
                    "output": str(output),
                    "log": str(output.with_name("run.log")),
                    "episodes": int(raw["experiment"]["episodes_per_task_model"])
                    * len(raw["experiment"]["models"]),
                    "status": "pending",
                }
            )
    return runs


def preflight(raw: dict[str, Any]) -> dict[str, Any]:
    contract = validate_protocol_contract(raw)
    pilot_path = Path(raw["experiment"]["pilot_gate"])
    prediction_path = Path(raw["experiment"]["prediction_status"])
    pilot = _read_json(pilot_path)
    prediction = _read_json(prediction_path)
    input_paths = [
        Path(raw["task"][task][name])
        for task in raw["experiment"]["tasks"]
        for name in (
            "normalization",
            "micro_q4d_checkpoint",
            "dense_checkpoint",
            "no_action_checkpoint",
        )
    ]
    missing = [str(path) for path in input_paths if not path.exists()]
    checks = {
        "protocol_contract": bool(contract["passed"]),
        "planner_pilot_passed": bool(pilot.get("passed")),
        "prediction_matrix_complete": bool(prediction.get("passed"))
        and int(prediction.get("passed_runs", 0)) == 24,
        "all_inputs_present": not missing,
        "pick_cube_pilot_passed": bool(pilot["tasks"]["pick_cube"]["passed"]),
    }
    return {
        "checks": checks,
        "protocol_contract": contract,
        "missing_inputs": missing,
        "pilot_gate": str(pilot_path),
        "prediction_status": str(prediction_path),
        "input_sha256": {
            str(path): _sha256(path) for path in input_paths if path.exists()
        },
        "passed": all(checks.values()),
    }


def _report_complete(path: Path, expected_episodes: int, maximum_batches: int | None) -> bool:
    try:
        report = _read_json(path)
    except (FileNotFoundError, json.JSONDecodeError):
        return False
    protocol = report.get("protocol", {})
    return bool(
        report.get("passed")
        and len(report.get("episodes", [])) == expected_episodes
        and protocol.get("models") == ["q4d", "dense", "no_action"]
        and protocol.get("methods") == ["random_shooting"]
        and protocol.get("budgets_ms") == [100.0]
        and protocol.get("episodes_per_condition") == 30
        and protocol.get("maximum_batches_per_cycle") == maximum_batches
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/submission_v1/definitive_mpc.toml"),
    )
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--phase", choices=("all", *PHASES), default="all")
    args = parser.parse_args()
    raw = _read_toml(args.config)
    output_root = Path(raw["experiment"]["output_root"])
    runs = prepare_configs(raw)
    flight = preflight(raw)
    status = {
        "experiment_id": raw["experiment"]["id"],
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "git": _git_state(),
        "preflight": flight,
        "primary_episodes": 360,
        "fixed_count_control_episodes": 360,
        "runs": runs,
        "status": "ready" if flight["passed"] else "blocked",
    }
    status_path = output_root / "run_status.json"
    _write_json(status_path, status)
    print(json.dumps(status, indent=2), flush=True)
    if args.preflight_only:
        return
    if not flight["passed"]:
        raise RuntimeError("definitive MPC preflight failed; see run_status.json")
    if not status["git"]["working_tree_clean"]:
        raise RuntimeError("definitive MPC must start from a clean exact commit")

    selected = PHASES if args.phase == "all" else (args.phase,)
    failures = []
    for run in status["runs"]:
        if run["phase"] not in selected:
            continue
        output = Path(run["output"])
        maximum_batches = (
            int(raw["fixed_count_control"]["maximum_batches"])
            if run["phase"] == "fixed_count"
            else None
        )
        if _report_complete(output, int(run["episodes"]), maximum_batches):
            run["status"] = "passed"
            _write_json(status_path, status)
            continue
        run["status"] = "running"
        _write_json(status_path, status)
        log_path = Path(run["log"])
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as log:
            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/evaluate_mpc.py",
                    "--config",
                    run["config"],
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        run["return_code"] = result.returncode
        run["status"] = (
            "passed"
            if _report_complete(output, int(run["episodes"]), maximum_batches)
            else "failed"
        )
        failures.extend([f"{run['phase']}:{run['task']}"] if run["status"] == "failed" else [])
        _write_json(status_path, status)
    status["status"] = "failed" if failures else "complete"
    status["failures"] = failures
    status["completed_at_utc"] = datetime.now(UTC).isoformat()
    _write_json(status_path, status)
    if failures:
        raise RuntimeError("incomplete definitive MPC runs: " + ", ".join(failures))


if __name__ == "__main__":
    main()
