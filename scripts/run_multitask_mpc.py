#!/usr/bin/env python3
"""Run the four resumable PushCube-sized closed-loop MPC matrices."""

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

TASKS = {
    "place_sphere": {
        "config": Path("configs/mpc_place_sphere.toml"),
        "pilot": Path("artifacts/datasets/place_sphere_pilot_v2/pilot_gate.json"),
    },
    "stack_cube": {
        "config": Path("configs/mpc_stack_cube.toml"),
        "pilot": Path("artifacts/datasets/stack_cube_pilot_v2/pilot_gate.json"),
    },
    "pull_cube": {
        "config": Path("configs/mpc_pull_cube.toml"),
        "pilot": Path("artifacts/datasets/pull_cube_pilot_v1/pilot_gate.json"),
    },
    "pick_cube": {
        "config": Path("configs/mpc_pick_cube.toml"),
        "pilot": Path("artifacts/datasets/pick_cube_pilot_v1/pilot_gate.json"),
    },
}
MODELS = ("q4d", "dense", "no_action")
METHODS = ("random_shooting", "cem")
BUDGETS_MS = (50, 100, 200)
EPISODES_PER_CONDITION = 10
EXPECTED_EPISODES = (
    len(MODELS) * len(METHODS) * len(BUDGETS_MS) * EPISODES_PER_CONDITION
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        return tomllib.load(stream)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_status(path: Path, status: dict[str, Any]) -> None:
    status["updated_at"] = datetime.now(UTC).isoformat()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _preflight(task: str) -> tuple[dict[str, Any], Path, str]:
    metadata = TASKS[task]
    pilot = _read_json(metadata["pilot"])
    if not pilot.get("passed"):
        raise RuntimeError(f"{task}: latest pilot gate did not pass")
    raw = _read_toml(metadata["config"])
    paths = raw["paths"]
    normalization = Path(paths["normalization"])
    checkpoints = [
        Path(paths[name])
        for name in (
            "micro_q4d_checkpoint",
            "dense_checkpoint",
            "no_action_checkpoint",
        )
    ]
    missing = [str(path) for path in (normalization, *checkpoints) if not path.exists()]
    if missing:
        raise RuntimeError(f"{task}: missing MPC inputs: {', '.join(missing)}")
    if int(raw["model"]["horizon"]) != 8:
        raise RuntimeError(f"{task}: MPC horizon is not H=8")
    dataset_root = normalization.parent
    freeze = _read_json(dataset_root / "freeze_record.json")
    normalization_hash = _sha256(normalization)
    if normalization_hash != freeze["files"]["normalization.json"]:
        raise RuntimeError(f"{task}: frozen normalization hash changed")
    return raw, Path(paths["output"]), normalization_hash


def _completed(report_path: Path) -> bool:
    if not report_path.exists():
        return False
    report = _read_json(report_path)
    protocol = report.get("protocol", {})
    return bool(
        report.get("passed")
        and protocol.get("planning_label") == "oracle-object-query planning"
        and set(protocol.get("models", [])) == set(MODELS)
        and set(protocol.get("methods", [])) == set(METHODS)
        and protocol.get("budgets_ms") == [float(value) for value in BUDGETS_MS]
        and protocol.get("episodes_per_condition") == EPISODES_PER_CONDITION
    )


def _progress(output: Path) -> int:
    progress = output.with_name("episodes.json")
    return len(_read_json(progress)) if progress.exists() else 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("tasks", choices=tuple(TASKS), nargs="*")
    parser.add_argument(
        "--status",
        type=Path,
        default=Path("artifacts/planning/multitask_mpc_status.json"),
    )
    args = parser.parse_args()
    selected_tasks = args.tasks or list(TASKS)
    status: dict[str, Any] = {
        "phase": "running",
        "planning_label": "oracle-object-query planning",
        "execution": "sequential_single_gpu_cpu_physics",
        "episodes_per_task": EXPECTED_EPISODES,
        "tasks": {},
    }
    _write_status(args.status, status)
    failures: list[str] = []

    for task in selected_tasks:
        try:
            _, output, normalization_hash = _preflight(task)
        except Exception as error:
            failures.append(task)
            status["tasks"][task] = {
                "phase": "preflight_failed",
                "error": f"{type(error).__name__}: {error}",
            }
            _write_status(args.status, status)
            continue

        log_path = output.with_name("run.log")
        if _completed(output):
            status["tasks"][task] = {
                "phase": "complete",
                "report": str(output),
                "completed_episodes": EXPECTED_EPISODES,
                "resumed": True,
            }
            _write_status(args.status, status)
            print(f"task={task} already_complete=true", flush=True)
            continue

        command = [
            sys.executable,
            "scripts/evaluate_mpc.py",
            "--config",
            str(TASKS[task]["config"]),
            "--episodes",
            str(EPISODES_PER_CONDITION),
            "--models",
            *MODELS,
            "--methods",
            *METHODS,
            "--budgets-ms",
            *(str(value) for value in BUDGETS_MS),
        ]
        output.parent.mkdir(parents=True, exist_ok=True)
        status["tasks"][task] = {
            "phase": "running",
            "config": str(TASKS[task]["config"]),
            "report": str(output),
            "log": str(log_path),
            "normalization_sha256": normalization_hash,
            "completed_episodes": _progress(output),
            "expected_episodes": EXPECTED_EPISODES,
        }
        _write_status(args.status, status)
        print(f"task={task} phase=running command={' '.join(command)}", flush=True)
        with log_path.open("a", encoding="utf-8") as log:
            log.write(
                f"\n[{datetime.now(UTC).isoformat()}] command={' '.join(command)}\n"
            )
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)

        try:
            _, _, normalization_after = _preflight(task)
            if normalization_after != normalization_hash:
                raise RuntimeError("normalization changed during MPC evaluation")
            if not _completed(output):
                raise RuntimeError("MPC report is missing or failed implementation checks")
        except Exception as error:
            failures.append(task)
            status["tasks"][task] = {
                **status["tasks"][task],
                "phase": "incomplete",
                "return_code": completed.returncode,
                "completed_episodes": _progress(output),
                "error": f"{type(error).__name__}: {error}",
            }
        else:
            status["tasks"][task] = {
                **status["tasks"][task],
                "phase": "complete",
                "return_code": completed.returncode,
                "completed_episodes": EXPECTED_EPISODES,
            }
        _write_status(args.status, status)

    status["phase"] = "attention_required" if failures else "complete"
    status["failed_or_incomplete_tasks"] = failures
    _write_status(args.status, status)
    if failures:
        raise RuntimeError(f"incomplete MPC tasks: {', '.join(failures)}")


if __name__ == "__main__":
    main()
