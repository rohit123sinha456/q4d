#!/usr/bin/env python3
"""Run the four resumable 10-seed gripper-aware Q4D pilot matrices."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TASKS = (
    "place_sphere",
    "stack_cube",
    "pick_cube",
    "pull_cube",
)
CONFIGS = {
    task: Path(f"configs/submission_v1/mpc_{task}_gripper_pilot.toml")
    for task in TASKS
}
CRITICAL_RELEASE_TASKS = {"place_sphere", "stack_cube"}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_status(path: Path, value: dict[str, Any]) -> None:
    value["updated_at"] = datetime.now(UTC).isoformat()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _report_path(task: str) -> Path:
    with (ROOT / CONFIGS[task]).open("rb") as stream:
        raw = tomllib.load(stream)
    path = Path(raw["paths"]["output"])
    return path if path.is_absolute() else ROOT / path


def _complete(report: dict[str, Any]) -> bool:
    protocol = report.get("protocol", {})
    return bool(
        report.get("passed")
        and protocol.get("models") == ["q4d"]
        and protocol.get("methods") == ["random_shooting"]
        and protocol.get("budgets_ms") == [100.0]
        and protocol.get("episodes_per_condition") == 10
        and protocol.get("action_space") == "gripper_schedules"
        and len(report.get("episodes", [])) == 10
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("tasks", nargs="*", choices=TASKS)
    parser.add_argument(
        "--status",
        type=Path,
        default=Path(
            "artifacts/submission_v1/planning/gripper_aware_pilot_v1/status.json"
        ),
    )
    args = parser.parse_args()
    selected = tuple(args.tasks) if args.tasks else TASKS
    status: dict[str, Any] = {
        "phase": "running",
        "protocol": "checklist_item_3_gripper_aware_pilot_v1",
        "tasks": {},
    }
    _write_status(args.status, status)

    for task in selected:
        report_path = _report_path(task)
        if report_path.is_file() and _complete(_read_json(report_path)):
            report = _read_json(report_path)
            status["tasks"][task] = {
                "phase": "complete",
                "resumed": True,
                "successes": sum(row["success"] for row in report["episodes"]),
                "report": str(report_path),
            }
            _write_status(args.status, status)
            continue

        log_path = report_path.with_name("run.log")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            "scripts/evaluate_mpc.py",
            "--config",
            str(CONFIGS[task]),
        ]
        status["tasks"][task] = {
            "phase": "running",
            "config": str(CONFIGS[task]),
            "report": str(report_path),
            "log": str(log_path),
        }
        _write_status(args.status, status)
        print(f"task={task} phase=running", flush=True)
        with log_path.open("a", encoding="utf-8") as log:
            log.write(
                f"\n[{datetime.now(UTC).isoformat()}] command={' '.join(command)}\n"
            )
            completed = subprocess.run(
                command,
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if completed.returncode != 0 or not report_path.is_file():
            status["phase"] = "attention_required"
            status["tasks"][task]["phase"] = "failed"
            status["tasks"][task]["return_code"] = completed.returncode
            _write_status(args.status, status)
            raise RuntimeError(f"{task} pilot execution failed; inspect {log_path}")
        report = _read_json(report_path)
        if not _complete(report):
            raise RuntimeError(f"{task} pilot report is incomplete or invalid")
        successes = sum(row["success"] for row in report["episodes"])
        status["tasks"][task] = {
            **status["tasks"][task],
            "phase": "complete",
            "return_code": completed.returncode,
            "successes": successes,
        }
        _write_status(args.status, status)
        print(f"task={task} phase=complete successes={successes}/10", flush=True)
        if task in CRITICAL_RELEASE_TASKS and successes == 0:
            status["phase"] = "stopped_zero_release_task_success"
            _write_status(args.status, status)
            raise RuntimeError(f"{task} has zero successes; stop and diagnose")

    gate_command = [sys.executable, "scripts/evaluate_gripper_pilot.py"]
    completed = subprocess.run(gate_command, cwd=ROOT, check=False)
    status["phase"] = "complete" if completed.returncode == 0 else "gate_failed"
    status["gate_return_code"] = completed.returncode
    _write_status(args.status, status)
    if completed.returncode:
        raise RuntimeError("pilot completed but failed the frozen physical-behavior gate")


if __name__ == "__main__":
    main()
