#!/usr/bin/env python3
"""Run the frozen four-task prediction matrix sequentially on one GPU."""

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

TASK_CONFIGS = {
    "place_sphere": Path("configs/scale_experiment_place_sphere.toml"),
    "stack_cube": Path("configs/scale_experiment_stack_cube.toml"),
    "pull_cube": Path("configs/scale_experiment_pull_cube.toml"),
    "pick_cube": Path("configs/scale_experiment_pick_cube.toml"),
}
FROZEN_FILES = ("splits.json", "normalization.json")
HORIZONS = (1, 2, 4, 8)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        return tomllib.load(stream)


def _write_status(path: Path, status: dict[str, Any]) -> None:
    status["updated_at"] = datetime.now(UTC).isoformat()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_frozen(config_path: Path) -> tuple[Path, Path, dict[str, str]]:
    raw = _read_toml(config_path)
    experiment = raw["experiment"]
    dataset_root = Path(experiment["dataset_root"])
    output_root = Path(experiment["output_root"])
    manifest = _read_json(dataset_root / "manifest.json")
    loader_report = _read_json(dataset_root / "loader_report.json")
    freeze_record = _read_json(dataset_root / "freeze_record.json")

    if not manifest.get("complete"):
        raise RuntimeError(f"{dataset_root}: collection is incomplete")
    if int(manifest.get("fragments", 0)) < int(experiment["minimum_fragments"]):
        raise RuntimeError(f"{dataset_root}: collection is below the fragment gate")
    if not loader_report.get("passed"):
        raise RuntimeError(f"{dataset_root}: loader report did not pass")
    if [int(value) for value in experiment["horizons"]] != list(HORIZONS):
        raise RuntimeError(f"{config_path}: horizons must remain 1, 2, 4, 8")

    hashes = {name: _sha256(dataset_root / name) for name in FROZEN_FILES}
    expected = freeze_record["files"]
    mismatches = [name for name, digest in hashes.items() if digest != expected[name]]
    if mismatches:
        raise RuntimeError(
            f"{dataset_root}: frozen artifacts changed: {', '.join(mismatches)}"
        )
    return dataset_root, output_root, hashes


def _progress(output_root: Path) -> dict[str, int | bool]:
    passed_reports = 0
    total_reports = len(HORIZONS) * 4
    micro_q4d_runs = 0
    neural_runs = 0
    for horizon in HORIZONS:
        root = output_root / f"h{horizon}"
        reports = (
            root / "non_neural.json",
            root / "no_action" / "report.json",
            root / "micro_q4d" / "report.json",
            root / "dense" / "report.json",
        )
        for index, report_path in enumerate(reports):
            if report_path.exists() and _read_json(report_path).get("passed"):
                passed_reports += 1
                if index > 0:
                    neural_runs += 1
                if index == 2:
                    micro_q4d_runs += 1
    return {
        "passed_stage_reports": passed_reports,
        "total_stage_reports": total_reports,
        "completed_micro_q4d_runs": micro_q4d_runs,
        "completed_neural_runs": neural_runs,
        "h8_grid_exists": (output_root / "h8" / "n_m_scaling.json").exists(),
        "gate_report_exists": (output_root / "gate_report.json").exists(),
    }


def _matrix_finished(output_root: Path) -> bool:
    progress = _progress(output_root)
    return bool(
        progress["passed_stage_reports"] == progress["total_stage_reports"]
        and progress["h8_grid_exists"]
        and progress["gate_report_exists"]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("tasks", choices=tuple(TASK_CONFIGS), nargs="*")
    parser.add_argument(
        "--status",
        type=Path,
        default=Path("artifacts/experiments/multitask_scale_status.json"),
    )
    args = parser.parse_args()
    status: dict[str, Any] = {
        "phase": "running",
        "execution": "sequential_single_gpu",
        "tasks": {},
    }
    _write_status(args.status, status)
    failures: list[str] = []

    tasks = args.tasks or list(TASK_CONFIGS)
    for task in tasks:
        config_path = TASK_CONFIGS[task]
        try:
            dataset_root, output_root, frozen_before = _verify_frozen(config_path)
        except Exception as error:
            failures.append(task)
            status["tasks"][task] = {
                "phase": "preflight_failed",
                "error": f"{type(error).__name__}: {error}",
            }
            _write_status(args.status, status)
            continue

        output_root.mkdir(parents=True, exist_ok=True)
        log_path = output_root / "matrix.log"
        status["tasks"][task] = {
            "phase": "running",
            "config": str(config_path),
            "dataset_root": str(dataset_root),
            "output_root": str(output_root),
            "log": str(log_path),
            "frozen_hashes": frozen_before,
            "progress": _progress(output_root),
        }
        _write_status(args.status, status)
        command = [
            sys.executable,
            "scripts/run_scale_experiment.py",
            "--config",
            str(config_path),
        ]
        print(f"task={task} phase=running command={' '.join(command)}", flush=True)
        with log_path.open("a", encoding="utf-8") as log:
            log.write(
                f"\n[{datetime.now(UTC).isoformat()}] command={' '.join(command)}\n"
            )
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)

        try:
            _, _, frozen_after = _verify_frozen(config_path)
            if frozen_after != frozen_before:
                raise RuntimeError("frozen split or normalization changed during training")
        except Exception as error:
            failures.append(task)
            status["tasks"][task] = {
                **status["tasks"][task],
                "phase": "freeze_violation",
                "return_code": completed.returncode,
                "error": f"{type(error).__name__}: {error}",
                "progress": _progress(output_root),
            }
            _write_status(args.status, status)
            continue

        progress = _progress(output_root)
        if _matrix_finished(output_root):
            gate = _read_json(output_root / "gate_report.json")
            phase = "complete" if gate.get("passed") else "complete_gate_failed"
            status["tasks"][task] = {
                **status["tasks"][task],
                "phase": phase,
                "return_code": completed.returncode,
                "scientific_gate_passed": bool(gate.get("passed")),
                "progress": progress,
            }
            print(f"task={task} phase={phase}", flush=True)
        else:
            failures.append(task)
            status["tasks"][task] = {
                **status["tasks"][task],
                "phase": "incomplete",
                "return_code": completed.returncode,
                "progress": progress,
            }
            print(f"task={task} phase=incomplete", flush=True)
        _write_status(args.status, status)

    status["phase"] = "attention_required" if failures else "complete"
    status["failed_or_incomplete_tasks"] = failures
    _write_status(args.status, status)
    if failures:
        raise RuntimeError(f"incomplete task matrices: {', '.join(failures)}")


if __name__ == "__main__":
    main()
