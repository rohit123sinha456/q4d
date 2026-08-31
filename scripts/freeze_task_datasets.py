#!/usr/bin/env python3
"""Freeze each task as soon as its full collection is complete."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

EXPERIMENT_CONFIGS = {
    "pull_cube": Path("configs/scale_experiment_pull_cube.toml"),
    "pick_cube": Path("configs/scale_experiment_pick_cube.toml"),
    "place_sphere": Path("configs/scale_experiment_place_sphere.toml"),
    "stack_cube": Path("configs/scale_experiment_stack_cube.toml"),
}


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _dataset_is_complete(task: str) -> bool:
    manifest_path = Path("artifacts/datasets") / f"{task}_scale_v1" / "manifest.json"
    if not manifest_path.exists():
        return False
    manifest = _read(manifest_path)
    return (
        manifest.get("complete") is True
        and manifest.get("completed_states") == 2000
        and manifest.get("fragments") == 10000
        and manifest.get("passed_fragments") == 10000
        and manifest.get("passed_groups") == 2000
        and set(manifest.get("branch_counts", {}).values()) == {2000}
        and bool(manifest.get("observed_outcome_counts"))
        and all(manifest.get("observed_outcome_counts", {}).values())
    )


def _wait_for_collection(task: str, path: Path) -> None:
    while True:
        if _dataset_is_complete(task):
            return
        if path.exists():
            collection_status = _read(path)
            if collection_status.get("phase") == "failed":
                raise RuntimeError(
                    f"collection supervisor failed: {collection_status.get('error')}"
                )
        time.sleep(30)


def _freeze(task: str) -> dict[str, Any]:
    root = Path("artifacts/datasets") / f"{task}_scale_v1"
    subprocess.run(
        [
            sys.executable,
            "scripts/run_scale_experiment.py",
            "--config",
            str(EXPERIMENT_CONFIGS[task]),
            "--prepare-only",
        ],
        check=True,
    )
    report_path = root / "loader_report.json"
    report = _read(report_path)
    expected = (
        report.get("passed") is True
        and report.get("split_groups")
        == {"train": 1600, "validation": 200, "test": 200}
        and report.get("splits")
        == {"train": 8000, "validation": 1000, "test": 1000}
        and report.get("normalization", {}).get("source_files") == 8000
        and report.get("checks", {}).get("normalization_is_train_only") is True
    )
    if not expected:
        raise RuntimeError(f"{task} loader report failed frozen split requirements")
    files = ("splits.json", "normalization.json", "loader_report.json")
    record = {
        "task": task,
        "dataset_root": str(root),
        "frozen_at_utc": datetime.now(UTC).isoformat(),
        "hash_algorithm": "sha256",
        "files": {name: _sha256(root / name) for name in files},
        "split_groups": report["split_groups"],
        "split_fragments": report["splits"],
        "normalization_source_files": report["normalization"]["source_files"],
        "immutable_after_training_starts": ["splits.json", "normalization.json"],
    }
    _write(root / "freeze_record.json", record)
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("tasks", choices=tuple(EXPERIMENT_CONFIGS), nargs="+")
    parser.add_argument(
        "--collection-status",
        type=Path,
        default=Path("artifacts/datasets/full_collection_status.json"),
    )
    parser.add_argument(
        "--status",
        type=Path,
        default=Path("artifacts/datasets/freeze_status.json"),
    )
    args = parser.parse_args()
    status: dict[str, Any] = {"phase": "waiting_for_collections", "tasks": {}}
    _write(args.status, status)
    try:
        status["phase"] = "running"
        _write(args.status, status)
        for task in args.tasks:
            status["tasks"][task] = {"phase": "waiting_for_collection"}
            _write(args.status, status)
            _wait_for_collection(task, args.collection_status)
            status["tasks"][task] = {"phase": "preparing"}
            _write(args.status, status)
            record = _freeze(task)
            status["tasks"][task] = {
                "phase": "complete",
                "freeze_record": record,
            }
            _write(args.status, status)
    except BaseException as error:
        status["phase"] = "failed"
        status["error"] = f"{type(error).__name__}: {error}"
        _write(args.status, status)
        raise
    status["phase"] = "complete"
    _write(args.status, status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
