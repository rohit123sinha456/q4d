#!/usr/bin/env python3
"""Run and merge the frozen four-shard, 2,000-state task collections."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

TASK_CONFIGS = {
    "pull_cube": Path("configs/scale_pull_cube.toml"),
    "pick_cube": Path("configs/scale_pick_cube.toml"),
    "place_sphere": Path("configs/scale_place_sphere.toml"),
    "stack_cube": Path("configs/scale_stack_cube.toml"),
}
SHARD_STARTS = (0, 500, 1000, 1500)


def _write_status(path: Path, status: dict[str, Any]) -> None:
    status["updated_at"] = datetime.now(UTC).isoformat()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _complete_manifest(root: Path) -> dict[str, Any] | None:
    path = root / "manifest.json"
    if not path.exists():
        return None
    report = json.loads(path.read_text(encoding="utf-8"))
    checks = report.get("checks", {})
    valid = (
        report.get("complete") is True
        and report.get("completed_states") == 2000
        and report.get("fragments") == 10000
        and report.get("passed_fragments") == 10000
        and report.get("passed_groups") == 2000
        and set(report.get("branch_counts", {}).values()) == {2000}
        and all(report.get("observed_outcome_counts", {}).values())
        and checks
        and all(checks.values())
    )
    return report if valid else None


def _run_task(task: str, status_path: Path, status: dict[str, Any]) -> None:
    root = Path("artifacts/datasets") / f"{task}_scale_v1"
    existing = _complete_manifest(root)
    if existing is not None:
        status["tasks"][task] = {"phase": "complete", "resumed": True}
        _write_status(status_path, status)
        print(f"task={task} already_complete=true", flush=True)
        return

    root.mkdir(parents=True, exist_ok=True)
    processes: list[tuple[int, subprocess.Popen[bytes], Any]] = []
    try:
        for shard, start in enumerate(SHARD_STARTS):
            log = (root / f"collection_shard{shard}.log").open("ab")
            command = [
                sys.executable,
                "-u",
                "scripts/generate_point_tracks.py",
                "--config",
                str(TASK_CONFIGS[task]),
                "--profile",
                "scaled",
                "--start-state",
                str(start),
                "--states",
                "500",
                "--output-dir",
                str(root),
                "--manifest-name",
                f"manifest_shard{shard}.json",
                "--resume",
                "--checkpoint-every",
                "10",
            ]
            process = subprocess.Popen(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            processes.append((shard, process, log))
        status["tasks"][task] = {
            "phase": "collecting",
            "shards": [
                {"shard": shard, "start_state": SHARD_STARTS[shard], "pid": process.pid}
                for shard, process, _ in processes
            ],
        }
        _write_status(status_path, status)
        print(f"task={task} phase=collecting", flush=True)

        failures = []
        for shard, process, _ in processes:
            return_code = process.wait()
            print(f"task={task} shard={shard} exit_code={return_code}", flush=True)
            if return_code != 0:
                failures.append((shard, return_code))
        if failures:
            raise RuntimeError(f"task {task} shard failures: {failures}")
    except BaseException:
        for _, process, _ in processes:
            if process.poll() is None:
                process.terminate()
        raise
    finally:
        for _, _, log in processes:
            log.close()

    status["tasks"][task] = {"phase": "merging"}
    _write_status(status_path, status)
    manifests = [root / f"manifest_shard{shard}.json" for shard in range(4)]
    merge_command = [
        sys.executable,
        "scripts/merge_collection_shards.py",
        "--root",
        str(root),
        "--states",
        "2000",
        "--branches",
        "5",
        *(str(path) for path in manifests),
    ]
    with (root / "merge.log").open("ab") as log:
        subprocess.run(
            merge_command,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    report = _complete_manifest(root)
    if report is None:
        raise RuntimeError(f"merged {task} manifest failed full-dataset validation")
    status["tasks"][task] = {
        "phase": "complete",
        "state_groups": report["completed_states"],
        "fragments": report["fragments"],
        "passed_fragments": report["passed_fragments"],
    }
    _write_status(status_path, status)
    print(f"task={task} phase=complete", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("tasks", choices=tuple(TASK_CONFIGS), nargs="+")
    parser.add_argument(
        "--status",
        type=Path,
        default=Path("artifacts/datasets/full_collection_status.json"),
    )
    args = parser.parse_args()
    status: dict[str, Any] = {"phase": "running", "tasks": {}}
    _write_status(args.status, status)
    try:
        for task in args.tasks:
            _run_task(task, args.status, status)
    except BaseException as error:
        status["phase"] = "failed"
        status["error"] = f"{type(error).__name__}: {error}"
        _write_status(args.status, status)
        raise
    status["phase"] = "complete"
    _write_status(args.status, status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
