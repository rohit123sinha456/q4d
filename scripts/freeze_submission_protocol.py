"""Validate and snapshot the frozen submission protocol without overwriting outputs."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "submission_protocol_v1.toml"
REQUIRED_TASKS = {"pull_cube", "pick_cube", "place_sphere", "stack_cube"}
REQUIRED_MODELS = {"micro_q4d", "no_action", "dense"}


def _read_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        return tomllib.load(stream)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _run(*command: str) -> tuple[int, str]:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    output = completed.stdout
    if completed.stderr:
        output += completed.stderr
    return completed.returncode, output.strip()


def validate_protocol(raw: dict[str, Any]) -> None:
    protocol = raw["protocol"]
    scope = raw["scope"]
    training = raw["training"]
    planning = raw["planning"]
    statistics = raw["statistics"]
    thresholds = raw["thresholds"]

    if protocol["id"] != "submission_v1":
        raise ValueError("protocol id must remain submission_v1")
    if set(scope["tasks"]) != REQUIRED_TASKS:
        raise ValueError("submission_v1 must contain exactly the four added tasks")
    if set(scope["models"]) != REQUIRED_MODELS:
        raise ValueError("all three matched neural models are required")
    if scope["primary_horizon"] != 8:
        raise ValueError("H=8 is the frozen primary horizon")
    if scope["primary_planner"] != "random_shooting":
        raise ValueError("random shooting is the frozen primary planner")
    if scope["primary_budget_ms"] != 100.0:
        raise ValueError("100 ms is the frozen primary planning budget")
    if training["independent_seeds_per_task_model"] != 3:
        raise ValueError("three independent training seeds are required")
    if planning["pilot_episodes_per_task"] != 10:
        raise ValueError("the pilot requires ten episodes per task")
    if planning["definitive_episodes_per_task_model"] != 30:
        raise ValueError("the definitive matrix requires 30 episodes per condition")
    if statistics["confidence_level"] != 0.95 or statistics["alpha"] != 0.05:
        raise ValueError("submission_v1 freezes 95% intervals and alpha=0.05")
    if thresholds["cache_equivalence_tolerance_m"] != 0.0005:
        raise ValueError("mixed-precision equivalence tolerance must remain 0.5 mm")

    for task in REQUIRED_TASKS:
        seeds = training[f"{task}_seeds"]
        if len(seeds) != 3 or len(set(seeds)) != 3:
            raise ValueError(f"{task} must have three unique training seeds")
        pilot = set(planning[f"{task}_pilot_seeds"])
        start = planning[f"{task}_definitive_seed_start"]
        definitive = set(range(start, start + planning["definitive_seed_count"]))
        if len(pilot) != 10 or pilot & definitive:
            raise ValueError(f"{task} pilot and definitive seeds must be complete and disjoint")


def _package_versions() -> list[str]:
    rows = []
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if name:
            rows.append(f"{name}=={distribution.version}")
    return sorted(set(rows), key=str.casefold)


def _resolve_repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    config_path = args.config.resolve()
    raw = _read_toml(config_path)
    validate_protocol(raw)

    configured_output = _resolve_repo_path(raw["protocol"]["protocol_output"])
    output = args.output.resolve() if args.output else configured_output
    if output.exists():
        raise FileExistsError(f"refusing to overwrite frozen protocol output: {output}")
    output.mkdir(parents=True)

    shutil.copy2(config_path, output / config_path.name)
    lock_path = _resolve_repo_path(raw["environment"]["package_lock"])
    shutil.copy2(lock_path, output / lock_path.name)

    git_code, git_commit = _run("git", "rev-parse", "HEAD")
    status_code, git_status = _run("git", "status", "--short")
    diff_code, git_diff = _run("git", "diff", "--binary", "HEAD")
    nvidia_code, nvidia_smi = _run("nvidia-smi", "-q")

    (output / "git_commit.txt").write_text(git_commit + "\n", encoding="utf-8")
    (output / "git_status.txt").write_text(git_status + "\n", encoding="utf-8")
    (output / "git_diff.patch").write_text(git_diff + "\n", encoding="utf-8")
    (output / "pip_freeze.txt").write_text(
        "\n".join(_package_versions()) + "\n", encoding="utf-8"
    )
    (output / "nvidia_smi.txt").write_text(nvidia_smi + "\n", encoding="utf-8")

    inputs = [config_path, lock_path]
    inputs.extend(_resolve_repo_path(path) for path in raw["data"]["split_files"])
    inputs.extend(_resolve_repo_path(path) for path in raw["data"]["normalization_files"])
    hashes = {
        str(path.relative_to(ROOT)): _sha256(path)
        for path in inputs
        if path.exists() and path.is_file()
    }
    missing_inputs = [str(path) for path in inputs if not path.is_file()]

    report = {
        "protocol_id": raw["protocol"]["id"],
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "config": str(config_path),
        "output": str(output),
        "git": {
            "return_code": git_code,
            "commit": git_commit,
            "expected_base_commit": raw["protocol"]["base_commit"],
            "status_return_code": status_code,
            "working_tree_clean": not bool(git_status),
            "diff_return_code": diff_code,
            "diff_sha256": _sha256(output / "git_diff.patch"),
        },
        "runtime": {
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
        },
        "nvidia_smi_return_code": nvidia_code,
        "input_sha256": hashes,
        "missing_inputs": missing_inputs,
        "checks": {
            "protocol_valid": True,
            "base_commit_matches": git_commit == raw["protocol"]["base_commit"],
            "required_inputs_present": not missing_inputs,
            "nvidia_smi_available": nvidia_code == 0,
        },
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
