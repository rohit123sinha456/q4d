#!/usr/bin/env python3
"""Run real one-state label audits for every supported ManiSkill task adapter."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from q4d_wam.data import validate_training_file
from q4d_wam.tasks import get_task_adapter, supported_task_ids

TASK_CONFIGS = (
    Path("configs/tasks/push_cube.toml"),
    Path("configs/tasks/pull_cube.toml"),
    Path("configs/tasks/pick_cube.toml"),
    Path("configs/tasks/place_sphere.toml"),
    Path("configs/tasks/stack_cube.toml"),
)
BRANCHES = ("success", "weak", "off_target", "failure", "no_op")
PARITY_KEYS = (
    "rgb",
    "depth_mm",
    "intrinsic_cv",
    "extrinsic_cv",
    "cam2world_gl",
    "actions",
    "point_pixels_uv",
    "xyz0_world_m",
    "point_rgb",
    "target_tracks_world_m",
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _task_output(output_root: Path, config: Path) -> Path:
    return output_root / config.stem


def _run_collection(config: Path, output: Path) -> None:
    command = [
        sys.executable,
        "scripts/generate_point_tracks.py",
        "--config",
        str(config),
        "--profile",
        "scaled",
        "--states",
        "1",
        "--output-dir",
        str(output),
        "--resume",
        "--checkpoint-every",
        "1",
    ]
    print("run " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def _verify_task(config: Path, output: Path) -> dict[str, Any]:
    manifest = _read_json(output / "manifest.json")
    records = manifest["records"]
    env_ids = {record["environment"] for record in records}
    if len(env_ids) != 1:
        raise RuntimeError(f"{config}: records contain multiple environments")
    env_id = next(iter(env_ids))
    adapter = get_task_adapter(env_id)
    action_signatures = set()
    schemas = []
    for record in records:
        training_path = output / record["training_file"]
        schemas.append(validate_training_file(training_path))
        with np.load(training_path, allow_pickle=False) as archive:
            action_signatures.add(hashlib.sha256(archive["actions"].tobytes()).hexdigest())
    records_by_branch = {record["branch"]: record for record in records}
    preparation_grasped = {
        record["branch"]: bool(record.get("preparation_grasped"))
        for record in records
        if "preparation_grasped" in record
    }
    checks = {
        "manifest_complete": bool(manifest["complete"]),
        "five_fragments": manifest["fragments"] == len(BRANCHES),
        "expected_branches": set(records_by_branch) == set(BRANCHES),
        "all_physical_checks_pass": all(record["passed"] for record in records),
        "adapter_recorded": all(
            record.get("task_adapter") == adapter.name for record in records
        ),
        "object_points_visible": all(
            record["category_counts"]["object"] > 0 for record in records
        ),
        "robot_points_visible": all(
            record["category_counts"]["robot"] > 0 for record in records
        ),
        "five_distinct_action_chunks": len(action_signatures) == len(BRANCHES),
        "schemas_consistent": len(
            {
                (schema["points"], schema["horizon"], schema["action_dimensions"])
                for schema in schemas
            }
        )
        == 1,
        "success_branch_moves_object": (
            records_by_branch["success"]["primary_object_displacement_m"] > 0.002
        ),
        "grasp_preparation_succeeds": all(
            preparation_grasped.get(branch, False)
            for branch in ("success", "off_target")
        )
        if preparation_grasped
        else True,
    }
    group = manifest["groups"][0]
    checks["sibling_inputs_identical"] = bool(
        group["initial_observations_identical"]
    )
    report = {
        "environment": env_id,
        "adapter": adapter.name,
        "output": str(output),
        "fragments": len(records),
        "schema": schemas[0],
        "success_branch": {
            "task_success": records_by_branch["success"]["task_success"],
            "primary_object_displacement_m": records_by_branch["success"][
                "primary_object_displacement_m"
            ],
        },
        "checks": checks,
    }
    report["passed"] = all(checks.values())
    return report


def _pushcube_parity(
    candidate_root: Path, reference_root: Path, tolerance: float
) -> dict[str, Any]:
    branches = {}
    for branch in BRANCHES:
        name = f"state_000000__{branch}.train.npz"
        candidate = candidate_root / name
        reference = reference_root / name
        if not reference.exists():
            return {
                "passed": False,
                "reason": f"missing reference fragment: {reference}",
            }
        key_results = {}
        with np.load(candidate, allow_pickle=False) as actual, np.load(
            reference, allow_pickle=False
        ) as expected:
            for key in PARITY_KEYS:
                same_shape = actual[key].shape == expected[key].shape
                if not same_shape:
                    maximum_difference = None
                    passed = False
                elif np.issubdtype(actual[key].dtype, np.number):
                    maximum_difference = float(
                        np.max(
                            np.abs(
                                actual[key].astype(np.float64)
                                - expected[key].astype(np.float64)
                            )
                        )
                    )
                    passed = maximum_difference <= tolerance
                else:
                    maximum_difference = None
                    passed = bool(np.array_equal(actual[key], expected[key]))
                key_results[key] = {
                    "same_shape": same_shape,
                    "maximum_difference": maximum_difference,
                    "passed": passed,
                }
        branches[branch] = {
            "keys": key_results,
            "passed": all(result["passed"] for result in key_results.values()),
        }
    return {
        "reference": str(reference_root),
        "candidate": str(candidate_root),
        "tolerance": tolerance,
        "branches": branches,
        "passed": all(result["passed"] for result in branches.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/task_adapter_verification"),
    )
    parser.add_argument(
        "--pushcube-reference",
        type=Path,
        default=Path("artifacts/datasets/pushcube_scale_v1"),
    )
    parser.add_argument("--parity-tolerance", type=float, default=1e-6)
    parser.add_argument("--skip-collection", action="store_true")
    args = parser.parse_args()
    if args.parity_tolerance < 0:
        raise ValueError("parity tolerance cannot be negative")
    if len(TASK_CONFIGS) != 5 or len(supported_task_ids()) != 5:
        raise RuntimeError("the verification protocol requires exactly five task adapters")

    task_reports = []
    for config in TASK_CONFIGS:
        output = _task_output(args.output_root, config)
        if not args.skip_collection:
            _run_collection(config, output)
        task_reports.append(_verify_task(config, output))

    parity = _pushcube_parity(
        _task_output(args.output_root, TASK_CONFIGS[0]),
        args.pushcube_reference,
        args.parity_tolerance,
    )
    report = {
        "supported_tasks": list(supported_task_ids()),
        "task_reports": task_reports,
        "pushcube_parity": parity,
    }
    report["passed"] = all(item["passed"] for item in task_reports) and bool(
        parity["passed"]
    )
    output = args.output_root / "report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise RuntimeError("task-adapter verification failed")


if __name__ == "__main__":
    main()
