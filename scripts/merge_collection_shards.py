#!/usr/bin/env python3
"""Validate and merge non-overlapping scaled-collection shard manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path("artifacts/datasets/pushcube_scale_v1")
    )
    parser.add_argument("--states", type=int, default=2000)
    parser.add_argument("--branches", type=int, default=5)
    parser.add_argument("manifests", type=Path, nargs="+")
    args = parser.parse_args()
    shard_reports = [
        json.loads(path.read_text(encoding="utf-8")) for path in args.manifests
    ]
    if not all(report.get("complete") for report in shard_reports):
        raise RuntimeError("one or more shard manifests are incomplete")
    records_by_file: dict[str, dict[str, Any]] = {}
    groups_by_id: dict[str, dict[str, Any]] = {}
    for report in shard_reports:
        for record in report["records"]:
            name = record["training_file"]
            if name in records_by_file:
                raise RuntimeError(f"duplicate training fragment across shards: {name}")
            records_by_file[name] = record
        for group in report["groups"]:
            group_id = group["group_id"]
            if group_id in groups_by_id:
                raise RuntimeError(f"duplicate state group across shards: {group_id}")
            groups_by_id[group_id] = group
    expected_groups = {f"state_{index:06d}" for index in range(args.states)}
    if set(groups_by_id) != expected_groups:
        missing = sorted(expected_groups - set(groups_by_id))[:10]
        extra = sorted(set(groups_by_id) - expected_groups)[:10]
        raise RuntimeError(f"state coverage mismatch; missing={missing}, extra={extra}")
    if len(records_by_file) != args.states * args.branches:
        raise RuntimeError("merged fragment count does not match states times branches")
    records = sorted(records_by_file.values(), key=lambda record: record["training_file"])
    groups = [groups_by_id[group_id] for group_id in sorted(groups_by_id)]
    missing_files = [
        record[key]
        for record in records
        for key in ("training_file", "audit_file", "record_file")
        if not (args.root / record[key]).exists()
    ]
    if missing_files:
        raise RuntimeError(f"merged records reference missing files: {missing_files[:10]}")
    outcomes = ("successful", "weak", "off_target", "no_motion")
    branches = sorted({record["branch"] for record in records})
    summary = {
        "profile": "scaled",
        "requested_states": args.states,
        "completed_states": len(groups),
        "complete": True,
        "branches_per_state": args.branches,
        "fragments": len(records),
        "passed_fragments": sum(record["passed"] for record in records),
        "passed_groups": sum(group["passed"] for group in groups),
        "mean_cube_displacement_m": sum(
            record["cube_displacement_m"] for record in records
        )
        / len(records),
        "branch_counts": {
            branch: sum(record["branch"] == branch for record in records)
            for branch in branches
        },
        "branch_success_counts": {
            branch: sum(
                record["branch"] == branch and record["task_success"]
                for record in records
            )
            for branch in branches
        },
        "branch_mean_cube_displacement_m": {
            branch: sum(
                record["cube_displacement_m"]
                for record in records
                if record["branch"] == branch
            )
            / sum(record["branch"] == branch for record in records)
            for branch in branches
        },
        "observed_outcome_counts": {
            outcome: sum(record["observed_outcome"] == outcome for record in records)
            for outcome in outcomes
        },
        "intended_outcome_matches": sum(
            record.get("outcome_match", True) for record in records
        ),
        "groups": groups,
        "records": records,
        "source_manifests": [str(path) for path in args.manifests],
    }
    checks = {
        "fragment_count": summary["fragments"] == args.states * args.branches,
        "state_group_count": summary["completed_states"] == args.states,
        "all_physical_checks_pass": summary["passed_fragments"] == summary["fragments"],
        "all_group_identity_checks_pass": summary["passed_groups"] == args.states,
        "all_outcome_classes_present": all(summary["observed_outcome_counts"].values()),
        "all_referenced_files_exist": not missing_files,
    }
    summary["checks"] = checks
    summary["passed"] = all(checks.values())
    temporary = args.root / "manifest.json.tmp"
    temporary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.root / "manifest.json")
    print(json.dumps({key: value for key, value in summary.items() if key != "records"}, indent=2))
    if not summary["passed"]:
        raise RuntimeError("merged collection manifest failed validation")


if __name__ == "__main__":
    main()
