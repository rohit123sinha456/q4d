"""Frozen readiness gate for one 100-state task pilot."""

from __future__ import annotations

from statistics import median
from typing import Any

BRANCHES = ("success", "weak", "off_target", "failure", "no_op")
OUTCOMES = ("successful", "weak", "off_target", "no_motion")


def evaluate_pilot_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    records = list(manifest.get("records", []))
    groups = list(manifest.get("groups", []))
    by_branch = {
        branch: [record for record in records if record.get("branch") == branch]
        for branch in BRANCHES
    }
    branch_successes = {
        branch: sum(bool(record.get("task_success")) for record in selected)
        for branch, selected in by_branch.items()
    }
    success_records = by_branch["success"]
    initial_distances = [
        float(record["initial_task_distance_m"]) for record in success_records
    ]
    final_distances = [
        float(record["final_task_distance_m"]) for record in success_records
    ]
    outcome_counts = {
        outcome: sum(record.get("observed_outcome") == outcome for record in records)
        for outcome in OUTCOMES
    }
    intended_outcome_matches = sum(
        bool(record.get("outcome_match")) for record in records
    )
    checks = {
        "collection_complete": bool(manifest.get("complete")),
        "state_groups_100": len(groups) == 100
        and int(manifest.get("completed_states", -1)) == 100,
        "fragments_500": len(records) == 500
        and int(manifest.get("fragments", -1)) == 500,
        "five_balanced_branches": all(len(by_branch[branch]) == 100 for branch in BRANCHES),
        "all_physical_and_schema_checks_pass": all(
            bool(record.get("passed")) for record in records
        )
        and int(manifest.get("passed_fragments", -1)) == 500,
        "all_sibling_groups_pass": all(bool(group.get("passed")) for group in groups)
        and int(manifest.get("passed_groups", -1)) == 100,
        "object_and_robot_visible": all(
            int(record.get("category_counts", {}).get("object", 0)) > 0
            and int(record.get("category_counts", {}).get("robot", 0)) > 0
            for record in records
        ),
        "success_branch_at_least_90_percent": branch_successes["success"] >= 90,
        "weak_false_success_at_most_5_percent": branch_successes["weak"] <= 5,
        "failure_false_success_at_most_5_percent": branch_successes["failure"] <= 5,
        "no_op_false_success_at_most_5_percent": branch_successes["no_op"] <= 5,
        "off_target_false_success_at_most_20_percent": (
            branch_successes["off_target"] <= 20
        ),
        "intended_outcomes_at_least_95_percent": intended_outcome_matches >= 475,
        "all_outcome_classes_present": all(outcome_counts.values()),
        "success_branch_median_task_distance_improves": bool(initial_distances)
        and median(final_distances) < median(initial_distances),
    }
    report = {
        "environment": sorted(
            {str(record.get("environment")) for record in records}
        ),
        "task_adapters": sorted(
            {str(record.get("task_adapter")) for record in records}
        ),
        "collection": {
            "state_groups": len(groups),
            "fragments": len(records),
            "branch_counts": {
                branch: len(selected) for branch, selected in by_branch.items()
            },
            "outcome_counts": outcome_counts,
            "intended_outcome_matches": intended_outcome_matches,
        },
        "branch_successes": branch_successes,
        "success_branch": {
            "median_initial_task_distance_m": median(initial_distances)
            if initial_distances
            else None,
            "median_final_task_distance_m": median(final_distances)
            if final_distances
            else None,
        },
        "checks": checks,
    }
    report["passed"] = all(checks.values())
    return report
