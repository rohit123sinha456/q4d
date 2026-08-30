from __future__ import annotations

from q4d_wam.evaluation.pilot import BRANCHES, evaluate_pilot_manifest


def _manifest() -> dict:
    records = []
    outcomes = ("successful", "weak", "off_target", "no_motion")
    for state_index in range(100):
        for branch in BRANCHES:
            records.append(
                {
                    "environment": "Example-v1",
                    "task_adapter": "example",
                    "branch": branch,
                    "task_success": branch == "success",
                    "initial_task_distance_m": 0.2,
                    "final_task_distance_m": 0.01 if branch == "success" else 0.2,
                    "observed_outcome": outcomes[state_index % len(outcomes)],
                    "outcome_match": True,
                    "category_counts": {"object": 1, "robot": 1},
                    "passed": True,
                }
            )
    return {
        "complete": True,
        "completed_states": 100,
        "fragments": 500,
        "passed_fragments": 500,
        "passed_groups": 100,
        "records": records,
        "groups": [{"passed": True} for _ in range(100)],
    }


def test_frozen_pilot_gate_passes_complete_balanced_pilot() -> None:
    report = evaluate_pilot_manifest(_manifest())
    assert report["passed"]
    assert all(report["checks"].values())
    assert report["collection"]["intended_outcome_matches"] == 500


def test_frozen_pilot_gate_reports_failed_success_policy() -> None:
    manifest = _manifest()
    success_records = [
        record for record in manifest["records"] if record["branch"] == "success"
    ]
    for record in success_records[:11]:
        record["task_success"] = False
    report = evaluate_pilot_manifest(manifest)
    assert not report["passed"]
    assert not report["checks"]["success_branch_at_least_90_percent"]
