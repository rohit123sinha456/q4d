"""Evidence aggregation for the MVP stop/continue decision."""

from __future__ import annotations

from statistics import median
from typing import Any

PREDICTION_GROUPS = ("all", "moving", "contact", "object")


def _condition(report: dict[str, Any], model: str) -> dict[str, Any]:
    matches = [item for item in report["conditions"] if item["model"] == model]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one MPC condition for {model}")
    return matches[0]


def evaluate_stop_gate(
    *,
    no_action: dict[str, Any],
    q4d: dict[str, Any],
    dense: dict[str, Any],
    cache_grid: dict[str, Any],
    mpc: dict[str, Any],
    thresholds: dict[str, float],
) -> dict[str, Any]:
    """Evaluate the four original MVP continuation hypotheses."""
    action_rows = {}
    for group in PREDICTION_GROUPS:
        no_action_ade = float(no_action["test"]["groups"][group]["ade_m"])
        q4d_ade = float(q4d["test"]["groups"][group]["ade_m"])
        action_rows[group] = {
            "no_action_ade_m": no_action_ade,
            "q4d_ade_m": q4d_ade,
            "relative_reduction": 1.0 - q4d_ade / no_action_ade,
        }

    cache_speedups = [float(row["cache_speedup"]) for row in cache_grid["rows"]]
    dense_rows = {}
    for group in PREDICTION_GROUPS:
        q4d_ade = float(q4d["test"]["groups"][group]["ade_m"])
        dense_ade = float(dense["test"]["groups"][group]["ade_m"])
        dense_rows[group] = {
            "q4d_ade_m": q4d_ade,
            "dense_ade_m": dense_ade,
            "q4d_over_dense_ade": q4d_ade / dense_ade,
        }

    q4d_mpc = _condition(mpc, "q4d")
    no_action_mpc = _condition(mpc, "no_action")
    q4d_seeds = {
        row["seed"] for row in mpc["episodes"] if row["model"] == "q4d"
    }
    no_action_seeds = {
        row["seed"] for row in mpc["episodes"] if row["model"] == "no_action"
    }
    source_integrity = all(
        bool(report.get("passed"))
        for report in (no_action, q4d, dense, cache_grid, mpc)
    ) and q4d_seeds == no_action_seeds
    gates = {
        "action_conditioning_beats_no_action": all(
            row["q4d_ade_m"] < row["no_action_ade_m"]
            for row in action_rows.values()
        ),
        "scene_caching_improves_throughput": (
            min(cache_speedups) > thresholds["cache_min_speedup"]
            and bool(cache_grid["checks"]["cache_matches_reencoding"])
        ),
        "q4d_competitive_with_dense": (
            dense_rows["all"]["q4d_over_dense_ade"]
            <= thresholds["dense_overall_ade_ratio_max"]
            and dense_rows["contact"]["q4d_over_dense_ade"]
            <= thresholds["dense_task_ade_ratio_max"]
            and dense_rows["object"]["q4d_over_dense_ade"]
            <= thresholds["dense_task_ade_ratio_max"]
        ),
        "prediction_improvement_translates_to_mpc": (
            q4d_mpc["success_rate"] - no_action_mpc["success_rate"]
            >= thresholds["mpc_min_success_rate_margin"]
            and q4d_mpc["mean_final_cube_goal_distance_m"]
            < no_action_mpc["mean_final_cube_goal_distance_m"]
        ),
    }
    should_continue = source_integrity and all(gates.values())
    legacy_speedup = float(q4d["candidate_benchmark"]["cached_speedup"])
    return {
        "decision": "continue" if should_continue else "stop_and_diagnose",
        "continue": should_continue,
        "source_integrity": source_integrity,
        "gates": gates,
        "thresholds": thresholds,
        "evidence": {
            "action_conditioning": action_rows,
            "cache": {
                "configurations": len(cache_speedups),
                "minimum_speedup": min(cache_speedups),
                "median_speedup": median(cache_speedups),
                "maximum_speedup": max(cache_speedups),
                "maximum_physical_output_difference_m": max(
                    float(row["maximum_output_difference_m"])
                    for row in cache_grid["rows"]
                ),
            },
            "dense_comparison": dense_rows,
            "mpc": {
                "episodes_per_model": q4d_mpc["episodes"],
                "matched_seeds": sorted(q4d_seeds),
                "q4d_success_rate": q4d_mpc["success_rate"],
                "no_action_success_rate": no_action_mpc["success_rate"],
                "success_rate_margin": (
                    q4d_mpc["success_rate"] - no_action_mpc["success_rate"]
                ),
                "q4d_final_distance_m": q4d_mpc[
                    "mean_final_cube_goal_distance_m"
                ],
                "no_action_final_distance_m": no_action_mpc[
                    "mean_final_cube_goal_distance_m"
                ],
            },
        },
        "diagnostics": {
            "legacy_single_cache_speedup": legacy_speedup,
            "legacy_cache_result_disagrees": legacy_speedup <= 1.0,
            "cache_resolution": (
                "The exact-index nine-configuration grid is the gate measurement. "
                "It avoids FP16 nearest-point ambiguity, uses a physical equivalence "
                "tolerance, and shows a speedup in every configuration."
            ),
            "planning_limit": (
                "CEM success is variable across ten seeds; continue with random shooting "
                "as the trusted planner while diagnosing CEM convergence separately."
            ),
        },
        "next_action": (
            "Proceed to the next scoped experiment; retain matched controls and do not "
            "claim statistical finality from ten planning seeds."
            if should_continue
            else "Do not scale; diagnose each failed gate first."
        ),
    }
