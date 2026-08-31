#!/usr/bin/env python3
"""Validate the frozen four-task scale results and compute task macro-averages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean
from typing import Any

TASKS = ("place_sphere", "stack_cube", "pull_cube", "pick_cube")
HORIZONS = (1, 2, 4, 8)


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _mm(value: float) -> float:
    return 1000.0 * float(value)


def _mean(rows: list[dict[str, Any]], name: str) -> float:
    return fmean(float(row[name]) for row in rows)


def _task_report(task: str, experiments_root: Path) -> dict[str, Any]:
    root = experiments_root / f"{task}_scale_v1"
    rows: list[dict[str, Any]] = []
    report_passes: list[bool] = []
    for horizon in HORIZONS:
        horizon_root = root / f"h{horizon}"
        static = _read(horizon_root / "non_neural.json")
        no_action = _read(horizon_root / "no_action" / "report.json")
        q4d = _read(horizon_root / "micro_q4d" / "report.json")
        dense = _read(horizon_root / "dense" / "report.json")
        shuffled = q4d["action_shuffle"]["test"]["groups"]
        q4d_groups = q4d["test"]["groups"]
        no_action_groups = no_action["test"]["groups"]
        cache = q4d["candidate_benchmark"]
        dense_benchmark = dense["matched_candidate_benchmark"]
        reports_pass = all(
            bool(report["passed"]) for report in (static, no_action, q4d, dense)
        )
        report_passes.append(reports_pass)
        rows.append(
            {
                "horizon": horizon,
                "reports_pass": reports_pass,
                "q4d_ade_mm": _mm(q4d_groups["all"]["ade_m"]),
                "q4d_fde_mm": _mm(q4d_groups["all"]["fde_m"]),
                "q4d_contact_ade_mm": _mm(q4d_groups["contact"]["ade_m"]),
                "q4d_p95_error_mm": _mm(q4d["test"]["p95_point_time_error_m"]),
                "no_action_ade_mm": _mm(no_action_groups["all"]["ade_m"]),
                "shuffled_ade_mm": _mm(shuffled["all"]["ade_m"]),
                "shuffled_contact_ade_mm": _mm(shuffled["contact"]["ade_m"]),
                "cache_maximum_output_difference_normalized": float(
                    cache["maximum_output_difference_normalized"]
                ),
                "cache_speedup": float(cache["cached_speedup"]),
                "sparse_vs_dense_speedup": float(dense_benchmark["sparse_speedup"]),
                "q4d_beats_no_action": q4d_groups["all"]["ade_m"]
                < no_action_groups["all"]["ade_m"],
                "action_shuffle_hurts_overall": shuffled["all"]["ade_m"]
                > q4d_groups["all"]["ade_m"],
                "action_shuffle_hurts_contact": shuffled["contact"]["ade_m"]
                > q4d_groups["contact"]["ade_m"],
                "cache_matches_reencoding": bool(
                    q4d["checks"]["cache_matches_reencoding"]
                ),
                "cache_is_faster": cache["cached_speedup"] > 1.0,
                "sparse_is_faster_than_dense": dense_benchmark["sparse_speedup"]
                > 1.0,
            }
        )

    grid = _read(root / "h8" / "n_m_scaling.json")
    gate = _read(root / "gate_report.json")
    checks = {
        "all_model_reports_pass": all(report_passes),
        "q4d_beats_no_action_every_horizon": all(
            row["q4d_beats_no_action"] for row in rows
        ),
        "action_shuffle_hurts_overall_every_horizon": all(
            row["action_shuffle_hurts_overall"] for row in rows
        ),
        "action_shuffle_hurts_contact_every_horizon": all(
            row["action_shuffle_hurts_contact"] for row in rows
        ),
        "cache_matches_reencoding_every_horizon": all(
            row["cache_matches_reencoding"] for row in rows
        ),
        "cache_is_faster_every_horizon": all(
            row["cache_is_faster"] for row in rows
        ),
        "sparse_is_faster_than_dense_every_horizon": all(
            row["sparse_is_faster_than_dense"] for row in rows
        ),
        "h8_grid_passes_memory_contract": bool(grid["passed"]),
    }
    return {
        "task": task,
        "horizons": rows,
        "checks": checks,
        "scientific_gate": {
            **gate["gate"],
            "passed": bool(gate["passed"]),
            "failed_checks": [
                name for name, passed in gate["checks"].items() if not passed
            ],
        },
    }


def _macro_report(tasks: list[dict[str, Any]]) -> dict[str, Any]:
    metric_names = (
        "q4d_ade_mm",
        "q4d_fde_mm",
        "q4d_contact_ade_mm",
        "q4d_p95_error_mm",
        "no_action_ade_mm",
        "shuffled_ade_mm",
        "shuffled_contact_ade_mm",
        "cache_maximum_output_difference_normalized",
        "cache_speedup",
        "sparse_vs_dense_speedup",
    )
    by_horizon = []
    for index, horizon in enumerate(HORIZONS):
        task_rows = [task["horizons"][index] for task in tasks]
        by_horizon.append(
            {
                "horizon": horizon,
                **{name: _mean(task_rows, name) for name in metric_names},
            }
        )
    task_means = []
    for task in tasks:
        task_means.append(
            {
                "task": task["task"],
                **{name: _mean(task["horizons"], name) for name in metric_names},
            }
        )
    check_names = tuple(tasks[0]["checks"])
    return {
        "replicate_unit": "task",
        "individual_points_used_as_replicates": False,
        "task_count": len(tasks),
        "macro_average_by_horizon": by_horizon,
        "per_task_horizon_mean": task_means,
        "overall_task_macro_average": {
            name: _mean(task_means, name) for name in metric_names
        },
        "task_pass_rates": {
            name: fmean(float(task["checks"][name]) for task in tasks)
            for name in check_names
        },
    }


def _mark(value: bool) -> str:
    return "PASS" if value else "FAIL"


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Four-task prediction validation",
        "",
        "Each task is an independent replicate. Individual points are not treated "
        "as independent replicates.",
        "",
    ]
    for task in report["tasks"]:
        lines.extend(
            [
                f"## {task['task']}",
                "",
                "| H | ADE (mm) | FDE (mm) | Contact ADE (mm) | p95 (mm) | "
                "No-action ADE (mm) | Shuffled ADE (mm) | Shuffled contact ADE "
                "(mm) | Cache speedup | Sparse/dense speedup |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in task["horizons"]:
            lines.append(
                f"| {row['horizon']} | {row['q4d_ade_mm']:.3f} | "
                f"{row['q4d_fde_mm']:.3f} | {row['q4d_contact_ade_mm']:.3f} | "
                f"{row['q4d_p95_error_mm']:.3f} | {row['no_action_ade_mm']:.3f} | "
                f"{row['shuffled_ade_mm']:.3f} | "
                f"{row['shuffled_contact_ade_mm']:.3f} | {row['cache_speedup']:.3f}x | "
                f"{row['sparse_vs_dense_speedup']:.3f}x |"
            )
        lines.extend(["", "Checks:", ""])
        for name, passed in task["checks"].items():
            lines.append(f"- {_mark(passed)}: `{name}`")
        gate = task["scientific_gate"]
        failed = ", ".join(gate["failed_checks"]) or "none"
        lines.extend(
            [
                "",
                f"Scientific gate: {_mark(gate['passed'])}. Failed checks: {failed}.",
                "",
            ]
        )

    macro = report["macro_average"]
    lines.extend(
        [
            "## Task-level macro-average",
            "",
            "| H | ADE (mm) | FDE (mm) | Contact ADE (mm) | p95 (mm) | "
            "No-action ADE (mm) | Shuffled ADE (mm) | Cache speedup | "
            "Sparse/dense speedup |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in macro["macro_average_by_horizon"]:
        lines.append(
            f"| {row['horizon']} | {row['q4d_ade_mm']:.3f} | "
            f"{row['q4d_fde_mm']:.3f} | {row['q4d_contact_ade_mm']:.3f} | "
            f"{row['q4d_p95_error_mm']:.3f} | {row['no_action_ade_mm']:.3f} | "
            f"{row['shuffled_ade_mm']:.3f} | {row['cache_speedup']:.3f}x | "
            f"{row['sparse_vs_dense_speedup']:.3f}x |"
        )
    lines.extend(["", "Task pass rates:", ""])
    for name, rate in macro["task_pass_rates"].items():
        lines.append(f"- `{name}`: {100.0 * rate:.1f}%")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiments-root", type=Path, default=Path("artifacts/experiments")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/experiments/multitask_validation.json"),
    )
    args = parser.parse_args()
    tasks = [_task_report(task, args.experiments_root) for task in TASKS]
    report = {
        "validation_complete": True,
        "thresholds_changed_after_test_data": False,
        "tasks": tasks,
        "macro_average": _macro_report(tasks),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    markdown_path = args.output.with_suffix(".md")
    markdown_path.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
