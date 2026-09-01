#!/usr/bin/env python3
"""Summarize frozen submission-v1 H=8 multi-seed prediction results."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import statistics
import tomllib
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

MODELS = ("no_action", "micro_q4d", "dense")
METRICS = ("ade_m", "fde_m", "contact_ade_m", "object_ade_m", "p95_error_m")


def _read_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        return tomllib.load(stream)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def report_path(raw: dict[str, Any], task: str, seed: int, model: str) -> Path:
    if seed == int(raw["seeds"][task][0]):
        return Path(raw["existing_seed_one"][task]) / model / "report.json"
    return (
        Path(raw["experiment"]["output_root"])
        / task
        / f"seed_{seed}"
        / model
        / "report.json"
    )


def extract_metrics(report: dict[str, Any]) -> dict[str, float]:
    groups = report["test"]["groups"]
    return {
        "ade_m": float(groups["all"]["ade_m"]),
        "fde_m": float(groups["all"]["fde_m"]),
        "contact_ade_m": float(groups["contact"]["ade_m"]),
        "object_ade_m": float(groups["object"]["ade_m"]),
        "p95_error_m": float(report["test"]["p95_point_time_error_m"]),
    }


def percentile_interval(values: np.ndarray, confidence: float) -> list[float]:
    alpha = 1.0 - confidence
    return [
        float(np.quantile(values, alpha / 2.0)),
        float(np.quantile(values, 1.0 - alpha / 2.0)),
    ]


def bootstrap_mean_interval(
    values: list[float], repetitions: int, confidence: float, rng: np.random.Generator
) -> list[float]:
    source = np.asarray(values, dtype=np.float64)
    samples = rng.choice(source, size=(repetitions, len(source)), replace=True)
    return percentile_interval(samples.mean(axis=1), confidence)


def hierarchical_interval(
    by_task: dict[str, list[float]],
    repetitions: int,
    confidence: float,
    rng: np.random.Generator,
    statistic: Callable[[np.ndarray], float] | None = None,
) -> list[float]:
    """Resample tasks, then replicates within each sampled task."""
    tasks = tuple(by_task)
    statistic = statistic or (lambda values: float(values.mean()))
    draws = np.empty(repetitions, dtype=np.float64)
    for index in range(repetitions):
        sampled_tasks = rng.choice(tasks, size=len(tasks), replace=True)
        task_values = []
        for task in sampled_tasks:
            source = np.asarray(by_task[str(task)], dtype=np.float64)
            sampled = rng.choice(source, size=len(source), replace=True)
            task_values.append(statistic(sampled))
        draws[index] = float(np.mean(task_values))
    return percentile_interval(draws, confidence)


def exact_sign_flip_pvalue(differences: list[float]) -> float:
    """One-sided exact p-value for an alternative with mean difference below zero."""
    values = np.asarray(differences, dtype=np.float64)
    observed = float(values.mean())
    at_least_as_extreme = 0
    total = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(values)):
        permuted = float(np.mean(values * np.asarray(signs)))
        at_least_as_extreme += permuted <= observed + 1e-15
        total += 1
    return at_least_as_extreme / total


def _summary(
    values: list[float], repetitions: int, confidence: float, rng: np.random.Generator
) -> dict[str, Any]:
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "sample_standard_deviation": statistics.stdev(values) if len(values) > 1 else None,
        "confidence_interval": bootstrap_mean_interval(
            values, repetitions, confidence, rng
        ),
    }


def collect_rows(raw: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    rows = []
    failures = []
    for task in raw["experiment"]["tasks"]:
        for seed in (int(value) for value in raw["seeds"][task]):
            for model in MODELS:
                path = report_path(raw, task, seed, model)
                try:
                    report = _read_json(path)
                    metrics = extract_metrics(report)
                    if not bool(report["passed"]) or not all(
                        math.isfinite(value) for value in metrics.values()
                    ):
                        failures.append(f"failed or non-finite report: {path}")
                    rows.append(
                        {
                            "task": task,
                            "seed": seed,
                            "model": model,
                            "source": "preserved_seed_one"
                            if seed == int(raw["seeds"][task][0])
                            else "submission_v1_new_run",
                            "report": str(path),
                            **metrics,
                        }
                    )
                except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError) as error:
                    failures.append(f"unreadable report {path}: {error}")
    return rows, failures


def _lookup(rows: list[dict[str, Any]]) -> dict[tuple[str, int, str], dict[str, Any]]:
    return {(row["task"], row["seed"], row["model"]): row for row in rows}


def summarize(raw: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    repetitions = int(raw["statistics"]["bootstrap_repetitions"])
    confidence = float(raw["statistics"]["confidence_level"])
    rng = np.random.default_rng(int(raw["statistics"]["bootstrap_seed"]))
    tasks = tuple(raw["experiment"]["tasks"])
    lookup = _lookup(rows)
    per_task: dict[str, Any] = {}
    for task in tasks:
        task_models = {}
        for model in MODELS:
            model_rows = [row for row in rows if row["task"] == task and row["model"] == model]
            task_models[model] = {
                metric: _summary(
                    [float(row[metric]) for row in model_rows],
                    repetitions,
                    confidence,
                    rng,
                )
                for metric in METRICS
            }
        paired = []
        for seed in (int(value) for value in raw["seeds"][task]):
            q4d = lookup[(task, seed, "micro_q4d")]
            no_action = lookup[(task, seed, "no_action")]
            dense = lookup[(task, seed, "dense")]
            paired.append(
                {
                    "seed": seed,
                    "q4d_minus_no_action_ade_m": q4d["ade_m"] - no_action["ade_m"],
                    "q4d_relative_ade_improvement_percent": 100.0
                    * (1.0 - q4d["ade_m"] / no_action["ade_m"]),
                    "q4d_dense_ade_ratio": q4d["ade_m"] / dense["ade_m"],
                    "q4d_dense_contact_ade_ratio": q4d["contact_ade_m"]
                    / dense["contact_ade_m"],
                    "q4d_dense_object_ade_ratio": q4d["object_ade_m"]
                    / dense["object_ade_m"],
                }
            )
        per_task[task] = {"models": task_models, "paired": paired}

    task_macro: dict[str, Any] = {"models": {}}
    for model in MODELS:
        task_macro["models"][model] = {}
        for metric in METRICS:
            by_task = {
                task: [
                    float(row[metric])
                    for row in rows
                    if row["task"] == task and row["model"] == model
                ]
                for task in tasks
            }
            task_means = [statistics.fmean(by_task[task]) for task in tasks]
            task_macro["models"][model][metric] = {
                "mean": statistics.fmean(task_means),
                "sample_standard_deviation_across_task_means": statistics.stdev(
                    task_means
                ),
                "hierarchical_confidence_interval": hierarchical_interval(
                    by_task, repetitions, confidence, rng
                ),
            }

    paired_metric_names = (
        "q4d_minus_no_action_ade_m",
        "q4d_relative_ade_improvement_percent",
        "q4d_dense_ade_ratio",
        "q4d_dense_contact_ade_ratio",
        "q4d_dense_object_ade_ratio",
    )
    paired_macro = {}
    for name in paired_metric_names:
        by_task = {
            task: [float(row[name]) for row in per_task[task]["paired"]] for task in tasks
        }
        task_means = [statistics.fmean(by_task[task]) for task in tasks]
        paired_macro[name] = {
            "mean": statistics.fmean(task_means),
            "sample_standard_deviation_across_task_means": statistics.stdev(task_means),
            "hierarchical_confidence_interval": hierarchical_interval(
                by_task, repetitions, confidence, rng
            ),
        }

    q4d_rows = [row for row in rows if row["model"] == "micro_q4d"]
    shuffle_rows = []
    decoding_rows = []
    for row in q4d_rows:
        q4d_report = _read_json(Path(row["report"]))
        dense_report = _read_json(
            report_path(raw, row["task"], int(row["seed"]), "dense")
        )
        shuffled = extract_metrics(q4d_report["action_shuffle"])
        shuffle_rows.append(
            {
                "task": row["task"],
                "seed": row["seed"],
                "shuffled_minus_unshuffled_ade_m": shuffled["ade_m"] - row["ade_m"],
                "shuffled_minus_unshuffled_contact_ade_m": shuffled["contact_ade_m"]
                - row["contact_ade_m"],
            }
        )
        benchmark = dense_report["matched_candidate_benchmark"]
        decoding_rows.append(
            {
                "task": row["task"],
                "seed": row["seed"],
                "sparse_speedup": float(benchmark["sparse_speedup"]),
                "sparse_cached_milliseconds": float(
                    benchmark["sparse_cached_milliseconds"]
                ),
                "dense_cached_milliseconds": float(
                    benchmark["dense_cached_milliseconds"]
                ),
                "same_scene_and_candidate_actions": bool(
                    benchmark["same_scene_and_candidate_actions"]
                ),
            }
        )

    shuffle_summary = {}
    for name in (
        "shuffled_minus_unshuffled_ade_m",
        "shuffled_minus_unshuffled_contact_ade_m",
    ):
        by_task = {
            task: [float(row[name]) for row in shuffle_rows if row["task"] == task]
            for task in tasks
        }
        shuffle_summary[name] = {
            "mean": statistics.fmean(value for values in by_task.values() for value in values),
            "hierarchical_confidence_interval": hierarchical_interval(
                by_task, repetitions, confidence, rng
            ),
        }
    decoding_by_task = {
        task: [
            float(row["sparse_speedup"])
            for row in decoding_rows
            if row["task"] == task
        ]
        for task in tasks
    }
    decoding_summary = {
        "mean_sparse_speedup": statistics.fmean(
            value for values in decoding_by_task.values() for value in values
        ),
        "hierarchical_confidence_interval": hierarchical_interval(
            decoding_by_task, repetitions, confidence, rng
        ),
    }

    differences = [
        float(row["q4d_minus_no_action_ade_m"])
        for task in tasks
        for row in per_task[task]["paired"]
    ]
    primary = paired_macro["q4d_minus_no_action_ade_m"]
    primary_p = exact_sign_flip_pvalue(differences)
    thresholds = _read_toml(Path(raw["experiment"]["protocol"]))["thresholds"]
    gates = {
        "primary_q4d_vs_no_action": primary["mean"] < 0
        and primary["hierarchical_confidence_interval"][1] < 0
        and primary_p < 0.05,
        "q4d_dense_overall_ade_ratio": paired_macro["q4d_dense_ade_ratio"][
            "hierarchical_confidence_interval"
        ][1]
        <= float(thresholds["q4d_dense_overall_ade_ratio_max"]),
        "q4d_dense_contact_ade_ratio": paired_macro[
            "q4d_dense_contact_ade_ratio"
        ]["hierarchical_confidence_interval"][1]
        <= float(thresholds["q4d_dense_contact_ade_ratio_max"]),
        "q4d_dense_object_ade_ratio": paired_macro["q4d_dense_object_ade_ratio"][
            "hierarchical_confidence_interval"
        ][1]
        <= float(thresholds["q4d_dense_object_ade_ratio_max"]),
        "action_shuffle_degrades_overall_ade": shuffle_summary[
            "shuffled_minus_unshuffled_ade_m"
        ]["hierarchical_confidence_interval"][0]
        > 0,
        "all_decoding_benchmarks_matched": all(
            row["same_scene_and_candidate_actions"] for row in decoding_rows
        ),
    }
    return {
        "per_seed": rows,
        "per_task": per_task,
        "task_macro": task_macro,
        "paired_task_macro": paired_macro,
        "primary_test": {
            "name": "one-sided exact paired sign-flip test",
            "replicates": len(differences),
            "q4d_minus_no_action_ade_m": primary,
            "p_value": primary_p,
        },
        "action_shuffle": {"per_seed": shuffle_rows, "summary": shuffle_summary},
        "decoding_benchmark": {
            "per_seed": decoding_rows,
            "summary": decoding_summary,
        },
        "gates": gates,
        "passed": all(gates.values()),
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Submission-v1 H=8 multi-seed prediction results",
        "",
        "All distances are reported in millimetres below. Replicates are task × "
        "training seed, not points.",
        "",
        "| Task | Seed | Model | ADE | FDE | Contact ADE | Object ADE | p95 |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["per_seed"]:
        lines.append(
            f"| {row['task']} | {row['seed']} | {row['model']} | "
            f"{1000 * row['ade_m']:.3f} | {1000 * row['fde_m']:.3f} | "
            f"{1000 * row['contact_ade_m']:.3f} | {1000 * row['object_ade_m']:.3f} | "
            f"{1000 * row['p95_error_m']:.3f} |"
        )
    lines.extend(["", "## Frozen gates", ""])
    for name, passed in report["gates"].items():
        lines.append(f"- {'PASS' if passed else 'FAIL'}: `{name}`")
    primary = report["primary_test"]
    lines.extend(
        [
            "",
            f"Primary exact sign-flip p-value: {primary['p_value']:.6g}.",
            "",
            f"Overall gate: {'PASS' if report['passed'] else 'FAIL'}.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/submission_v1/multiseed_h8.toml"),
    )
    args = parser.parse_args()
    raw = _read_toml(args.config)
    rows, failures = collect_rows(raw)
    expected = len(raw["experiment"]["tasks"]) * 3 * len(MODELS)
    if failures or len(rows) != expected:
        raise RuntimeError("multi-seed matrix is incomplete:\n" + "\n".join(failures))
    report = {
        "experiment_id": raw["experiment"]["id"],
        "horizon": int(raw["experiment"]["horizon"]),
        "training_runs": len(rows),
        "additional_training_runs": 24,
        **summarize(raw, rows),
    }
    output_root = Path(raw["experiment"]["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    output = output_root / "summary.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (output_root / "summary.md").write_text(_markdown(report), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise RuntimeError("one or more frozen multi-seed prediction gates failed")


if __name__ == "__main__":
    main()
