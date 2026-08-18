#!/usr/bin/env python3
"""Aggregate item-12 results and evaluate its prediction/compute gate."""

from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        return tomllib.load(stream)


def _millimetres(report: dict[str, Any], group: str, metric: str = "ade_m") -> float:
    return 1000.0 * float(report["test"]["groups"][group][metric])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/scale_experiment.toml"))
    args = parser.parse_args()
    raw = _read_toml(args.config)
    experiment = raw["experiment"]
    dataset_root = Path(experiment["dataset_root"])
    output_root = Path(experiment["output_root"])
    collection = _read_json(dataset_root / "manifest.json")
    rows = []
    checks: dict[str, bool] = {
        "collection_complete": bool(collection["complete"]),
        "minimum_fragment_count": collection["fragments"]
        >= int(experiment["minimum_fragments"]),
        "successful_outcomes_present": collection["observed_outcome_counts"]["successful"]
        > 0,
        "weak_outcomes_present": collection["observed_outcome_counts"]["weak"] > 0,
        "off_target_outcomes_present": collection["observed_outcome_counts"]["off_target"]
        > 0,
        "failed_or_no_motion_outcomes_present": collection["observed_outcome_counts"][
            "no_motion"
        ]
        > 0,
    }
    for horizon in (int(value) for value in experiment["horizons"]):
        horizon_root = output_root / f"h{horizon}"
        static = _read_json(horizon_root / "non_neural.json")
        no_action = _read_json(horizon_root / "no_action" / "report.json")
        q4d = _read_json(horizon_root / "micro_q4d" / "report.json")
        dense = _read_json(horizon_root / "dense" / "report.json")
        shuffled = q4d["action_shuffle"]
        cache = q4d["candidate_benchmark"]
        dense_benchmark = dense["matched_candidate_benchmark"]
        row = {
            "horizon": horizon,
            "static_ade_mm": 1000.0
            * static["baselines"]["static"]["groups"]["all"]["ade_m"],
            "no_action_ade_mm": _millimetres(no_action, "all"),
            "dense_ade_mm": _millimetres(dense, "all"),
            "q4d_ade_mm": _millimetres(q4d, "all"),
            "q4d_fde_mm": _millimetres(q4d, "all", "fde_m"),
            "q4d_contact_ade_mm": _millimetres(q4d, "contact"),
            "shuffled_q4d_ade_mm": 1000.0
            * shuffled["test"]["groups"]["all"]["ade_m"],
            "shuffled_q4d_contact_ade_mm": 1000.0
            * shuffled["test"]["groups"]["contact"]["ade_m"],
            "cache_speedup": cache["cached_speedup"],
            "sparse_vs_dense_speedup": dense_benchmark["sparse_speedup"],
            "q4d_cached_latency_ms": cache["cached_milliseconds"],
            "dense_cached_latency_ms": dense_benchmark["dense_cached_milliseconds"],
            "q4d_training_peak_reserved_mib": q4d["training"]["memory"][
                "peak_reserved_mib"
            ],
            "dense_training_peak_reserved_mib": dense["training"]["memory"][
                "peak_reserved_mib"
            ],
        }
        rows.append(row)
        checks[f"h{horizon}_reports_pass"] = all(
            report["passed"] for report in (static, no_action, q4d, dense)
        )
        checks[f"h{horizon}_q4d_beats_no_action_all"] = (
            row["q4d_ade_mm"] < row["no_action_ade_mm"]
        )
        checks[f"h{horizon}_action_shuffle_hurts_all"] = (
            row["shuffled_q4d_ade_mm"] > row["q4d_ade_mm"]
        )
        checks[f"h{horizon}_action_shuffle_hurts_contact"] = (
            row["shuffled_q4d_contact_ade_mm"] > row["q4d_contact_ade_mm"]
        )
        checks[f"h{horizon}_cache_is_faster"] = row["cache_speedup"] > 1.0
        checks[f"h{horizon}_sparse_is_faster_than_dense"] = (
            row["sparse_vs_dense_speedup"] > 1.0
        )
    scaling = _read_json(output_root / "h8" / "n_m_scaling.json")
    checks["larger_n_m_grid_passes_memory_contract"] = bool(scaling["passed"])
    prediction_checks = [
        value
        for name, value in checks.items()
        if "q4d_beats" in name or "action_shuffle_hurts" in name
    ]
    compute_checks = [
        value
        for name, value in checks.items()
        if "cache_is_faster" in name or "sparse_is_faster" in name
    ]
    report = {
        "collection": {
            "fragments": collection["fragments"],
            "state_groups": collection["completed_states"],
            "outcomes": collection["observed_outcome_counts"],
        },
        "horizon_results": rows,
        "checks": checks,
        "gate": {
            "prediction_hypothesis_survives": all(prediction_checks),
            "computational_hypothesis_survives": all(compute_checks),
            "all_integrity_checks_pass": all(checks.values()),
        },
    }
    report["passed"] = all(report["gate"].values())
    output = output_root / "gate_report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise RuntimeError("the item-12 scientific gate did not pass")


if __name__ == "__main__":
    main()
