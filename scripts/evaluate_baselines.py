#!/usr/bin/env python3
"""Evaluate simple non-neural trajectory predictors on the frozen test split."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import tomllib
from pathlib import Path
from typing import Any

import numpy as np
import torch

from q4d_wam.baselines import (
    ActionKnnBaseline,
    MeanDisplacementBaseline,
    SceneKnnBaseline,
    StaticBaseline,
)
from q4d_wam.data import (
    DataLoaderConfig,
    NormalizationStats,
    SplitManifest,
    TrackDataset,
    build_dataloader,
    trajectory_group_id,
)
from q4d_wam.evaluation import TrajectoryMetricAccumulator, load_audit_metadata


def _read_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        return tomllib.load(stream)


def _manifest_digest(manifest: SplitManifest) -> str:
    names = "\n".join(manifest.all_files).encode()
    return hashlib.sha256(names).hexdigest()


def _evaluate(
    baseline: object,
    dataset: TrackDataset,
    loader: torch.utils.data.DataLoader[dict[str, torch.Tensor]],
    *,
    horizon: int,
    moving_threshold_m: float,
    compute_geometry_metrics: bool,
) -> dict[str, Any]:
    accumulator = TrajectoryMetricAccumulator(
        horizon=horizon,
        moving_threshold_m=moving_threshold_m,
        compute_geometry_metrics=compute_geometry_metrics,
    )
    prediction_seconds = 0.0
    episodes = 0
    queries = 0
    audit_files = []
    for batch in loader:
        point_groups, body_indices, batch_audits = load_audit_metadata(dataset, batch)
        start = time.perf_counter()
        prediction = baseline.predict(batch)  # type: ignore[attr-defined]
        prediction_seconds += time.perf_counter() - start
        accumulator.update(
            prediction,
            batch["target_world_m"],
            batch["query_xyz_world_m"],
            point_groups=point_groups,
            body_indices=body_indices,
        )
        episodes += len(batch["sample_id"])
        queries += batch["query_xyz_world_m"].shape[0] * batch["query_xyz_world_m"].shape[1]
        audit_files.extend(batch_audits)
    metrics = accumulator.report()
    metrics["runtime"] = {
        "prediction_seconds": prediction_seconds,
        "episodes_per_second": episodes / prediction_seconds,
        "queries_per_second": queries / prediction_seconds,
        "note": "CPU predictor time only; data loading and metric computation excluded.",
    }
    metrics["audit_files"] = audit_files
    return metrics


def _matched_retrieval_comparison(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    comparison = {}
    for group_name in ("all", "moving", "contact", "object"):
        scene_ade = results["scene_knn"]["groups"][group_name]["ade_m"]
        action_ade = results["action_knn"]["groups"][group_name]["ade_m"]
        comparison[group_name] = {
            "scene_knn_ade_m": scene_ade,
            "action_knn_ade_m": action_ade,
            "action_minus_scene_ade_m": action_ade - scene_ade,
            "action_relative_change_percent": 100.0 * (action_ade / scene_ade - 1.0),
            "action_is_better": action_ade < scene_ade,
        }
    action_wins_overall = comparison["all"]["action_is_better"]
    return {
        "lower_is_better": True,
        "groups": comparison,
        "conclusion": (
            "Action retrieval beats matched scene-only retrieval overall, confirming "
            "that the grouped corpus exposes action-dependent futures."
            if action_wins_overall
            else "Action retrieval does not beat matched scene-only retrieval overall; "
            "collect counterfactual action branches before learned action conditioning."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/evaluation.toml"))
    args = parser.parse_args()
    evaluation = _read_toml(args.config)["evaluation"]
    data_config = _read_toml(Path(evaluation["data_config"]))["dataset"]

    dataset_root = Path(data_config["root"])
    manifest = SplitManifest.load(evaluation["split_manifest"])
    normalization = NormalizationStats.load(evaluation["normalization"])
    split_name = str(evaluation["split"])
    evaluation_queries = int(evaluation["evaluation_queries"])
    requested_horizon = (
        int(evaluation["horizon"]) if "horizon" in evaluation else None
    )
    training_dataset = TrackDataset(
        manifest.files(dataset_root, "train"),
        normalization,
        num_queries=evaluation_queries,
        horizon=requested_horizon,
        cache_size=int(data_config["cache_size"]),
    )
    evaluation_dataset = TrackDataset(
        manifest.files(dataset_root, split_name),
        normalization,
        num_queries=evaluation_queries,
        horizon=requested_horizon,
        cache_size=int(data_config["cache_size"]),
    )
    loader = build_dataloader(
        evaluation_dataset,
        DataLoaderConfig(
            batch_size=int(evaluation["batch_size"]),
            num_workers=int(evaluation["num_workers"]),
            pin_memory=False,
            persistent_workers=False,
        ),
        shuffle=False,
        seed=manifest.seed,
    )
    first_sample = evaluation_dataset[0]
    horizon = first_sample["target_world_m"].shape[1]

    fit_start = time.perf_counter()
    mean_baseline = MeanDisplacementBaseline.fit(training_dataset)
    mean_fit_seconds = time.perf_counter() - fit_start
    fit_start = time.perf_counter()
    scene_knn_baseline = SceneKnnBaseline.fit(
        training_dataset, neighbors=int(evaluation["knn_neighbors"])
    )
    scene_knn_fit_seconds = time.perf_counter() - fit_start
    fit_start = time.perf_counter()
    knn_baseline = ActionKnnBaseline.fit(
        training_dataset, neighbors=int(evaluation["knn_neighbors"])
    )
    knn_fit_seconds = time.perf_counter() - fit_start
    baselines = [StaticBaseline(), mean_baseline, scene_knn_baseline, knn_baseline]
    results = {
        baseline.name: _evaluate(
            baseline,
            evaluation_dataset,
            loader,
            horizon=horizon,
            moving_threshold_m=float(evaluation["moving_threshold_m"]),
            compute_geometry_metrics=bool(
                evaluation.get("compute_geometry_metrics", True)
            ),
        )
        for baseline in baselines
    }

    all_reports_finite = all(
        all(
            value is None or np.isfinite(value)
            for group in result["groups"].values()
            for key, value in group.items()
            if key != "points"
        )
        for result in results.values()
    )
    report = {
        "protocol": {
            "dataset_root": str(dataset_root),
            "fit_split": "train",
            "evaluation_split": split_name,
            "split_seed": manifest.seed,
            "split_manifest_sha256": _manifest_digest(manifest),
            "training_fragments": len(training_dataset),
            "evaluation_fragments": len(evaluation_dataset),
            "training_state_groups": len(
                {trajectory_group_id(path) for path in training_dataset.files}
            ),
            "evaluation_state_groups": len(
                {trajectory_group_id(path) for path in evaluation_dataset.files}
            ),
            "queries_per_episode": evaluation_queries,
            "horizon": horizon,
            "moving_threshold_m": float(evaluation["moving_threshold_m"]),
            "privileged_data_policy": (
                "Audit labels only stratify reported metrics; no baseline receives them."
            ),
        },
        "fit_seconds": {
            mean_baseline.name: mean_fit_seconds,
            scene_knn_baseline.name: scene_knn_fit_seconds,
            knn_baseline.name: knn_fit_seconds,
        },
        "baselines": results,
        "matched_retrieval_action_effect": _matched_retrieval_comparison(results),
        "checks": {
            "train_and_evaluation_disjoint": not (
                set(manifest.train) & set(getattr(manifest, split_name))
            ),
            "train_and_evaluation_groups_disjoint": not (
                {trajectory_group_id(name) for name in manifest.train}
                & {trajectory_group_id(name) for name in getattr(manifest, split_name)}
            ),
            "normalization_is_train_only": set(normalization.source_files)
            == set(manifest.train),
            "all_evaluation_points_used": evaluation_queries
            == first_sample["scene_xyz"].shape[0],
            "contact_points_evaluated": results["static"]["groups"]["contact"]["points"] > 0,
            "moving_points_evaluated": results["static"]["groups"]["moving"]["points"] > 0,
            "metrics_are_finite": all_reports_finite,
        },
    }
    report["passed"] = all(report["checks"].values())
    output = Path(evaluation["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        failed = [name for name, passed in report["checks"].items() if not passed]
        raise RuntimeError(f"baseline evaluation failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
