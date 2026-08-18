#!/usr/bin/env python3
"""Freeze splits/statistics and verify GPU-friendly batching for trajectory fragments."""

from __future__ import annotations

import argparse
import json
import time
import tomllib
from pathlib import Path
from typing import Any

import numpy as np
import torch

from q4d_wam.data import (
    DataLoaderConfig,
    SplitManifest,
    TrackDataset,
    build_dataloader,
    compute_normalization,
    discover_training_files,
    move_batch_to_device,
    trajectory_group_id,
    validate_training_file,
)
from q4d_wam.labels import CATEGORY_OBJECT


def _tensor_shapes(batch: dict[str, torch.Tensor]) -> dict[str, dict[str, Any]]:
    return {
        key: {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "pinned": value.is_pinned(),
        }
        for key, value in batch.items()
    }


def _batch_bytes(batch: dict[str, torch.Tensor]) -> int:
    return sum(value.numel() * value.element_size() for value in batch.values())


def _query_object_coverage(dataset: TrackDataset) -> dict[str, Any]:
    object_counts = []
    missing_audits = []
    for index, training_path in enumerate(dataset.files):
        audit_path = training_path.with_name(training_path.name.replace(".train.npz", ".audit.npz"))
        if not audit_path.exists():
            missing_audits.append(audit_path.name)
            continue
        query_indices = dataset[index]["query_indices"].numpy()
        with np.load(audit_path, allow_pickle=False) as audit:
            categories = audit["point_categories"]
        object_counts.append(int(np.sum(categories[query_indices] == CATEGORY_OBJECT)))
    return {
        "audited_files": len(object_counts),
        "missing_audit_files": missing_audits,
        "episodes_with_object_query": int(np.sum(np.asarray(object_counts) > 0)),
        "minimum_object_queries": min(object_counts) if object_counts else None,
        "mean_object_queries": float(np.mean(object_counts)) if object_counts else None,
        "note": "Audit-only check; body categories are never returned by TrackDataset.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/data.toml"))
    args = parser.parse_args()
    with args.config.open("rb") as stream:
        raw = tomllib.load(stream)

    dataset_config = raw["dataset"]
    loader_raw = raw["loader"]
    root = Path(dataset_config["root"])
    files = discover_training_files(root)
    split = SplitManifest.create(
        files,
        seed=int(dataset_config["split_seed"]),
        train_fraction=float(dataset_config["train_fraction"]),
        validation_fraction=float(dataset_config["validation_fraction"]),
        test_fraction=float(dataset_config["test_fraction"]),
    )
    split_path = root / "splits.json"
    split.save(split_path)

    schema_records = [validate_training_file(path) for path in files]
    unique_shapes = {
        (record["points"], record["horizon"], record["action_dimensions"])
        for record in schema_records
    }
    if len(unique_shapes) != 1:
        raise RuntimeError(f"inconsistent dataset sample shapes: {sorted(unique_shapes)}")

    training_files = split.files(root, "train")
    normalization = compute_normalization(training_files)
    normalization_path = root / "normalization.json"
    normalization.save(normalization_path)

    datasets = {
        name: TrackDataset(
            split.files(root, name),
            normalization,
            num_queries=int(dataset_config["num_queries"]),
            cache_size=int(dataset_config["cache_size"]),
        )
        for name in ("train", "validation", "test")
    }
    loader_config = DataLoaderConfig(
        batch_size=int(loader_raw["batch_size"]),
        num_workers=int(loader_raw["num_workers"]),
        prefetch_factor=int(loader_raw["prefetch_factor"]),
        pin_memory=bool(loader_raw["pin_memory"]),
        persistent_workers=bool(loader_raw["persistent_workers"]),
    )
    train_loader = build_dataloader(
        datasets["train"],
        loader_config,
        shuffle=True,
        seed=split.seed,
        drop_last=True,
    )

    warmup_start = time.perf_counter()
    batch = next(iter(train_loader))
    warmup_seconds = time.perf_counter() - warmup_start
    batch_bytes = _batch_bytes(batch)

    epoch_start = time.perf_counter()
    epoch_batches = 0
    epoch_samples = 0
    for epoch_batch in train_loader:
        epoch_batches += 1
        epoch_samples += len(epoch_batch["sample_id"])
    epoch_seconds = time.perf_counter() - epoch_start

    gpu_report: dict[str, Any]
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.synchronize(device)
        transfer_start = time.perf_counter()
        gpu_batch = move_batch_to_device(batch, device, non_blocking=True)
        torch.cuda.synchronize(device)
        transfer_seconds = time.perf_counter() - transfer_start
        gpu_report = {
            "available": True,
            "device": torch.cuda.get_device_name(device),
            "all_tensors_on_cuda": all(value.is_cuda for value in gpu_batch.values()),
            "transfer_milliseconds": transfer_seconds * 1000,
            "batch_mebibytes": batch_bytes / 2**20,
            "cuda_allocated_mebibytes": torch.cuda.memory_allocated(device) / 2**20,
        }
        del gpu_batch
    else:
        gpu_report = {"available": False}

    query_object_coverage = _query_object_coverage(datasets["train"])
    report = {
        "dataset_root": str(root),
        "training_archives": len(files),
        "sample_shape": {
            "points": next(iter(unique_shapes))[0],
            "horizon": next(iter(unique_shapes))[1],
            "action_dimensions": next(iter(unique_shapes))[2],
            "queries": int(dataset_config["num_queries"]),
        },
        "splits": split.to_dict()["counts"],
        "split_groups": split.to_dict()["group_counts"],
        "normalization": {
            "fit_scope": "training split only",
            "source_files": len(normalization.source_files),
            "constant_action_channels": list(normalization.constant_action_channels),
            "path": str(normalization_path),
        },
        "query_object_coverage": query_object_coverage,
        "loader": {
            "batch_size": loader_config.batch_size,
            "num_workers": loader_config.num_workers,
            "pin_memory_requested": loader_config.pin_memory,
            "all_batch_tensors_pinned": all(value.is_pinned() for value in batch.values()),
            "warmup_seconds": warmup_seconds,
            "epoch_batches": epoch_batches,
            "epoch_samples": epoch_samples,
            "epoch_seconds": epoch_seconds,
            "samples_per_second": epoch_samples / epoch_seconds,
            "batch": _tensor_shapes(batch),
        },
        "gpu_transfer": gpu_report,
        "checks": {
            "split_is_exhaustive": len(set(split.all_files)) == len(files),
            "split_groups_are_disjoint": not (
                {trajectory_group_id(name) for name in split.train}
                & {trajectory_group_id(name) for name in split.validation}
                or {trajectory_group_id(name) for name in split.train}
                & {trajectory_group_id(name) for name in split.test}
                or {trajectory_group_id(name) for name in split.validation}
                & {trajectory_group_id(name) for name in split.test}
            ),
            "normalization_is_train_only": set(normalization.source_files) == set(split.train),
            "schema_is_consistent": len(unique_shapes) == 1,
            "no_privileged_batch_fields": not any(
                "body" in key or "segmentation" in key or "local" in key for key in batch
            ),
            "gpu_transfer_passed": bool(gpu_report.get("all_tensors_on_cuda", False)),
            "object_queries_preserved": (
                query_object_coverage["minimum_object_queries"] or 0
            )
            > 0,
        },
    }
    report["passed"] = all(report["checks"].values())
    report_path = root / "loader_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        failed = [name for name, passed in report["checks"].items() if not passed]
        raise RuntimeError(f"dataset preparation failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
