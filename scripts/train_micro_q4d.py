#!/usr/bin/env python3
"""Train, evaluate, and benchmark the checklist-item-9 micro-Q4D model."""

from __future__ import annotations

import argparse
import inspect
import itertools
import json
import random
import time
import tomllib
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from q4d_wam.data import (
    DataLoaderConfig,
    NormalizationStats,
    SplitManifest,
    TrackDataset,
    build_dataloader,
    trajectory_group_id,
)
from q4d_wam.evaluation import TrajectoryMetricAccumulator, load_audit_metadata
from q4d_wam.models import MicroQ4D
from q4d_wam.training import CudaMemoryBudget, bytes_to_mib


def _read_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        return tomllib.load(stream)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _model_inputs(
    batch: dict[str, Tensor], device: torch.device
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return tuple(
        batch[key].to(device=device, non_blocking=True)
        for key in ("scene_xyz", "scene_rgb", "actions", "query_xyz")
    )  # type: ignore[return-value]


@torch.no_grad()
def _validation_loss(
    model: nn.Module,
    loader: torch.utils.data.DataLoader[dict[str, Tensor]],
    device: torch.device,
    use_amp: bool,
) -> float:
    model.eval()
    total = 0.0
    values = 0
    for batch in loader:
        target = batch["target_displacement"].to(device=device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            prediction = model(*_model_inputs(batch, device))
            loss = torch.nn.functional.mse_loss(prediction, target, reduction="sum")
        total += float(loss)
        values += target.numel()
    return total / values


def _sibling_separation(groups: dict[str, list[Tensor]]) -> dict[str, float]:
    distances = []
    maximum = 0.0
    for values in groups.values():
        for first, second in itertools.combinations(values, 2):
            point_time_distance = torch.linalg.vector_norm(first - second, dim=-1)
            distances.append(float(point_time_distance.mean()))
            maximum = max(maximum, float(point_time_distance.max()))
    return {
        "mean_pairwise_point_time_distance_m": float(np.mean(distances)),
        "maximum_point_time_distance_m": maximum,
    }


@torch.no_grad()
def _test_metrics(
    model: MicroQ4D,
    dataset: TrackDataset,
    loader: torch.utils.data.DataLoader[dict[str, Tensor]],
    normalization: NormalizationStats,
    device: torch.device,
    *,
    moving_threshold_m: float,
    use_amp: bool,
    compute_geometry_metrics: bool = True,
    action_permutation: Tensor | None = None,
) -> dict[str, Any]:
    model.eval()
    accumulator = TrajectoryMetricAccumulator(
        horizon=model.horizon,
        moving_threshold_m=moving_threshold_m,
        compute_geometry_metrics=compute_geometry_metrics,
    )
    displacement_mean = normalization.displacement_mean_m.to(device)
    displacement_scale = normalization.displacement_scale_m.to(device)
    prediction_seconds = 0.0
    sibling_predictions: dict[str, list[Tensor]] = {}
    sibling_targets: dict[str, list[Tensor]] = {}
    for batch in loader:
        groups, body_indices, _ = load_audit_metadata(dataset, batch)
        model_batch = batch
        if action_permutation is not None:
            source_indices = action_permutation[batch["sample_id"].to(torch.long)]
            shuffled_actions = torch.stack(
                [dataset[int(index)]["actions"] for index in source_indices]
            )
            model_batch = {**batch, "actions": shuffled_actions}
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            normalized = model(*_model_inputs(model_batch, device))
        displacement = normalized.to(torch.float32) * displacement_scale + displacement_mean
        query_world = batch["query_xyz_world_m"].to(device=device, non_blocking=True)
        prediction = query_world[:, :, None, :] + displacement
        torch.cuda.synchronize(device)
        prediction_seconds += time.perf_counter() - start
        prediction_cpu = prediction.cpu()
        accumulator.update(
            prediction_cpu,
            batch["target_world_m"],
            batch["query_xyz_world_m"],
            point_groups=groups,
            body_indices=body_indices,
        )
        for row, sample_id in enumerate(batch["sample_id"]):
            group_id = trajectory_group_id(dataset.files[int(sample_id)])
            sibling_predictions.setdefault(group_id, []).append(prediction_cpu[row])
            sibling_targets.setdefault(group_id, []).append(batch["target_world_m"][row])

    report = accumulator.report()
    report["runtime"] = {
        "prediction_seconds": prediction_seconds,
        "fragments_per_second": len(dataset) / prediction_seconds,
        "queries_per_second": len(dataset) * dataset.num_queries / prediction_seconds,
        "device": torch.cuda.get_device_name(device),
    }
    report["predicted_sibling_separation"] = _sibling_separation(sibling_predictions)
    report["target_sibling_separation"] = _sibling_separation(sibling_targets)
    return report


def _measure_cuda(
    function: Callable[[], Tensor], device: torch.device, repetitions: int
) -> tuple[float, float]:
    for _ in range(5):
        function()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repetitions):
        function()
    end.record()
    torch.cuda.synchronize(device)
    milliseconds = start.elapsed_time(end) / repetitions
    peak_mib = torch.cuda.max_memory_allocated(device) / 2**20
    return milliseconds, peak_mib


@torch.no_grad()
def _candidate_benchmark(
    model: MicroQ4D,
    dataset: TrackDataset,
    device: torch.device,
    *,
    candidates: int,
    repetitions: int,
    use_amp: bool,
) -> dict[str, Any]:
    model.eval()
    sample = dataset[0]
    scene_xyz = sample["scene_xyz"][None].to(device)
    scene_rgb = sample["scene_rgb"][None].to(device)
    query_xyz = sample["query_xyz"][None].to(device)
    branch_actions = torch.stack(
        [dataset[index]["actions"] for index in range(min(4, len(dataset)))]
    ).to(device)
    repeats = (candidates + len(branch_actions) - 1) // len(branch_actions)
    candidate_actions = branch_actions.repeat(repeats, 1, 1)[:candidates][None]
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
        scene_cache = model.encode_scene(scene_xyz, scene_rgb)
        query_cache = model.encode_queries(scene_cache, query_xyz)

    def cached() -> Tensor:
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            return model.predict_candidates(query_cache, candidate_actions)

    expanded_scene_xyz = scene_xyz.expand(candidates, -1, -1)
    expanded_scene_rgb = scene_rgb.expand(candidates, -1, -1)
    expanded_query_xyz = query_xyz.expand(candidates, -1, -1)

    def reencoded() -> Tensor:
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            return model(
                expanded_scene_xyz,
                expanded_scene_rgb,
                candidate_actions.squeeze(0),
                expanded_query_xyz,
            )

    cached_output = cached().squeeze(0)
    reencoded_output = reencoded()
    maximum_difference = float((cached_output - reencoded_output).abs().max())
    cached_ms, cached_memory = _measure_cuda(cached, device, repetitions)
    reencoded_ms, reencoded_memory = _measure_cuda(reencoded, device, repetitions)
    return {
        "candidate_branches": candidates,
        "queries_per_branch": dataset.num_queries,
        "repetitions": repetitions,
        "cached_milliseconds": cached_ms,
        "reencoded_milliseconds": reencoded_ms,
        "cached_candidates_per_second": candidates / (cached_ms / 1000.0),
        "reencoded_candidates_per_second": candidates / (reencoded_ms / 1000.0),
        "cached_speedup": reencoded_ms / cached_ms,
        "cached_peak_cuda_memory_mib": cached_memory,
        "reencoded_peak_cuda_memory_mib": reencoded_memory,
        "maximum_output_difference_normalized": maximum_difference,
        "note": "Both paths batch all candidates; cached path reuses scene and query encoding.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/micro_q4d.toml"))
    args = parser.parse_args()
    raw = _read_toml(args.config)
    paths = raw["paths"]
    model_config = raw["model"]
    training = raw["training"]
    evaluation = raw["evaluation"]
    data_config = _read_toml(Path(paths["data_config"]))["dataset"]
    seed = int(training["seed"])
    _seed_everything(seed)
    if not torch.cuda.is_available():
        raise RuntimeError("micro-Q4D training requires CUDA")
    device = torch.device("cuda:0")
    use_amp = bool(training["amp"])
    micro_batch_size = int(training["micro_batch_size"])
    accumulation_steps = int(training["gradient_accumulation_steps"])
    if micro_batch_size <= 0 or accumulation_steps <= 0:
        raise ValueError("micro batch size and accumulation steps must be positive")
    memory_budget = CudaMemoryBudget(
        total_mib=bytes_to_mib(torch.cuda.get_device_properties(device).total_memory),
        budget_mib=float(training["memory_budget_mib"]),
        minimum_headroom_mib=float(training["minimum_headroom_mib"]),
    )
    root = Path(data_config["root"])
    manifest = SplitManifest.load(paths["split_manifest"])
    normalization = NormalizationStats.load(paths["normalization"])
    train_dataset = TrackDataset(
        manifest.files(root, "train"),
        normalization,
        num_queries=int(training["queries"]),
        horizon=int(model_config["horizon"]),
        cache_size=int(data_config["cache_size"]),
    )
    validation_dataset = TrackDataset(
        manifest.files(root, "validation"),
        normalization,
        num_queries=int(training["queries"]),
        horizon=int(model_config["horizon"]),
        cache_size=int(data_config["cache_size"]),
    )
    loader_config = DataLoaderConfig(
        batch_size=micro_batch_size,
        num_workers=int(training["num_workers"]),
        pin_memory=True,
        persistent_workers=int(training["num_workers"]) > 0,
    )
    train_loader = build_dataloader(
        train_dataset, loader_config, shuffle=True, seed=seed, drop_last=False
    )
    validation_loader = build_dataloader(
        validation_dataset, loader_config, shuffle=False, seed=seed
    )
    model = MicroQ4D(
        action_dimensions=int(model_config["action_dimensions"]),
        horizon=int(model_config["horizon"]),
        width=int(model_config["width"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    output_dir = Path(paths["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best.pt"
    best_validation = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history = []
    training_start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(1, int(training["epochs"]) + 1):
        model.train()
        training_total = 0.0
        training_values = 0
        optimizer.zero_grad(set_to_none=True)
        batch_count = len(train_loader)
        for batch_index, batch in enumerate(train_loader):
            target = batch["target_displacement"].to(device=device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                prediction = model(*_model_inputs(batch, device))
                loss = torch.nn.functional.mse_loss(prediction, target)
            accumulation_start = (batch_index // accumulation_steps) * accumulation_steps
            accumulation_size = min(
                accumulation_steps, batch_count - accumulation_start
            )
            scaler.scale(loss / accumulation_size).backward()
            update_due = (batch_index + 1) % accumulation_steps == 0
            final_batch = batch_index + 1 == batch_count
            if update_due or final_batch:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(training["gradient_clip_norm"])
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            training_total += float(loss.detach()) * target.numel()
            training_values += target.numel()
        training_loss = training_total / training_values
        validation_loss = _validation_loss(model, validation_loader, device, use_amp)
        history.append(
            {"epoch": epoch, "training_mse": training_loss, "validation_mse": validation_loss}
        )
        if validation_loss < best_validation:
            best_validation = validation_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            epochs_without_improvement += 1
        if epoch == 1 or epoch % 10 == 0:
            print(
                f"epoch={epoch:03d} train_mse={training_loss:.6f} "
                f"validation_mse={validation_loss:.6f}"
            )
        if epochs_without_improvement >= int(training["patience"]):
            print(f"early_stop epoch={epoch} best_epoch={best_epoch}")
            break
        memory_budget.assert_peak(
            allocated_mib=bytes_to_mib(torch.cuda.max_memory_allocated(device)),
            reserved_mib=bytes_to_mib(torch.cuda.max_memory_reserved(device)),
        )

    torch.cuda.synchronize(device)
    training_seconds = time.perf_counter() - training_start
    peak_memory_mib = torch.cuda.max_memory_allocated(device) / 2**20
    peak_reserved_mib = torch.cuda.max_memory_reserved(device) / 2**20
    training_memory = memory_budget.report(
        allocated_mib=peak_memory_mib, reserved_mib=peak_reserved_mib
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    test_dataset = TrackDataset(
        manifest.files(root, "test"),
        normalization,
        num_queries=int(evaluation["queries"]),
        horizon=int(model_config["horizon"]),
        cache_size=int(data_config["cache_size"]),
    )
    test_loader = build_dataloader(
        test_dataset,
        DataLoaderConfig(
            batch_size=int(evaluation["batch_size"]),
            num_workers=0,
            pin_memory=True,
            persistent_workers=False,
        ),
        shuffle=False,
        seed=seed,
    )
    metrics = _test_metrics(
        model,
        test_dataset,
        test_loader,
        normalization,
        device,
        moving_threshold_m=float(evaluation["moving_threshold_m"]),
        use_amp=use_amp,
        compute_geometry_metrics=bool(evaluation.get("compute_geometry_metrics", True)),
    )
    shuffled_metrics = None
    action_permutation = None
    if bool(evaluation.get("action_shuffle", False)):
        generator = torch.Generator().manual_seed(
            int(evaluation.get("action_shuffle_seed", seed + 1))
        )
        action_permutation = torch.randperm(len(test_dataset), generator=generator)
        while torch.any(action_permutation == torch.arange(len(test_dataset))):
            action_permutation = torch.randperm(len(test_dataset), generator=generator)
        shuffled_metrics = _test_metrics(
            model,
            test_dataset,
            test_loader,
            normalization,
            device,
            moving_threshold_m=float(evaluation["moving_threshold_m"]),
            use_amp=use_amp,
            compute_geometry_metrics=bool(
                evaluation.get("compute_geometry_metrics", True)
            ),
            action_permutation=action_permutation,
        )
    benchmark_dataset = TrackDataset(
        manifest.files(root, "test"),
        normalization,
        num_queries=int(evaluation.get("benchmark_queries", evaluation["queries"])),
        horizon=int(model_config["horizon"]),
        cache_size=int(data_config["cache_size"]),
    )
    benchmark = _candidate_benchmark(
        model,
        benchmark_dataset,
        device,
        candidates=int(evaluation["candidate_branches"]),
        repetitions=int(evaluation["benchmark_repetitions"]),
        use_amp=use_amp,
    )
    non_neural = json.loads(Path(paths["reference_baselines"]).read_text(encoding="utf-8"))
    no_action = json.loads(Path(paths["no_action_report"]).read_text(encoding="utf-8"))
    no_action_groups = no_action["test"]["groups"]
    comparisons = {
        group: {
            "micro_q4d_ade_m": metrics["groups"][group]["ade_m"],
            "no_action_ade_m": no_action_groups[group]["ade_m"],
            "relative_reduction_percent": 100.0
            * (
                1.0
                - metrics["groups"][group]["ade_m"] / no_action_groups[group]["ade_m"]
            ),
        }
        for group in ("all", "moving", "contact", "object")
    }
    report = {
        "model": {
            "name": "micro_q4d",
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "width": model.width,
            "horizon": model.horizon,
            "action_dimensions": model.action_dimensions,
            "forward_parameters": list(inspect.signature(model.forward).parameters),
            "cache_stages": ["encode_scene", "encode_queries", "predict_candidates"],
        },
        "training": {
            "device": torch.cuda.get_device_name(device),
            "amp": use_amp,
            "epochs_completed": len(history),
            "best_epoch": best_epoch,
            "best_validation_mse": best_validation,
            "training_seconds": training_seconds,
            "micro_batch_size": micro_batch_size,
            "gradient_accumulation_steps": accumulation_steps,
            "effective_batch_size": micro_batch_size * accumulation_steps,
            "memory": training_memory,
            "training_fragments": len(train_dataset),
            "validation_fragments": len(validation_dataset),
        },
        "test": metrics,
        "action_shuffle": (
            {
                "test": shuffled_metrics,
                "permutation_has_no_fixed_points": bool(
                    torch.all(action_permutation != torch.arange(len(test_dataset)))
                ),
                "all_ade_relative_change_percent": 100.0
                * (
                    shuffled_metrics["groups"]["all"]["ade_m"]
                    / metrics["groups"]["all"]["ade_m"]
                    - 1.0
                ),
                "contact_ade_relative_change_percent": 100.0
                * (
                    shuffled_metrics["groups"]["contact"]["ade_m"]
                    / metrics["groups"]["contact"]["ade_m"]
                    - 1.0
                ),
            }
            if shuffled_metrics is not None and action_permutation is not None
            else None
        ),
        "no_action_comparison": comparisons,
        "candidate_benchmark": benchmark,
        "references": {
            "action_knn": non_neural["baselines"]["action_knn"]["groups"],
            "no_action_neural": no_action_groups,
        },
        "checks": {
            "cuda_training": device.type == "cuda",
            "actions_present_in_forward": "actions" in inspect.signature(model.forward).parameters,
            "counterfactual_predictions_diverge": metrics["predicted_sibling_separation"][
                "mean_pairwise_point_time_distance_m"
            ]
            > 0,
            "cache_matches_reencoding": benchmark["maximum_output_difference_normalized"]
            < 1e-3,
            "checkpoint_exists": checkpoint_path.exists(),
            "metrics_are_finite": bool(np.isfinite(metrics["groups"]["all"]["ade_m"])),
            "training_within_memory_budget": max(peak_memory_mib, peak_reserved_mib)
            <= memory_budget.budget_mib,
        },
        "scientific_checks": {
            f"beats_no_action_{group}": comparisons[group]["relative_reduction_percent"] > 0
            for group in comparisons
        },
    }
    report["passed"] = all(report["checks"].values())
    (output_dir / "history.json").write_text(
        json.dumps(history, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        failed = [name for name, passed in report["checks"].items() if not passed]
        raise RuntimeError(f"micro-Q4D training failed checks: {', '.join(failed)}")


if __name__ == "__main__":
    main()
