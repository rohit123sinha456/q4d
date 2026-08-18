#!/usr/bin/env python3
"""Train and evaluate the parameter-matched dense point-future baseline."""

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
from torch import Tensor

from q4d_wam.data import (
    DataLoaderConfig,
    NormalizationStats,
    SplitManifest,
    TrackDataset,
    build_dataloader,
    trajectory_group_id,
)
from q4d_wam.evaluation import TrajectoryMetricAccumulator, load_audit_metadata
from q4d_wam.models import (
    DensePointFutureModel,
    MicroQ4D,
    dense_query_set_is_complete,
)
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


def _inputs(
    batch: dict[str, Tensor], device: torch.device
) -> tuple[Tensor, Tensor, Tensor]:
    return tuple(
        batch[key].to(device=device, non_blocking=True)
        for key in ("scene_xyz", "scene_rgb", "actions")
    )  # type: ignore[return-value]


def _all_scene_points_are_queried(dataset: TrackDataset) -> bool:
    """Return whether a sample queries every scene point exactly once."""
    sample = dataset[0]
    return dense_query_set_is_complete(sample["scene_xyz"], sample["query_indices"])


def _prediction_in_query_order(
    model: DensePointFutureModel, batch: dict[str, Tensor], device: torch.device
) -> Tensor:
    dense = model(*_inputs(batch, device))
    indices = batch["query_indices"].to(device=device, non_blocking=True)
    return torch.gather(
        dense,
        dim=1,
        index=indices[:, :, None, None].expand(-1, -1, model.horizon, 3),
    )


@torch.no_grad()
def _validation_loss(
    model: DensePointFutureModel,
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
            prediction = _prediction_in_query_order(model, batch, device)
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
    model: DensePointFutureModel,
    dataset: TrackDataset,
    loader: torch.utils.data.DataLoader[dict[str, Tensor]],
    normalization: NormalizationStats,
    device: torch.device,
    *,
    moving_threshold_m: float,
    use_amp: bool,
    compute_geometry_metrics: bool = True,
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
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            normalized = _prediction_in_query_order(model, batch, device)
        displacement = normalized.float() * displacement_scale + displacement_mean
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
    return (
        start.elapsed_time(end) / repetitions,
        torch.cuda.max_memory_allocated(device) / 2**20,
    )


@torch.no_grad()
def _matched_candidate_benchmark(
    dense: DensePointFutureModel,
    sparse: MicroQ4D,
    dense_dataset: TrackDataset,
    sparse_dataset: TrackDataset,
    device: torch.device,
    *,
    candidates: int,
    repetitions: int,
    use_amp: bool,
) -> dict[str, Any]:
    dense.eval()
    sparse.eval()
    dense_sample = dense_dataset[0]
    sparse_sample = sparse_dataset[0]
    scene_xyz = dense_sample["scene_xyz"][None].to(device)
    scene_rgb = dense_sample["scene_rgb"][None].to(device)
    branch_actions = torch.stack(
        [dense_dataset[index]["actions"] for index in range(min(4, len(dense_dataset)))]
    ).to(device)
    repeats = (candidates + len(branch_actions) - 1) // len(branch_actions)
    actions = branch_actions.repeat(repeats, 1, 1)[:candidates][None]
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
        dense_scene = dense.encode_scene(scene_xyz, scene_rgb)
        dense_queries = dense.encode_dense_queries(dense_scene)
        sparse_scene = sparse.encode_scene(scene_xyz, scene_rgb)
        sparse_queries = sparse.encode_queries(
            sparse_scene, sparse_sample["query_xyz"][None].to(device)
        )

    def run_dense() -> Tensor:
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            return dense.predict_candidates(dense_queries, actions)

    def run_sparse() -> Tensor:
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            return sparse.predict_candidates(sparse_queries, actions)

    dense_output = run_dense()
    sparse_output = run_sparse()
    dense_ms, dense_memory = _measure_cuda(run_dense, device, repetitions)
    sparse_ms, sparse_memory = _measure_cuda(run_sparse, device, repetitions)
    dense_queries_count = dense_output.shape[2]
    sparse_queries_count = sparse_output.shape[2]
    return {
        "candidate_branches": candidates,
        "repetitions": repetitions,
        "dense_queries_per_branch": dense_queries_count,
        "sparse_queries_per_branch": sparse_queries_count,
        "output_trajectory_ratio": dense_queries_count / sparse_queries_count,
        "dense_cached_milliseconds": dense_ms,
        "sparse_cached_milliseconds": sparse_ms,
        "sparse_speedup": dense_ms / sparse_ms,
        "dense_candidates_per_second": candidates / (dense_ms / 1000.0),
        "sparse_candidates_per_second": candidates / (sparse_ms / 1000.0),
        "dense_peak_cuda_memory_mib": dense_memory,
        "sparse_peak_cuda_memory_mib": sparse_memory,
        "same_scene_and_candidate_actions": bool(
            torch.equal(dense_sample["scene_xyz"], sparse_sample["scene_xyz"])
        ),
        "note": (
            "Both paths cache the same scene and decode the same action branches; "
            "only the number of requested output points differs."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/dense_baseline.toml"))
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
        raise RuntimeError("dense baseline training requires CUDA")
    device = torch.device("cuda:0")
    use_amp = bool(training["amp"])
    micro_batch_size = int(training["micro_batch_size"])
    accumulation_steps = int(training["gradient_accumulation_steps"])
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
    model = DensePointFutureModel(
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
                prediction = _prediction_in_query_order(model, batch, device)
                loss = torch.nn.functional.mse_loss(prediction, target)
            accumulation_start = (batch_index // accumulation_steps) * accumulation_steps
            accumulation_size = min(
                accumulation_steps, batch_count - accumulation_start
            )
            scaler.scale(loss / accumulation_size).backward()
            update_due = (batch_index + 1) % accumulation_steps == 0
            if update_due or batch_index + 1 == batch_count:
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
    peak_memory_mib = bytes_to_mib(torch.cuda.max_memory_allocated(device))
    peak_reserved_mib = bytes_to_mib(torch.cuda.max_memory_reserved(device))
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
    sparse_model = MicroQ4D(
        action_dimensions=model.action_dimensions,
        horizon=model.horizon,
        width=model.width,
    ).to(device)
    sparse_model.load_state_dict(
        torch.load(paths["micro_q4d_checkpoint"], map_location=device, weights_only=True)
    )
    sparse_dataset = TrackDataset(
        manifest.files(root, "test"),
        normalization,
        num_queries=int(evaluation["sparse_queries"]),
        horizon=int(model_config["horizon"]),
        cache_size=int(data_config["cache_size"]),
    )
    benchmark = _matched_candidate_benchmark(
        model,
        sparse_model,
        test_dataset,
        sparse_dataset,
        device,
        candidates=int(evaluation["candidate_branches"]),
        repetitions=int(evaluation["benchmark_repetitions"]),
        use_amp=use_amp,
    )
    evaluation_scene_points = int(test_dataset[0]["scene_xyz"].shape[0])
    sparse_benchmark_points = int(sparse_dataset[0]["query_indices"].shape[0])
    expected_output_ratio = evaluation_scene_points / sparse_benchmark_points
    micro_report = json.loads(Path(paths["micro_q4d_report"]).read_text(encoding="utf-8"))
    no_action_report = json.loads(
        Path(paths["no_action_report"]).read_text(encoding="utf-8")
    )
    dense_groups = metrics["groups"]
    micro_groups = micro_report["test"]["groups"]
    comparisons = {
        group: {
            "dense_ade_m": dense_groups[group]["ade_m"],
            "micro_q4d_ade_m": micro_groups[group]["ade_m"],
            "dense_minus_micro_ade_m": dense_groups[group]["ade_m"]
            - micro_groups[group]["ade_m"],
            "dense_relative_change_percent": 100.0
            * (dense_groups[group]["ade_m"] / micro_groups[group]["ade_m"] - 1.0),
        }
        for group in ("all", "moving", "contact", "object")
    }
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    sparse_parameter_count = sum(parameter.numel() for parameter in sparse_model.parameters())
    report = {
        "model": {
            "name": "matched_dense_point_future",
            "parameters": parameter_count,
            "matched_micro_q4d_parameters": sparse_parameter_count,
            "width": model.width,
            "horizon": model.horizon,
            "action_dimensions": model.action_dimensions,
            "forward_parameters": list(inspect.signature(model.forward).parameters),
            "output_protocol": (
                f"all {evaluation_scene_points} visible scene points for every action branch"
            ),
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
            "training_queries_per_fragment": train_dataset.num_queries,
        },
        "test": metrics,
        "micro_q4d_comparison": comparisons,
        "matched_candidate_benchmark": benchmark,
        "references": {
            "micro_q4d": micro_groups,
            "no_action_neural": no_action_report["test"]["groups"],
        },
        "checks": {
            "cuda_training": device.type == "cuda",
            "parameter_count_exactly_matched": parameter_count == sparse_parameter_count,
            "dense_training_uses_all_points": _all_scene_points_are_queried(
                train_dataset
            ),
            "dense_evaluation_uses_all_points": _all_scene_points_are_queried(
                test_dataset
            ),
            "counterfactual_predictions_diverge": metrics[
                "predicted_sibling_separation"
            ]["mean_pairwise_point_time_distance_m"]
            > 0,
            "same_benchmark_inputs": benchmark["same_scene_and_candidate_actions"],
            "expected_output_ratio": bool(
                np.isclose(
                    benchmark["output_trajectory_ratio"], expected_output_ratio
                )
            ),
            "checkpoint_exists": checkpoint_path.exists(),
            "metrics_are_finite": bool(np.isfinite(dense_groups["all"]["ade_m"])),
            "training_within_memory_budget": max(peak_memory_mib, peak_reserved_mib)
            <= memory_budget.budget_mib,
        },
        "scientific_observations": {
            "dense_beats_micro_all_ade": dense_groups["all"]["ade_m"]
            < micro_groups["all"]["ade_m"],
            "sparse_decoding_is_faster": benchmark["sparse_speedup"] > 1.0,
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
        raise RuntimeError(f"dense baseline training failed checks: {', '.join(failed)}")


if __name__ == "__main__":
    main()
