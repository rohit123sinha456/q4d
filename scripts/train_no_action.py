#!/usr/bin/env python3
"""Train and evaluate the checklist-item-8 action-free trajectory baseline."""

from __future__ import annotations

import argparse
import inspect
import json
import random
import time
import tomllib
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
from q4d_wam.models import NoActionTrajectoryModel


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


def _model_inputs(batch: dict[str, Tensor], device: torch.device) -> tuple[Tensor, Tensor, Tensor]:
    return tuple(
        batch[key].to(device=device, non_blocking=True)
        for key in ("scene_xyz", "scene_rgb", "query_xyz")
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


@torch.no_grad()
def _test_metrics(
    model: NoActionTrajectoryModel,
    dataset: TrackDataset,
    loader: torch.utils.data.DataLoader[dict[str, Tensor]],
    normalization: NormalizationStats,
    device: torch.device,
    *,
    moving_threshold_m: float,
    use_amp: bool,
    compute_geometry_metrics: bool = True,
) -> tuple[dict[str, Any], float]:
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
    for batch in loader:
        groups, body_indices, _ = load_audit_metadata(dataset, batch)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            normalized = model(*_model_inputs(batch, device))
        displacement = normalized.to(torch.float32) * displacement_scale + displacement_mean
        query_world = batch["query_xyz_world_m"].to(device=device, non_blocking=True)
        prediction = query_world[:, :, None, :] + displacement
        if device.type == "cuda":
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

    maximum_sibling_difference = 0.0
    for predictions in sibling_predictions.values():
        reference = predictions[0]
        differences = [
            float((prediction - reference).abs().max()) for prediction in predictions[1:]
        ]
        if differences:
            maximum_sibling_difference = max(maximum_sibling_difference, *differences)
    report = accumulator.report()
    report["runtime"] = {
        "prediction_seconds": prediction_seconds,
        "fragments_per_second": len(dataset) / prediction_seconds,
        "queries_per_second": len(dataset) * dataset.num_queries / prediction_seconds,
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
    }
    report["maximum_sibling_prediction_difference_m"] = maximum_sibling_difference
    return report, maximum_sibling_difference


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/no_action.toml"))
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
        raise RuntimeError("item 8 is configured for CUDA, but torch.cuda.is_available() is false")
    device = torch.device("cuda:0")
    use_amp = bool(training["amp"])
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
        batch_size=int(training["batch_size"]),
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

    model = NoActionTrajectoryModel(
        horizon=int(model_config["horizon"]), width=int(model_config["width"])
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
        for batch in train_loader:
            target = batch["target_displacement"].to(device=device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                prediction = model(*_model_inputs(batch, device))
                loss = torch.nn.functional.mse_loss(prediction, target)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(training["gradient_clip_norm"])
            )
            scaler.step(optimizer)
            scaler.update()
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

    torch.cuda.synchronize(device)
    training_seconds = time.perf_counter() - training_start
    peak_memory_mib = torch.cuda.max_memory_allocated(device) / 2**20
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
    metrics, sibling_difference = _test_metrics(
        model,
        test_dataset,
        test_loader,
        normalization,
        device,
        moving_threshold_m=float(evaluation["moving_threshold_m"]),
        use_amp=use_amp,
        compute_geometry_metrics=bool(evaluation.get("compute_geometry_metrics", True)),
    )
    reference = json.loads(Path(paths["reference_baselines"]).read_text(encoding="utf-8"))
    report = {
        "model": {
            "name": "no_action",
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "width": model.width,
            "horizon": model.horizon,
            "forward_parameters": list(inspect.signature(model.forward).parameters),
            "accepts_actions": "actions" in inspect.signature(model.forward).parameters,
        },
        "training": {
            "device": torch.cuda.get_device_name(device),
            "amp": use_amp,
            "epochs_completed": len(history),
            "best_epoch": best_epoch,
            "best_validation_mse": best_validation,
            "training_seconds": training_seconds,
            "peak_cuda_memory_mib": peak_memory_mib,
            "training_fragments": len(train_dataset),
            "validation_fragments": len(validation_dataset),
        },
        "test": metrics,
        "references": {
            name: reference["baselines"][name]["groups"]
            for name in ("static", "scene_knn", "action_knn")
        },
        "checks": {
            "cuda_training": device.type == "cuda",
            "actions_absent_from_forward": "actions"
            not in inspect.signature(model.forward).parameters,
            "grouped_test_predictions_identical": sibling_difference == 0.0,
            "all_metrics_finite": bool(
                np.isfinite(metrics["groups"]["all"]["ade_m"])
            ),
            "checkpoint_exists": checkpoint_path.exists(),
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
        raise RuntimeError(f"no-action training failed checks: {', '.join(failed)}")


if __name__ == "__main__":
    main()
