#!/usr/bin/env python3
"""Measure real micro-Q4D optimizer steps against the configured 8 GB contract."""

from __future__ import annotations

import argparse
import json
import random
import time
import tomllib
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
)
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


def _optimizer_state_mib(optimizer: torch.optim.Optimizer) -> float:
    size = 0
    for state in optimizer.state.values():
        for value in state.values():
            if isinstance(value, Tensor):
                size += value.numel() * value.element_size()
    return bytes_to_mib(size)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/memory_8gb.toml"))
    args = parser.parse_args()
    audit_config = _read_toml(args.config)
    raw = _read_toml(Path(audit_config["paths"]["model_config"]))
    paths = raw["paths"]
    model_config = raw["model"]
    training = raw["training"]
    data_config = _read_toml(Path(paths["data_config"]))["dataset"]
    audit = audit_config["audit"]
    seed = int(training["seed"])
    _seed_everything(seed)
    if not torch.cuda.is_available():
        raise RuntimeError("the 8 GB training audit requires CUDA")
    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)
    budget = CudaMemoryBudget(
        total_mib=bytes_to_mib(properties.total_memory),
        budget_mib=float(training["memory_budget_mib"]),
        minimum_headroom_mib=float(training["minimum_headroom_mib"]),
    )
    micro_batch_size = int(training["micro_batch_size"])
    accumulation_steps = int(training["gradient_accumulation_steps"])
    effective_batch_size = micro_batch_size * accumulation_steps

    root = Path(data_config["root"])
    manifest = SplitManifest.load(paths["split_manifest"])
    normalization = NormalizationStats.load(paths["normalization"])
    dataset = TrackDataset(
        manifest.files(root, "train"),
        normalization,
        num_queries=int(training["queries"]),
        cache_size=int(data_config["cache_size"]),
    )
    loader = build_dataloader(
        dataset,
        DataLoaderConfig(
            batch_size=micro_batch_size,
            num_workers=int(training["num_workers"]),
            pin_memory=True,
            persistent_workers=int(training["num_workers"]) > 0,
        ),
        shuffle=True,
        seed=seed,
        drop_last=True,
    )
    loader_iterator = iter(loader)

    def next_batch() -> dict[str, Tensor]:
        nonlocal loader_iterator
        try:
            return next(loader_iterator)
        except StopIteration:
            loader_iterator = iter(loader)
            return next(loader_iterator)

    first_batch = next_batch()
    all_cpu_tensors_pinned = all(value.is_pinned() for value in first_batch.values())
    model = MicroQ4D(
        action_dimensions=int(model_config["action_dimensions"]),
        horizon=int(model_config["horizon"]),
        width=int(model_config["width"]),
    ).to(device)
    checkpoint = Path(audit_config["paths"]["checkpoint"])
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=bool(training["amp"]))

    def optimizer_step(initial_batch: dict[str, Tensor] | None = None) -> None:
        optimizer.zero_grad(set_to_none=True)
        for micro_step in range(accumulation_steps):
            batch = initial_batch if micro_step == 0 and initial_batch is not None else next_batch()
            scene_xyz = batch["scene_xyz"].to(device, non_blocking=True)
            scene_rgb = batch["scene_rgb"].to(device, non_blocking=True)
            actions = batch["actions"].to(device, non_blocking=True)
            query_xyz = batch["query_xyz"].to(device, non_blocking=True)
            target = batch["target_displacement"].to(device, non_blocking=True)
            with torch.autocast(
                device_type="cuda", dtype=torch.float16, enabled=bool(training["amp"])
            ):
                prediction = model(scene_xyz, scene_rgb, actions, query_xyz)
                loss = torch.nn.functional.mse_loss(prediction, target)
            scaler.scale(loss / accumulation_steps).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(training["gradient_clip_norm"])
        )
        scaler.step(optimizer)
        scaler.update()

    for warmup_index in range(int(audit["warmup_steps"])):
        optimizer_step(first_batch if warmup_index == 0 else None)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    allocated_by_step = []
    reserved_by_step = []
    measured_start = time.perf_counter()
    for _ in range(int(audit["optimizer_steps"])):
        optimizer_step()
        torch.cuda.synchronize(device)
        allocated_by_step.append(bytes_to_mib(torch.cuda.memory_allocated(device)))
        reserved_by_step.append(bytes_to_mib(torch.cuda.memory_reserved(device)))
    measured_seconds = time.perf_counter() - measured_start
    peak_allocated_mib = bytes_to_mib(torch.cuda.max_memory_allocated(device))
    peak_reserved_mib = bytes_to_mib(torch.cuda.max_memory_reserved(device))
    memory = budget.report(
        allocated_mib=peak_allocated_mib, reserved_mib=peak_reserved_mib
    )
    steady_state_growth_mib = max(
        allocated_by_step[-1] - allocated_by_step[0],
        reserved_by_step[-1] - reserved_by_step[0],
    )
    report = {
        "device": {
            "name": torch.cuda.get_device_name(device),
            "physical_total_mib": bytes_to_mib(properties.total_memory),
            "compute_capability": f"{properties.major}.{properties.minor}",
        },
        "configuration": {
            "amp": bool(training["amp"]),
            "micro_batch_size": micro_batch_size,
            "gradient_accumulation_steps": accumulation_steps,
            "effective_batch_size": effective_batch_size,
            "scene_points": first_batch["scene_xyz"].shape[1],
            "queries": first_batch["query_xyz"].shape[1],
            "horizon": first_batch["target_displacement"].shape[2],
            "width": int(model_config["width"]),
        },
        "model": {
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "parameter_mib": bytes_to_mib(
                sum(
                    parameter.numel() * parameter.element_size()
                    for parameter in model.parameters()
                )
            ),
            "optimizer_state_mib": _optimizer_state_mib(optimizer),
        },
        "measured_training": {
            **memory,
            "optimizer_steps": int(audit["optimizer_steps"]),
            "micro_batches": int(audit["optimizer_steps"]) * accumulation_steps,
            "samples": int(audit["optimizer_steps"]) * effective_batch_size,
            "seconds": measured_seconds,
            "samples_per_second": int(audit["optimizer_steps"])
            * effective_batch_size
            / measured_seconds,
            "allocated_mib_by_step": allocated_by_step,
            "reserved_mib_by_step": reserved_by_step,
            "steady_state_growth_mib": steady_state_growth_mib,
        },
        "checks": {
            "rtx_4060_8gb_detected": "4060" in torch.cuda.get_device_name(device)
            and 7500 <= bytes_to_mib(properties.total_memory) <= 9000,
            "amp_enabled": bool(training["amp"]),
            "pinned_host_batches": all_cpu_tensors_pinned,
            "gradient_accumulation_active": accumulation_steps > 1,
            "effective_batch_is_32": effective_batch_size == 32,
            "peak_allocated_within_budget": peak_allocated_mib <= budget.budget_mib,
            "peak_reserved_within_budget": peak_reserved_mib <= budget.budget_mib,
            "configured_headroom_preserved": budget.configured_headroom_mib
            >= budget.minimum_headroom_mib,
            "steady_state_memory_stable": steady_state_growth_mib
            <= float(audit["maximum_steady_state_growth_mib"]),
        },
    }
    report["passed"] = all(report["checks"].values())
    output = Path(audit_config["paths"]["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        failed = [name for name, passed in report["checks"].items() if not passed]
        raise RuntimeError(f"8 GB memory audit failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
