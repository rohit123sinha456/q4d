#!/usr/bin/env python3
"""Benchmark cached Q4D candidate decoding over feasible scene/query sizes."""

from __future__ import annotations

import argparse
import json
import time
import tomllib
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from q4d_wam.data import NormalizationStats, SplitManifest, TrackDataset
from q4d_wam.labels import farthest_point_indices
from q4d_wam.models import MicroQ4D


def _read_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        return tomllib.load(stream)


def _measure(
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
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/scale_experiment.toml"))
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--scene-points", type=int, nargs="+", default=[128, 256, 512])
    parser.add_argument("--query-points", type=int, nargs="+", default=[32, 64, 128])
    parser.add_argument("--repetitions", type=int, default=30)
    args = parser.parse_args()
    raw = _read_toml(args.config)
    experiment = raw["experiment"]
    model_config = raw["model"]
    evaluation = raw["evaluation"]
    dataset_root = Path(experiment["dataset_root"])
    output_root = Path(experiment["output_root"])
    checkpoint = output_root / f"h{args.horizon}" / "micro_q4d" / "best.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"missing trained checkpoint: {checkpoint}")
    if not torch.cuda.is_available():
        raise RuntimeError("scale benchmark requires CUDA")
    device = torch.device("cuda:0")
    manifest = SplitManifest.load(dataset_root / "splits.json")
    normalization = NormalizationStats.load(dataset_root / "normalization.json")
    dataset = TrackDataset(
        manifest.files(dataset_root, "test"),
        normalization,
        num_queries=max(args.query_points),
        horizon=args.horizon,
        cache_size=8,
    )
    model = MicroQ4D(
        action_dimensions=int(model_config["action_dimensions"]),
        horizon=args.horizon,
        width=int(model_config["width"]),
    ).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval()
    sample = dataset[0]
    full_scene_xyz = sample["scene_xyz"].to(device)
    full_scene_rgb = sample["scene_rgb"].to(device)
    branch_actions = torch.stack(
        [dataset[index]["actions"] for index in range(min(5, len(dataset)))]
    ).to(device)
    candidates = int(evaluation["candidate_branches"])
    repeats = (candidates + len(branch_actions) - 1) // len(branch_actions)
    candidate_actions = branch_actions.repeat(repeats, 1, 1)[:candidates][None]
    use_amp = bool(raw["training"]["amp"])
    rows = []
    for scene_points in args.scene_points:
        if scene_points > len(full_scene_xyz):
            continue
        scene_indices = farthest_point_indices(full_scene_xyz, scene_points)
        scene_xyz = full_scene_xyz[scene_indices][None]
        scene_rgb = full_scene_rgb[scene_indices][None]
        for query_points in args.query_points:
            if query_points > scene_points:
                continue
            query_indices = torch.arange(query_points, device=device)[None]
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                scene_cache = model.encode_scene(scene_xyz, scene_rgb)
                query_cache = model.encode_query_indices(scene_cache, query_indices)

            def cached(query_cache: Any = query_cache) -> Tensor:
                with torch.autocast(
                    device_type="cuda", dtype=torch.float16, enabled=use_amp
                ):
                    return model.predict_candidates(query_cache, candidate_actions)

            expanded_xyz = scene_xyz.expand(candidates, -1, -1)
            expanded_rgb = scene_rgb.expand(candidates, -1, -1)
            expanded_indices = query_indices.expand(candidates, -1)

            def reencoded(
                expanded_xyz: Tensor = expanded_xyz,
                expanded_rgb: Tensor = expanded_rgb,
                expanded_indices: Tensor = expanded_indices,
            ) -> Tensor:
                with torch.autocast(
                    device_type="cuda", dtype=torch.float16, enabled=use_amp
                ):
                    reencoded_scene = model.encode_scene(expanded_xyz, expanded_rgb)
                    reencoded_queries = model.encode_query_indices(
                        reencoded_scene, expanded_indices
                    )
                    return model.decode(
                        reencoded_queries, candidate_actions.squeeze(0)
                    )

            cached_output = cached().squeeze(0)
            reencoded_output = reencoded()
            difference = float((cached_output - reencoded_output).abs().max())
            physical_difference = float(
                (
                    (cached_output.float() - reencoded_output.float())
                    * normalization.displacement_scale_m.to(device)
                )
                .abs()
                .max()
            )
            cached_ms, cached_memory = _measure(
                cached, device, repetitions=args.repetitions
            )
            reencoded_ms, reencoded_memory = _measure(
                reencoded, device, repetitions=args.repetitions
            )
            row = {
                "scene_points": scene_points,
                "query_points": query_points,
                "candidate_branches": candidates,
                "cached_milliseconds": cached_ms,
                "reencoded_milliseconds": reencoded_ms,
                "cache_speedup": reencoded_ms / cached_ms,
                "cached_candidates_per_second": candidates / (cached_ms / 1000.0),
                "reencoded_candidates_per_second": candidates
                / (reencoded_ms / 1000.0),
                "cached_peak_cuda_memory_mib": cached_memory,
                "reencoded_peak_cuda_memory_mib": reencoded_memory,
                "maximum_output_difference_normalized": difference,
                "maximum_output_difference_m": physical_difference,
            }
            rows.append(row)
            print(json.dumps(row), flush=True)

    budget_mib = float(raw["training"]["memory_budget_mib"])
    report = {
        "device": torch.cuda.get_device_name(device),
        "horizon": args.horizon,
        "repetitions": args.repetitions,
        "rows": rows,
        "checks": {
            "all_requested_scene_sizes_tested": {row["scene_points"] for row in rows}
            == set(args.scene_points),
            "all_requested_query_sizes_tested": {row["query_points"] for row in rows}
            == set(args.query_points),
            "cache_matches_reencoding": all(
                row["maximum_output_difference_m"]
                <= float(evaluation["cache_equivalence_tolerance_m"])
                for row in rows
            ),
            "cache_equivalence_tolerance_m": float(
                evaluation["cache_equivalence_tolerance_m"]
            ),
            "all_runs_within_memory_budget": all(
                max(
                    row["cached_peak_cuda_memory_mib"],
                    row["reencoded_peak_cuda_memory_mib"],
                )
                <= budget_mib
                for row in rows
            ),
        },
        "generated_at_unix_seconds": time.time(),
    }
    report["passed"] = all(report["checks"].values())
    output = output_root / f"h{args.horizon}" / "n_m_scaling.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise RuntimeError("one or more N/M scaling checks failed")


if __name__ == "__main__":
    main()
