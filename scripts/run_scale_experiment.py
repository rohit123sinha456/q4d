#!/usr/bin/env python3
"""Run the resumable checklist-item-12 PushCube experiment matrix."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any


def _read_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        return tomllib.load(stream)


def _quote(value: str | Path) -> str:
    escaped = str(value).replace("\\", "/").replace('"', '\\"')
    return f'"{escaped}"'


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")
    return path


def _passed(report_path: Path) -> bool:
    if not report_path.exists():
        return False
    try:
        return bool(json.loads(report_path.read_text(encoding="utf-8"))["passed"])
    except (json.JSONDecodeError, KeyError):
        return False


def _run(script: str, config: Path, report: Path, *, force: bool) -> None:
    if not force and _passed(report):
        print(f"skip passed={report}", flush=True)
        return
    command = [sys.executable, script, "--config", str(config)]
    print("run " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def _evaluation_config(
    *, data_config: Path, dataset_root: Path, output: Path, horizon: int, raw: dict[str, Any]
) -> str:
    evaluation = raw["evaluation"]
    return f"""
[evaluation]
data_config = {_quote(data_config)}
split_manifest = {_quote(dataset_root / 'splits.json')}
normalization = {_quote(dataset_root / 'normalization.json')}
output = {_quote(output)}
split = "test"
evaluation_queries = {raw['model']['dense_queries']}
horizon = {horizon}
batch_size = {evaluation['batch_size']}
num_workers = 2
knn_neighbors = 3
moving_threshold_m = {evaluation['moving_threshold_m']}
compute_geometry_metrics = false
"""


def _no_action_config(
    *, data_config: Path, dataset_root: Path, horizon_root: Path, horizon: int,
    raw: dict[str, Any]
) -> str:
    training = raw["training"]
    evaluation = raw["evaluation"]
    return f"""
[paths]
data_config = {_quote(data_config)}
split_manifest = {_quote(dataset_root / 'splits.json')}
normalization = {_quote(dataset_root / 'normalization.json')}
reference_baselines = {_quote(horizon_root / 'non_neural.json')}
output_dir = {_quote(horizon_root / 'no_action')}

[model]
width = {raw['model']['width']}
horizon = {horizon}

[training]
seed = {raw['experiment']['seed']}
epochs = {training['epochs']}
patience = {training['patience']}
batch_size = {training['batch_size']}
num_workers = {training['num_workers']}
learning_rate = {training['learning_rate']}
weight_decay = {training['weight_decay']}
gradient_clip_norm = {training['gradient_clip_norm']}
amp = true
queries = {raw['model']['sparse_queries']}

[evaluation]
batch_size = {evaluation['batch_size']}
queries = {raw['model']['dense_queries']}
moving_threshold_m = {evaluation['moving_threshold_m']}
compute_geometry_metrics = false
"""


def _micro_config(
    *, data_config: Path, dataset_root: Path, horizon_root: Path, horizon: int,
    raw: dict[str, Any]
) -> str:
    training = raw["training"]
    evaluation = raw["evaluation"]
    return f"""
[paths]
data_config = {_quote(data_config)}
split_manifest = {_quote(dataset_root / 'splits.json')}
normalization = {_quote(dataset_root / 'normalization.json')}
reference_baselines = {_quote(horizon_root / 'non_neural.json')}
no_action_report = {_quote(horizon_root / 'no_action' / 'report.json')}
output_dir = {_quote(horizon_root / 'micro_q4d')}

[model]
width = {raw['model']['width']}
horizon = {horizon}
action_dimensions = {raw['model']['action_dimensions']}

[training]
seed = {raw['experiment']['seed']}
epochs = {training['epochs']}
patience = {training['patience']}
micro_batch_size = {training['batch_size']}
gradient_accumulation_steps = 1
num_workers = {training['num_workers']}
learning_rate = {training['learning_rate']}
weight_decay = {training['weight_decay']}
gradient_clip_norm = {training['gradient_clip_norm']}
amp = true
queries = {raw['model']['sparse_queries']}
memory_budget_mib = {training['memory_budget_mib']}
minimum_headroom_mib = {training['minimum_headroom_mib']}

[evaluation]
batch_size = {evaluation['batch_size']}
queries = {raw['model']['dense_queries']}
benchmark_queries = {raw['model']['sparse_queries']}
moving_threshold_m = {evaluation['moving_threshold_m']}
candidate_branches = {evaluation['candidate_branches']}
benchmark_repetitions = {evaluation['benchmark_repetitions']}
compute_geometry_metrics = false
action_shuffle = true
action_shuffle_seed = {evaluation['action_shuffle_seed']}
"""


def _dense_config(
    *, data_config: Path, dataset_root: Path, horizon_root: Path, horizon: int,
    raw: dict[str, Any]
) -> str:
    training = raw["training"]
    evaluation = raw["evaluation"]
    return f"""
[paths]
data_config = {_quote(data_config)}
split_manifest = {_quote(dataset_root / 'splits.json')}
normalization = {_quote(dataset_root / 'normalization.json')}
reference_baselines = {_quote(horizon_root / 'non_neural.json')}
no_action_report = {_quote(horizon_root / 'no_action' / 'report.json')}
micro_q4d_report = {_quote(horizon_root / 'micro_q4d' / 'report.json')}
micro_q4d_checkpoint = {_quote(horizon_root / 'micro_q4d' / 'best.pt')}
output_dir = {_quote(horizon_root / 'dense')}

[model]
width = {raw['model']['width']}
horizon = {horizon}
action_dimensions = {raw['model']['action_dimensions']}

[training]
seed = {raw['experiment']['seed']}
epochs = {training['epochs']}
patience = {training['patience']}
micro_batch_size = {training['batch_size']}
gradient_accumulation_steps = 1
num_workers = {training['num_workers']}
learning_rate = {training['learning_rate']}
weight_decay = {training['weight_decay']}
gradient_clip_norm = {training['gradient_clip_norm']}
amp = true
queries = {raw['model']['dense_queries']}
memory_budget_mib = {training['memory_budget_mib']}
minimum_headroom_mib = {training['minimum_headroom_mib']}

[evaluation]
batch_size = {evaluation['batch_size']}
queries = {raw['model']['dense_queries']}
sparse_queries = {raw['model']['sparse_queries']}
moving_threshold_m = {evaluation['moving_threshold_m']}
candidate_branches = {evaluation['candidate_branches']}
benchmark_repetitions = {evaluation['benchmark_repetitions']}
compute_geometry_metrics = false
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/scale_experiment.toml"))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--horizons", type=int, nargs="*")
    args = parser.parse_args()
    raw = _read_toml(args.config)
    experiment = raw["experiment"]
    dataset_root = Path(experiment["dataset_root"])
    data_config = Path(experiment["data_config"])
    output_root = Path(experiment["output_root"])
    manifest_path = dataset_root / "manifest.json"
    if not manifest_path.exists():
        raise RuntimeError("scaled collection manifest is missing")
    collection = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not collection.get("complete") or collection.get("fragments", 0) < int(
        experiment["minimum_fragments"]
    ):
        raise RuntimeError("scaled collection has not reached the configured fragment gate")

    preparation_report = dataset_root / "loader_report.json"
    if args.force or not _passed(preparation_report):
        subprocess.run(
            [sys.executable, "scripts/prepare_dataset.py", "--config", str(data_config)],
            check=True,
        )
    if args.prepare_only:
        return

    horizons = args.horizons or [int(value) for value in experiment["horizons"]]
    config_root = output_root / "generated_configs"
    for horizon in horizons:
        horizon_root = output_root / f"h{horizon}"
        baseline_report = horizon_root / "non_neural.json"
        baseline_config = _write(
            config_root / f"h{horizon}_non_neural.toml",
            _evaluation_config(
                data_config=data_config,
                dataset_root=dataset_root,
                output=baseline_report,
                horizon=horizon,
                raw=raw,
            ),
        )
        no_action_config = _write(
            config_root / f"h{horizon}_no_action.toml",
            _no_action_config(
                data_config=data_config,
                dataset_root=dataset_root,
                horizon_root=horizon_root,
                horizon=horizon,
                raw=raw,
            ),
        )
        micro_config = _write(
            config_root / f"h{horizon}_micro_q4d.toml",
            _micro_config(
                data_config=data_config,
                dataset_root=dataset_root,
                horizon_root=horizon_root,
                horizon=horizon,
                raw=raw,
            ),
        )
        dense_config = _write(
            config_root / f"h{horizon}_dense.toml",
            _dense_config(
                data_config=data_config,
                dataset_root=dataset_root,
                horizon_root=horizon_root,
                horizon=horizon,
                raw=raw,
            ),
        )
        _run("scripts/evaluate_baselines.py", baseline_config, baseline_report, force=args.force)
        _run(
            "scripts/train_no_action.py",
            no_action_config,
            horizon_root / "no_action" / "report.json",
            force=args.force,
        )
        _run(
            "scripts/train_micro_q4d.py",
            micro_config,
            horizon_root / "micro_q4d" / "report.json",
            force=args.force,
        )
        _run(
            "scripts/train_dense_baseline.py",
            dense_config,
            horizon_root / "dense" / "report.json",
            force=args.force,
        )
    if 8 in horizons:
        scaling_report = output_root / "h8" / "n_m_scaling.json"
        if args.force or not _passed(scaling_report):
            subprocess.run(
                [
                    sys.executable,
                    "scripts/benchmark_scale_grid.py",
                    "--config",
                    str(args.config),
                    "--horizon",
                    "8",
                ],
                check=True,
            )
    if set(horizons) == {int(value) for value in experiment["horizons"]}:
        subprocess.run(
            [
                sys.executable,
                "scripts/summarize_scale_experiment.py",
                "--config",
                str(args.config),
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
