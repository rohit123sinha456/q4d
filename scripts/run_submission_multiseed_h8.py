#!/usr/bin/env python3
"""Generate and run the resumable submission-v1 H=8 multi-seed matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

MODELS = ("no_action", "micro_q4d", "dense")
TRAINING_SCRIPTS = {
    "no_action": "scripts/train_no_action.py",
    "micro_q4d": "scripts/train_micro_q4d.py",
    "dense": "scripts/train_dense_baseline.py",
}


def _read_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        return tomllib.load(stream)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _quote(value: str | Path) -> str:
    return '"' + str(value).replace("\\", "/").replace('"', '\\"') + '"'


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_state() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"], check=True, capture_output=True, text=True
    ).stdout
    return {"commit": commit, "working_tree_clean": not bool(status.strip())}


def _report_passed(path: Path) -> bool:
    try:
        return bool(_read_json(path)["passed"])
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return False


def _seed_one_config(config_root: Path, model: str) -> Path:
    return config_root / f"h8_{model}.toml"


def _training_settings(raw: dict[str, Any], model: str) -> dict[str, Any]:
    training = raw["training"]
    batch_key = "batch_size" if model == "no_action" else "micro_batch_size"
    return {
        "epochs": int(training["epochs"]),
        "patience": int(training["patience"]),
        "batch_size": int(training[batch_key]),
        "learning_rate": float(training["learning_rate"]),
        "weight_decay": float(training["weight_decay"]),
        "gradient_clip_norm": float(training["gradient_clip_norm"]),
        "amp": bool(training["amp"]),
    }


def validate_protocol_contract(raw: dict[str, Any]) -> dict[str, bool]:
    """Check the executable matrix against the locked submission protocol."""
    protocol = _read_toml(Path(raw["experiment"]["protocol"]))
    frozen_training = protocol["training"]
    configured_training = raw["training"]
    tasks = list(raw["experiment"]["tasks"])
    checks = {
        "protocol_is_locked": protocol["protocol"]["status"]
        == "locked_before_new_test_results",
        "tasks_match": tasks == list(protocol["scope"]["tasks"]),
        "models_match": list(raw["experiment"]["models"])
        == list(protocol["scope"]["models"]),
        "horizon_matches": int(raw["experiment"]["horizon"])
        == int(protocol["scope"]["primary_horizon"]),
        "seeds_match": all(
            list(raw["seeds"][task])
            == list(frozen_training[f"{task}_seeds"])
            for task in tasks
        ),
        "training_matches": all(
            configured_training[name] == frozen_training[name]
            for name in (
                "epochs",
                "patience",
                "batch_size",
                "learning_rate",
                "weight_decay",
                "gradient_clip_norm",
                "amp",
            )
        ),
        "width_matches": int(raw["model"]["width"])
        == int(frozen_training["model_width"]),
        "split_files_match": [
            raw["task_inputs"][task]["split_manifest"] for task in tasks
        ]
        == list(protocol["data"]["split_files"]),
        "normalization_files_match": [
            raw["task_inputs"][task]["normalization"] for task in tasks
        ]
        == list(protocol["data"]["normalization_files"]),
    }
    checks["passed"] = all(checks.values())
    return checks


def validate_seed_one(raw: dict[str, Any]) -> dict[str, Any]:
    """Prove that each preserved H=8 seed-one run matches the frozen protocol."""
    expected_training = raw["training"]
    expected = {
        "epochs": int(expected_training["epochs"]),
        "patience": int(expected_training["patience"]),
        "batch_size": int(expected_training["batch_size"]),
        "learning_rate": float(expected_training["learning_rate"]),
        "weight_decay": float(expected_training["weight_decay"]),
        "gradient_clip_norm": float(expected_training["gradient_clip_norm"]),
        "amp": bool(expected_training["amp"]),
    }
    checks: dict[str, Any] = {}
    for task in raw["experiment"]["tasks"]:
        seed = int(raw["seeds"][task][0])
        source_root = Path(raw["existing_seed_one"][task])
        config_root = Path(raw["task_inputs"][task]["seed_one_configs"])
        for model in MODELS:
            config_path = _seed_one_config(config_root, model)
            config = _read_toml(config_path)
            report_path = source_root / model / "report.json"
            key = f"{task}_{model}"
            checks[key] = {
                "seed": int(config["training"]["seed"]),
                "seed_matches": int(config["training"]["seed"]) == seed,
                "horizon_matches": int(config["model"]["horizon"])
                == int(raw["experiment"]["horizon"]),
                "training_matches": _training_settings(config, model) == expected,
                "width_matches": int(config["model"]["width"])
                == int(raw["model"]["width"]),
                "split_matches": config["paths"]["split_manifest"]
                == raw["task_inputs"][task]["split_manifest"],
                "normalization_matches": config["paths"]["normalization"]
                == raw["task_inputs"][task]["normalization"],
                "report_passed": _report_passed(report_path),
                "config_sha256": _sha256(config_path),
                "report_sha256": _sha256(report_path),
                "checkpoint_sha256": _sha256(source_root / model / "best.pt"),
            }
            checks[key]["passed"] = all(
                value
                for name, value in checks[key].items()
                if name.endswith("_matches") or name == "report_passed"
            )
    return {
        "checks": checks,
        "passed": all(bool(check["passed"]) for check in checks.values()),
    }


def _paths(raw: dict[str, Any], task: str, seed: int) -> dict[str, Path]:
    root = Path(raw["experiment"]["output_root"]) / task / f"seed_{seed}"
    return {model: root / model for model in MODELS}


def _common_paths(raw: dict[str, Any], task: str) -> dict[str, str]:
    return raw["task_inputs"][task]


def build_config(raw: dict[str, Any], task: str, seed: int, model: str) -> str:
    """Return one training config without changing the frozen optimizer protocol."""
    if model not in MODELS:
        raise ValueError(f"unknown model: {model}")
    inputs = _common_paths(raw, task)
    outputs = _paths(raw, task, seed)
    training = raw["training"]
    evaluation = raw["evaluation"]
    model_raw = raw["model"]
    horizon = int(raw["experiment"]["horizon"])
    paths = [
        "[paths]",
        f"data_config = {_quote(inputs['data_config'])}",
        f"split_manifest = {_quote(inputs['split_manifest'])}",
        f"normalization = {_quote(inputs['normalization'])}",
        f"reference_baselines = {_quote(inputs['reference_baselines'])}",
    ]
    if model in {"micro_q4d", "dense"}:
        paths.append(f"no_action_report = {_quote(outputs['no_action'] / 'report.json')}")
    if model == "dense":
        paths.extend(
            [
                f"micro_q4d_report = {_quote(outputs['micro_q4d'] / 'report.json')}",
                f"micro_q4d_checkpoint = {_quote(outputs['micro_q4d'] / 'best.pt')}",
            ]
        )
    paths.append(f"output_dir = {_quote(outputs[model])}")
    model_lines = [
        "",
        "[model]",
        f"width = {int(model_raw['width'])}",
        f"horizon = {horizon}",
    ]
    if model != "no_action":
        model_lines.append(f"action_dimensions = {int(model_raw['action_dimensions'])}")
    batch_name = "batch_size" if model == "no_action" else "micro_batch_size"
    training_lines = [
        "",
        "[training]",
        f"seed = {seed}",
        f"epochs = {int(training['epochs'])}",
        f"patience = {int(training['patience'])}",
        f"{batch_name} = {int(training['batch_size'])}",
    ]
    if model != "no_action":
        training_lines.append("gradient_accumulation_steps = 1")
    training_lines.extend(
        [
            f"num_workers = {int(training['num_workers'])}",
            f"learning_rate = {float(training['learning_rate'])}",
            f"weight_decay = {float(training['weight_decay'])}",
            f"gradient_clip_norm = {float(training['gradient_clip_norm'])}",
            f"amp = {str(bool(training['amp'])).lower()}",
            "queries = "
            + str(
                int(
                    model_raw["dense_queries"]
                    if model == "dense"
                    else model_raw["sparse_queries"]
                )
            ),
        ]
    )
    if model != "no_action":
        training_lines.extend(
            [
                f"memory_budget_mib = {int(training['memory_budget_mib'])}",
                f"minimum_headroom_mib = {int(training['minimum_headroom_mib'])}",
            ]
        )
    evaluation_lines = [
        "",
        "[evaluation]",
        f"batch_size = {int(evaluation['batch_size'])}",
        f"queries = {int(model_raw['dense_queries'])}",
    ]
    if model == "micro_q4d":
        evaluation_lines.append(f"benchmark_queries = {int(model_raw['sparse_queries'])}")
    if model == "dense":
        evaluation_lines.append(f"sparse_queries = {int(model_raw['sparse_queries'])}")
    evaluation_lines.append(
        f"moving_threshold_m = {float(evaluation['moving_threshold_m'])}"
    )
    if model != "no_action":
        evaluation_lines.extend(
            [
                f"candidate_branches = {int(evaluation['candidate_branches'])}",
                f"benchmark_repetitions = {int(evaluation['benchmark_repetitions'])}",
            ]
        )
    evaluation_lines.append(
        f"compute_geometry_metrics = {str(bool(evaluation['compute_geometry_metrics'])).lower()}"
    )
    if model == "micro_q4d":
        evaluation_lines.extend(
            [
                "action_shuffle = true",
                f"action_shuffle_seed = {int(evaluation['action_shuffle_seed'])}",
            ]
        )
    return "\n".join(paths + model_lines + training_lines + evaluation_lines) + "\n"


def prepare_matrix(raw: dict[str, Any]) -> dict[str, Any]:
    output_root = Path(raw["experiment"]["output_root"])
    config_root = output_root / "generated_configs"
    matrix = []
    for task in raw["experiment"]["tasks"]:
        for seed in (int(value) for value in raw["seeds"][task][1:]):
            for model in MODELS:
                config_path = _write(
                    config_root / task / f"seed_{seed}_{model}.toml",
                    build_config(raw, task, seed, model),
                )
                report = _paths(raw, task, seed)[model] / "report.json"
                matrix.append(
                    {
                        "task": task,
                        "seed": seed,
                        "model": model,
                        "config": str(config_path),
                        "config_sha256": _sha256(config_path),
                        "report": str(report),
                        "log": str(report.parent / "run.log"),
                        "status": "passed" if _report_passed(report) else "pending",
                    }
                )
    return {"runs": matrix, "count": len(matrix)}


def _write_status(path: Path, status: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/submission_v1/multiseed_h8.toml"),
    )
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    raw = _read_toml(args.config)
    protocol_contract = validate_protocol_contract(raw)
    if not protocol_contract["passed"]:
        raise RuntimeError("executable matrix does not match the locked protocol")
    git = _git_state()
    if not git["working_tree_clean"] and not args.prepare_only:
        raise RuntimeError("training must start from a clean exact commit")
    seed_one = validate_seed_one(raw)
    if not seed_one["passed"]:
        raise RuntimeError("an existing seed-one run does not match the frozen protocol")
    matrix = prepare_matrix(raw)
    if matrix["count"] != 24:
        raise RuntimeError(f"expected 24 additional runs, found {matrix['count']}")
    output_root = Path(raw["experiment"]["output_root"])
    status_path = output_root / "run_status.json"
    status = {
        "experiment_id": raw["experiment"]["id"],
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "git": git,
        "protocol": str(raw["experiment"]["protocol"]),
        "protocol_contract": protocol_contract,
        "seed_one_validation": seed_one,
        "input_sha256": {
            str(Path(raw["task_inputs"][task][name])): _sha256(
                Path(raw["task_inputs"][task][name])
            )
            for task in raw["experiment"]["tasks"]
            for name in ("split_manifest", "normalization")
        },
        "matrix": matrix["runs"],
    }
    _write_status(status_path, status)
    print(f"prepared additional_runs={matrix['count']} status={status_path}", flush=True)
    if args.prepare_only:
        return

    for index, run in enumerate(status["matrix"], start=1):
        report_path = Path(run["report"])
        if _report_passed(report_path):
            run["status"] = "passed"
            print(
                f"[{index:02d}/24] skip passed {run['task']} seed={run['seed']} "
                f"model={run['model']}",
                flush=True,
            )
            continue
        log_path = Path(run["log"])
        log_path.parent.mkdir(parents=True, exist_ok=True)
        run["status"] = "running"
        run["started_at_utc"] = datetime.now(UTC).isoformat()
        _write_status(status_path, status)
        print(
            f"[{index:02d}/24] run {run['task']} seed={run['seed']} "
            f"model={run['model']} log={log_path}",
            flush=True,
        )
        with log_path.open("w", encoding="utf-8") as log:
            result = subprocess.run(
                [
                    sys.executable,
                    TRAINING_SCRIPTS[run["model"]],
                    "--config",
                    run["config"],
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        run["finished_at_utc"] = datetime.now(UTC).isoformat()
        run["return_code"] = result.returncode
        run["status"] = "passed" if _report_passed(report_path) else "failed"
        if report_path.exists():
            run["report_sha256"] = _sha256(report_path)
        checkpoint_path = report_path.parent / "best.pt"
        if checkpoint_path.exists():
            run["checkpoint_sha256"] = _sha256(checkpoint_path)
        _write_status(status_path, status)
        print(f"[{index:02d}/24] {run['status']}", flush=True)

    status["completed_at_utc"] = datetime.now(UTC).isoformat()
    status["passed_runs"] = sum(run["status"] == "passed" for run in status["matrix"])
    status["failed_runs"] = sum(run["status"] == "failed" for run in status["matrix"])
    status["passed"] = status["passed_runs"] == 24
    _write_status(status_path, status)
    print(
        f"complete passed={status['passed_runs']} failed={status['failed_runs']}",
        flush=True,
    )
    if not status["passed"]:
        raise RuntimeError("one or more frozen training runs failed")


if __name__ == "__main__":
    main()
