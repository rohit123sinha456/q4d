from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from scripts.run_submission_multiseed_h8 import (
    MODELS,
    build_config,
    validate_protocol_contract,
    validate_seed_one,
)
from scripts.summarize_submission_multiseed_h8 import exact_sign_flip_pvalue

CONFIG = Path("configs/submission_v1/multiseed_h8.toml")


def _config() -> dict:
    with CONFIG.open("rb") as stream:
        return tomllib.load(stream)


def test_seed_one_runs_match_frozen_protocol() -> None:
    validation = validate_seed_one(_config())
    assert validation["passed"]
    assert len(validation["checks"]) == 12


def test_executable_matrix_matches_locked_protocol() -> None:
    validation = validate_protocol_contract(_config())
    assert validation["passed"]


@pytest.mark.parametrize("model", MODELS)
def test_new_seed_configs_keep_frozen_training_settings(model: str) -> None:
    raw = _config()
    generated = tomllib.loads(build_config(raw, "pull_cube", 2702, model))
    training = generated["training"]
    assert training["seed"] == 2702
    assert generated["model"]["horizon"] == 8
    assert generated["model"]["width"] == 128
    assert training["epochs"] == 30
    assert training["patience"] == 6
    assert training["learning_rate"] == pytest.approx(0.001)
    assert training["weight_decay"] == pytest.approx(0.0001)
    assert training["gradient_clip_norm"] == pytest.approx(1.0)
    assert training["amp"] is True
    batch_key = "batch_size" if model == "no_action" else "micro_batch_size"
    assert training[batch_key] == 32
    assert generated["paths"]["split_manifest"].endswith(
        "pull_cube_scale_v1/splits.json"
    )
    assert generated["paths"]["normalization"].endswith(
        "pull_cube_scale_v1/normalization.json"
    )


def test_q4d_config_enables_required_per_seed_benchmarks() -> None:
    generated = tomllib.loads(build_config(_config(), "pick_cube", 3703, "micro_q4d"))
    assert generated["evaluation"]["action_shuffle"] is True
    assert generated["evaluation"]["candidate_branches"] == 64
    assert generated["evaluation"]["benchmark_queries"] == 64


def test_dense_config_uses_same_seed_q4d_for_decoding_benchmark() -> None:
    generated = tomllib.loads(build_config(_config(), "stack_cube", 5702, "dense"))
    assert "seed_5702/micro_q4d/best.pt" in generated["paths"][
        "micro_q4d_checkpoint"
    ]
    assert generated["evaluation"]["sparse_queries"] == 64


def test_exact_sign_flip_pvalue() -> None:
    assert exact_sign_flip_pvalue([-1.0, -1.0, -1.0]) == pytest.approx(1 / 8)
    assert exact_sign_flip_pvalue([1.0, 1.0, 1.0]) == pytest.approx(1.0)
