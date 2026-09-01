from __future__ import annotations

import sys
import tomllib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from evaluate_mpc import _planner_config  # noqa: E402
from run_submission_definitive_mpc import (  # noqa: E402
    build_task_config,
    prepare_configs,
    preflight,
    validate_protocol_contract,
)

CONFIG = Path("configs/submission_v1/definitive_mpc.toml")


def _config() -> dict:
    with CONFIG.open("rb") as stream:
        return tomllib.load(stream)


def test_definitive_matrix_matches_frozen_protocol() -> None:
    assert validate_protocol_contract(_config())["passed"]


def test_primary_config_is_30_seed_three_model_wall_clock_matrix() -> None:
    generated = tomllib.loads(build_task_config(_config(), "pull_cube", "wall_clock"))
    assert generated["planning"]["seed"] == 13701
    assert generated["planning"]["episodes"] == 30
    assert generated["planning"]["models"] == ["q4d", "dense", "no_action"]
    assert generated["planning"]["methods"] == ["random_shooting"]
    assert generated["planning"]["budgets_ms"] == [100.0]
    assert "maximum_batches" not in generated["planning"]
    assert _planner_config(generated).maximum_batches is None


def test_fixed_count_control_executes_exactly_64_candidates_per_cycle() -> None:
    generated = tomllib.loads(build_task_config(_config(), "stack_cube", "fixed_count"))
    planner = _planner_config(generated)
    assert planner.candidates_per_batch == 64
    assert planner.maximum_batches == 1


def test_matrix_has_primary_and_fixed_control_for_all_tasks() -> None:
    runs = prepare_configs(_config())
    assert len(runs) == 8
    assert sum(run["episodes"] for run in runs if run["phase"] == "wall_clock") == 360
    assert sum(run["episodes"] for run in runs if run["phase"] == "fixed_count") == 360


def test_preflight_blocks_on_failed_pick_cube_pilot() -> None:
    flight = preflight(_config())
    assert flight["checks"]["prediction_matrix_complete"]
    assert not flight["checks"]["pick_cube_pilot_passed"]
    assert not flight["passed"]
