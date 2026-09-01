import tomllib
from pathlib import Path

import pytest

from scripts.freeze_submission_protocol import validate_protocol

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "submission_protocol_v1.toml"


def _protocol() -> dict[str, object]:
    with CONFIG.open("rb") as stream:
        return tomllib.load(stream)


def test_submission_protocol_v1_is_valid() -> None:
    validate_protocol(_protocol())


def test_primary_conditions_are_frozen() -> None:
    protocol = _protocol()
    scope = protocol["scope"]
    assert scope["primary_horizon"] == 8
    assert scope["primary_planner"] == "random_shooting"
    assert scope["primary_budget_ms"] == 100.0
    assert scope["secondary_horizons"] == [1, 2, 4]


def test_invalid_primary_budget_is_rejected() -> None:
    protocol = _protocol()
    protocol["scope"]["primary_budget_ms"] = 200.0
    with pytest.raises(ValueError, match="100 ms"):
        validate_protocol(protocol)


def test_pilot_and_definitive_seeds_are_disjoint() -> None:
    protocol = _protocol()
    planning = protocol["planning"]
    for task in protocol["scope"]["tasks"]:
        pilot = set(planning[f"{task}_pilot_seeds"])
        start = planning[f"{task}_definitive_seed_start"]
        definitive = set(range(start, start + planning["definitive_seed_count"]))
        assert pilot.isdisjoint(definitive)
