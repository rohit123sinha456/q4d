"""Typed configuration for small, reproducible Q4D experiments."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectConfig:
    seed: int
    output_dir: Path


@dataclass(frozen=True)
class ModelConfig:
    n_scene_points: int
    n_query_points: int
    horizon: int
    width: int


@dataclass(frozen=True)
class SimulationConfig:
    env_id: str
    num_envs: int
    obs_mode: str
    control_mode: str
    steps: int
    sim_backend: str
    render_backend: str


@dataclass(frozen=True)
class ExperimentConfig:
    project: ProjectConfig
    model: ModelConfig
    simulation: SimulationConfig


def load_config(path: str | Path) -> ExperimentConfig:
    """Load and validate an experiment TOML file."""
    config_path = Path(path)
    with config_path.open("rb") as stream:
        raw = tomllib.load(stream)

    project = ProjectConfig(
        seed=int(raw["project"]["seed"]),
        output_dir=Path(raw["project"]["output_dir"]),
    )
    model = ModelConfig(**raw["model"])
    simulation = SimulationConfig(**raw["simulation"])

    if model.n_query_points > model.n_scene_points:
        raise ValueError("n_query_points cannot exceed n_scene_points")
    if min(model.n_scene_points, model.n_query_points, model.horizon, model.width) <= 0:
        raise ValueError("model sizes and horizon must be positive")
    if simulation.num_envs <= 0 or simulation.steps <= 0:
        raise ValueError("num_envs and steps must be positive")
    if not simulation.env_id.endswith("-v1"):
        raise ValueError("expected a versioned ManiSkill environment id")

    return ExperimentConfig(project=project, model=model, simulation=simulation)

