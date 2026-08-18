"""Simple trajectory baselines used before fitting learned models."""

from q4d_wam.baselines.non_neural import (
    ActionKnnBaseline,
    MeanDisplacementBaseline,
    SceneKnnBaseline,
    StaticBaseline,
)

__all__ = [
    "ActionKnnBaseline",
    "MeanDisplacementBaseline",
    "SceneKnnBaseline",
    "StaticBaseline",
]
