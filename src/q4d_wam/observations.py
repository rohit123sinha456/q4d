"""Utilities for inspecting nested ManiSkill observations without assuming camera names."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def tensor_leaves(value: Any, prefix: str = "obs") -> dict[str, Any]:
    """Flatten tensor/array leaves from nested mappings and sequences."""
    leaves: dict[str, Any] = {}
    if isinstance(value, Mapping):
        for key, child in value.items():
            leaves.update(tensor_leaves(child, f"{prefix}.{key}"))
    elif isinstance(value, list | tuple):
        for index, child in enumerate(value):
            leaves.update(tensor_leaves(child, f"{prefix}.{index}"))
    elif hasattr(value, "shape"):
        leaves[prefix] = value
    return leaves


def leaf_summary(value: Any) -> dict[str, dict[str, Any]]:
    """Return JSON-safe shape, dtype, and device metadata for observation leaves."""
    summary: dict[str, dict[str, Any]] = {}
    for name, leaf in tensor_leaves(value).items():
        summary[name] = {
            "shape": list(leaf.shape),
            "dtype": str(getattr(leaf, "dtype", "unknown")),
            "device": str(getattr(leaf, "device", "cpu")),
        }
    return summary


def find_leaf(value: Any, suffix: str) -> Any | None:
    """Find the first nested tensor/array whose dotted name ends with suffix."""
    suffix = suffix.lower()
    for name, leaf in tensor_leaves(value).items():
        if name.lower().endswith(suffix):
            return leaf
    return None
