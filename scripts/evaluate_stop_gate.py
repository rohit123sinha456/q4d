#!/usr/bin/env python3
"""Aggregate completed MVP artifacts into a machine-readable stop/continue decision."""

from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path
from typing import Any

from q4d_wam.evaluation.decision import evaluate_stop_gate


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"missing gate input: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/stop_gate.toml"))
    args = parser.parse_args()
    with args.config.open("rb") as stream:
        raw = tomllib.load(stream)
    paths = raw["paths"]
    report = evaluate_stop_gate(
        no_action=_load_json(Path(paths["no_action_report"])),
        q4d=_load_json(Path(paths["q4d_report"])),
        dense=_load_json(Path(paths["dense_report"])),
        cache_grid=_load_json(Path(paths["cache_grid_report"])),
        mpc=_load_json(Path(paths["mpc_report"])),
        thresholds={key: float(value) for key, value in raw["thresholds"].items()},
    )
    output = Path(paths["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps(report, indent=2))
    if not report["continue"]:
        raise RuntimeError("MVP stop/continue gate failed")


if __name__ == "__main__":
    main()
