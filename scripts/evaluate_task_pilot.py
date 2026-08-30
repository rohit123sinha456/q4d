#!/usr/bin/env python3
"""Evaluate one 100-state task pilot against the frozen pre-scale gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from q4d_wam.evaluation.pilot import evaluate_pilot_manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    manifest_path = args.root / "manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(f"pilot manifest does not exist: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    report = evaluate_pilot_manifest(manifest)
    output = args.output or args.root / "pilot_gate.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        print("task pilot failed the frozen pre-scale gate", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
