from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).parents[1]
TASKS = {
    "pull_cube": ("PullCube-v1", 2701, 16, 3601),
    "pick_cube": ("PickCube-v1", 3701, 24, 4601),
    "place_sphere": ("PlaceSphere-v1", 4701, 24, 5601),
    "stack_cube": ("StackCube-v1", 5701, 24, 6601),
}


def _load(name: str) -> dict:
    with (ROOT / "configs" / name).open("rb") as stream:
        return tomllib.load(stream)


def test_multitask_configs_preserve_pushcube_scale_and_isolate_outputs() -> None:
    push_scale = _load("scale_pushcube.toml")
    push_data = _load("data_scale.toml")
    push_experiment = _load("scale_experiment.toml")
    push_mpc = _load("mpc.toml")
    roots = set()

    for slug, (env_id, seed, approach_steps, planning_seed) in TASKS.items():
        dataset_root = f"artifacts/datasets/{slug}_scale_v1"
        experiment_root = f"artifacts/experiments/{slug}_scale_v1"
        scale = _load(f"scale_{slug}.toml")
        data = _load(f"data_scale_{slug}.toml")
        experiment = _load(f"scale_experiment_{slug}.toml")
        mpc = _load(f"mpc_{slug}.toml")

        assert scale["project"] == {"seed": seed, "output_dir": dataset_root}
        assert scale["model"] == push_scale["model"]
        assert scale["simulation"] == {
            **push_scale["simulation"],
            "env_id": env_id,
        }
        assert scale["labels"] == {
            **push_scale["labels"],
            "approach_max_steps": approach_steps,
        }

        assert data["dataset"] == {
            **push_data["dataset"],
            "root": dataset_root,
            "split_seed": seed,
        }
        assert data["loader"] == push_data["loader"]

        assert experiment["experiment"] == {
            **push_experiment["experiment"],
            "name": f"{slug}_scale_v1",
            "dataset_root": dataset_root,
            "data_config": f"configs/data_scale_{slug}.toml",
            "output_root": experiment_root,
            "seed": seed,
        }
        assert experiment["model"] == push_experiment["model"]
        assert experiment["training"] == push_experiment["training"]
        assert experiment["evaluation"] == push_experiment["evaluation"]

        assert mpc["model"] == push_mpc["model"]
        assert mpc["simulation"] == {
            **push_mpc["simulation"],
            "env_id": env_id,
            "approach_max_steps": approach_steps,
        }
        assert mpc["planning"] == {
            **push_mpc["planning"],
            "seed": planning_seed,
        }
        assert mpc["paths"] == {
            "normalization": f"{dataset_root}/normalization.json",
            "micro_q4d_checkpoint": f"{experiment_root}/h8/micro_q4d/best.pt",
            "dense_checkpoint": f"{experiment_root}/h8/dense/best.pt",
            "no_action_checkpoint": f"{experiment_root}/h8/no_action/best.pt",
            "output": f"artifacts/planning/{slug}_mpc_v1/report.json",
        }
        roots.add(dataset_root)

    assert len(roots) == len(TASKS)
