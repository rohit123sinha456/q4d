#!/usr/bin/env python3
"""Record and replay a short ManiSkill PushCube smoke episode."""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from q4d_wam.config import load_config
from q4d_wam.observations import find_leaf, leaf_summary


def to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def scalar_list(value: Any) -> list[float]:
    return to_numpy(value).reshape(-1).astype(float).tolist()


def first_image(value: Any) -> np.ndarray:
    image = to_numpy(value)
    while image.ndim > 3:
        image = image[0]
    return image


def create_env(config: Any, mode: str) -> tuple[Any, dict[str, Any]]:
    simulation = config.simulation
    if mode == "gpu-rgbd":
        overrides = {
            "num_envs": simulation.num_envs,
            "obs_mode": "rgbd",
            "sim_backend": "physx_cuda",
            "render_backend": "sapien_cuda",
            "render_mode": "sensors",
        }
    elif mode == "cpu-rgbd":
        software_icd = Path("/usr/share/vulkan/icd.d/lvp_icd.json")
        if software_icd.exists():
            os.environ.setdefault("VK_ICD_FILENAMES", str(software_icd))
        overrides = {
            "num_envs": 1,
            "obs_mode": "rgbd",
            "sim_backend": "physx_cpu",
            "render_backend": "sapien_cpu",
            "render_mode": "sensors",
        }
    else:
        software_icd = Path("/usr/share/vulkan/icd.d/lvp_icd.json")
        if software_icd.exists():
            os.environ.setdefault("VK_ICD_FILENAMES", str(software_icd))
        overrides = {
            "num_envs": 1,
            "obs_mode": "state",
            "sim_backend": "physx_cpu",
            "render_backend": "none",
            "render_mode": None,
        }

    kwargs = {
        "control_mode": simulation.control_mode,
        **overrides,
    }
    import mani_skill.envs  # noqa: F401 -- registers environments after backend selection

    return gym.make(simulation.env_id, **kwargs), kwargs


def run_episode(config_path: Path, mode: str, replay_path: Path | None) -> Path:
    config = load_config(config_path)
    output_dir = config.project.output_dir / mode
    output_dir.mkdir(parents=True, exist_ok=True)

    env, env_kwargs = create_env(config, mode)
    try:
        env.action_space.seed(config.project.seed)
        obs, reset_info = env.reset(seed=config.project.seed)
        initial_summary = leaf_summary(obs)

        if replay_path is None:
            actions = [env.action_space.sample() for _ in range(config.simulation.steps)]
        else:
            with np.load(replay_path) as replay:
                actions = [action for action in replay["actions"]]

        rewards: list[list[float]] = []
        terminated_steps: list[list[float]] = []
        truncated_steps: list[list[float]] = []
        start = time.perf_counter()
        for action in actions:
            obs, reward, terminated, truncated, _ = env.step(action)
            rewards.append(scalar_list(reward))
            terminated_steps.append(scalar_list(terminated))
            truncated_steps.append(scalar_list(truncated))
        elapsed = time.perf_counter() - start

        rgb = find_leaf(obs, ".rgb")
        depth = find_leaf(obs, ".depth")
        if mode.endswith("rgbd") and (rgb is None or depth is None):
            raise RuntimeError("RGB-D mode returned no .rgb or .depth observation leaves")

        episode_path = output_dir / ("episode_replay.npz" if replay_path else "episode.npz")
        np.savez_compressed(
            episode_path,
            seed=np.asarray(config.project.seed),
            actions=np.stack([to_numpy(action) for action in actions]),
            rewards=np.asarray(rewards),
            terminated=np.asarray(terminated_steps),
            truncated=np.asarray(truncated_steps),
        )

        if rgb is not None:
            from PIL import Image

            rgb_image = first_image(rgb)
            if np.issubdtype(rgb_image.dtype, np.floating):
                rgb_image = np.clip(rgb_image * 255.0, 0, 255)
            Image.fromarray(rgb_image.astype(np.uint8)).save(output_dir / "last_rgb.png")
        if depth is not None:
            np.save(output_dir / "last_depth.npy", first_image(depth))

        unwrapped = env.unwrapped
        report = {
            "mode": mode,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "mani_skill": __import__("mani_skill").__version__,
            "sapien": __import__("sapien").__version__,
            "environment": config.simulation.env_id,
            "environment_kwargs": env_kwargs,
            "environment_device": str(getattr(unwrapped, "device", "unknown")),
            "scene_device": str(getattr(getattr(unwrapped, "scene", None), "device", "unknown")),
            "steps": len(actions),
            "elapsed_seconds": elapsed,
            "steps_per_second": len(actions) * env_kwargs["num_envs"] / elapsed,
            "initial_observation": initial_summary,
            "final_observation": leaf_summary(obs),
            "reset_info_keys": sorted(reset_info),
            "rgb_present": rgb is not None,
            "depth_present": depth is not None,
            "episode_path": str(episode_path),
            "replayed_from": str(replay_path) if replay_path else None,
        }
        report_path = output_dir / ("replay_report.json" if replay_path else "report.json")
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
        return episode_path
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/smoke.toml"))
    parser.add_argument(
        "--mode", choices=("cpu-state", "cpu-rgbd", "gpu-rgbd"), default="gpu-rgbd"
    )
    parser.add_argument("--replay", type=Path)
    args = parser.parse_args()
    run_episode(args.config, args.mode, args.replay)


if __name__ == "__main__":
    main()
