#!/usr/bin/env python3
"""Inspect ManiSkill segmentation IDs and body views for the current PushCube scene."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import gymnasium as gym
import torch


def _json_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def main() -> None:
    software_icd = Path("/usr/share/vulkan/icd.d/lvp_icd.json")
    if software_icd.exists():
        os.environ.setdefault("VK_ICD_FILENAMES", str(software_icd))

    import mani_skill.envs  # noqa: F401 -- registers environments after Vulkan selection

    env = gym.make(
        "PushCube-v1",
        num_envs=1,
        obs_mode="rgb+depth+segmentation",
        control_mode="pd_ee_delta_pose",
        sim_backend="physx_cpu",
        render_backend="sapien_cpu",
        render_mode="sensors",
    )
    try:
        observation, _ = env.reset(seed=7)
        unwrapped = env.unwrapped
        camera_name = next(iter(observation["sensor_data"]))
        segmentation = observation["sensor_data"][camera_name]["segmentation"]
        unique_ids, counts = torch.unique(segmentation, return_counts=True)

        actor_views = []
        for name, actor in sorted(unwrapped.scene.actor_views.items()):
            actor_views.append(
                {
                    "name": name,
                    "per_scene_id": _json_value(actor.per_scene_id),
                    "pose": _json_value(actor.pose.raw_pose),
                    "body_type": getattr(actor, "px_body_type", None),
                }
            )

        entities = []
        for entity in unwrapped.scene.sub_scenes[0].entities:
            entities.append(
                {
                    "name": entity.name,
                    "per_scene_id": entity.per_scene_id,
                    "pose": [float(value) for value in (*entity.pose.p, *entity.pose.q)],
                    "components": [type(component).__name__ for component in entity.components],
                }
            )

        robot_links = []
        for link in unwrapped.agent.robot.links:
            robot_links.append(
                {
                    "name": link.name,
                    "per_scene_id": _json_value(link.per_scene_id),
                    "pose": _json_value(link.pose.raw_pose),
                }
            )

        print(
            json.dumps(
                {
                    "visible_segmentation": dict(
                        zip(
                            map(str, _json_value(unique_ids)),
                            _json_value(counts),
                            strict=True,
                        )
                    ),
                    "actor_views": actor_views,
                    "entities": entities,
                    "robot_links": robot_links,
                },
                indent=2,
            )
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
