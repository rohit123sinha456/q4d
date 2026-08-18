#!/usr/bin/env python3
"""Generate privileged persistent 3D point-trajectory labels for PushCube."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from q4d_wam.geometry import backproject_depth_cv, camera_cv_to_world
from q4d_wam.labels import (
    CATEGORY_GOAL,
    CATEGORY_OBJECT,
    CATEGORY_ROBOT,
    CATEGORY_STATIC,
    CATEGORY_UNKNOWN,
    attach_points_to_bodies,
    reconstruct_rigid_tracks,
    stratified_point_indices,
)

CATEGORY_NAMES = {
    CATEGORY_STATIC: "static",
    CATEGORY_ROBOT: "robot",
    CATEGORY_OBJECT: "object",
    CATEGORY_GOAL: "goal",
    CATEGORY_UNKNOWN: "unknown",
}

COUNTERFACTUAL_BRANCHES = ("success", "perturbed", "no_op", "failure")
SCALED_BRANCHES = ("success", "weak", "off_target", "failure", "no_op")


@dataclass(frozen=True)
class LabelConfig:
    num_points: int
    horizon: int
    max_depth_m: float
    object_quota: int
    robot_quota: int
    goal_quota: int
    approach_max_steps: int
    contact_distance_m: float


@dataclass(frozen=True)
class BodyEntry:
    segmentation_id: int
    name: str
    category: int
    entity: Any


@dataclass(frozen=True)
class InitialState:
    """Post-approach simulator state shared by every counterfactual branch."""

    observation: Any
    simulator_state: dict[str, Any]
    approach_steps: int


def _load_settings(config_path: Path) -> tuple[dict[str, Any], LabelConfig]:
    with config_path.open("rb") as stream:
        raw = tomllib.load(stream)
    return raw, LabelConfig(**raw["labels"])


def _pose_matrix(entity: Any) -> torch.Tensor:
    return torch.as_tensor(entity.pose.to_transformation_matrix(), dtype=torch.float32)


def _build_body_registry(env: Any) -> list[BodyEntry]:
    unwrapped = env.unwrapped
    robot_ids = {
        int(segmentation_id)
        for link in unwrapped.agent.robot.links
        for segmentation_id in link.per_scene_id
    }
    object_id = int(unwrapped.obj.per_scene_id[0])
    goal_id = int(unwrapped.goal_region.per_scene_id[0])
    entries = []
    for entity in unwrapped.scene.sub_scenes[0].entities:
        component_names = {type(component).__name__ for component in entity.components}
        if "RenderBodyComponent" not in component_names:
            continue
        segmentation_id = int(entity.per_scene_id)
        if segmentation_id in robot_ids:
            category = CATEGORY_ROBOT
        elif segmentation_id == object_id:
            category = CATEGORY_OBJECT
        elif segmentation_id == goal_id:
            category = CATEGORY_GOAL
        else:
            category = CATEGORY_STATIC
        entries.append(BodyEntry(segmentation_id, entity.name, category, entity))
    return sorted(entries, key=lambda entry: entry.segmentation_id)


def _body_poses(entries: list[BodyEntry]) -> torch.Tensor:
    return torch.stack([_pose_matrix(entry.entity) for entry in entries])


def _make_action(env: Any, target_world: torch.Tensor, gripper: float = -1.0) -> np.ndarray:
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    tcp_world = env.unwrapped.agent.tcp.pose.p[0].detach().cpu()
    delta_world = target_world.detach().cpu() - tcp_world
    # Panda normalized actions map [-1, 1] to [-0.1, 0.1] metres.
    action[..., :3] = np.clip(delta_world.numpy() / 0.1, -1.0, 1.0)
    action[..., -1] = gripper
    return action


def _move_to(env: Any, target_world: torch.Tensor, max_steps: int) -> tuple[Any, int]:
    observation = None
    for step in range(max_steps):
        action = _make_action(env, target_world)
        observation, _, _, _, _ = env.step(action)
        tcp_world = env.unwrapped.agent.tcp.pose.p[0].detach().cpu()
        if torch.linalg.vector_norm(tcp_world - target_world.detach().cpu()) < 0.006:
            return observation, step + 1
    return observation, max_steps


def _approach_cube(env: Any, max_steps: int) -> tuple[Any, int]:
    cube = env.unwrapped.obj.pose.p[0].detach().cpu()
    above = cube + torch.tensor([-0.05, 0.0, 0.08])
    observation, first_steps = _move_to(env, above, max_steps // 2)
    behind = cube + torch.tensor([-0.05, 0.0, 0.0])
    observation, second_steps = _move_to(env, behind, max_steps - first_steps)
    return observation, first_steps + second_steps


def _prepare_initial_state(env: Any, seed: int, max_steps: int) -> InitialState:
    """Reset and approach once, then snapshot the identical branch point."""
    env.reset(seed=seed)
    observation, approach_steps = _approach_cube(env, max_steps)
    if observation is None:
        raise RuntimeError("approach controller produced no observation")
    return InitialState(
        observation=copy.deepcopy(observation),
        simulator_state=copy.deepcopy(env.unwrapped.get_state_dict()),
        approach_steps=approach_steps,
    )


def _category_tensor(segmentation_ids: torch.Tensor, entries: list[BodyEntry]) -> torch.Tensor:
    categories = torch.full_like(segmentation_ids, CATEGORY_UNKNOWN, dtype=torch.long)
    for entry in entries:
        categories[segmentation_ids == entry.segmentation_id] = entry.category
    return categories


def _contact_region(
    tracks: torch.Tensor, categories: torch.Tensor, distance: float
) -> torch.Tensor:
    robot = tracks[categories == CATEGORY_ROBOT]
    result = torch.zeros(len(tracks), dtype=torch.bool)
    non_robot_indices = torch.nonzero(categories != CATEGORY_ROBOT, as_tuple=False).squeeze(-1)
    if len(robot) == 0 or len(non_robot_indices) == 0:
        return result
    non_robot = tracks[non_robot_indices]
    minimum = torch.full((len(non_robot),), torch.inf)
    for time_index in range(tracks.shape[1]):
        distances = torch.cdist(non_robot[:, time_index], robot[:, time_index])
        minimum = torch.minimum(minimum, distances.min(dim=1).values)
    result[non_robot_indices] = minimum < distance
    return result


def _rigidity_error(tracks: torch.Tensor, body_indices: torch.Tensor) -> float:
    maximum = 0.0
    for body_index in torch.unique(body_indices[body_indices >= 0]):
        body_tracks = tracks[body_indices == body_index]
        if len(body_tracks) < 2:
            continue
        # Exclude cdist's diagonal: its float32 square-root kernel can report a small
        # nonzero self-distance for larger point sets and create a false rigidity error.
        initial_distances = torch.pdist(body_tracks[:, 0])
        for time_index in range(1, tracks.shape[1]):
            distances = torch.pdist(body_tracks[:, time_index])
            maximum = max(maximum, float((distances - initial_distances).abs().max()))
    return maximum


def _save_visualization(
    path: Path, tracks: torch.Tensor, categories: torch.Tensor, contact: torch.Tensor
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xyz = tracks.detach().cpu().numpy()
    categories_np = categories.cpu().numpy()
    contact_np = contact.cpu().numpy()
    colors = {
        CATEGORY_STATIC: "#999999",
        CATEGORY_ROBOT: "#2474b5",
        CATEGORY_OBJECT: "#d62728",
        CATEGORY_GOAL: "#2ca02c",
        CATEGORY_UNKNOWN: "#000000",
    }

    figure = plt.figure(figsize=(12, 5), dpi=160)
    axis_3d = figure.add_subplot(121, projection="3d")
    axis_top = figure.add_subplot(122)
    for category, name in CATEGORY_NAMES.items():
        mask = categories_np == category
        if not np.any(mask):
            continue
        initial = xyz[mask, 0]
        axis_3d.scatter(*initial.T, s=8, c=colors[category], label=name, alpha=0.8)
        axis_top.scatter(initial[:, 0], initial[:, 1], s=8, c=colors[category], label=name)
        if category in (CATEGORY_ROBOT, CATEGORY_OBJECT):
            for trajectory in xyz[mask]:
                axis_3d.plot(*trajectory.T, c=colors[category], linewidth=0.7, alpha=0.7)
                axis_top.plot(trajectory[:, 0], trajectory[:, 1], c=colors[category], alpha=0.5)

    contact_start = xyz[contact_np, 0]
    if len(contact_start):
        axis_3d.scatter(*contact_start.T, s=30, facecolors="none", edgecolors="magenta")
        axis_top.scatter(
            contact_start[:, 0],
            contact_start[:, 1],
            s=30,
            facecolors="none",
            edgecolors="magenta",
            label="contact region",
        )
    axis_3d.set_xlabel("world x (m)")
    axis_3d.set_ylabel("world y (m)")
    axis_3d.set_zlabel("world z (m)")
    axis_3d.set_title("Persistent rigid point trajectories")
    axis_top.set_xlabel("world x (m)")
    axis_top.set_ylabel("world y (m)")
    axis_top.set_title("Top view")
    axis_top.set_aspect("equal", adjustable="box")
    axis_top.grid(alpha=0.25)
    axis_top.legend(loc="best", fontsize=8)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def _generate_episode(
    env: Any,
    settings: dict[str, Any],
    label_config: LabelConfig,
    seed: int,
    state_index: int,
    branch: str,
    output_dir: Path,
    initial_state: InitialState | None = None,
    enforce_outcome_checks: bool = True,
) -> dict[str, Any]:
    known_branches = set(COUNTERFACTUAL_BRANCHES) | set(SCALED_BRANCHES)
    if branch not in known_branches:
        raise ValueError(f"unknown counterfactual branch: {branch}")
    if initial_state is None:
        initial_state = _prepare_initial_state(
            env, seed, label_config.approach_max_steps
        )
    else:
        env.unwrapped.set_state_dict(copy.deepcopy(initial_state.simulator_state))
    observation = initial_state.observation
    approach_steps = initial_state.approach_steps

    entries = _build_body_registry(env)
    body_segmentation_ids = torch.tensor([entry.segmentation_id for entry in entries])
    body_categories = torch.tensor([entry.category for entry in entries])
    body_names = [entry.name for entry in entries]

    camera_name = next(iter(observation["sensor_data"]))
    sensor = observation["sensor_data"][camera_name]
    calibration = observation["sensor_param"][camera_name]
    depth_mm = sensor["depth"]
    rgb = sensor["rgb"]
    segmentation = sensor["segmentation"]
    points_cv, valid = backproject_depth_cv(
        depth_mm, calibration["intrinsic_cv"], max_depth_m=label_config.max_depth_m
    )
    points_world = camera_cv_to_world(points_cv, calibration["extrinsic_cv"])

    segmentation_image = segmentation.squeeze(-1)
    known = torch.isin(segmentation_image, body_segmentation_ids)
    usable = valid & known
    flat_world = points_world[usable]
    flat_rgb = rgb[usable]
    flat_segmentation = segmentation_image[usable].to(torch.long)
    flat_categories = _category_tensor(flat_segmentation, entries)

    rows, columns = torch.meshgrid(
        torch.arange(depth_mm.shape[1]), torch.arange(depth_mm.shape[2]), indexing="ij"
    )
    pixels_uv = torch.stack((columns, rows), dim=-1)[None].expand(depth_mm.shape[0], -1, -1, -1)
    flat_pixels_uv = pixels_uv[usable]

    selected = stratified_point_indices(
        flat_world,
        flat_categories,
        label_config.num_points,
        {
            CATEGORY_OBJECT: label_config.object_quota,
            CATEGORY_ROBOT: label_config.robot_quota,
            CATEGORY_GOAL: label_config.goal_quota,
        },
    )
    xyz0 = flat_world[selected]
    point_rgb = flat_rgb[selected]
    point_segmentation_ids = flat_segmentation[selected]
    point_categories = flat_categories[selected]
    point_pixels_uv = flat_pixels_uv[selected]

    initial_body_poses = _body_poses(entries)
    attachments = attach_points_to_bodies(
        xyz0, point_segmentation_ids, body_segmentation_ids, initial_body_poses
    )
    body_pose_sequence = [initial_body_poses]
    cube_centers = [env.unwrapped.obj.pose.p[0].detach().cpu().clone()]
    actions = []
    goal = env.unwrapped.goal_region.pose.p[0].detach().cpu()
    cube_start = env.unwrapped.obj.pose.p[0].detach().cpu().clone()
    rng = np.random.default_rng(seed + 10_000)
    perturb_sign = -1.0 if state_index % 2 else 1.0
    lateral_offset = perturb_sign * float(rng.uniform(0.03, 0.06))
    success_lateral_jitter = 0.0
    success_standoff = 0.12
    weak_scale = float(rng.uniform(0.30, 0.50))
    weak_steps = int(rng.integers(3, max(4, label_config.horizon // 2 + 1)))
    off_target_offset = perturb_sign * float(rng.uniform(0.18, 0.25))
    failure_offset = float(rng.uniform(-0.04, 0.04))
    branch_targets = {
        "success": goal
        + torch.tensor([-success_standoff, success_lateral_jitter, 0.0]),
        "perturbed": goal + torch.tensor([-0.12, lateral_offset, 0.0]),
        "weak": goal + torch.tensor([-success_standoff, success_lateral_jitter, 0.0]),
        "off_target": cube_start
        + torch.tensor([float(rng.uniform(0.05, 0.08)), off_target_offset, 0.0]),
        "failure": cube_start + torch.tensor([-0.16, failure_offset, 0.0]),
    }
    final_info: dict[str, Any] = {}
    success_reached = False
    for time_index in range(label_config.horizon):
        if (
            branch == "no_op"
            or (branch == "success" and success_reached)
            or (branch == "weak" and time_index >= weak_steps)
        ):
            action = np.zeros(env.action_space.shape, dtype=np.float32)
            action[..., -1] = -1.0
        else:
            action = _make_action(env, branch_targets[branch])
            if branch == "weak":
                action[..., :3] *= weak_scale
        _, _, _, _, final_info = env.step(action)
        success_reached |= bool(torch.as_tensor(final_info.get("success", False)).any())
        actions.append(torch.from_numpy(action.reshape(-1)).clone())
        body_pose_sequence.append(_body_poses(entries))
        cube_centers.append(env.unwrapped.obj.pose.p[0].detach().cpu().clone())

    body_pose_sequence_tensor = torch.stack(body_pose_sequence)
    tracks = reconstruct_rigid_tracks(
        attachments.local_xyz_m, attachments.body_indices, body_pose_sequence_tensor
    )
    contact = _contact_region(tracks, point_categories, label_config.contact_distance_m)
    actions_tensor = torch.stack(actions)
    cube_centers_tensor = torch.stack(cube_centers)
    task_success_final = bool(torch.as_tensor(final_info.get("success", False)).any())

    static_mask = point_categories == CATEGORY_STATIC
    object_mask = point_categories == CATEGORY_OBJECT
    robot_mask = point_categories == CATEGORY_ROBOT
    static_displacement = (
        torch.linalg.vector_norm(tracks[static_mask, -1] - tracks[static_mask, 0], dim=-1)
        if static_mask.any()
        else torch.zeros(1)
    )
    cube_displacement = torch.linalg.vector_norm(cube_centers_tensor[-1] - cube_centers_tensor[0])
    initial_reconstruction_error = torch.linalg.vector_norm(tracks[:, 0] - xyz0, dim=-1)

    group_id = f"state_{state_index:06d}"
    stem = f"{group_id}__{branch}"
    training_path = output_dir / f"{stem}.train.npz"
    audit_path = output_dir / f"{stem}.audit.npz"
    np.savez_compressed(
        training_path,
        rgb=_numpy(rgb[0]),
        depth_mm=_numpy(depth_mm[0]),
        intrinsic_cv=_numpy(calibration["intrinsic_cv"][0]),
        extrinsic_cv=_numpy(calibration["extrinsic_cv"][0]),
        cam2world_gl=_numpy(calibration["cam2world_gl"][0]),
        actions=_numpy(actions_tensor),
        point_pixels_uv=_numpy(point_pixels_uv),
        xyz0_world_m=_numpy(xyz0),
        point_rgb=_numpy(point_rgb),
        target_tracks_world_m=_numpy(tracks[:, 1:]),
    )
    np.savez_compressed(
        audit_path,
        body_segmentation_ids=_numpy(body_segmentation_ids),
        body_categories=_numpy(body_categories),
        point_segmentation_ids=_numpy(point_segmentation_ids),
        point_categories=_numpy(point_categories),
        body_indices=_numpy(attachments.body_indices),
        local_xyz_m=_numpy(attachments.local_xyz_m),
        body_pose_sequence_world=_numpy(body_pose_sequence_tensor),
        tracks_world_m=_numpy(tracks),
        contact_region=_numpy(contact),
        cube_centers_world_m=_numpy(cube_centers_tensor),
    )

    counts = {
        CATEGORY_NAMES[category]: int((point_categories == category).sum())
        for category in CATEGORY_NAMES
    }
    record = {
        "state_index": state_index,
        "group_id": group_id,
        "branch": branch,
        "seed": seed,
        "environment": settings["simulation"]["env_id"],
        "approach_steps": approach_steps,
        "horizon": label_config.horizon,
        "num_points": len(xyz0),
        "category_counts": counts,
        "contact_region_points": int(contact.sum()),
        "cube_displacement_m": float(cube_displacement),
        "task_success": task_success_final,
        "task_success_ever": success_reached,
        "lateral_perturbation_m": lateral_offset if branch == "perturbed" else 0.0,
        "success_lateral_jitter_m": success_lateral_jitter,
        "success_standoff_m": success_standoff,
        "weak_action_scale": weak_scale if branch == "weak" else 1.0,
        "weak_active_steps": weak_steps if branch == "weak" else 0,
        "off_target_lateral_offset_m": (
            off_target_offset if branch == "off_target" else 0.0
        ),
        "object_point_final_displacement_mean_m": float(
            torch.linalg.vector_norm(
                tracks[object_mask, -1] - tracks[object_mask, 0], dim=-1
            ).mean()
        ),
        "robot_point_final_displacement_mean_m": float(
            torch.linalg.vector_norm(tracks[robot_mask, -1] - tracks[robot_mask, 0], dim=-1).mean()
        ),
        "static_displacement_max_m": float(static_displacement.max()),
        "initial_reconstruction_error_max_m": float(initial_reconstruction_error.max()),
        "rigidity_error_max_m": _rigidity_error(tracks, attachments.body_indices),
        "all_tracks_finite": bool(torch.isfinite(tracks).all()),
        "body_names": body_names,
        "training_file": training_path.name,
        "audit_file": audit_path.name,
    }
    if record["task_success"]:
        observed_outcome = "successful"
    elif record["cube_displacement_m"] <= 0.002:
        observed_outcome = "no_motion"
    elif record["cube_displacement_m"] <= 0.08:
        observed_outcome = "weak"
    else:
        observed_outcome = "off_target"
    record["observed_outcome"] = observed_outcome
    checks = {
        "all_points_attached": bool(attachments.known_body.all()),
        "initial_reconstruction": record["initial_reconstruction_error_max_m"] < 1e-5,
        "static_points_fixed": record["static_displacement_max_m"] < 1e-6,
        "rigid_distances_preserved": record["rigidity_error_max_m"] < 1e-5,
        "object_points_present": counts["object"] > 0,
        "robot_points_present": counts["robot"] > 0,
        "tracks_finite": record["all_tracks_finite"],
    }
    outcome_checks = {}
    if branch == "success":
        outcome_checks["intended_motion_present"] = record["cube_displacement_m"] > 0.002
        outcome_checks["task_success_achieved"] = record["task_success"]
    if branch in ("no_op", "failure", "weak"):
        outcome_checks["non_success_outcome"] = not record["task_success"]
    if branch in ("weak", "off_target"):
        outcome_checks["intended_motion_present"] = record["cube_displacement_m"] > 0.002
    if enforce_outcome_checks:
        checks.update(outcome_checks)
    record["checks"] = checks
    record["outcome_checks"] = outcome_checks
    record["outcome_match"] = all(outcome_checks.values())
    record["passed"] = all(checks.values())
    record_path = output_dir / f"{stem}.record.json"
    record["record_file"] = record_path.name
    record_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    if state_index == 0:
        _save_visualization(
            output_dir / f"trajectory_labels_{branch}.png", tracks, point_categories, contact
        )
    return record


def _numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy()


def _validate_group_identity(
    output_dir: Path,
    records: list[dict[str, Any]],
    expected_branches: tuple[str, ...] = COUNTERFACTUAL_BRANCHES,
) -> dict[str, Any]:
    initial_keys = (
        "rgb",
        "depth_mm",
        "intrinsic_cv",
        "extrinsic_cv",
        "cam2world_gl",
        "point_pixels_uv",
        "xyz0_world_m",
        "point_rgb",
    )
    reference_path = output_dir / records[0]["training_file"]
    with np.load(reference_path, allow_pickle=False) as archive:
        reference = {key: archive[key] for key in initial_keys}
    observations_identical = True
    action_signatures = set()
    for record in records:
        path = output_dir / record["training_file"]
        with np.load(path, allow_pickle=False) as archive:
            observations_identical &= all(
                np.array_equal(reference[key], archive[key]) for key in initial_keys
            )
            action_signatures.add(hashlib.sha256(archive["actions"].tobytes()).hexdigest())
    branches = {record["branch"] for record in records}
    return {
        "group_id": records[0]["group_id"],
        "branches": sorted(branches),
        "initial_observations_identical": observations_identical,
        "distinct_action_chunks": len(action_signatures),
        "passed": observations_identical
        and branches == set(expected_branches)
        and len(action_signatures) == len(expected_branches),
    }


def _record_is_complete(output_dir: Path, record: dict[str, Any]) -> bool:
    return bool(record.get("passed")) and all(
        (output_dir / record[key]).exists()
        for key in ("training_file", "audit_file", "record_file")
    )


def _load_completed_group(
    output_dir: Path, state_index: int, branches: tuple[str, ...]
) -> list[dict[str, Any]] | None:
    records = []
    for branch in branches:
        path = output_dir / f"state_{state_index:06d}__{branch}.record.json"
        if not path.exists():
            return None
        record = json.loads(path.read_text(encoding="utf-8"))
        if not _record_is_complete(output_dir, record):
            return None
        if "observed_outcome" not in record:
            if record["task_success"]:
                record["observed_outcome"] = "successful"
            elif record["cube_displacement_m"] <= 0.002:
                record["observed_outcome"] = "no_motion"
            elif record["cube_displacement_m"] <= 0.08:
                record["observed_outcome"] = "weak"
            else:
                record["observed_outcome"] = "off_target"
        records.append(record)
    return records


def _summary(
    *,
    requested_states: int,
    branches: tuple[str, ...],
    records: list[dict[str, Any]],
    groups: list[dict[str, Any]],
    profile: str,
    complete: bool,
) -> dict[str, Any]:
    return {
        "profile": profile,
        "requested_states": requested_states,
        "completed_states": len(groups),
        "complete": complete,
        "branches_per_state": len(branches),
        "fragments": len(records),
        "passed_fragments": sum(record["passed"] for record in records),
        "passed_groups": sum(group["passed"] for group in groups),
        "mean_cube_displacement_m": float(
            np.mean([record["cube_displacement_m"] for record in records])
        ),
        "branch_counts": {
            branch: sum(record["branch"] == branch for record in records)
            for branch in branches
        },
        "branch_success_counts": {
            branch: sum(
                record["branch"] == branch and record["task_success"]
                for record in records
            )
            for branch in branches
        },
        "branch_mean_cube_displacement_m": {
            branch: float(
                np.mean(
                    [
                        record["cube_displacement_m"]
                        for record in records
                        if record["branch"] == branch
                    ]
                )
            )
            for branch in branches
        },
        "observed_outcome_counts": {
            outcome: sum(record.get("observed_outcome") == outcome for record in records)
            for outcome in ("successful", "weak", "off_target", "no_motion")
        },
        "intended_outcome_matches": sum(
            record.get("outcome_match", True) for record in records
        ),
        "groups": groups,
        "records": records,
    }


def _write_manifest(
    output_dir: Path, summary: dict[str, Any], manifest_name: str = "manifest.json"
) -> None:
    temporary = output_dir / f"{manifest_name}.tmp"
    temporary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output_dir / manifest_name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/smoke.toml"))
    parser.add_argument("--states", type=int, default=1)
    parser.add_argument("--start-state", type=int, default=0)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--profile", choices=("counterfactual", "scaled"), default="counterfactual"
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--manifest-name", default="manifest.json")
    args = parser.parse_args()

    if args.states <= 0:
        raise ValueError("states must be positive")
    if args.start_state < 0:
        raise ValueError("start-state cannot be negative")
    if args.checkpoint_every <= 0:
        raise ValueError("checkpoint-every must be positive")
    if Path(args.manifest_name).name != args.manifest_name:
        raise ValueError("manifest-name must be a filename without directories")
    settings, label_config = _load_settings(args.config)
    base_seed = args.seed if args.seed is not None else int(settings["project"]["seed"])
    output_dir = args.output_dir or Path(settings["project"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    branches = SCALED_BRANCHES if args.profile == "scaled" else COUNTERFACTUAL_BRANCHES

    software_icd = Path("/usr/share/vulkan/icd.d/lvp_icd.json")
    if software_icd.exists():
        os.environ.setdefault("VK_ICD_FILENAMES", str(software_icd))
    import mani_skill.envs  # noqa: F401 -- registration after Vulkan configuration

    env = gym.make(
        settings["simulation"]["env_id"],
        num_envs=1,
        obs_mode="rgb+depth+segmentation",
        control_mode=settings["simulation"]["control_mode"],
        sim_backend="physx_cpu",
        render_backend="sapien_cpu",
        render_mode="sensors",
    )
    records = []
    groups = []
    try:
        for state_index in range(args.start_state, args.start_state + args.states):
            group_records = (
                _load_completed_group(output_dir, state_index, branches)
                if args.resume
                else None
            )
            if group_records is None:
                initial_state = _prepare_initial_state(
                    env, base_seed + state_index, label_config.approach_max_steps
                )
                group_records = []
                for branch in branches:
                    record = _generate_episode(
                        env,
                        settings,
                        label_config,
                        base_seed + state_index,
                        state_index,
                        branch,
                        output_dir,
                        initial_state,
                        enforce_outcome_checks=args.profile != "scaled",
                    )
                    records.append(record)
                    group_records.append(record)
                    print(
                        f"state={state_index} branch={branch} passed={record['passed']} "
                        f"success={record['task_success']} "
                        f"cube_motion={record['cube_displacement_m']:.4f}m",
                        flush=True,
                    )
            else:
                records.extend(group_records)
                print(f"state={state_index} resumed=true", flush=True)
            group = _validate_group_identity(output_dir, group_records, branches)
            groups.append(group)
            if not group["passed"]:
                raise RuntimeError(f"counterfactual group failed identity checks: {group}")
            if len(groups) % args.checkpoint_every == 0:
                _write_manifest(
                    output_dir,
                    _summary(
                        requested_states=args.states,
                        branches=branches,
                        records=records,
                        groups=groups,
                        profile=args.profile,
                        complete=False,
                    ),
                    args.manifest_name,
                )
    finally:
        env.close()

    summary = _summary(
        requested_states=args.states,
        branches=branches,
        records=records,
        groups=groups,
        profile=args.profile,
        complete=True,
    )
    _write_manifest(output_dir, summary, args.manifest_name)
    print(json.dumps({key: value for key, value in summary.items() if key != "records"}, indent=2))
    if summary["passed_fragments"] != summary["fragments"]:
        raise RuntimeError("one or more generated fragments failed label validation")


if __name__ == "__main__":
    main()
