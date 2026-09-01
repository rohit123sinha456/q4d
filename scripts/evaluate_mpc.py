#!/usr/bin/env python3
"""Evaluate random-shooting and CEM MPC through a task adapter."""

from __future__ import annotations

import argparse
import json
import os
import time
import tomllib
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from generate_point_tracks import (
    _build_body_registry,
    _category_tensor,
    _select_camera_name,
)
from PIL import Image, ImageDraw
from torch import Tensor

from q4d_wam.data import NormalizationStats
from q4d_wam.geometry import backproject_depth_cv, camera_cv_to_world
from q4d_wam.labels import (
    CATEGORY_GOAL,
    CATEGORY_OBJECT,
    CATEGORY_ROBOT,
    stratified_point_indices,
)
from q4d_wam.models import DensePointFutureModel, MicroQ4D, NoActionTrajectoryModel
from q4d_wam.planning import (
    DEFAULT_GRIPPER_SCHEDULES,
    CachedCubeCost,
    PlannerConfig,
    cem,
    random_shooting,
    sample_random_action_sequences,
)
from q4d_wam.tasks import TaskAdapter, get_task_adapter


def _read_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        return tomllib.load(stream)


def _planner_config(raw: dict[str, Any]) -> PlannerConfig:
    model = raw["model"]
    planning = raw["planning"]
    return PlannerConfig(
        horizon=int(model["horizon"]),
        action_dimensions=int(model["action_dimensions"]),
        candidates_per_batch=int(planning["candidates_per_batch"]),
        elite_fraction=float(planning["elite_fraction"]),
        initial_std_xy=float(planning["initial_std_xy"]),
        initial_std_z=float(planning["initial_std_z"]),
        minimum_std=float(planning["minimum_std"]),
        action_space=str(planning.get("action_space", "translation_only")),
        gripper_schedules=tuple(
            planning.get("gripper_schedules", DEFAULT_GRIPPER_SCHEDULES)
        ),
        minimum_schedule_probability=float(
            planning.get("minimum_schedule_probability", 0.05)
        ),
    )


def _cost_kwargs(planning: dict[str, Any]) -> dict[str, float | int | bool]:
    return {
        "action_penalty": float(planning["action_penalty"]),
        "settling_penalty": float(planning.get("settling_penalty", 0.0)),
        "settling_steps": int(planning.get("settling_steps", 2)),
        "use_amp": bool(planning["amp"]),
    }


def _rgb_frame(observation: dict[str, Any], env: Any, adapter: TaskAdapter) -> np.ndarray:
    camera_name = _select_camera_name(observation, env, adapter)
    rgb = torch.as_tensor(observation["sensor_data"][camera_name]["rgb"])
    while rgb.ndim > 3 and rgb.shape[0] == 1:
        rgb = rgb[0]
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError("trajectory RGB frame must have shape [H, W, 3]")
    rgb = rgb.detach().cpu()
    if rgb.is_floating_point() and float(rgb.max()) <= 1.0:
        rgb = rgb * 255.0
    return rgb.clamp(0, 255).to(torch.uint8).numpy()


def _write_contact_sheet(
    frames: list[tuple[str, np.ndarray]], path: Path, *, columns: int = 4
) -> None:
    if not frames:
        raise ValueError("contact sheet needs at least one trajectory frame")
    frame_height, frame_width = frames[0][1].shape[:2]
    label_height = 28
    rows = (len(frames) + columns - 1) // columns
    sheet = Image.new(
        "RGB",
        (columns * frame_width, rows * (frame_height + label_height)),
        color=(20, 20, 20),
    )
    draw = ImageDraw.Draw(sheet)
    for index, (label, frame) in enumerate(frames):
        if frame.shape != (frame_height, frame_width, 3):
            raise ValueError("all trajectory frames must share one RGB shape")
        column = index % columns
        row = index // columns
        x = column * frame_width
        y = row * (frame_height + label_height)
        sheet.paste(Image.fromarray(frame), (x, y + label_height))
        draw.text((x + 3, y + 4), label[:32], fill=(255, 255, 255))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def _primary_object_behavior(env: Any, adapter: TaskAdapter) -> dict[str, Any]:
    obj = adapter.primary_object(env)
    position = adapter.primary_object_position(env)
    velocity = getattr(obj, "linear_velocity", None)
    speed = None
    if velocity is not None:
        velocity_tensor = torch.as_tensor(velocity).detach().cpu().float()
        while velocity_tensor.ndim > 1 and velocity_tensor.shape[0] == 1:
            velocity_tensor = velocity_tensor[0]
        if velocity_tensor.shape == (3,):
            speed = float(torch.linalg.vector_norm(velocity_tensor))
    grasped = None
    is_grasped = getattr(adapter, "is_grasped", None)
    if callable(is_grasped):
        grasped = bool(is_grasped(env))
    return {
        "position_world_m": position.tolist(),
        "speed_m_per_s": speed,
        "is_grasped": grasped,
    }


def _scene_from_observation(
    observation: dict[str, Any],
    env: Any,
    adapter: TaskAdapter,
    *,
    num_points: int,
    max_depth_m: float,
    quotas: dict[int, int],
    object_query_limit: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    entries = _build_body_registry(env, adapter)
    body_ids = torch.tensor([entry.segmentation_id for entry in entries])
    camera_name = _select_camera_name(observation, env, adapter)
    sensor = observation["sensor_data"][camera_name]
    calibration = observation["sensor_param"][camera_name]
    points_cv, valid = backproject_depth_cv(
        sensor["depth"], calibration["intrinsic_cv"], max_depth_m=max_depth_m
    )
    points_world = camera_cv_to_world(points_cv, calibration["extrinsic_cv"])
    segmentation = sensor["segmentation"].squeeze(-1)
    usable = valid & torch.isin(segmentation, body_ids)
    flat_world = points_world[usable]
    flat_rgb = sensor["rgb"][usable].float() / 255.0
    flat_categories = _category_tensor(segmentation[usable].long(), entries)
    selected = stratified_point_indices(
        flat_world, flat_categories, num_points, quotas
    )
    scene_world = flat_world[selected]
    scene_rgb = flat_rgb[selected]
    categories = flat_categories[selected]
    object_indices = torch.nonzero(
        categories == CATEGORY_OBJECT, as_tuple=False
    ).squeeze(-1)
    if len(object_indices) == 0:
        raise RuntimeError("current observation contains no sampled cube points")
    if len(object_indices) > object_query_limit:
        object_indices = object_indices[:object_query_limit]
    goal_world = adapter.goal_world_m(env)
    return scene_world.cpu(), scene_rgb.cpu(), object_indices.cpu(), goal_world


def _load_models(
    raw: dict[str, Any], device: torch.device
) -> dict[str, MicroQ4D | NoActionTrajectoryModel]:
    paths = raw["paths"]
    model_config = raw["model"]
    arguments = {
        "action_dimensions": int(model_config["action_dimensions"]),
        "horizon": int(model_config["horizon"]),
        "width": int(model_config["width"]),
    }
    q4d = MicroQ4D(**arguments).to(device).eval()
    dense = DensePointFutureModel(**arguments).to(device).eval()
    no_action = NoActionTrajectoryModel(
        horizon=arguments["horizon"], width=arguments["width"]
    ).to(device).eval()
    q4d.load_state_dict(
        torch.load(paths["micro_q4d_checkpoint"], map_location=device, weights_only=True)
    )
    dense.load_state_dict(
        torch.load(paths["dense_checkpoint"], map_location=device, weights_only=True)
    )
    no_action.load_state_dict(
        torch.load(
            paths["no_action_checkpoint"], map_location=device, weights_only=True
        )
    )
    return {"q4d": q4d, "dense": dense, "no_action": no_action}


@torch.no_grad()
def _warmup_models(
    models: dict[str, MicroQ4D | NoActionTrajectoryModel],
    normalization: NormalizationStats,
    raw: dict[str, Any],
    device: torch.device,
) -> None:
    """Initialize the complete CPU-observation-to-cost CUDA path before timing."""
    model_config = raw["model"]
    scene_points = int(model_config["scene_points"])
    scene = torch.zeros(scene_points, 3)
    rgb = torch.zeros_like(scene)
    goal = torch.zeros(3)
    planning = raw["planning"]
    planner_config = _planner_config(raw)
    actions = sample_random_action_sequences(
        planner_config,
        device=device,
        generator=torch.Generator(device=device).manual_seed(0),
    )
    for name, model in models.items():
        if name == "dense":
            query_counts = [scene_points]
        else:
            query_limit = min(
                int(model_config["object_query_limit"]), scene_points
            )
            # Visible object counts vary with viewpoint. Warm representative decoder
            # shapes so their one-time CUDA setup is never charged to an MPC cycle.
            query_counts = sorted(
                {min(count, query_limit) for count in (16, 20, 24, 29, 32, 48)}
            )
        for query_count in query_counts:
            query_indices = torch.arange(query_count)
            cost = CachedCubeCost(
                model,
                normalization,
                dense_output=name == "dense",
                **_cost_kwargs(planning),
            )
            cost.prepare(scene, rgb, query_indices, goal)
            cost(actions)
    torch.cuda.synchronize(device)


def _condition_key(model: str, method: str, budget_ms: float, seed: int) -> str:
    return f"{model}:{method}:{budget_ms:g}:{seed}"


@torch.no_grad()
def _warmup_observed_scene(
    env: Any,
    adapter: TaskAdapter,
    models: dict[str, MicroQ4D | NoActionTrajectoryModel],
    normalization: NormalizationStats,
    raw: dict[str, Any],
    device: torch.device,
) -> None:
    """Warm the same renderer-to-model data path used by the first real cycle."""
    simulation = raw["simulation"]
    model_config = raw["model"]
    planning = raw["planning"]
    env.reset(seed=int(planning["seed"]))
    observation, _ = adapter.prepare(env, int(simulation["approach_max_steps"]))
    if observation is None:
        raise RuntimeError("warm-up approach did not return an observation")
    scene_world, scene_rgb, object_indices, goal_world = _scene_from_observation(
        observation,
        env,
        adapter,
        num_points=int(model_config["scene_points"]),
        max_depth_m=float(simulation["max_depth_m"]),
        quotas={
            CATEGORY_OBJECT: int(simulation["object_quota"]),
            CATEGORY_ROBOT: int(simulation["robot_quota"]),
            CATEGORY_GOAL: int(simulation["goal_quota"]),
        },
        object_query_limit=int(model_config["object_query_limit"]),
    )
    planner_config = _planner_config(raw)
    actions = sample_random_action_sequences(
        planner_config,
        device=device,
        generator=torch.Generator(device=device).manual_seed(1),
    )
    for name, model in models.items():
        cost = CachedCubeCost(
            model,
            normalization,
            dense_output=name == "dense",
            **_cost_kwargs(planning),
        )
        cost.prepare(scene_world, scene_rgb, object_indices, goal_world)
        cost(actions)
    torch.cuda.synchronize(device)


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _episode(
    env: Any,
    model: MicroQ4D | NoActionTrajectoryModel,
    normalization: NormalizationStats,
    raw: dict[str, Any],
    *,
    model_name: str,
    method: str,
    budget_ms: float,
    episode_seed: int,
    device: torch.device,
    adapter: TaskAdapter,
    visualization_path: Path | None = None,
) -> dict[str, Any]:
    model_config = raw["model"]
    simulation = raw["simulation"]
    planning = raw["planning"]
    env.reset(seed=episode_seed)
    observation, approach_steps = adapter.prepare(
        env, int(simulation["approach_max_steps"])
    )
    if observation is None:
        raise RuntimeError("approach controller did not return an observation")
    planner_config = _planner_config(raw)
    cost = CachedCubeCost(
        model,
        normalization,
        dense_output=model_name == "dense",
        **_cost_kwargs(planning),
    )
    planner = random_shooting if method == "random_shooting" else cem
    cycles = []
    trajectory_frames: list[tuple[str, np.ndarray]] = []
    if visualization_path is not None:
        trajectory_frames.append(("initial", _rgb_frame(observation, env, adapter)))
    success = False
    termination_reason = "control_cycle_limit"
    initial_distance = adapter.task_distance_m(env)
    initial_behavior = _primary_object_behavior(env, adapter)
    initial_position = torch.tensor(initial_behavior["position_world_m"])
    for cycle in range(int(simulation["control_cycles"])):
        try:
            scene_world, scene_rgb, object_indices, goal_world = (
                _scene_from_observation(
                    observation,
                    env,
                    adapter,
                    num_points=int(model_config["scene_points"]),
                    max_depth_m=float(simulation["max_depth_m"]),
                    quotas={
                        CATEGORY_OBJECT: int(simulation["object_quota"]),
                        CATEGORY_ROBOT: int(simulation["robot_quota"]),
                        CATEGORY_GOAL: int(simulation["goal_quota"]),
                    },
                    object_query_limit=int(model_config["object_query_limit"]),
                )
            )
        except RuntimeError as error:
            if str(error) != "current observation contains no sampled cube points":
                raise
            termination_reason = "object_not_visible"
            break
        torch.cuda.synchronize(device)
        planning_start = time.perf_counter()
        previous_encodes = cost.scene_encode_count
        cost.prepare(scene_world, scene_rgb, object_indices, goal_world)
        torch.cuda.synchronize(device)
        result = planner(
            cost,
            planner_config,
            budget_ms=budget_ms,
            device=device,
            seed=episode_seed * 100 + cycle,
            started_at_s=planning_start,
        )
        if cost.scene_encode_count != previous_encodes + 1:
            raise RuntimeError("scene must be encoded exactly once per control cycle")
        action = result.first_action.detach().cpu().numpy().astype(np.float32)
        observation, _, _, _, info = env.step(action)
        success = bool(torch.as_tensor(info.get("success", False)).any())
        task_distance = adapter.task_distance_m(env)
        behavior = _primary_object_behavior(env, adapter)
        object_displacement = float(
            torch.linalg.vector_norm(
                torch.tensor(behavior["position_world_m"]) - initial_position
            )
        )
        cycles.append(
            {
                "cycle": cycle,
                "planning_ms": result.elapsed_ms,
                "budget_overrun_ms": result.budget_overrun_ms,
                "candidates_evaluated": result.candidates_evaluated,
                "batches_evaluated": result.batches_evaluated,
                "predicted_cost": result.predicted_cost,
                "selected_gripper_schedule": result.gripper_schedule,
                "executed_gripper_command": float(action[-1]),
                "executed_first_action": action.tolist(),
                "primary_object_position_world_m": behavior["position_world_m"],
                "primary_object_speed_m_per_s": behavior["speed_m_per_s"],
                "primary_object_is_grasped": behavior["is_grasped"],
                "primary_object_displacement_m": object_displacement,
                "cube_goal_distance_m": task_distance,
                "task_distance_m": task_distance,
                "object_query_points": len(object_indices),
                "scene_encodes": 1,
                "success": success,
            }
        )
        if visualization_path is not None:
            label = (
                f"c{cycle} {result.gripper_schedule[:14]} "
                f"g={float(action[-1]):+.0f} d={task_distance:.3f}"
            )
            trajectory_frames.append((label, _rgb_frame(observation, env, adapter)))
        if success:
            termination_reason = "success"
            break
    final_distance = adapter.task_distance_m(env)
    final_behavior = _primary_object_behavior(env, adapter)
    if visualization_path is not None:
        _write_contact_sheet(trajectory_frames, visualization_path)
    return {
        "task_adapter": adapter.name,
        "model": model_name,
        "method": method,
        "budget_ms": budget_ms,
        "seed": episode_seed,
        "approach_steps": approach_steps,
        "success": success,
        "termination_reason": termination_reason,
        "control_cycles": len(cycles),
        "initial_cube_goal_distance_m": initial_distance,
        "final_cube_goal_distance_m": final_distance,
        "initial_task_distance_m": initial_distance,
        "final_task_distance_m": final_distance,
        "distance_improvement_m": initial_distance - final_distance,
        "initial_object_behavior": initial_behavior,
        "final_object_behavior": final_behavior,
        "closed_command_executed": any(
            cycle["executed_gripper_command"] < 0 for cycle in cycles
        ),
        "release_command_executed": any(
            cycle["executed_gripper_command"] > 0 for cycle in cycles
        ),
        "trajectory_visualization": (
            str(visualization_path) if visualization_path is not None else None
        ),
        "cycles": cycles,
    }


def _summarize(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    conditions = sorted(
        {(record["model"], record["method"], record["budget_ms"]) for record in records}
    )
    summary = []
    for model, method, budget in conditions:
        selected = [
            record
            for record in records
            if (record["model"], record["method"], record["budget_ms"])
            == (model, method, budget)
        ]
        cycles = [cycle for record in selected for cycle in record["cycles"]]
        planning_ms = [float(cycle["planning_ms"]) for cycle in cycles]
        budget_overruns_ms = [
            float(cycle["budget_overrun_ms"]) for cycle in cycles
        ]
        candidates = [int(cycle["candidates_evaluated"]) for cycle in cycles]
        total_planning_seconds = sum(planning_ms) / 1000.0
        if not cycles or total_planning_seconds <= 0:
            raise RuntimeError("each MPC condition must contain measured planning cycles")
        summary.append(
            {
                "model": model,
                "method": method,
                "budget_ms": budget,
                "episodes": len(selected),
                "successes": sum(record["success"] for record in selected),
                "success_rate": float(np.mean([record["success"] for record in selected])),
                "mean_final_cube_goal_distance_m": float(
                    np.mean([record["final_cube_goal_distance_m"] for record in selected])
                ),
                "mean_final_task_distance_m": float(
                    np.mean([record["final_task_distance_m"] for record in selected])
                ),
                "mean_control_cycles": float(
                    np.mean([record["control_cycles"] for record in selected])
                ),
                "mean_candidates_per_cycle": float(
                    np.mean(candidates)
                ),
                "candidate_throughput_per_second": float(
                    sum(candidates) / total_planning_seconds
                ),
                "mean_planning_ms": float(np.mean(planning_ms)),
                "p50_planning_ms": float(np.quantile(planning_ms, 0.50)),
                "p95_planning_ms": float(np.quantile(planning_ms, 0.95)),
                "budget_overrun_cycles": sum(value > 0 for value in budget_overruns_ms),
                "budget_overrun_rate": float(
                    np.mean([value > 0 for value in budget_overruns_ms])
                ),
                "p50_budget_overrun_ms": float(
                    np.quantile(budget_overruns_ms, 0.50)
                ),
                "p95_budget_overrun_ms": float(
                    np.quantile(budget_overruns_ms, 0.95)
                ),
                "maximum_budget_overrun_ms": max(budget_overruns_ms),
                "object_visibility_failures": sum(
                    record["termination_reason"] == "object_not_visible"
                    for record in selected
                ),
                "object_visibility_failure_rate": float(
                    np.mean(
                        [
                            record["termination_reason"] == "object_not_visible"
                            for record in selected
                        ]
                    )
                ),
                "selected_gripper_schedule_counts": {
                    name: sum(
                        cycle.get(
                            "selected_gripper_schedule",
                            "unrecorded_legacy_schedule",
                        )
                        == name
                        for cycle in cycles
                    )
                    for name in sorted(
                        {
                            cycle.get(
                                "selected_gripper_schedule",
                                "unrecorded_legacy_schedule",
                            )
                            for cycle in cycles
                        }
                    )
                },
            }
        )
    return summary


def _matched_comparisons(summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compare Q4D and dense under exactly the same planner and time budget."""
    indexed = {
        (condition["model"], condition["method"], condition["budget_ms"]): condition
        for condition in summary
    }
    comparisons = []
    pairs = sorted({(item["method"], item["budget_ms"]) for item in summary})
    for method, budget in pairs:
        q4d = indexed.get(("q4d", method, budget))
        dense = indexed.get(("dense", method, budget))
        if q4d is None or dense is None:
            continue
        comparisons.append(
            {
                "method": method,
                "budget_ms": budget,
                "q4d_minus_dense_success_rate": (
                    q4d["success_rate"] - dense["success_rate"]
                ),
                "q4d_minus_dense_final_distance_m": (
                    q4d["mean_final_cube_goal_distance_m"]
                    - dense["mean_final_cube_goal_distance_m"]
                ),
                "q4d_over_dense_candidate_throughput": (
                    q4d["candidate_throughput_per_second"]
                    / dense["candidate_throughput_per_second"]
                ),
                "q4d_minus_dense_p50_planning_ms": (
                    q4d["p50_planning_ms"] - dense["p50_planning_ms"]
                ),
                "q4d_minus_dense_p95_planning_ms": (
                    q4d["p95_planning_ms"] - dense["p95_planning_ms"]
                ),
            }
        )
    return comparisons


def _valid_executed_actions(records: list[dict[str, Any]]) -> bool:
    for record in records:
        for cycle in record["cycles"]:
            action = np.asarray(cycle["executed_first_action"], dtype=np.float64)
            if (
                action.shape != (7,)
                or not np.isfinite(action).all()
                or np.any(action < -1.0)
                or np.any(action > 1.0)
            ):
                return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/mpc.toml"))
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--methods", nargs="+")
    parser.add_argument("--budgets-ms", type=float, nargs="+")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    raw = _read_toml(args.config)
    planning = raw["planning"]
    planner_config = _planner_config(raw)
    models_requested = args.models or list(planning["models"])
    methods = args.methods or list(planning["methods"])
    budgets = args.budgets_ms or [float(value) for value in planning["budgets_ms"]]
    episodes = args.episodes or int(planning["episodes"])
    if not set(models_requested) <= {"q4d", "dense", "no_action"}:
        raise ValueError("models must contain q4d, dense, and/or no_action")
    if not set(methods) <= {"random_shooting", "cem"}:
        raise ValueError("methods must contain only random_shooting and/or cem")
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("MPC evaluation requires CUDA")
    device = torch.device("cuda:0")
    output = args.output or Path(raw["paths"]["output"])
    save_visualizations = bool(
        raw.get("visualization", {}).get("save_episode_contact_sheets", False)
    )
    progress_path = output.with_name("episodes.json")
    progress = (
        json.loads(progress_path.read_text(encoding="utf-8"))
        if progress_path.exists()
        else {}
    )
    normalization = NormalizationStats.load(raw["paths"]["normalization"])
    models = _load_models(raw, device)
    software_icd = Path("/usr/share/vulkan/icd.d/lvp_icd.json")
    if software_icd.exists():
        os.environ.setdefault("VK_ICD_FILENAMES", str(software_icd))
    import mani_skill.envs  # noqa: F401

    adapter = get_task_adapter(raw["simulation"]["env_id"])
    env = gym.make(
        raw["simulation"]["env_id"],
        num_envs=1,
        obs_mode="rgb+depth+segmentation",
        control_mode=raw["simulation"]["control_mode"],
        sim_backend="physx_cpu",
        render_backend="sapien_cpu",
        render_mode="sensors",
    )
    # Simulator initialization can lazily initialize GPU-adjacent libraries, so
    # warm model inference only after the environment exists.
    _warmup_models(models, normalization, raw, device)
    _warmup_observed_scene(env, adapter, models, normalization, raw, device)
    try:
        for model_name in models_requested:
            for method in methods:
                for budget in budgets:
                    for episode_index in range(episodes):
                        episode_seed = int(planning["seed"]) + episode_index
                        key = _condition_key(
                            model_name, method, budget, episode_seed
                        )
                        if key in progress:
                            print(f"skip completed={key}", flush=True)
                            continue
                        record = _episode(
                            env,
                            models[model_name],
                            normalization,
                            raw,
                            model_name=model_name,
                            method=method,
                            budget_ms=budget,
                            episode_seed=episode_seed,
                            device=device,
                            adapter=adapter,
                            visualization_path=(
                                output.parent
                                / "visualizations"
                                / (
                                    f"{model_name}_{method}_{budget:g}ms_"
                                    f"seed{episode_seed}.png"
                                )
                                if save_visualizations
                                else None
                            ),
                        )
                        progress[key] = record
                        _write_json_atomic(progress_path, progress)
                        print(
                            f"condition={key} success={record['success']} "
                            f"cycles={record['control_cycles']} "
                            f"final_distance={record['final_cube_goal_distance_m']:.4f}m",
                            flush=True,
                        )
    finally:
        env.close()
    records = list(progress.values())
    summary = _summarize(records)
    matched_comparisons = _matched_comparisons(summary)
    expected = len(models_requested) * len(methods) * len(budgets) * episodes
    report_seeds = [int(planning["seed"]) + index for index in range(episodes)]
    report = {
        "protocol": {
            "environment": raw["simulation"]["env_id"],
            "task_adapter": adapter.name,
            "models": models_requested,
            "methods": methods,
            "budgets_ms": budgets,
            "episodes_per_condition": episodes,
            "seeds": report_seeds,
            "action_space": planner_config.action_space,
            "candidate_gripper_schedules": (
                list(planner_config.gripper_schedules)
                if planner_config.action_space == "gripper_schedules"
                else ["translation_only_hold_closed"]
            ),
            "settling_penalty": float(planning.get("settling_penalty", 0.0)),
            "settling_steps": int(planning.get("settling_steps", 2)),
            "episode_contact_sheets": save_visualizations,
            "neutralized_constant_action_channels": list(
                normalization.constant_action_channels
            ),
            "cost": (
                "final visible primary-object centroid distance to task goal plus "
                "late-horizon centroid-motion settling penalty plus action penalty"
                if float(planning.get("settling_penalty", 0.0)) > 0
                else "final visible primary-object centroid distance to task goal plus "
                "action penalty"
            ),
            "predictor_inputs": (
                "normalized scene XYZ/RGB; action-conditioned models additionally "
                "receive executable action sequences"
            ),
            "privileged_cost_only": (
                "task-adapter semantics and segmentation select identical primary-object "
                "points for both models"
            ),
            "planning_label": "oracle-object-query planning",
            "oracle_object_query_disclosure": (
                "Privileged simulator segmentation selects primary-object query points "
                "for planning cost evaluation; this is not deployable perception."
            ),
            "receding_horizon": "execute first action, reobserve, and replan",
        },
        "conditions": summary,
        "matched_comparisons": matched_comparisons,
        "episodes": records,
        "checks": {
            "all_conditions_completed": len(records) == expected,
            "scene_encoded_once_per_cycle": all(
                cycle["scene_encodes"] == 1
                for record in records
                for cycle in record["cycles"]
            ),
            "only_first_action_executed": all(
                len(record["cycles"]) == record["control_cycles"] for record in records
            ),
            "valid_executable_7d_actions": _valid_executed_actions(records),
            "shared_gripper_schedule_library": bool(
                planner_config.action_space == "translation_only"
                or set(planner_config.gripper_schedules)
                == set(DEFAULT_GRIPPER_SCHEDULES)
            ),
            "metrics_are_finite": all(
                np.isfinite(condition[name])
                for condition in summary
                for name in (
                    "mean_final_task_distance_m",
                    "p50_planning_ms",
                    "p95_planning_ms",
                    "candidate_throughput_per_second",
                    "p95_budget_overrun_ms",
                )
            ),
            "matched_episode_seeds": all(
                {
                    record["seed"]
                    for record in records
                    if record["model"] == model
                    and record["method"] == method
                    and record["budget_ms"] == budget
                }
                == set(report_seed for report_seed in report_seeds)
                for model in models_requested
                for method in methods
                for budget in budgets
            ),
            "object_visibility_failures_accounted": sum(
                condition["object_visibility_failures"] for condition in summary
            )
            == sum(
                record["termination_reason"] == "object_not_visible"
                for record in records
            ),
            "checkpoints_exist": all(
                Path(raw["paths"][name]).exists()
                for name in (
                    "micro_q4d_checkpoint",
                    "dense_checkpoint",
                    "no_action_checkpoint",
                )
            ),
        },
    }
    report["passed"] = all(report["checks"].values())
    _write_json_atomic(output, report)
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise RuntimeError("MPC evaluation failed implementation checks")


if __name__ == "__main__":
    main()
