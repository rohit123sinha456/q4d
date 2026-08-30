"""ManiSkill task adapters for collection, semantic labeling, and planning.

The adapters deliberately depend only on the small public surface shared by the
supported ManiSkill tabletop environments. Privileged simulator entities are used for
label construction and audit metadata, never as model inputs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch import Tensor

from q4d_wam.labels import CATEGORY_GOAL, CATEGORY_OBJECT

COUNTERFACTUAL_BRANCHES = ("success", "perturbed", "no_op", "failure")
SCALED_BRANCHES = ("success", "weak", "off_target", "failure", "no_op")
ALL_BRANCHES = frozenset(COUNTERFACTUAL_BRANCHES) | frozenset(SCALED_BRANCHES)


@dataclass(frozen=True)
class SemanticEntity:
    """A simulator entity and its privileged audit category."""

    entity: Any
    category: int
    role: str


@dataclass(frozen=True)
class BranchPlan:
    """Absolute TCP waypoints and gripper commands for one action branch."""

    targets_world_m: tuple[Tensor | None, ...]
    gripper_commands: tuple[float, ...]
    position_scales: tuple[float, ...]
    stop_after_success: bool = False
    metadata: dict[str, float | int | str | bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        length = len(self.targets_world_m)
        if length == 0:
            raise ValueError("branch plan cannot be empty")
        if len(self.gripper_commands) != length or len(self.position_scales) != length:
            raise ValueError("branch plan fields must share one horizon")

    @property
    def horizon(self) -> int:
        return len(self.targets_world_m)


def _position(entity: Any) -> Tensor:
    return entity.pose.p[0].detach().cpu().float().clone()


def make_delta_pose_action(
    env: Any,
    target_world_m: Tensor | None,
    *,
    gripper: float,
    position_scale: float = 1.0,
) -> np.ndarray:
    """Create the normalized Panda delta-pose action used by the MVP."""

    action = np.zeros(env.action_space.shape, dtype=np.float32)
    if target_world_m is not None:
        tcp_world = env.unwrapped.agent.tcp.pose.p[0].detach().cpu()
        delta_world = target_world_m.detach().cpu() - tcp_world
        action[..., :3] = np.clip(delta_world.numpy() / 0.1, -1.0, 1.0)
        action[..., :3] *= float(position_scale)
    action[..., -1] = float(gripper)
    return action


def move_tcp_to(
    env: Any,
    target_world_m: Tensor,
    max_steps: int,
    *,
    gripper: float,
) -> tuple[Any, int]:
    """Move toward an absolute TCP waypoint using the configured delta controller."""

    observation = None
    for step in range(max_steps):
        action = make_delta_pose_action(env, target_world_m, gripper=gripper)
        observation, _, _, _, _ = env.step(action)
        tcp_world = env.unwrapped.agent.tcp.pose.p[0].detach().cpu()
        if torch.linalg.vector_norm(tcp_world - target_world_m.detach().cpu()) < 0.006:
            return observation, step + 1
    return observation, max_steps


class TaskAdapter(ABC):
    """Environment-specific semantics needed by the generic Q4D pipeline."""

    env_id: str
    name: str
    counterfactual_branches = COUNTERFACTUAL_BRANCHES
    scaled_branches = SCALED_BRANCHES

    @abstractmethod
    def semantic_entities(self, env: Any) -> tuple[SemanticEntity, ...]:
        """Return privileged object/goal entities used only for labels and audits."""

    @abstractmethod
    def primary_object(self, env: Any) -> Any:
        """Return the entity whose visible points are scored by the planner."""

    @abstractmethod
    def goal_world_m(self, env: Any) -> Tensor:
        """Return the task-space target for the primary-object centroid."""

    @abstractmethod
    def prepare(self, env: Any, max_steps: int) -> tuple[Any, int]:
        """Move to the shared branch point and return its latest observation."""

    @abstractmethod
    def make_branch_plan(
        self,
        env: Any,
        branch: str,
        horizon: int,
        *,
        seed: int,
        state_index: int,
    ) -> BranchPlan:
        """Create one deterministic executable counterfactual action plan."""

    def tracked_entities(self, env: Any) -> tuple[Any, ...]:
        """Return dynamic task entities whose centers are useful audit metadata."""

        return (self.primary_object(env),)

    def action(
        self,
        env: Any,
        plan: BranchPlan,
        time_index: int,
        *,
        success_reached: bool,
    ) -> np.ndarray:
        if time_index < 0 or time_index >= plan.horizon:
            raise IndexError("time index is outside the branch plan")
        target = plan.targets_world_m[time_index]
        if plan.stop_after_success and success_reached:
            target = None
        return make_delta_pose_action(
            env,
            target,
            gripper=plan.gripper_commands[time_index],
            position_scale=plan.position_scales[time_index],
        )

    def task_distance_m(self, env: Any) -> float:
        return float(
            torch.linalg.vector_norm(
                _position(self.primary_object(env)) - self.goal_world_m(env)
            )
        )

    def primary_object_position(self, env: Any) -> Tensor:
        return _position(self.primary_object(env))

    def branch_names(self, profile: str) -> tuple[str, ...]:
        if profile == "counterfactual":
            return self.counterfactual_branches
        if profile == "scaled":
            return self.scaled_branches
        raise ValueError(f"unknown collection profile: {profile}")

    def validate_branch(self, branch: str) -> None:
        if branch not in ALL_BRANCHES:
            raise ValueError(f"unknown counterfactual branch: {branch}")


class PushCubeAdapter(TaskAdapter):
    """Exact adapter for the original PushCube MVP behavior."""

    env_id = "PushCube-v1"
    name = "push_cube"

    def semantic_entities(self, env: Any) -> tuple[SemanticEntity, ...]:
        unwrapped = env.unwrapped
        return (
            SemanticEntity(unwrapped.obj, CATEGORY_OBJECT, "primary_object"),
            SemanticEntity(unwrapped.goal_region, CATEGORY_GOAL, "goal"),
        )

    def primary_object(self, env: Any) -> Any:
        return env.unwrapped.obj

    def goal_world_m(self, env: Any) -> Tensor:
        return _position(env.unwrapped.goal_region)

    def prepare(self, env: Any, max_steps: int) -> tuple[Any, int]:
        cube = _position(env.unwrapped.obj)
        above = cube + torch.tensor([-0.05, 0.0, 0.08])
        observation, first_steps = move_tcp_to(
            env, above, max_steps // 2, gripper=-1.0
        )
        behind = cube + torch.tensor([-0.05, 0.0, 0.0])
        observation, second_steps = move_tcp_to(
            env, behind, max_steps - first_steps, gripper=-1.0
        )
        return observation, first_steps + second_steps

    def make_branch_plan(
        self,
        env: Any,
        branch: str,
        horizon: int,
        *,
        seed: int,
        state_index: int,
    ) -> BranchPlan:
        self.validate_branch(branch)
        goal = self.goal_world_m(env)
        cube_start = self.primary_object_position(env)
        rng = np.random.default_rng(seed + 10_000)
        perturb_sign = -1.0 if state_index % 2 else 1.0
        lateral_offset = perturb_sign * float(rng.uniform(0.03, 0.06))
        success_lateral_jitter = 0.0
        success_standoff = 0.12
        weak_scale = float(rng.uniform(0.30, 0.50))
        weak_steps = int(rng.integers(3, max(4, horizon // 2 + 1)))
        off_target_offset = perturb_sign * float(rng.uniform(0.18, 0.25))
        failure_offset = float(rng.uniform(-0.04, 0.04))
        targets = {
            "success": goal
            + torch.tensor([-success_standoff, success_lateral_jitter, 0.0]),
            "perturbed": goal + torch.tensor([-0.12, lateral_offset, 0.0]),
            "weak": goal
            + torch.tensor([-success_standoff, success_lateral_jitter, 0.0]),
            "off_target": cube_start
            + torch.tensor([float(rng.uniform(0.05, 0.08)), off_target_offset, 0.0]),
            "failure": cube_start + torch.tensor([-0.16, failure_offset, 0.0]),
        }
        waypoints: list[Tensor | None] = []
        scales = []
        for time_index in range(horizon):
            inactive = branch == "no_op" or (
                branch == "weak" and time_index >= weak_steps
            )
            waypoints.append(None if inactive else targets[branch])
            scales.append(weak_scale if branch == "weak" and not inactive else 1.0)
        return BranchPlan(
            tuple(waypoints),
            tuple([-1.0] * horizon),
            tuple(scales),
            stop_after_success=branch == "success",
            metadata={
                "lateral_perturbation_m": lateral_offset
                if branch == "perturbed"
                else 0.0,
                "success_lateral_jitter_m": success_lateral_jitter,
                "success_standoff_m": success_standoff,
                "weak_action_scale": weak_scale if branch == "weak" else 1.0,
                "weak_active_steps": weak_steps if branch == "weak" else 0,
                "off_target_lateral_offset_m": off_target_offset
                if branch == "off_target"
                else 0.0,
            },
        )


class PullCubeAdapter(PushCubeAdapter):
    """Planar pulling counterpart to the PushCube adapter."""

    env_id = "PullCube-v1"
    name = "pull_cube"

    def prepare(self, env: Any, max_steps: int) -> tuple[Any, int]:
        cube = self.primary_object_position(env)
        above = cube + torch.tensor([0.05, 0.0, 0.08])
        observation, first_steps = move_tcp_to(
            env, above, max_steps // 2, gripper=-1.0
        )
        in_front = cube + torch.tensor([0.05, 0.0, 0.0])
        observation, second_steps = move_tcp_to(
            env, in_front, max_steps - first_steps, gripper=-1.0
        )
        return observation, first_steps + second_steps

    def make_branch_plan(
        self,
        env: Any,
        branch: str,
        horizon: int,
        *,
        seed: int,
        state_index: int,
    ) -> BranchPlan:
        self.validate_branch(branch)
        goal = self.goal_world_m(env)
        cube_start = self.primary_object_position(env)
        rng = np.random.default_rng(seed + 10_000)
        sign = -1.0 if state_index % 2 else 1.0
        lateral_offset = sign * float(rng.uniform(0.03, 0.06))
        weak_scale = float(rng.uniform(0.30, 0.50))
        weak_steps = int(rng.integers(3, max(4, horizon // 2 + 1)))
        off_target_offset = sign * float(rng.uniform(0.18, 0.25))
        failure_offset = float(rng.uniform(-0.04, 0.04))
        # The goal region is 0.20 m behind the initial cube center and has a
        # 0.10 m radius. A 0.12 m standoff leaves the cube just outside the
        # success boundary after eight controller steps on the CPU backend.
        standoff = 0.09
        targets = {
            "success": goal + torch.tensor([standoff, 0.0, 0.0]),
            "perturbed": goal + torch.tensor([standoff, lateral_offset, 0.0]),
            "weak": goal + torch.tensor([standoff, 0.0, 0.0]),
            "off_target": cube_start
            + torch.tensor([-float(rng.uniform(0.05, 0.08)), off_target_offset, 0.0]),
            "failure": cube_start + torch.tensor([0.16, failure_offset, 0.0]),
        }
        waypoints: list[Tensor | None] = []
        scales = []
        for time_index in range(horizon):
            inactive = branch == "no_op" or (
                branch == "weak" and time_index >= weak_steps
            )
            waypoints.append(None if inactive else targets[branch])
            scales.append(weak_scale if branch == "weak" and not inactive else 1.0)
        return BranchPlan(
            tuple(waypoints),
            tuple([-1.0] * horizon),
            tuple(scales),
            stop_after_success=branch == "success",
            metadata={
                "lateral_perturbation_m": lateral_offset
                if branch == "perturbed"
                else 0.0,
                "success_standoff_m": standoff,
                "weak_action_scale": weak_scale if branch == "weak" else 1.0,
                "weak_active_steps": weak_steps if branch == "weak" else 0,
                "off_target_lateral_offset_m": off_target_offset
                if branch == "off_target"
                else 0.0,
            },
        )


class GraspAndPlaceAdapter(TaskAdapter):
    """Shared preparation and branches for single-object grasp-and-place tasks."""

    grasp_clearance_m = 0.10
    release_at_destination = True
    destination_clearance_m = 0.10
    transit_step_adjustment = 0
    release_step_adjustment = 0
    hold_settle_steps = 0
    staging_clearance_m: float | None = None
    staged_closed_steps = 0
    staged_open_steps = 0
    staged_gripper_command = 1.0

    def adjust_destination(
        self, obj: Tensor, destination: Tensor, branch: str
    ) -> Tensor:
        """Adjust an object-center destination for task-specific release dynamics."""

        return destination

    def adjust_staging_destination(self, obj: Tensor, destination: Tensor) -> Tensor:
        """Adjust the common pre-branch staging point without biasing placement."""

        return destination

    def prepare(self, env: Any, max_steps: int) -> tuple[Any, int]:
        obj = self.primary_object_position(env)
        above = obj + torch.tensor([0.0, 0.0, self.grasp_clearance_m])
        first_budget = max(1, max_steps // (4 if self.staging_clearance_m else 2))
        observation, first_steps = move_tcp_to(
            env, above, first_budget, gripper=1.0
        )
        grasp = obj + torch.tensor([0.0, 0.0, 0.01])
        if self.staging_clearance_m is None:
            second_budget = max(1, max_steps - first_steps - 2)
        else:
            second_budget = max(1, max_steps // 4)
        observation, second_steps = move_tcp_to(
            env, grasp, second_budget, gripper=1.0
        )
        if self.staging_clearance_m is None:
            close_steps = max(1, max_steps - first_steps - second_steps)
        else:
            close_steps = max(1, max_steps // 8)
        for _ in range(close_steps):
            action = make_delta_pose_action(env, grasp, gripper=-1.0)
            observation, _, _, _, _ = env.step(action)
        used_steps = first_steps + second_steps + close_steps
        if self.staging_clearance_m is None:
            return observation, used_steps

        obj = self.primary_object_position(env)
        tcp = env.unwrapped.agent.tcp.pose.p[0].detach().cpu().float().clone()
        destination = self.adjust_staging_destination(obj, self.goal_world_m(env))
        staging_target = destination + (tcp - obj) + torch.tensor(
            [0.0, 0.0, self.staging_clearance_m]
        )
        staging_budget = max(1, max_steps - used_steps)
        observation, staging_steps = move_tcp_to(
            env, staging_target, staging_budget, gripper=-1.0
        )
        settle_steps = max(0, staging_budget - staging_steps)
        for _ in range(settle_steps):
            action = make_delta_pose_action(env, staging_target, gripper=-1.0)
            observation, _, _, _, _ = env.step(action)
        return observation, used_steps + staging_steps + settle_steps

    def make_branch_plan(
        self,
        env: Any,
        branch: str,
        horizon: int,
        *,
        seed: int,
        state_index: int,
    ) -> BranchPlan:
        self.validate_branch(branch)
        obj = self.primary_object_position(env)
        goal = self.goal_world_m(env)
        tcp = env.unwrapped.agent.tcp.pose.p[0].detach().cpu().float().clone()
        tcp_from_object = tcp - obj
        rng = np.random.default_rng(seed + 20_000)
        sign = -1.0 if state_index % 2 else 1.0
        small_lateral = sign * float(rng.uniform(0.03, 0.06))
        large_lateral = sign * float(rng.uniform(0.12, 0.18))
        weak_scale = float(rng.uniform(0.35, 0.55))
        lift = obj + tcp_from_object + torch.tensor([0.0, 0.0, 0.10])
        perturbed_goal = goal + torch.tensor([0.0, small_lateral, 0.0])
        off_target_goal = goal + torch.tensor([0.0, large_lateral, 0.02])
        failure_goal = obj + torch.tensor([-0.12, large_lateral, 0.02])

        transit_steps = max(1, horizon // 2 + self.transit_step_adjustment)
        release_steps = max(1, horizon // 4 + self.release_step_adjustment)
        waypoints: list[Tensor | None] = []
        grippers: list[float] = []
        scales: list[float] = []
        for time_index in range(horizon):
            if branch == "no_op":
                target = None
                gripper = -1.0
                scale = 1.0
            elif branch == "failure":
                target = failure_goal
                gripper = 1.0
                scale = 1.0
            elif branch == "weak":
                target = lift if time_index < max(1, horizon // 3) else None
                gripper = -1.0
                scale = weak_scale if target is not None else 1.0
            else:
                destination = {
                    "success": goal,
                    "perturbed": perturbed_goal,
                    "off_target": off_target_goal,
                }[branch]
                destination = self.adjust_destination(obj, destination, branch)
                placement_target = destination + tcp_from_object
                if self.staging_clearance_m is not None:
                    closed_steps = min(self.staged_closed_steps, horizon)
                    open_end = min(
                        closed_steps + self.staged_open_steps,
                        horizon,
                    )
                    if time_index < closed_steps:
                        target = placement_target
                        gripper = -1.0
                    elif time_index < open_end:
                        target = placement_target
                        gripper = self.staged_gripper_command
                    else:
                        target = None
                        gripper = self.staged_gripper_command
                elif not self.release_at_destination:
                    target = (
                        None
                        if time_index >= horizon - self.hold_settle_steps
                        else placement_target
                    )
                    gripper = -1.0
                else:
                    if time_index < transit_steps:
                        target = placement_target + torch.tensor(
                            [0.0, 0.0, self.destination_clearance_m]
                        )
                    else:
                        target = placement_target
                    gripper = 1.0 if time_index >= horizon - release_steps else -1.0
                scale = 1.0
            waypoints.append(target)
            grippers.append(gripper)
            scales.append(scale)
        return BranchPlan(
            tuple(waypoints),
            tuple(grippers),
            tuple(scales),
            stop_after_success=False,
            metadata={
                "lateral_perturbation_m": small_lateral
                if branch == "perturbed"
                else 0.0,
                "weak_action_scale": weak_scale if branch == "weak" else 1.0,
                "weak_active_steps": max(1, horizon // 3)
                if branch == "weak"
                else 0,
                "off_target_lateral_offset_m": large_lateral
                if branch == "off_target"
                else 0.0,
                "preparation_grasped": self.is_grasped(env),
                "preparation_staged": self.staging_clearance_m is not None,
            },
        )

    def is_grasped(self, env: Any) -> bool:
        value = env.unwrapped.agent.is_grasping(self.primary_object(env))
        return bool(torch.as_tensor(value).any())


class PickCubeAdapter(GraspAndPlaceAdapter):
    env_id = "PickCube-v1"
    name = "pick_cube"
    # PickCube succeeds while the cube remains grasped; it additionally requires
    # the arm to settle. Holding the goal is therefore preferable to releasing.
    release_at_destination = False
    hold_settle_steps = 1

    def semantic_entities(self, env: Any) -> tuple[SemanticEntity, ...]:
        unwrapped = env.unwrapped
        return (
            SemanticEntity(unwrapped.cube, CATEGORY_OBJECT, "primary_object"),
            SemanticEntity(unwrapped.goal_site, CATEGORY_GOAL, "goal"),
        )

    def primary_object(self, env: Any) -> Any:
        return env.unwrapped.cube

    def goal_world_m(self, env: Any) -> Tensor:
        return _position(env.unwrapped.goal_site)


class PlaceSphereAdapter(GraspAndPlaceAdapter):
    env_id = "PlaceSphere-v1"
    name = "place_sphere"
    staging_clearance_m = 0.06
    staged_closed_steps = 3
    staged_open_steps = 2
    staged_gripper_command = 0.75

    def semantic_entities(self, env: Any) -> tuple[SemanticEntity, ...]:
        unwrapped = env.unwrapped
        return (
            SemanticEntity(unwrapped.obj, CATEGORY_OBJECT, "primary_object"),
            SemanticEntity(unwrapped.bin, CATEGORY_GOAL, "goal_support"),
        )

    def primary_object(self, env: Any) -> Any:
        return env.unwrapped.obj

    def goal_world_m(self, env: Any) -> Tensor:
        unwrapped = env.unwrapped
        goal = _position(unwrapped.bin)
        goal[2] += float(unwrapped.block_half_size[0] + unwrapped.radius)
        return goal


class StackCubeAdapter(GraspAndPlaceAdapter):
    env_id = "StackCube-v1"
    name = "stack_cube"
    staging_clearance_m = 0.06
    staged_closed_steps = 2
    staged_open_steps = 2
    staged_gripper_command = 0.75

    def adjust_staging_destination(self, obj: Tensor, destination: Tensor) -> Tensor:
        approach_xy = destination[:2] - obj[:2]
        norm = torch.linalg.vector_norm(approach_xy)
        if norm <= 1e-6:
            return destination
        adjusted = destination.clone()
        # Stage slightly short to avoid striking cube B during transport. The
        # branch policy then places at the exact center instead of compounding
        # this offset from an almost-zero approach direction.
        adjusted[:2] -= 0.02 * approach_xy / norm
        return adjusted

    def semantic_entities(self, env: Any) -> tuple[SemanticEntity, ...]:
        unwrapped = env.unwrapped
        return (
            SemanticEntity(unwrapped.cubeA, CATEGORY_OBJECT, "primary_object"),
            SemanticEntity(unwrapped.cubeB, CATEGORY_GOAL, "goal_support"),
        )

    def primary_object(self, env: Any) -> Any:
        return env.unwrapped.cubeA

    def goal_world_m(self, env: Any) -> Tensor:
        unwrapped = env.unwrapped
        goal = _position(unwrapped.cubeB)
        half_size = torch.as_tensor(unwrapped.cube_half_size).flatten()
        goal[2] += float(half_size[-1] * 2)
        return goal

    def tracked_entities(self, env: Any) -> tuple[Any, ...]:
        return (env.unwrapped.cubeA, env.unwrapped.cubeB)


_ADAPTERS: dict[str, TaskAdapter] = {
    adapter.env_id: adapter
    for adapter in (
        PushCubeAdapter(),
        PullCubeAdapter(),
        PickCubeAdapter(),
        PlaceSphereAdapter(),
        StackCubeAdapter(),
    )
}


def supported_task_ids() -> tuple[str, ...]:
    return tuple(_ADAPTERS)


def get_task_adapter(env_id: str) -> TaskAdapter:
    try:
        return _ADAPTERS[env_id]
    except KeyError as error:
        supported = ", ".join(supported_task_ids())
        raise ValueError(
            f"no task adapter for {env_id!r}; supported tasks: {supported}"
        ) from error
