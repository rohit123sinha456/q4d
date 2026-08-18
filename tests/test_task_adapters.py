from types import SimpleNamespace

import numpy as np
import pytest
import torch

from q4d_wam.labels import CATEGORY_GOAL, CATEGORY_OBJECT
from q4d_wam.tasks import get_task_adapter, supported_task_ids
from q4d_wam.tasks.adapters import PushCubeAdapter


class _Actor:
    def __init__(self, name: str, position: list[float], segmentation_id: int):
        self.name = name
        self.pose = SimpleNamespace(p=torch.tensor([position], dtype=torch.float32))
        self.per_scene_id = torch.tensor([segmentation_id])


def _push_env() -> SimpleNamespace:
    cube = _Actor("cube", [0.0, 0.0, 0.02], 10)
    goal = _Actor("goal_region", [0.2, 0.0, 0.001], 11)
    tcp = SimpleNamespace(
        pose=SimpleNamespace(p=torch.tensor([[-0.05, 0.0, 0.02]]))
    )
    unwrapped = SimpleNamespace(
        obj=cube,
        goal_region=goal,
        agent=SimpleNamespace(tcp=tcp),
    )
    return SimpleNamespace(
        unwrapped=unwrapped,
        action_space=SimpleNamespace(shape=(7,)),
    )


def test_registry_exposes_exactly_five_supported_tasks() -> None:
    assert supported_task_ids() == (
        "PushCube-v1",
        "PullCube-v1",
        "PickCube-v1",
        "PlaceSphere-v1",
        "StackCube-v1",
    )
    assert {get_task_adapter(env_id).name for env_id in supported_task_ids()} == {
        "push_cube",
        "pull_cube",
        "pick_cube",
        "place_sphere",
        "stack_cube",
    }


def test_unknown_task_fails_with_supported_task_list() -> None:
    with pytest.raises(ValueError, match="no task adapter"):
        get_task_adapter("Unsupported-v1")


def test_pushcube_semantics_and_success_plan_preserve_mvp_action() -> None:
    env = _push_env()
    adapter = PushCubeAdapter()
    semantics = adapter.semantic_entities(env)
    assert [(item.role, item.category) for item in semantics] == [
        ("primary_object", CATEGORY_OBJECT),
        ("goal", CATEGORY_GOAL),
    ]

    plan = adapter.make_branch_plan(env, "success", 8, seed=1701, state_index=0)
    action = adapter.action(env, plan, 0, success_reached=False)

    expected_target = torch.tensor([0.08, 0.0, 0.001])
    expected_delta = np.clip(
        (expected_target - torch.tensor([-0.05, 0.0, 0.02])).numpy() / 0.1,
        -1.0,
        1.0,
    )
    np.testing.assert_allclose(action[:3], expected_delta)
    assert action[-1] == -1.0
    assert plan.stop_after_success
    stopped = adapter.action(env, plan, 1, success_reached=True)
    np.testing.assert_array_equal(stopped[:6], np.zeros(6, dtype=np.float32))
    assert stopped[-1] == -1.0


def test_pushcube_branch_plans_are_deterministic_and_distinct() -> None:
    env = _push_env()
    adapter = PushCubeAdapter()
    first = adapter.make_branch_plan(env, "off_target", 8, seed=1701, state_index=2)
    second = adapter.make_branch_plan(env, "off_target", 8, seed=1701, state_index=2)
    for left, right in zip(
        first.targets_world_m, second.targets_world_m, strict=True
    ):
        torch.testing.assert_close(left, right)
    assert first.metadata == second.metadata

    signatures = set()
    for branch in adapter.scaled_branches:
        plan = adapter.make_branch_plan(env, branch, 8, seed=1701, state_index=2)
        actions = np.stack(
            [
                adapter.action(env, plan, index, success_reached=False)
                for index in range(plan.horizon)
            ]
        )
        signatures.add(actions.tobytes())
    assert len(signatures) == len(adapter.scaled_branches)


def test_every_adapter_uses_same_model_facing_branch_contract() -> None:
    for env_id in supported_task_ids():
        adapter = get_task_adapter(env_id)
        assert adapter.counterfactual_branches == (
            "success",
            "perturbed",
            "no_op",
            "failure",
        )
        assert adapter.scaled_branches == (
            "success",
            "weak",
            "off_target",
            "failure",
            "no_op",
        )
