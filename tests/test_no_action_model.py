import inspect

import pytest
import torch

from q4d_wam.models import NoActionTrajectoryModel


def test_no_action_model_shape_and_interface() -> None:
    model = NoActionTrajectoryModel(horizon=4, width=16)
    scene_xyz = torch.randn(2, 12, 3)
    scene_rgb = torch.rand(2, 12, 3)
    query_xyz = scene_xyz[:, :5]

    prediction = model(scene_xyz, scene_rgb, query_xyz)

    assert prediction.shape == (2, 5, 4, 3)
    assert list(inspect.signature(model.forward).parameters) == [
        "scene_xyz",
        "scene_rgb",
        "query_xyz",
    ]


def test_no_action_model_validates_shapes() -> None:
    model = NoActionTrajectoryModel(horizon=2, width=8)

    with pytest.raises(ValueError, match="scene_rgb"):
        model(torch.zeros(1, 4, 3), torch.zeros(1, 5, 3), torch.zeros(1, 2, 3))


def test_no_action_cached_candidates_match_direct_prediction() -> None:
    torch.manual_seed(12)
    model = NoActionTrajectoryModel(horizon=2, width=8).eval()
    scene_xyz = torch.randn(1, 8, 3)
    scene_rgb = torch.rand(1, 8, 3)
    query_indices = torch.tensor([[1, 3, 6]])
    query_xyz = scene_xyz[:, query_indices[0]]
    actions = torch.randn(1, 5, 2, 7)

    direct = model(scene_xyz, scene_rgb, query_xyz)
    scene = model.encode_scene(scene_xyz, scene_rgb)
    queries = model.encode_query_indices(scene, query_indices)
    candidates = model.predict_candidates(queries, actions)

    assert candidates.shape == (1, 5, 3, 2, 3)
    torch.testing.assert_close(candidates[:, 0], direct)
    torch.testing.assert_close(candidates[:, 1:], candidates[:, :1].expand(-1, 4, -1, -1, -1))
