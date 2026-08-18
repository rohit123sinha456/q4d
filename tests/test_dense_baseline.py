import torch

from q4d_wam.models import (
    DensePointFutureModel,
    MicroQ4D,
    dense_query_set_is_complete,
)


def _inputs() -> tuple[torch.Tensor, ...]:
    scene_xyz = torch.randn(2, 12, 3)
    scene_rgb = torch.rand(2, 12, 3)
    actions = torch.randn(2, 4, 7)
    return scene_xyz, scene_rgb, actions


def test_dense_baseline_is_exactly_parameter_matched() -> None:
    dense = DensePointFutureModel(action_dimensions=7, horizon=4, width=16)
    sparse = MicroQ4D(action_dimensions=7, horizon=4, width=16)

    assert sum(parameter.numel() for parameter in dense.parameters()) == sum(
        parameter.numel() for parameter in sparse.parameters()
    )
    assert dense.state_dict().keys() == sparse.state_dict().keys()


def test_dense_forward_matches_explicit_all_point_query_decode() -> None:
    torch.manual_seed(0)
    model = DensePointFutureModel(action_dimensions=7, horizon=4, width=16).eval()
    scene_xyz, scene_rgb, actions = _inputs()

    direct = model(scene_xyz, scene_rgb, actions)
    scene = model.encode_scene(scene_xyz, scene_rgb)
    cached = model.decode(model.encode_dense_queries(scene), actions)

    assert direct.shape == (2, 12, 4, 3)
    torch.testing.assert_close(direct, cached)


def test_dense_prediction_is_action_conditioned() -> None:
    torch.manual_seed(1)
    model = DensePointFutureModel(action_dimensions=7, horizon=4, width=16).eval()
    scene_xyz, scene_rgb, actions = _inputs()

    prediction = model(scene_xyz, scene_rgb, actions)
    no_op_prediction = model(scene_xyz, scene_rgb, torch.zeros_like(actions))

    assert not torch.allclose(prediction, no_op_prediction)


def test_dense_coverage_check_uses_actual_scene_size() -> None:
    scene_xyz = torch.zeros(512, 3)

    assert dense_query_set_is_complete(scene_xyz, torch.randperm(512))
    assert not dense_query_set_is_complete(scene_xyz, torch.arange(256))
