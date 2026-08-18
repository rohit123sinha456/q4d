import torch

from q4d_wam.models import MicroQ4D


def _inputs() -> tuple[torch.Tensor, ...]:
    scene_xyz = torch.randn(2, 12, 3)
    scene_rgb = torch.rand(2, 12, 3)
    actions = torch.randn(2, 4, 7)
    query_xyz = scene_xyz[:, :5]
    return scene_xyz, scene_rgb, actions, query_xyz


def test_forward_and_cached_decode_are_equivalent() -> None:
    torch.manual_seed(0)
    model = MicroQ4D(action_dimensions=7, horizon=4, width=16).eval()
    scene_xyz, scene_rgb, actions, query_xyz = _inputs()

    direct = model(scene_xyz, scene_rgb, actions, query_xyz)
    scene_cache = model.encode_scene(scene_xyz, scene_rgb)
    query_cache = model.encode_queries(scene_cache, query_xyz)
    cached = model.decode(query_cache, actions)

    assert direct.shape == (2, 5, 4, 3)
    torch.testing.assert_close(cached, direct)


def test_action_changes_prediction_and_candidates_share_cache() -> None:
    torch.manual_seed(1)
    model = MicroQ4D(action_dimensions=7, horizon=4, width=16).eval()
    scene_xyz, scene_rgb, actions, query_xyz = _inputs()
    queries = model.encode_queries(model.encode_scene(scene_xyz, scene_rgb), query_xyz)
    candidates = torch.stack((actions, torch.zeros_like(actions), -actions), dim=1)

    prediction = model.predict_candidates(queries, candidates)

    assert prediction.shape == (2, 3, 5, 4, 3)
    torch.testing.assert_close(prediction[:, 0], model.decode(queries, actions))
    assert not torch.allclose(prediction[:, 0], prediction[:, 1])


def test_exact_scene_index_queries_match_coordinate_queries() -> None:
    torch.manual_seed(2)
    model = MicroQ4D(action_dimensions=7, horizon=4, width=16).eval()
    scene_xyz, scene_rgb, _, _ = _inputs()
    scene = model.encode_scene(scene_xyz, scene_rgb)
    indices = torch.tensor([[1, 4, 7], [2, 5, 8]])
    query_xyz = torch.gather(
        scene_xyz, 1, indices[..., None].expand(-1, -1, 3)
    )

    by_index = model.encode_query_indices(scene, indices)
    by_coordinate = model.encode_queries(scene, query_xyz)

    torch.testing.assert_close(by_index.query_features, by_coordinate.query_features)
