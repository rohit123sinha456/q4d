import torch

from q4d_wam.baselines import (
    ActionKnnBaseline,
    MeanDisplacementBaseline,
    SceneKnnBaseline,
    StaticBaseline,
)


class _TinyDataset:
    def __init__(self, samples: list[dict[str, torch.Tensor]]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.samples[index]


def _sample(action: float, displacement: float) -> dict[str, torch.Tensor]:
    initial = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    delta = torch.tensor([[[displacement, 0.0, 0.0]], [[displacement, 0.0, 0.0]]])
    return {
        "sample_id": torch.tensor(0),
        "scene_xyz": initial,
        "scene_rgb": torch.tensor([[1.0, 0, 0], [0.0, 1, 0]]),
        "actions": torch.tensor([[action]]),
        "query_indices": torch.tensor([0, 1]),
        "query_xyz": initial,
        "query_xyz_world_m": initial,
        "target_world_m": initial[:, None, :] + delta,
    }


def test_static_and_mean_displacement_baselines() -> None:
    dataset = _TinyDataset([_sample(0.0, 1.0), _sample(1.0, 3.0)])
    batch = {key: value.unsqueeze(0) for key, value in dataset[0].items()}

    static_prediction = StaticBaseline().predict(batch)
    mean_prediction = MeanDisplacementBaseline.fit(dataset).predict(batch)  # type: ignore[arg-type]

    torch.testing.assert_close(static_prediction[:, :, 0], batch["query_xyz_world_m"])
    torch.testing.assert_close(
        mean_prediction[:, :, 0, 0], batch["query_xyz_world_m"][..., 0] + 2.0
    )


def test_action_knn_retrieves_nearest_action_and_visible_point() -> None:
    dataset = _TinyDataset([_sample(0.0, 1.0), _sample(10.0, 4.0)])
    query = _sample(9.0, 0.0)
    batch = {key: value.unsqueeze(0) for key, value in query.items()}

    prediction = ActionKnnBaseline.fit(dataset, neighbors=1).predict(batch)  # type: ignore[arg-type]

    torch.testing.assert_close(
        prediction[:, :, 0, 0], batch["query_xyz_world_m"][..., 0] + 4.0
    )


def test_scene_knn_is_action_free() -> None:
    first = _sample(0.0, 1.0)
    second = _sample(10.0, 4.0)
    second["scene_xyz"] = second["scene_xyz"] + 20.0
    query = _sample(999.0, 0.0)
    batch = {key: value.unsqueeze(0) for key, value in query.items()}

    prediction = SceneKnnBaseline.fit(
        _TinyDataset([first, second]), neighbors=1  # type: ignore[arg-type]
    ).predict(batch)

    torch.testing.assert_close(
        prediction[:, :, 0, 0], batch["query_xyz_world_m"][..., 0] + 1.0
    )
