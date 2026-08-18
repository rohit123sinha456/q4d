"""Non-neural controls for persistent 3D trajectory prediction."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from q4d_wam.data import TrackDataset


def _query_rgb(batch: dict[str, Tensor]) -> Tensor:
    indices = batch["query_indices"]
    return torch.gather(
        batch["scene_rgb"],
        dim=1,
        index=indices[..., None].expand(-1, -1, batch["scene_rgb"].shape[-1]),
    )


def _ensure_batched(batch: dict[str, Tensor]) -> dict[str, Tensor]:
    if batch["query_xyz_world_m"].ndim == 2:
        return {key: value.unsqueeze(0) for key, value in batch.items()}
    return batch


def _scene_descriptor(batch: dict[str, Tensor]) -> Tensor:
    visible = torch.cat((batch["scene_xyz"], batch["scene_rgb"]), dim=-1)
    return torch.cat((visible.mean(dim=1), visible.std(dim=1)), dim=-1)


def _retrieve_trajectories(
    batch: dict[str, Tensor],
    reference_vectors: Tensor,
    training_reference_vectors: Tensor,
    training_query_features: Tensor,
    training_displacements_m: Tensor,
    neighbors: int,
) -> Tensor:
    initial = batch["query_xyz_world_m"]
    query_features = torch.cat((batch["query_xyz"], _query_rgb(batch)), dim=-1).cpu()
    episode_distances = torch.cdist(reference_vectors.cpu(), training_reference_vectors)
    episode_neighbors = episode_distances.topk(neighbors, largest=False, sorted=True).indices
    predictions = []
    for batch_index in range(len(initial)):
        retrieved = []
        for episode_index in episode_neighbors[batch_index]:
            point_matches = torch.cdist(
                query_features[batch_index], training_query_features[episode_index]
            ).argmin(dim=1)
            retrieved.append(training_displacements_m[episode_index, point_matches])
        mean_displacement = torch.stack(retrieved).mean(dim=0).to(initial)
        predictions.append(initial[batch_index, :, None, :] + mean_displacement)
    return torch.stack(predictions)


@dataclass(frozen=True)
class StaticBaseline:
    """Predict that every visible material point remains at its initial location."""

    name: str = "static"

    def predict(self, batch: dict[str, Tensor]) -> Tensor:
        batch = _ensure_batched(batch)
        initial = batch["query_xyz_world_m"]
        horizon = batch["actions"].shape[1]
        return initial[:, :, None, :].expand(-1, -1, horizon, -1).clone()


@dataclass(frozen=True)
class MeanDisplacementBaseline:
    """Apply the training set's mean trajectory displacement to every query."""

    mean_displacement_m: Tensor
    name: str = "train_mean_displacement"

    @classmethod
    def fit(cls, dataset: TrackDataset) -> MeanDisplacementBaseline:
        total: Tensor | None = None
        count = 0
        for index in range(len(dataset)):
            sample = dataset[index]
            displacement = (
                sample["target_world_m"] - sample["query_xyz_world_m"][:, None, :]
            ).to(torch.float64)
            total = displacement.sum(dim=0) if total is None else total + displacement.sum(dim=0)
            count += displacement.shape[0]
        if total is None or count == 0:
            raise ValueError("cannot fit a mean baseline on an empty dataset")
        return cls((total / count).to(torch.float32))

    def predict(self, batch: dict[str, Tensor]) -> Tensor:
        batch = _ensure_batched(batch)
        initial = batch["query_xyz_world_m"]
        displacement = self.mean_displacement_m.to(initial)
        if displacement.shape[0] != batch["actions"].shape[1]:
            raise ValueError("baseline and batch horizons differ")
        return initial[:, :, None, :] + displacement[None, None, :, :]


@dataclass(frozen=True)
class SceneKnnBaseline:
    """Retrieve point trajectories from episodes with a similar visible scene."""

    training_scene_descriptors: Tensor
    training_query_features: Tensor
    training_displacements_m: Tensor
    neighbors: int
    name: str = "scene_knn"

    @classmethod
    def fit(cls, dataset: TrackDataset, *, neighbors: int = 3) -> SceneKnnBaseline:
        if neighbors <= 0:
            raise ValueError("neighbors must be positive")
        descriptors = []
        features = []
        displacements = []
        for index in range(len(dataset)):
            sample = _ensure_batched(dataset[index])
            descriptors.append(_scene_descriptor(sample).squeeze(0))
            features.append(
                torch.cat((sample["query_xyz"], _query_rgb(sample)), dim=-1).squeeze(0)
            )
            displacements.append(
                (
                    sample["target_world_m"]
                    - sample["query_xyz_world_m"][:, :, None, :]
                ).squeeze(0)
            )
        if not descriptors:
            raise ValueError("cannot fit scene KNN on an empty dataset")
        return cls(
            training_scene_descriptors=torch.stack(descriptors),
            training_query_features=torch.stack(features),
            training_displacements_m=torch.stack(displacements),
            neighbors=min(neighbors, len(descriptors)),
        )

    def predict(self, batch: dict[str, Tensor]) -> Tensor:
        batch = _ensure_batched(batch)
        return _retrieve_trajectories(
            batch,
            _scene_descriptor(batch),
            self.training_scene_descriptors,
            self.training_query_features,
            self.training_displacements_m,
            self.neighbors,
        )


@dataclass(frozen=True)
class ActionKnnBaseline:
    """Retrieve trajectories from training episodes with similar action chunks.

    Episodes are selected in normalized action space. Each evaluation query then uses
    the closest visible training query in normalized XYZ plus RGB feature space. No
    segmentation, body identity, or simulator state is used.
    """

    training_actions: Tensor
    training_query_features: Tensor
    training_displacements_m: Tensor
    neighbors: int
    name: str = "action_knn"

    @classmethod
    def fit(cls, dataset: TrackDataset, *, neighbors: int = 3) -> ActionKnnBaseline:
        if neighbors <= 0:
            raise ValueError("neighbors must be positive")
        actions = []
        features = []
        displacements = []
        for index in range(len(dataset)):
            sample = _ensure_batched(dataset[index])
            actions.append(sample["actions"].flatten(start_dim=1).squeeze(0))
            features.append(
                torch.cat((sample["query_xyz"], _query_rgb(sample)), dim=-1).squeeze(0)
            )
            displacements.append(
                (
                    sample["target_world_m"]
                    - sample["query_xyz_world_m"][:, :, None, :]
                ).squeeze(0)
            )
        if not actions:
            raise ValueError("cannot fit action KNN on an empty dataset")
        return cls(
            training_actions=torch.stack(actions),
            training_query_features=torch.stack(features),
            training_displacements_m=torch.stack(displacements),
            neighbors=min(neighbors, len(actions)),
        )

    def predict(self, batch: dict[str, Tensor]) -> Tensor:
        batch = _ensure_batched(batch)
        action_vectors = batch["actions"].flatten(start_dim=1).cpu()
        return _retrieve_trajectories(
            batch,
            action_vectors,
            self.training_actions,
            self.training_query_features,
            self.training_displacements_m,
            self.neighbors,
        )
