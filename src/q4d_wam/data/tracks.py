"""Dataset loading for compact persistent-trajectory fragments."""

from __future__ import annotations

import hashlib
import json
import random
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from q4d_wam.labels import farthest_point_indices

REQUIRED_TRAINING_KEYS = frozenset(
    {
        "rgb",
        "depth_mm",
        "intrinsic_cv",
        "extrinsic_cv",
        "cam2world_gl",
        "actions",
        "point_pixels_uv",
        "xyz0_world_m",
        "point_rgb",
        "target_tracks_world_m",
    }
)

PRIVILEGED_KEYS = frozenset(
    {
        "body_categories",
        "body_indices",
        "body_pose_sequence_world",
        "body_segmentation_ids",
        "contact_region",
        "cube_centers_world_m",
        "local_xyz_m",
        "point_categories",
        "point_segmentation_ids",
        "primary_object_centers_world_m",
        "tracked_entity_centers_world_m",
        "tracked_entity_names",
        "tracks_world_m",
    }
)


def discover_training_files(root: str | Path) -> list[Path]:
    """Find only model-facing files and never privileged audit archives."""
    root = Path(root)
    files = sorted(root.glob("*.train.npz"))
    if not files:
        raise FileNotFoundError(f"no *.train.npz files found under {root}")
    return files


def trajectory_group_id(path: str | Path) -> str:
    """Return the shared initial-state ID encoded before a double underscore."""
    name = Path(path).name
    suffix = ".train.npz"
    if not name.endswith(suffix):
        raise ValueError(f"expected a {suffix} file, got {name}")
    return name[: -len(suffix)].split("__", maxsplit=1)[0]


def _stable_rank(path: Path, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{path.name}".encode()).hexdigest()


@dataclass(frozen=True)
class SplitManifest:
    seed: int
    train: tuple[str, ...]
    validation: tuple[str, ...]
    test: tuple[str, ...]

    @classmethod
    def create(
        cls,
        files: list[Path],
        *,
        seed: int,
        train_fraction: float,
        validation_fraction: float,
        test_fraction: float,
    ) -> SplitManifest:
        fractions = train_fraction + validation_fraction + test_fraction
        if not np.isclose(fractions, 1.0):
            raise ValueError(f"split fractions must sum to one, got {fractions}")
        grouped: dict[str, list[Path]] = {}
        for path in files:
            grouped.setdefault(trajectory_group_id(path), []).append(path)
        if len(grouped) < 3:
            raise ValueError("at least three state groups are required for split construction")
        ordered_groups = sorted(
            grouped, key=lambda group_id: _stable_rank(Path(group_id), seed)
        )
        train_count = max(1, int(len(ordered_groups) * train_fraction))
        validation_count = max(1, int(len(ordered_groups) * validation_fraction))
        if train_count + validation_count >= len(ordered_groups):
            validation_count = 1
            train_count = len(ordered_groups) - 2

        def group_files(group_ids: list[str]) -> tuple[str, ...]:
            return tuple(
                path.name
                for group_id in group_ids
                for path in sorted(grouped[group_id], key=lambda item: item.name)
            )

        return cls(
            seed=seed,
            train=group_files(ordered_groups[:train_count]),
            validation=group_files(
                ordered_groups[train_count : train_count + validation_count]
            ),
            test=group_files(ordered_groups[train_count + validation_count :]),
        )

    @property
    def all_files(self) -> tuple[str, ...]:
        return self.train + self.validation + self.test

    def files(self, root: str | Path, split: str) -> list[Path]:
        names = getattr(self, split)
        return [Path(root) / name for name in names]

    def to_dict(self) -> dict[str, Any]:
        group_counts = {
            "train": len({trajectory_group_id(name) for name in self.train}),
            "validation": len({trajectory_group_id(name) for name in self.validation}),
            "test": len({trajectory_group_id(name) for name in self.test}),
        }
        return {
            "seed": self.seed,
            "counts": {
                "train": len(self.train),
                "validation": len(self.validation),
                "test": len(self.test),
            },
            "group_counts": group_counts,
            "train": list(self.train),
            "validation": list(self.validation),
            "test": list(self.test),
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> SplitManifest:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            seed=int(raw["seed"]),
            train=tuple(raw["train"]),
            validation=tuple(raw["validation"]),
            test=tuple(raw["test"]),
        )


class _RunningMoments:
    def __init__(self, feature_count: int):
        self.count = 0
        self.total = torch.zeros(feature_count, dtype=torch.float64)
        self.total_squared = torch.zeros(feature_count, dtype=torch.float64)

    def update(self, values: Tensor) -> None:
        values = values.detach().to(torch.float64).reshape(-1, self.total.numel())
        self.count += len(values)
        self.total += values.sum(dim=0)
        self.total_squared += (values * values).sum(dim=0)

    def finish(self, epsilon: float) -> tuple[Tensor, Tensor, Tensor]:
        if self.count == 0:
            raise ValueError("cannot compute moments from no values")
        mean = self.total / self.count
        variance = torch.clamp(self.total_squared / self.count - mean.square(), min=0)
        raw_std = torch.sqrt(variance)
        scale = torch.clamp(raw_std, min=epsilon)
        return mean.to(torch.float32), scale.to(torch.float32), raw_std.to(torch.float32)


@dataclass(frozen=True)
class NormalizationStats:
    xyz_mean_m: Tensor
    xyz_scale_m: Tensor
    action_mean: Tensor
    action_scale: Tensor
    displacement_mean_m: Tensor
    displacement_scale_m: Tensor
    constant_action_channels: tuple[int, ...]
    source_files: tuple[str, ...]
    epsilon: float

    def to_dict(self) -> dict[str, Any]:
        def values(tensor: Tensor) -> list[float]:
            return [float(value) for value in tensor]

        return {
            "fit_scope": "training split only",
            "source_file_count": len(self.source_files),
            "source_files": list(self.source_files),
            "epsilon": self.epsilon,
            "xyz_mean_m": values(self.xyz_mean_m),
            "xyz_scale_m": values(self.xyz_scale_m),
            "action_mean": values(self.action_mean),
            "action_scale": values(self.action_scale),
            "displacement_mean_m": values(self.displacement_mean_m),
            "displacement_scale_m": values(self.displacement_scale_m),
            "constant_action_channels": list(self.constant_action_channels),
            "rgb_transform": "uint8 / 255",
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> NormalizationStats:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            xyz_mean_m=torch.tensor(raw["xyz_mean_m"], dtype=torch.float32),
            xyz_scale_m=torch.tensor(raw["xyz_scale_m"], dtype=torch.float32),
            action_mean=torch.tensor(raw["action_mean"], dtype=torch.float32),
            action_scale=torch.tensor(raw["action_scale"], dtype=torch.float32),
            displacement_mean_m=torch.tensor(
                raw["displacement_mean_m"], dtype=torch.float32
            ),
            displacement_scale_m=torch.tensor(
                raw["displacement_scale_m"], dtype=torch.float32
            ),
            constant_action_channels=tuple(raw["constant_action_channels"]),
            source_files=tuple(raw["source_files"]),
            epsilon=float(raw["epsilon"]),
        )


def validate_training_file(path: str | Path) -> dict[str, Any]:
    """Validate schema, fixed dimensions, finiteness, and privilege separation."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as archive:
        keys = frozenset(archive.files)
        missing = REQUIRED_TRAINING_KEYS - keys
        privileged = PRIVILEGED_KEYS & keys
        if missing:
            raise ValueError(f"{path.name} is missing keys: {sorted(missing)}")
        if privileged:
            raise ValueError(f"{path.name} contains privileged keys: {sorted(privileged)}")

        xyz = archive["xyz0_world_m"]
        rgb = archive["point_rgb"]
        actions = archive["actions"]
        targets = archive["target_tracks_world_m"]
        pixels = archive["point_pixels_uv"]
        if xyz.ndim != 2 or xyz.shape[-1] != 3:
            raise ValueError(f"{path.name}: xyz0_world_m must be [N, 3]")
        if rgb.shape != xyz.shape:
            raise ValueError(f"{path.name}: point_rgb must match xyz0_world_m")
        if pixels.shape != (len(xyz), 2):
            raise ValueError(f"{path.name}: point_pixels_uv must be [N, 2]")
        if targets.ndim != 3 or targets.shape[0] != len(xyz) or targets.shape[-1] != 3:
            raise ValueError(f"{path.name}: target_tracks_world_m must be [N, H, 3]")
        if actions.ndim != 2 or actions.shape[0] != targets.shape[1]:
            raise ValueError(f"{path.name}: actions must be [H, A] and match target horizon")
        for key in ("xyz0_world_m", "actions", "target_tracks_world_m"):
            if not np.isfinite(archive[key]).all():
                raise ValueError(f"{path.name}: {key} contains non-finite values")
        return {
            "points": len(xyz),
            "horizon": targets.shape[1],
            "action_dimensions": actions.shape[1],
            "keys": sorted(keys),
        }


def compute_normalization(
    training_files: list[Path], epsilon: float = 1e-6
) -> NormalizationStats:
    """Fit streaming statistics using the training split and no validation/test files."""
    if not training_files:
        raise ValueError("training_files cannot be empty")
    xyz_moments = _RunningMoments(3)
    action_moments: _RunningMoments | None = None
    displacement_moments = _RunningMoments(3)
    for path in training_files:
        validate_training_file(path)
        with np.load(path, allow_pickle=False) as archive:
            xyz = torch.from_numpy(archive["xyz0_world_m"]).to(torch.float32)
            actions = torch.from_numpy(archive["actions"]).to(torch.float32)
            targets = torch.from_numpy(archive["target_tracks_world_m"]).to(torch.float32)
        if action_moments is None:
            action_moments = _RunningMoments(actions.shape[-1])
        xyz_moments.update(xyz)
        action_moments.update(actions)
        displacement_moments.update(targets - xyz[:, None, :])

    xyz_mean, xyz_scale, _ = xyz_moments.finish(epsilon)
    assert action_moments is not None
    action_mean, action_scale, action_raw_std = action_moments.finish(epsilon)
    displacement_mean, displacement_scale, _ = displacement_moments.finish(epsilon)
    constant_channels = tuple(
        int(index) for index in torch.nonzero(action_raw_std < epsilon).flatten()
    )
    return NormalizationStats(
        xyz_mean_m=xyz_mean,
        xyz_scale_m=xyz_scale,
        action_mean=action_mean,
        action_scale=action_scale,
        displacement_mean_m=displacement_mean,
        displacement_scale_m=displacement_scale,
        constant_action_channels=constant_channels,
        source_files=tuple(path.name for path in training_files),
        epsilon=epsilon,
    )


class TrackDataset(Dataset[dict[str, Tensor]]):
    """Model-facing point trajectories with no access to audit archives."""

    def __init__(
        self,
        files: list[Path],
        normalization: NormalizationStats,
        *,
        num_queries: int,
        horizon: int | None = None,
        cache_size: int = 0,
        include_metric_targets: bool = True,
    ):
        if not files:
            raise ValueError("dataset files cannot be empty")
        if num_queries <= 0:
            raise ValueError("num_queries must be positive")
        if horizon is not None and horizon <= 0:
            raise ValueError("horizon must be positive when provided")
        self.files = list(files)
        self.normalization = normalization
        self.num_queries = num_queries
        self.horizon = horizon
        self.cache_size = max(0, cache_size)
        self.include_metric_targets = include_metric_targets
        self._cache: OrderedDict[int, dict[str, np.ndarray]] = OrderedDict()
        self._sample_cache: OrderedDict[int, dict[str, Tensor]] = OrderedDict()

    def __len__(self) -> int:
        return len(self.files)

    def _load(self, index: int) -> dict[str, np.ndarray]:
        if index in self._cache:
            self._cache.move_to_end(index)
            return self._cache[index]
        path = self.files[index]
        validate_training_file(path)
        with np.load(path, allow_pickle=False) as archive:
            raw = {key: archive[key] for key in REQUIRED_TRAINING_KEYS}
        if self.cache_size:
            self._cache[index] = raw
            self._cache.move_to_end(index)
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return raw

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        if index in self._sample_cache:
            self._sample_cache.move_to_end(index)
            return self._sample_cache[index]
        raw = self._load(index)
        xyz_world = torch.from_numpy(raw["xyz0_world_m"]).to(torch.float32)
        rgb = torch.from_numpy(raw["point_rgb"]).to(torch.float32) / 255.0
        actions = torch.from_numpy(raw["actions"]).to(torch.float32)
        targets_world = torch.from_numpy(raw["target_tracks_world_m"]).to(torch.float32)
        if self.horizon is not None:
            if self.horizon > len(actions):
                raise ValueError(
                    f"requested horizon {self.horizon} exceeds stored horizon {len(actions)}"
                )
            actions = actions[: self.horizon]
            targets_world = targets_world[:, : self.horizon]
        # Geometry-only FPS misses the small PushCube because it occupies a dense central
        # region. Standardized RGB adds visible appearance diversity without privileged
        # segmentation or body labels.
        xyz_features = (xyz_world - xyz_world.mean(dim=0)) / xyz_world.std(dim=0).clamp_min(
            1e-6
        )
        rgb_features = (rgb - rgb.mean(dim=0)) / rgb.std(dim=0).clamp_min(1e-6)
        visible_features = torch.cat((xyz_features, rgb_features), dim=-1)
        query_indices = farthest_point_indices(
            visible_features, min(self.num_queries, len(xyz_world))
        )
        query_world = xyz_world[query_indices]
        query_targets_world = targets_world[query_indices]
        displacement = query_targets_world - query_world[:, None, :]

        stats = self.normalization
        batch = {
            "sample_id": torch.tensor(index, dtype=torch.long),
            "scene_xyz": (xyz_world - stats.xyz_mean_m) / stats.xyz_scale_m,
            "scene_rgb": rgb,
            "actions": (actions - stats.action_mean) / stats.action_scale,
            "query_indices": query_indices,
            "query_xyz": (query_world - stats.xyz_mean_m) / stats.xyz_scale_m,
            "target_displacement": (
                displacement - stats.displacement_mean_m
            )
            / stats.displacement_scale_m,
        }
        if self.include_metric_targets:
            batch["query_xyz_world_m"] = query_world
            batch["target_world_m"] = query_targets_world
        if self.cache_size:
            self._sample_cache[index] = batch
            self._sample_cache.move_to_end(index)
            # Once the deterministic model-facing tensors exist, retaining the larger raw
            # RGB-D archive in the same worker only wastes memory.
            self._cache.pop(index, None)
            while len(self._sample_cache) > self.cache_size:
                self._sample_cache.popitem(last=False)
        return batch


@dataclass(frozen=True)
class DataLoaderConfig:
    batch_size: int
    num_workers: int
    prefetch_factor: int = 2
    pin_memory: bool = True
    persistent_workers: bool = True


def _seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_dataloader(
    dataset: TrackDataset,
    config: DataLoaderConfig,
    *,
    shuffle: bool,
    seed: int,
    drop_last: bool = False,
) -> DataLoader[dict[str, Tensor]]:
    generator = torch.Generator().manual_seed(seed)
    worker_options: dict[str, Any] = {}
    if config.num_workers > 0:
        worker_options = {
            "prefetch_factor": config.prefetch_factor,
            "persistent_workers": config.persistent_workers,
        }
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=drop_last,
        worker_init_fn=_seed_worker,
        generator=generator,
        **worker_options,
    )


def move_batch_to_device(
    batch: Mapping[str, Tensor], device: torch.device | str, *, non_blocking: bool = True
) -> dict[str, Tensor]:
    return {
        key: value.to(device=device, non_blocking=non_blocking)
        for key, value in batch.items()
    }
