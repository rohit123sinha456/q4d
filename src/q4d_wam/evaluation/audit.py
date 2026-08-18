"""Privileged metric stratification kept separate from model-facing inputs."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from q4d_wam.data import TrackDataset
from q4d_wam.labels import CATEGORY_GOAL, CATEGORY_OBJECT, CATEGORY_ROBOT, CATEGORY_STATIC

CATEGORY_NAMES = {
    CATEGORY_STATIC: "static",
    CATEGORY_ROBOT: "robot",
    CATEGORY_OBJECT: "object",
    CATEGORY_GOAL: "goal",
}


def load_audit_metadata(
    dataset: TrackDataset, batch: dict[str, Tensor]
) -> tuple[dict[str, Tensor], Tensor, list[str]]:
    """Load evaluation-only categories, contacts, body IDs, and branch masks."""
    category_rows = []
    contact_rows = []
    body_rows = []
    branch_names = []
    audit_files = []
    for row, sample_id in enumerate(batch["sample_id"]):
        training_path = dataset.files[int(sample_id)]
        audit_path = training_path.with_name(
            training_path.name.replace(".train.npz", ".audit.npz")
        )
        if not audit_path.exists():
            raise FileNotFoundError(f"missing audit archive for evaluation: {audit_path}")
        indices = batch["query_indices"][row].numpy()
        with np.load(audit_path, allow_pickle=False) as audit:
            category_rows.append(torch.from_numpy(audit["point_categories"][indices]))
            contact_rows.append(torch.from_numpy(audit["contact_region"][indices]))
            body_rows.append(torch.from_numpy(audit["body_indices"][indices]))
        stem = training_path.name.removesuffix(".train.npz")
        branch_names.append(stem.split("__", maxsplit=1)[1] if "__" in stem else "single")
        audit_files.append(audit_path.name)

    categories = torch.stack(category_rows).to(torch.long)
    groups = {
        name: categories == category_id for category_id, name in CATEGORY_NAMES.items()
    }
    groups["contact"] = torch.stack(contact_rows).to(torch.bool)
    for branch_name in sorted(set(branch_names)):
        row_mask = torch.tensor([name == branch_name for name in branch_names])
        groups[f"branch_{branch_name}"] = row_mask[:, None].expand_as(categories)
    return groups, torch.stack(body_rows).to(torch.long), audit_files
