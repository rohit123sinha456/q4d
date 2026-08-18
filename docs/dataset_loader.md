# Dataset loader and batching

Checklist item 6 turns the PushCube trajectory fragments into a reproducible
PyTorch input pipeline. Its settings live in `configs/data.toml`, and the preparation
entry point is:

```bash
python scripts/prepare_dataset.py
```

The command validates every model-facing archive, freezes the split manifest, fits
normalization statistics, exercises a complete loader epoch, and verifies a pinned,
non-blocking transfer to CUDA. It writes these reproducibility artifacts beside the
dataset:

- `splits.json`: exact filenames in the train, validation, and test partitions.
- `normalization.json`: statistics and the exact training files used to fit them.
- `loader_report.json`: schema, batching, query-coverage, and GPU checks.

## Split and privilege rules

Only files matching `*.train.npz` are discoverable by the loader. The matching
`.audit.npz` files contain simulator-only labels and are never opened by
`TrackDataset`. A seeded SHA-256 ranking assigns complete initial-state groups to the
80/10/10 train/validation/test partitions. The split is independent of filesystem order
and all counterfactual siblings remain together. The current 25 groups produce 20/2/3
state groups and 80/8/12 fragments.

Normalization is fitted in streaming float64 accumulators using only the 80 training
archives. World XYZ, actions, and future displacement targets use per-channel z-scores.
RGB is converted from `uint8` to `[0, 1]`. Action channels that are constant in this
corpus are recorded explicitly and use the configured epsilon as their scale.

## Model-facing batch contract

With the current configuration, each batch contains:

| Field | Shape | Meaning |
| --- | --- | --- |
| `sample_id` | `[B]` | Index within the split. |
| `scene_xyz` | `[B, 256, 3]` | Normalized visible world points. |
| `scene_rgb` | `[B, 256, 3]` | Visible point colors in `[0, 1]`. |
| `actions` | `[B, 8, 7]` | Normalized future action chunk. |
| `query_indices` | `[B, 32]` | Selected indices in the visible point set. |
| `query_xyz` | `[B, 32, 3]` | Normalized query positions. |
| `target_displacement` | `[B, 32, 8, 3]` | Normalized future displacement labels. |
| `query_xyz_world_m` | `[B, 32, 3]` | Metric queries for evaluation and plots. |
| `target_world_m` | `[B, 32, 8, 3]` | Metric future trajectories for evaluation. |

The metric fields can be disabled when constructing `TrackDataset` if a training loop
does not need them.

Geometry-only farthest-point sampling omitted the small, centrally located cube in all
training episodes. Query selection therefore uses standardized visible XYZ and RGB as a
six-dimensional feature. It still uses no segmentation, body identity, or other
privileged label. An audit-only preparation check confirms that the selected queries
include at least two cube points in every training episode; those categories are never
returned to the model.

## GPU path

The default loader uses batch size 8, two persistent workers, two prefetched batches,
an in-worker archive cache, pinned host tensors, and deterministic worker seeds.
`move_batch_to_device(..., non_blocking=True)` moves the entire mapping to the GPU.
The verified RTX 4060 run transferred all tensors to CUDA and used about 0.104 MiB of
allocated device memory for one batch. These intentionally small settings leave most of
the 8 GB GPU available for the first model and optimizer.
