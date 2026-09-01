# Q4D-WAM

Q4D-WAM is a research implementation of an action-conditioned, queryable 4D world
model for manipulation planning. It learns future 3D trajectories for selected visible
scene points, reuses cached scene features to evaluate candidate action sequences, and
compares sparse Q4D prediction with action-free, retrieval, and parameter-matched dense
baselines.

The repository provides an end-to-end, tested pipeline for ManiSkill `PushCube-v1` and
task-adapter-based RGB-D collection and privileged rigid-track labels for `PullCube-v1`,
`PickCube-v1`, `PlaceSphere-v1`, and `StackCube-v1`. The model-facing archive schema is
shared across all five tasks.

> **Research status:** adapter integrity is verified for five tasks, and PushCube output
> is exactly backward-compatible with the pre-adapter collector. This is not yet a
> publication-ready multi-task result: joint training, strong task-solving branch
> policies for the four added tasks, multi-seed evaluation, and statistical reporting
> remain to be completed.

## What works today

| Capability | Status |
| --- | --- |
| PushCube RGB-D collection and rigid point-track labels | Implemented |
| Counterfactual branches from shared initial states | Implemented |
| Deterministic grouped 80/10/10 splits | Implemented |
| Static, KNN, no-action, sparse Q4D, and dense baselines | Implemented |
| H=1/2/4/8 experiment matrix | Implemented |
| Mixed-precision single-GPU training | Implemented |
| Cached candidate-action decoding and N/M benchmark | Implemented |
| Random-shooting and CEM MPC | Uses task-provided object and goal semantics |
| Five ManiSkill task adapters | Implemented and simulator-audited |
| Distributed data-parallel training | Not implemented |
| GPU-accelerated collection in `generate_point_tracks.py` | Not implemented; CPU backends are hard-coded |
| Joint multi-task training/evaluation | Not implemented |

The original development machine used Python 3.12, PyTorch 2.11.0 + CUDA 12.8,
ManiSkill 3.0.1, and an 8 GB RTX 4060. A native Linux GPU server is preferred for the
next stage. Official installation references are the
[ManiSkill installation guide](https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/installation.html)
and [PyTorch installation selector](https://pytorch.org/get-started/locally/).

## Repository layout

```text
configs/        TOML configurations for data, training, planning, and gates
docs/           protocol and implementation notes for checklist items 1-14
requirements/   direct requirements and the resolved CUDA 12.8 environment
scripts/        collection, training, evaluation, planning, and aggregation entry points
src/q4d_wam/    geometry, labels, data pipeline, models, metrics, and planners
tests/          unit tests
artifacts/      generated datasets, checkpoints, and reports; ignored by Git
```

## 1. Prepare a Linux GPU server

Recommended starting resources for a larger PushCube run:

- Ubuntu 24.04 or another supported native Linux distribution.
- Python 3.12.
- A recent NVIDIA driver compatible with the pinned CUDA 12.8 PyTorch wheel.
- At least one CUDA GPU. The scripts use one visible GPU per process.
- 32 GB host RAM for the existing 10,000-fragment experiment; use more RAM or reduce
  dataset caching when increasing the corpus.
- Fast local NVMe storage. The existing 10,000-fragment dataset occupies about 0.9 GB,
  excluding checkpoints and duplicated backups. Storage grows approximately with state
  count, branch count, point count, and horizon.
- Vulkan runtime support if simulator rendering will run on the server.

Verify the driver before creating the environment:

```bash
nvidia-smi
python3.12 --version
```

Install common system packages on Ubuntu:

```bash
sudo apt-get update
sudo apt-get install -y git python3.12 python3.12-venv libvulkan1 vulkan-tools
```

Clone the repository and create an isolated environment:

```bash
git clone <repository-url> q4d-wam
cd q4d-wam

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

For exact reproduction of the development environment:

```bash
python -m pip install -r requirements/lock-wsl-cu128.txt
python -m pip install -e .
```

Despite its filename, `lock-wsl-cu128.txt` is a normal Python requirements file and can
be used on a compatible Linux CUDA 12.8 machine. If the server requires a different CUDA
wheel, install the appropriate PyTorch build first, then install the direct project
requirements:

```bash
python -m pip install -r requirements/torch-cu128.txt
python -m pip install -r requirements/project.txt
python -m pip install -e .
```

Do not mix a new PyTorch/ManiSkill stack into the main paper run without recording it as
a separate environment condition.

## 2. Verify the installation

Run these checks from the repository root:

```bash
source .venv/bin/activate

python scripts/check_cuda.py --output artifacts/environment/cuda.json
pytest -q
ruff check .
```

The current repository should report 44 passing tests. Then check ManiSkill separately:

```bash
# State-only simulator check.
python scripts/smoke_maniskill.py --mode cpu-state

# CPU physics and CPU/software RGB-D rendering.
python scripts/smoke_maniskill.py --mode cpu-rgbd

# Desired native-Linux server path: CUDA physics and CUDA rendering.
timeout 120s python scripts/smoke_maniskill.py --mode gpu-rgbd
```

Also run the geometry audit before collecting a large dataset:

```bash
python scripts/verify_backprojection.py --config configs/smoke.toml
```

Do not start a large collection until CUDA, RGB-D, segmentation, backprojection, and the
test suite all pass on the target server.

## 3. Reproduce the existing PushCube pipeline

The following commands reproduce the current scaled protocol: 2,000 initial states,
five counterfactual branches per state, 10,000 fragments, 512 visible points, and a
maximum stored horizon of eight actions.

### 3.1 Collect trajectory fragments

Run a small pilot first:

```bash
python -u scripts/generate_point_tracks.py \
  --config configs/scale_pushcube.toml \
  --profile scaled \
  --states 10 \
  --output-dir artifacts/datasets/pushcube_server_pilot \
  --checkpoint-every 2
```

Inspect the pilot manifest and several `*.record.json` files. The manifest must be
complete, all physical checks must pass, sibling observations must be identical, and all
five action branches must be distinct.

Run or resume the full collection:

```bash
python -u scripts/generate_point_tracks.py \
  --config configs/scale_pushcube.toml \
  --profile scaled \
  --states 2000 \
  --resume \
  --checkpoint-every 10
```

`--resume` reuses only complete state groups. The five sibling branches for a state are
kept together so that later train/validation/test splits cannot leak the same initial
observation across partitions.

Important: `generate_point_tracks.py` currently constructs the environment with
`physx_cpu` and `sapien_cpu` directly. Changing the TOML backend fields does not make
collection GPU-accelerated. Parallel CPU collection or a code change is required to use
additional server compute for this stage.

### 3.2 Freeze the dataset split and normalization

```bash
python scripts/prepare_dataset.py --config configs/data_scale.toml
```

This writes `splits.json`, training-only `normalization.json`, and
`loader_report.json` beside the dataset. Treat these files as immutable for a named
experiment. Never refit normalization on validation or test fragments.

### 3.3 Run baselines and the H=1/2/4/8 matrix

```bash
python scripts/run_scale_experiment.py --config configs/scale_experiment.toml
```

The runner is resumable: a stage with an existing passing report is skipped. Use
`--force` only when intentionally replacing an experiment, preferably after selecting a
new output directory.

The runner executes, for every requested horizon:

1. static, scene-KNN, and action-KNN baselines;
2. the neural no-action baseline;
3. sparse action-conditioned micro-Q4D;
4. the parameter-matched dense point-future baseline;
5. action-shuffling and candidate-decoding benchmarks;
6. the H=8 N/M scaling grid.

The final item-12 summarizer is deliberately strict and exits non-zero when any frozen
scientific check fails. A failed summarizer is a research result, not a reason to edit a
threshold after inspecting the data. The historical H=4/H=8 single cache benchmark and
the later corrected exact-index grid disagree; preserve and report both until the fixed
gate has been rerun under one frozen protocol.

Useful partial runs are:

```bash
# Prepare data only.
python scripts/run_scale_experiment.py \
  --config configs/scale_experiment.toml \
  --prepare-only

# Run selected horizons.
python scripts/run_scale_experiment.py \
  --config configs/scale_experiment.toml \
  --horizons 4 8

# Re-run the corrected N/M cache grid.
python scripts/benchmark_scale_grid.py \
  --config configs/scale_experiment.toml \
  --horizon 8 \
  --scene-points 128 256 512 \
  --query-points 32 64 128 \
  --repetitions 30

# Evaluate the original fixed item-12 aggregate gate.
python scripts/summarize_scale_experiment.py \
  --config configs/scale_experiment.toml
```

### 3.4 Run closed-loop MPC

```bash
python scripts/evaluate_mpc.py \
  --config configs/mpc.toml \
  --episodes 10 \
  --models q4d dense no_action \
  --methods random_shooting cem \
  --budgets-ms 50 100 200
```

For a paper, increase the episode count and evaluate genuinely varied initial states.
Record actual elapsed time and budget overruns, not only the requested deadline. The
current planner uses privileged segmentation to select visible object points; results
must be described as oracle-object-query planning unless that dependency is replaced by
a deployable perception module.

### 3.5 Aggregate the MVP decision

The decision aggregator expects exactly one matched Q4D/no-action MPC condition, not the
full Q4D/dense/CEM matrix above. Generate that control report first:

```bash
python scripts/evaluate_mpc.py \
  --config configs/mpc.toml \
  --episodes 10 \
  --models q4d no_action \
  --methods random_shooting \
  --budgets-ms 100 \
  --output artifacts/planning/stop_gate_mpc/report.json
```

Update paths in `configs/stop_gate.toml` if a new experiment namespace was used, then run:

```bash
python scripts/evaluate_stop_gate.py --config configs/stop_gate.toml
python -m json.tool artifacts/decisions/mvp_stop_continue/report.json
```

This item-14 gate is an engineering continuation decision. It is not a substitute for
confidence intervals, multiple training seeds, the original item-12 gate, or held-out
multi-task evaluation.

## 4. Scale PushCube on a larger server

Create new configurations instead of overwriting the MVP protocol:

```bash
cp configs/scale_pushcube.toml configs/server_pushcube_v1.toml
cp configs/data_scale.toml configs/server_data_v1.toml
cp configs/scale_experiment.toml configs/server_experiment_v1.toml
cp configs/mpc.toml configs/server_mpc_v1.toml
cp configs/stop_gate.toml configs/server_stop_gate_v1.toml
```

Give every experiment a unique dataset root, output root, and planning/decision output.
Update all dependent paths consistently.

### Scaling controls

| Goal | Configuration or CLI control | Constraint |
| --- | --- | --- |
| More initial states | `--states` and `minimum_fragments` | Fragments = states × five branches for the current scaled profile |
| More visible points N | `labels.num_points`, `model.n_scene_points`, `dense_queries` | Dense queries should equal stored scene points |
| More sparse queries M | `dataset.num_queries`, `model.sparse_queries` | M must not exceed N |
| Longer horizon | collection `simulation.steps`, `labels.horizon`, model/experiment horizons | A trained horizon cannot exceed the stored horizon |
| Larger model | `model.width` | Re-run matched baselines and parameter audits |
| Larger effective batch | micro-batch/batch and gradient accumulation | Increase only after a memory audit |
| More training | `epochs`, `patience` | Keep selection based on validation data |
| Faster input pipeline | `num_workers`, `prefetch_factor`, `cache_size` | Every worker owns a cache; RAM can multiply quickly |
| More candidate actions | `candidate_branches` or MPC `candidates_per_batch` | Report latency and peak VRAM with accuracy |
| More robust planning estimate | MPC `--episodes` and seed range | Use matched seeds and varied initial states |

On a larger GPU, raise `training.memory_budget_mib` from the original 6 GiB cap while
retaining explicit headroom. Start with a conservative micro-batch, run a short audit,
then increase batch size, N, M, horizon, or width one at a time. Dense prediction grows
much faster than sparse query prediction as N increases.

For very large corpora, `data_scale.toml`'s `cache_size = 10000` can cause each persistent
worker to retain a large in-memory cache. Set a bounded value or zero before increasing
the dataset by an order of magnitude.

### Parallel collection shards

Collection can be split into non-overlapping state ranges. All shards must use the same
configuration and base seed, a shared output directory, distinct state ranges, and
distinct manifest names. Example for 20,000 states across four processes:

```bash
DATA_ROOT=artifacts/datasets/pushcube_server_v1
mkdir -p "$DATA_ROOT" logs

python -u scripts/generate_point_tracks.py --config configs/server_pushcube_v1.toml \
  --profile scaled --start-state 0 --states 5000 --output-dir "$DATA_ROOT" \
  --manifest-name manifest.shard0.json --resume > logs/collect_0.log 2>&1 &

python -u scripts/generate_point_tracks.py --config configs/server_pushcube_v1.toml \
  --profile scaled --start-state 5000 --states 5000 --output-dir "$DATA_ROOT" \
  --manifest-name manifest.shard1.json --resume > logs/collect_1.log 2>&1 &

python -u scripts/generate_point_tracks.py --config configs/server_pushcube_v1.toml \
  --profile scaled --start-state 10000 --states 5000 --output-dir "$DATA_ROOT" \
  --manifest-name manifest.shard2.json --resume > logs/collect_2.log 2>&1 &

python -u scripts/generate_point_tracks.py --config configs/server_pushcube_v1.toml \
  --profile scaled --start-state 15000 --states 5000 --output-dir "$DATA_ROOT" \
  --manifest-name manifest.shard3.json --resume > logs/collect_3.log 2>&1 &

wait

python scripts/merge_collection_shards.py \
  --root "$DATA_ROOT" \
  --states 20000 \
  --branches 5 \
  "$DATA_ROOT"/manifest.shard0.json \
  "$DATA_ROOT"/manifest.shard1.json \
  "$DATA_ROOT"/manifest.shard2.json \
  "$DATA_ROOT"/manifest.shard3.json
```

Begin with two processes and monitor CPU, RAM, storage bandwidth, and renderer stability
before increasing concurrency.

### Use multiple GPUs

Training scripts currently select `cuda:0` and do not implement DDP/FSDP. Multiple GPUs
can still run independent horizons or experimental seeds. `CUDA_VISIBLE_DEVICES` maps the
chosen physical GPU to the script's `cuda:0`.

Prepare the dataset once, then run one horizon per GPU:

```bash
python scripts/run_scale_experiment.py \
  --config configs/server_experiment_v1.toml \
  --prepare-only

mkdir -p logs
CUDA_VISIBLE_DEVICES=0 python scripts/run_scale_experiment.py \
  --config configs/server_experiment_v1.toml --horizons 1 > logs/h1.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python scripts/run_scale_experiment.py \
  --config configs/server_experiment_v1.toml --horizons 2 > logs/h2.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python scripts/run_scale_experiment.py \
  --config configs/server_experiment_v1.toml --horizons 4 > logs/h4.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python scripts/run_scale_experiment.py \
  --config configs/server_experiment_v1.toml --horizons 8 > logs/h8.log 2>&1 &
wait

python scripts/summarize_scale_experiment.py \
  --config configs/server_experiment_v1.toml
```

Do not launch multiple jobs that write to the same horizon/model output directory.

## 5. Multi-task ManiSkill extension

The task-adapter layer now separates semantic entities, preparation, branch actions,
tracked bodies, goals, and task-distance calculations from the collection and MPC
scripts. See [ManiSkill task adapters](docs/task_adapters.md) for the five supported
tasks, verification command, and exact scope of the real-simulator audit.

Do not combine the five pilot datasets and treat that as a multi-task result. The
remaining research work is to:

1. Validate the four added tasks' one-state-success policies across varied initial
   states and improve any task whose pilot success rate is inadequate.
2. Store explicit task/control/action metadata in a combined dataset manifest and add
   masks or separate heads before mixing incompatible action spaces.
3. Define task-balanced training-only normalization and grouped train/validation/test
   splits, including held-out poses, layouts, objects, or tasks where claimed.
4. Add task/contact-specific diagnostics and support more complex multi-object tracking
   where required.
5. Run per-task and joint training with multiple seeds, then evaluate matched
   closed-loop planning episodes and confidence intervals.
6. Replace privileged object segmentation with a deployable perception condition, or
   clearly report it as an oracle-query result.

A practical progression is:

```text
PushCube adapter parity
  -> one single-object task
  -> one contact-rich or multi-object task
  -> joint multi-task training
  -> held-out configuration/task evaluation
```

Keep per-task datasets and reports separate through the larger pilot stage. Create a
combined manifest only after every task has reliable success behavior and passes its
geometry and label audits.

## 6. Publication-grade experiment protocol

The additional pre-submission experiments are frozen in
[`docs/submission_protocol_v1.md`](docs/submission_protocol_v1.md), with the
machine-readable contract in `configs/submission_protocol_v1.toml`. New results use the
versioned `artifacts/submission_v1/` namespace; the initial study remains unchanged.
The corrected discrete gripper schedules and settling cost are documented in
[`docs/gripper_aware_mpc.md`](docs/gripper_aware_mpc.md).

Freeze the protocol before launching the expensive run. At minimum, define:

- tasks, robots, control modes, object/layout distributions, and train/test boundaries;
- dataset scale and the number of independent state groups per task;
- N, M, horizon, width, optimizer, stopping rule, and memory budget;
- at least three independent training seeds per learned model;
- paired planning seeds and enough episodes for confidence intervals;
- static, scene-KNN, action-KNN, no-action, action-shuffled, dense, and Q4D baselines;
- primary metrics and gate thresholds before reading final test results.

Recommended reporting:

- ADE, FDE, per-horizon error, p95 point error, and task/contact/object subgroups;
- per-task results and macro-average across tasks;
- mean and confidence interval across training seeds;
- bootstrap confidence intervals over independent state groups, not individual points;
- success rate and final task distance for closed-loop planning;
- actual p50/p95 planning latency, budget overruns, candidates per second, and peak VRAM;
- accuracy-versus-throughput curves over N, M, horizon, and candidate count;
- oracle-segmentation and deployable-perception planning as separate conditions;
- failure cases, including object loss from view and contact prediction errors.

Do not count the hundreds of points within one trajectory as independent experimental
replicates. The independent unit is normally an initial-state group, environment seed,
or training run.

### Freeze provenance for every run

The repository currently ignores `artifacts/`, so copy important outputs to durable
storage. Before a paper-scale run, make an initial Git commit and require a clean working
tree. Record code, environment, configuration, hardware, and source artifact hashes:

```bash
RUN_ID=$(date -u +%Y%m%dT%H%M%SZ)
META_DIR=run_metadata/$RUN_ID
mkdir -p "$META_DIR/configs"

git rev-parse HEAD > "$META_DIR/git_commit.txt"
git status --short > "$META_DIR/git_status.txt"
python -m pip freeze > "$META_DIR/pip_freeze.txt"
python --version > "$META_DIR/python.txt"
nvidia-smi -q > "$META_DIR/nvidia_smi.txt"
cp configs/server_*.toml "$META_DIR/configs/"
```

At completion, retain at least:

```text
dataset manifest, split manifest, and normalization statistics
all exact TOML configurations
training histories and best checkpoints
all baseline, ablation, cache, memory, and MPC reports
stdout/stderr logs and wall-clock timestamps
Git commit, package lock, GPU/driver information, and random seeds
SHA-256 hashes for datasets, checkpoints, and final reports
```

Use a new experiment ID for every protocol change. Never overwrite a result used in a
table or figure.

## 7. Troubleshooting

### `torch.cuda.is_available()` is false

Check `nvidia-smi`, the installed PyTorch CUDA build, container GPU passthrough if used,
and whether the job scheduler assigned a GPU. Run `python scripts/check_cuda.py` before
debugging the model.

### Rendering fails on a headless server

Run `vulkaninfo --summary`, verify that the NVIDIA Vulkan ICD is visible inside the job
or container, and test `cpu-state`, `cpu-rgbd`, and `gpu-rgbd` separately. Simulator
physics, rendering, and PyTorch CUDA are distinct failure domains.

### CUDA memory gate fails on a large GPU

The training configuration may still contain the original 6 GiB allocation cap. Set a
deliberate server-specific `memory_budget_mib` and `minimum_headroom_mib`; do not remove
the guard entirely.

### Host RAM grows every epoch

Reduce dataset `cache_size`, loader `num_workers`, or `prefetch_factor`. Persistent
workers keep separate caches.

### The final scale summarizer exits non-zero

Inspect every boolean in `gate_report.json`. A failed cache, shuffle, accuracy, memory,
or integrity check is expected to stop the fixed scientific gate. Preserve the failed
report and diagnose it before changing the protocol.

### A new ManiSkill environment crashes in collection or MPC

Changing only `simulation.env_id` is unsupported. Implement and test a task adapter as
described above.

## Further documentation

- [Implementation checklist](docs/checklist.md)
- [Scaled experiment protocol](docs/scale_experiment.md)
- [Dataset loader and split contract](docs/dataset_loader.md)
- [Trajectory label schema](docs/trajectory_labels.md)
- [Baselines and metrics](docs/baselines_and_metrics.md)
- [Micro-Q4D model](docs/micro_q4d.md)
- [Dense baseline](docs/dense_baseline.md)
- [MPC protocol](docs/mpc.md)
- [MVP stop/continue decision](docs/stop_continue_decision.md)

## License and release note

`pyproject.toml` currently identifies this as proprietary research code. Review the
repository license, ManiSkill/SAPIEN licenses, model/data redistribution rights, and
artifact provenance before publishing code, checkpoints, or datasets with a paper.
