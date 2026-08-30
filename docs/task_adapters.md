# ManiSkill task adapters

The collection and MPC entry points now select task semantics through
`q4d_wam.tasks.get_task_adapter(env_id)`. Five environments are supported:

| Environment | Adapter | Manipulated object | Goal semantics |
| --- | --- | --- | --- |
| `PushCube-v1` | `push_cube` | cube | goal region center |
| `PullCube-v1` | `pull_cube` | cube | goal region center |
| `PickCube-v1` | `pick_cube` | cube | goal site center |
| `PlaceSphere-v1` | `place_sphere` | sphere | bin placement height |
| `StackCube-v1` | `stack_cube` | lower cube A | top of support cube B |

Each adapter owns its semantic body mapping, preparation motion, five scaled branch
plans, primary-object selection, goal position, and task-distance calculation. The
collector automatically selects the camera that contains the most visible pixels from
the primary object. This is necessary for tasks such as `StackCube-v1`, where the first
registered camera does not always see cube A.

## Run a five-task adapter audit

From the repository root in the configured Python environment:

```bash
python -u scripts/verify_task_adapters.py \
  --output-root artifacts/task_adapter_verification
```

The command collects one shared initial state and five counterfactual fragments for
each task (25 fragments total), validates the training schema and physical labels, and
compares the new PushCube fragments with
`artifacts/datasets/pushcube_scale_v1`. It exits non-zero if any check fails and writes
the complete report to `artifacts/task_adapter_verification/report.json`.

To re-check already collected outputs without running ManiSkill again:

```bash
python scripts/verify_task_adapters.py \
  --output-root artifacts/task_adapter_verification \
  --skip-collection
```

The PushCube parity reference can be changed explicitly:

```bash
python scripts/verify_task_adapters.py \
  --pushcube-reference artifacts/datasets/pushcube_scale_v1
```

## Collect a particular task

The task configurations under `configs/tasks/` are small verification configurations.
Use one as the base for a uniquely named scaled experiment:

```bash
python -u scripts/generate_point_tracks.py \
  --config configs/tasks/pick_cube.toml \
  --profile scaled \
  --states 10 \
  --output-dir artifacts/datasets/pick_cube_pilot \
  --checkpoint-every 2
```

Keep task datasets separate while validating them. The training files retain the same
ten model-facing arrays as PushCube. Task identity, semantic entity trajectories, and
other privileged diagnostics stay in the record JSON and audit archive, preventing
them from leaking into model inputs.

## Verified scope and remaining work

The one-state real-simulator audit verifies that all five adapters produce complete
groups, visible robot/object points, distinct action chunks, consistent schemas,
identical sibling inputs, and meaningful primary-object motion. The PushCube candidate
matches the pre-adapter reference exactly for all model-facing arrays.

The short horizon-eight policies for PullCube, PickCube, PlaceSphere, and StackCube now
complete their task in the deterministic one-state real-simulator audit. This is a
readiness check, not evidence of reliability across randomized initial states or expert
demonstration quality. Before claiming multi-task results, validate the policies on a
larger pilot, collect many independent states, define task-balanced splits and
normalization, train multiple seeds, and report per-task and macro-averaged planning
metrics with confidence intervals.

## Frozen 100-state pilot gate

Apply this gate independently to each added task before collecting 2,000 states. These
criteria are frozen before the pilot results are inspected:

| Check | Required result |
| --- | --- |
| Collection size | 100 complete state groups and 500 fragments |
| Physical and schema checks | 500/500 fragments pass |
| Sibling identity and action diversity | 100/100 groups pass |
| Visibility | Primary-object and robot points are present in every fragment |
| Success policy | At least 90/100 `success` branches complete the environment task |
| Weak/failure/no-op controls | No more than 5% task-success false positives per branch |
| Off-target control | No more than 20% task-success false positives |
| Intended outcomes | At least 475/500 records have `outcome_match = true` |
| Outcome coverage | Successful, weak, off-target, and no-motion classes are all present |
| Task progress | Median final task distance is below median initial distance for the success branch |

A task that misses any criterion remains at policy-development scale. Preserve the pilot
report and revise the policy or outcome classifier before starting its full collection.

Evaluate a completed pilot with:

```bash
python scripts/evaluate_task_pilot.py \
  --root artifacts/datasets/<task>_pilot_v1
```

The command writes `pilot_gate.json` beside the collection manifest and exits non-zero
when any frozen criterion fails.
