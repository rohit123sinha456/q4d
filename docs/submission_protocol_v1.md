# Submission protocol v1

Status: **frozen on 1 September 2026, before examining any new submission-v1 test
result**. The machine-readable source of truth is
`configs/submission_protocol_v1.toml`. The current PushCube and four-task `*_scale_v1`
artifacts are prior exploratory evidence and remain unchanged; all new outputs go under
`artifacts/submission_v1/`.

This freeze is based on repository commit
`298538a238342271754ce2b07f6389a5d40b7aaf`. Result-producing runs must additionally
record their own clean Git commit and provenance snapshot. If code changes after this
freeze, the run snapshot—not this base commit—is the exact implementation identifier.

## Research question and claim boundary

Across PullCube-v1, PickCube-v1, PlaceSphere-v1, and StackCube-v1, does an
action-conditioned sparse queryable 4D predictor improve H=8 held-out 3D trajectory
prediction over a matched no-action model, remain within frozen accuracy margins of a
parameter-matched dense predictor, and convert sparse decoding throughput into better
100 ms random-shooting oracle-object-query planning outcomes?

The sole primary confirmatory comparison is H=8 task-macro overall ADE for Micro-Q4D
versus no-action. Planning, dense competitiveness, action shuffling, and compute claims
are prespecified supporting claims with their own gates. A claim is made only when its
gate passes; a failed task or metric is named rather than averaged away.

Allowed scope is limited to separately trained models on the four declared ManiSkill
tasks and their existing frozen splits. Planning always carries the label
**oracle-object-query planning**, because simulator segmentation supplies object query
points. The study does not establish deployable perception, joint or cross-task
generalization, general-purpose planning, or planner superiority outside H=8, 100 ms
random shooting. Action-conditioning is described as causal evidence only if both the
no-action and action-shuffle controls pass.

PushCube is the preserved initial study. Its translation-only MPC remains an ablation;
it is not pooled into the four-task submission-v1 confirmatory analysis.

## Frozen conditions

The primary prediction condition is H=8 with N=512 scene points, M=64 sparse queries,
and the existing per-task split and training-only normalization files. Micro-Q4D,
no-action, and dense use width 128, batch size 32, 30 epochs, patience 6, Adam settings
already encoded by the task experiment configs, AMP, and validation-loss checkpoint
selection. Each task/model has three independent training seeds: the unchanged existing
seed and the next two integers.

The primary planning condition is gripper-aware H=8 random shooting at 100 ms. The
planner compares all three models using identical episode and candidate-generation seeds
within task. The pilot uses ten seeds per task. The definitive experiment uses 30 new,
disjoint seeds per task/model and includes a fixed-candidate-count control. H=1/2/4,
50/200 ms, CEM, and other budgets are secondary repeated conditions.

The four task-specific training and planning seed lists are frozen in the TOML file.
Pilot and definitive planning seeds are deliberately disjoint so visually inspected
pilot trajectories cannot enter the definitive estimate.

## Replicates and metrics

For prediction, the replicate is task × training seed. For planning, it is task ×
episode seed. For the cache benchmark, it is an independently repeated paired timing
trial. Points, horizons, control cycles, candidates, budgets, and planners are repeated
measurements—not independent replicates. Task-macro summaries weight the four tasks
equally after within-task aggregation.

Primary prediction error is overall ADE in world metres at H=8. Frozen secondary
prediction metrics are overall FDE, contact ADE, object ADE, p95 point-time error, Q4D
improvement over no-action, and the Q4D/dense accuracy ratio. A point is “moving” only
when its maximum target displacement is greater than 1 mm.

The primary planning metric is the paired difference in episode distance improvement:
`(initial - final task distance)_Q4D - (initial - final task distance)_no-action`.
Secondary reporting includes success, initial/final distance, p50/p95 planning latency,
budget-overrun rate and positive magnitude, candidates per second, cycles, visibility
failures, selected gripper schedule, and termination reason. Cycle metrics may be shown
descriptively, but uncertainty is clustered at task × episode seed.

Cache and re-encoding paths use identical scenes, exact query indices, actions, batch
shapes, AMP, and synchronization. Warm-up is separate. Each configuration uses at least
five paired trials of 30 timed repetitions. Numerical differences are reported in metres
and must be at most 0.5 mm.

## Frozen inference and gates

All intervals are 95%. The main uncertainty estimator is a paired hierarchical
percentile bootstrap with 10,000 resamples and seed 240901: tasks are resampled first,
then matched replicate seeds/trials within task or configuration. Per-task success rates
also receive Wilson intervals. The primary test is a one-sided exact paired sign-flip
randomization test on the 12 H=8 task × training-seed overall-ADE differences. Secondary
families—prediction, planning, and compute—use Holm correction at familywise alpha 0.05.

The primary claim requires a negative mean Q4D-minus-no-action ADE difference, a paired
95% interval wholly below zero, and exact-test p < 0.05. Action shuffle must produce a
Holm-adjusted paired interval wholly above unshuffled Q4D ADE.

Dense competitiveness requires the upper 95% confidence bound of Q4D/dense ADE to be no
more than 1.25 overall and no more than 1.10 for both contact and object points. These
margins carry forward the documented MVP gate; they were not derived from new
submission-v1 results.

Cache and sparse-decoding speedups each require a paired 95% confidence lower bound
above 1.0. The planner pilot requires valid actions and at least one genuine success per
task. “Nontrivial” definitive success means at least 10% success for the named task.
Q4D beats no-action in planning only if either its paired distance-improvement interval
is wholly above zero or its superiority in matched success passes the Holm-adjusted
exact McNemar test. The passing branch must be named.

For the throughput-with-comparable-outcomes claim, Q4D/dense candidate throughput must
have a lower 95% bound above 1.0; the Q4D-minus-dense success lower bound must be at least
-0.10 and its final-distance upper bound at most +0.01 m.

## Exclusions and failure handling

There are no post-hoc exclusions. Every valid frozen test group and every requested
task/model/seed is reported. A missing or non-finite required prediction metric fails
the affected gate.

Every started planning episode is retained. Object loss is an unsuccessful episode with
the last valid distance. Control-cycle limits, timeouts, and budget overruns remain in
the analysis. An invalid action fails the implementation gate and blocks a positive
planning claim. A simulator error after the first executed action is a recorded failed
episode, not a replacement opportunity.

An infrastructure interruption before measurement may be rerun from scratch using the
same inputs and seed, with both logs retained. After measurement begins, OOM, NaN,
missing-checkpoint, and missing-metric failures are not replaced with a more favorable
seed. Zero-success tasks are reported and fail that task's positive planning claim.

## Environment and amendment rule

The freeze environment is Ubuntu 24.04.4 under WSL2 kernel 6.6.87.2, Python 3.12.3,
PyTorch 2.11.0+cu128 (CUDA runtime 12.8), ManiSkill 3.0.1, SAPIEN 3.0.3, and an NVIDIA
GeForce RTX 4060 Laptop GPU with 8,188 MiB and driver 592.82. Exact packages are pinned
in `requirements/lock-wsl-cu128.txt` and copied into the provenance snapshot. Raw timing
results from different hardware are not pooled.

Run the freeze/audit command before result-producing work:

```bash
/home/mrogwm/.venvs/q4d/bin/python scripts/freeze_submission_protocol.py
```

The default command creates the one-time protocol freeze and refuses to overwrite it.
After committing the protocol and each later implementation change, capture that run's
clean commit in its own directory, for example:

```bash
/home/mrogwm/.venvs/q4d/bin/python scripts/freeze_submission_protocol.py \
  --output artifacts/submission_v1/run_metadata/<versioned-run-id>
```

If the protocol must change after any new test result is examined, create a dated
amendment and a new protocol/output version, explain the reason without consulting
additional results, and rerun every affected condition. Never edit a gate in place.
