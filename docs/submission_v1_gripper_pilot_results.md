# Submission-v1 gripper-aware planner pilot

Date: 1 September 2026

## Decision

The 40-episode planner-validity pilot **fails** its frozen overall gate. Do not start
the definitive MPC matrix from these results. PullCube, PlaceSphere, and StackCube
pass their task gates; PickCube fails because none of its ten episodes end with the
cube grasped.

The pilot used Q4D, random shooting, horizon 8, a 100 ms planning budget, and ten
frozen episode seeds per task. The aggregate machine-readable result is
`artifacts/submission_v1/planning/gripper_aware_pilot_v1/pilot_gate.json`.

## Results

| Task | Simulator successes | Genuine successes | Improved episodes | Mean improvement | Median improvement | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| PullCube | 10/10 | 10/10 | 10/10 | 11.13 cm | 10.88 cm | Pass |
| PickCube | 1/10 | 0/10 | 9/10 | 2.30 cm | 1.03 cm | **Fail** |
| PlaceSphere | 5/10 | 5/10 | 6/10 | 1.19 cm | 5.57 cm | Pass |
| StackCube | 4/10 | 4/10 | 6/10 | 1.05 cm | 2.61 cm | Pass |

All 40 evaluated episodes used valid executable 7D actions. Every task received the
complete five-schedule candidate library, including closed and release candidates.
All 40 trajectory contact sheets were written. The final PlaceSphere and StackCube
successes released their objects, ended ungrasped, and satisfied the frozen 0.05 m/s
settling threshold.

## PickCube diagnosis

PickCube's only simulator success was seed 14610. It reduced cube-goal distance from
0.11760 m to 0.01047 m, but ended with `is_grasped=false`. The cube was grasped after
control cycle 10; cycle 11 selected `open_to_closed_halfway`, whose receding-horizon
first action is open (`+1`), and the cube was released immediately before the simulator
declared success.

This is consistent with the configured planner objective. It scores predicted final
object-centroid distance to the task goal plus translation-action regularization for
PickCube. It does not score grasp retention or penalize an executed release. The
candidate library is therefore semantically executable, but the PickCube objective is
not aligned with the checklist's requirement that the cube remain grasped. Changing
that objective or restricting schedules now would be a new, versioned protocol
amendment and must be followed by a fresh matched-seed pilot; it must not be applied to
these results post hoc.

## Visual inspection

Representative successful and failed trajectories were inspected as contact sheets:

- PullCube seed 13601 visibly moved the cube toward the target and agreed with its
  recorded success.
- PlaceSphere seed 15603 released the sphere at the receptacle and settled; seed 15601
  released it away from the receptacle. Both agreed with the recorded outcomes.
- StackCube seed 16601 formed and released a stable stack; seed 16603 did not form a
  stable stack. Both agreed with the recorded outcomes.
- PickCube seed 14610 visibly ended after release near the goal, confirming that its
  simulator success is not a genuine PickCube success. Seed 14602 remained a failure.

## Implementation incidents and provenance

The first PullCube attempt failed before executing an action because its training data
has a constant gripper channel and the newly introduced open command overflowed after
normalization in FP16. The failed attempt is preserved under
`artifacts/submission_v1/planning/gripper_aware_pilot_v1/pull_cube/`; the narrowly
scoped correction and versioned retry are documented in
`docs/submission_v1_pilot_amendment_1.md`.

The first aggregate evaluator incorrectly applied the placement/stacking settling
threshold to PullCube and PickCube. No episode output changed. The task-specific gate
correction is documented in `docs/submission_v1_pilot_amendment_2.md`.

Relevant commits are:

- `6a6d5864a4f6f9e4bf163b6e2a44f3b3b0af7b33`: frozen pilot harness and the first
  three completed task reports.
- `1b5965035d558c89d25cea855ae8e8e7d975a041`: PullCube constant-channel correction
  and versioned retry provenance.
- `f996b4c`: task-specific analysis-gate correction.

The final verification suite contains 67 passing tests.
