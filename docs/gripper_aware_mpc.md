# Gripper-aware MPC action space

Submission-v1 corrects the historical MPC sampler's action-space defect. The original
sampler generated only XYZ delta translations and set the seventh action channel to
`-1` at every horizon step. Those reports and `configs/mpc*.toml` files remain the
translation-only ablation and are not overwritten.

## Candidate construction

The corrected sampler jointly pairs each smooth bounded XYZ trajectory with one member
of a shared discrete gripper library:

| Schedule | H=8 commands |
| --- | --- |
| `hold_closed` | `-1 -1 -1 -1 -1 -1 -1 -1` |
| `hold_open` | `+1 +1 +1 +1 +1 +1 +1 +1` |
| `closed_to_open_halfway` | `-1 -1 -1 -1 +1 +1 +1 +1` |
| `closed_to_open_final_quarter` | `-1 -1 -1 -1 -1 -1 +1 +1` |
| `open_to_closed_halfway` | `+1 +1 +1 +1 -1 -1 -1 -1` |

Every candidate batch is shuffled but contains every schedule at least once. The same
sampler and schedule names are passed to Q4D, dense, and no-action. The model and episode
seed continue to produce identical first candidate batches across models; faster models
may evaluate more subsequent batches under the wall-clock budget. CEM retains the same
library and updates a categorical distribution from elite schedules while preserving at
least one candidate per schedule in every batch.

Generated actions have shape `[candidates, H, 7]`, finite values in `[-1, 1]`, XYZ in
channels 0–2, zero delta rotation in channels 3–5, and a library gripper command in
channel 6. The planner validates these invariants before model evaluation. The evaluator
also validates every executed 7D action and records the selected schedule and executed
gripper command per control cycle.

## Placement and stacking stability

PlaceSphere and StackCube add a final-state settling term with weight 1.0 over the last
two predicted centroid transitions:

```text
cost = final object-goal distance
     + mean final-two object-centroid motion
     + 0.0001 * mean squared XYZ action
```

PullCube and PickCube keep the settling weight at zero. This makes the primary corrected
condition explicit while retaining the final-distance-only comparison for the planned
ablation.

## Versioned configurations

The new pilot configurations are:

- `configs/submission_v1/mpc_pull_cube_gripper_pilot.toml`
- `configs/submission_v1/mpc_pick_cube_gripper_pilot.toml`
- `configs/submission_v1/mpc_place_sphere_gripper_pilot.toml`
- `configs/submission_v1/mpc_stack_cube_gripper_pilot.toml`

They write only below
`artifacts/submission_v1/planning/gripper_aware_pilot_v1/`. Their `[ablation]` sections
point to the preserved translation-only config and report for the same task. The pilot
defaults to Q4D only, yielding the checklist's 40 episodes; the evaluator applies the
identical library whenever dense and no-action are requested for later matched matrices.
