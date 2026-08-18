# Persistent 3D point-trajectory labels

Checklist item 5 is implemented for ManiSkill `PushCube-v1`.

## Label construction

For every episode:

1. Reset the scene with a deterministic seed.
2. Move the Panda TCP to the official PushCube pre-push geometry: 5 cm behind the cube.
3. Capture one RGB, metric depth, segmentation, and camera-calibration observation.
4. Backproject valid pixels within 1.5 m into world coordinates.
5. Preserve rare semantic regions with stratified farthest-point sampling:
   - up to 24 cube points;
   - 64 robot points;
   - 16 goal points;
   - remaining capacity from the visible scene.
6. Match each selected pixel's actor segmentation ID to a simulator entity.
7. Convert the point from world coordinates into that body's local frame.
8. Execute eight low-level `pd_ee_delta_pose` actions toward the push goal.
9. Record every attached body's pose after each action.
10. Transform the unchanged body-local point through those poses to produce its persistent
    world trajectory.

Segmentation IDs and body poses are privileged supervision. They are not included in the
model-facing training files.

## Scene mapping

The observed PushCube scene uses:

| IDs | Category |
| --- | --- |
| 1–15 | Panda links |
| 16 | table workspace |
| 17 | ground |
| 18 | cube |
| 19 | goal marker |

The registry is constructed from the live simulator entities instead of hard-coding these
values into the label algorithm.

## File separation

Each fragment produces two files.

`episode_NNNNNN.train.npz` contains only information usable by a future model:

- RGB and depth observation;
- camera intrinsics and extrinsics;
- eight normalized robot actions;
- sampled pixel coordinates and colors;
- initial world XYZ;
- eight future world XYZ targets.

`episode_NNNNNN.audit.npz` contains privileged data for label validation:

- point and body segmentation IDs;
- body indices and categories;
- body-local XYZ;
- body pose sequence;
- trajectories including time zero;
- approximate contact-region mask;
- cube-centre trajectory.

## Validation invariants

Every generated fragment checks that:

- all sampled points attach to known rendered bodies;
- reconstructing time zero returns the observed point within 10 micrometres;
- static points do not move;
- pairwise distances on each body are preserved within 10 micrometres;
- the cube moves at least 2 mm;
- object and robot points are present;
- every trajectory coordinate is finite.

## Grouped counterfactual development corpus

The current `pushcube_counterfactual_v1` corpus replays each initial state four times.
The initial RGB-D observation, calibration, sampled points, and point colors must be
byte-identical across siblings. Only the action chunk and its resulting future differ:

- `success`: push toward the goal and hold after the first success signal;
- `perturbed`: push toward a deterministically offset lateral target;
- `no_op`: hold the end effector in place;
- `failure`: retreat away from the cube.

The corpus contains:

| Property | Result |
| --- | ---: |
| Shared initial states | 25 |
| Branches per state | 4 |
| Fragments | 100 |
| Passed label validation | 100 / 100 |
| Passed sibling identity validation | 25 / 25 |
| Successful nominal pushes | 25 / 25 |
| Successful perturbed pushes | 25 / 25 |
| Successful no-op/retreat branches | 0 / 50 |
| Points per fragment | 256 |
| Future steps | 8 |
| Train/validation/test state groups | 20 / 2 / 3 |
| Train/validation/test fragments | 80 / 8 / 12 |

The contact-region flag is a sampling-time proximity heuristic based on the selected robot
surface points; it is not used to establish trajectory correctness.

Generate and prepare the corpus with:

```bash
python scripts/generate_point_tracks.py --config configs/counterfactual.toml --states 25
python scripts/prepare_dataset.py
```
