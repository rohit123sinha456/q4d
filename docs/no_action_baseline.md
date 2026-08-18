# No-action neural baseline

Checklist item 8 establishes the learned action-free ceiling before implementing an
action-conditioned Q4D model:

```bash
python scripts/train_no_action.py
```

## Architecture

`NoActionTrajectoryModel` has 86,680 parameters. A shared MLP encodes each normalized
XYZ+RGB scene point. Mean and max pooling form a global scene context, while nearest-point
lookup supplies a local visible feature for each query. A query MLP predicts eight
normalized 3D displacements.

Its forward signature is deliberately limited to `scene_xyz`, `scene_rgb`, and
`query_xyz`; no action argument exists. For the four siblings of a held-out state, the
model produces bit-identical predictions even though their targets diverge.

## Training

- 20 state groups / 80 branches for training;
- 2 groups / 8 branches for validation;
- 3 groups / 12 branches for final testing;
- 32 deterministic visible queries per training fragment;
- AdamW, mixed-precision CUDA, width 128, maximum 120 epochs;
- best-validation checkpoint selection.

The RTX 4060 run completed in 111.7 seconds, peaked at 23.4 MiB of allocated CUDA memory,
and selected epoch 106. Dense 256-query evaluation ran at approximately 194k queries/s;
this is model prediction time and excludes data loading and metric aggregation.

## Held-out result

| Predictor | All ADE | Moving ADE | Contact ADE | Object ADE | All FDE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Static | **21.01 mm** | 76.95 mm | 134.00 mm | **71.92 mm** | **27.56 mm** |
| Scene KNN | 21.37 mm | **64.95 mm** | **93.96 mm** | 73.39 mm | 28.07 mm |
| No-action neural | 21.96 mm | 65.22 mm | 94.61 mm | 72.63 mm | 28.58 mm |
| Action KNN | 3.79 mm | 13.84 mm | 36.94 mm | 15.47 mm | 5.65 mm |

The neural baseline learns a scene-conditioned average future: it improves substantially
over static prediction on moving and contact points, but introduces small false motion
on static points and cannot choose among sibling action outcomes. Its near-match to scene
KNN and large gap to action KNN are the intended controls. A future action-conditioned
model must materially beat these action-free results, especially on moving, contact, and
object points.

Full history, checkpoint, and metrics are saved under `artifacts/models/no_action_v1/`.
