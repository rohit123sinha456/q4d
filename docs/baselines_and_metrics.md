# Non-neural baselines and trajectory metrics

Checklist item 7 freezes the prediction evaluation protocol before introducing a
learned model. Run it with:

```bash
python scripts/evaluate_baselines.py
```

The evaluator fits reference statistics only on 20 training state groups (80 branches)
and evaluates three held-out groups (12 branches). It scores all 256 visible points rather
than the 32-query training subset, giving robust moving-object and contact coverage.
The exact split manifest is hashed into the report.

## Baselines

- `static`: repeats the initial point at all eight future times.
- `train_mean_displacement`: adds the mean training trajectory to every point. This is
  an action-free corpus-prior check.
- `scene_knn`: retrieves three training episodes using only an initial visible-scene
  descriptor, then transfers trajectories from nearest visible XYZ+RGB points.
- `action_knn`: uses the same point retrieval, but chooses training episodes by the
  normalized eight-step action chunk. Comparing this with `scene_knn` isolates whether
  the recorded action provides useful information beyond the initial scene.

None of these predictors receives segmentation, body identity, contact, or simulator
state. Audit fields are opened only after prediction to stratify metrics.

## Metrics

Primary errors are measured directly in world metres:

- average displacement error (ADE) over points and future times;
- final displacement error (FDE) at step eight;
- ADE at every horizon and the 95th-percentile point-time error;
- ADE/FDE for points moving more than 1 mm;
- contact, robot, object, static, and goal subgroup ADE/FDE;
- discrete acceleration error for temporal consistency;
- all-query pairwise-distance error;
- same-body pairwise-distance error as an audit-only rigidity diagnostic.

Same-body rigidity must not be interpreted as trajectory accuracy: a static prediction
is perfectly rigid while being wrong about body motion. ADE/FDE and the moving/contact
groups remain primary.

## Current held-out result

The values below are ADE in millimetres; lower is better.

| Baseline | All | Moving | Contact | Object | All FDE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Static | 21.01 | 76.95 | 134.00 | 71.92 | 27.56 |
| Train mean displacement | 28.00 | 75.43 | 127.77 | 72.14 | 36.54 |
| Scene KNN | 21.37 | 64.95 | 93.96 | 73.39 | 28.07 |
| Action KNN | **3.79** | **13.84** | **36.94** | **15.47** | **5.65** |

The grouped corpus reverses the confounding found in the original debug set. Action KNN
now reduces ADE relative to matched scene-only retrieval by 82% overall, 79% for moving
and object points, and 61% around contact. The action chunk therefore contains
identifiable information that the initial scene alone cannot supply.

The full-precision result and per-horizon arrays are saved to
`artifacts/evaluation/counterfactual_non_neural_baselines.json`.
