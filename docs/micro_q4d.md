# Micro-Q4D action-conditioned baseline

Checklist item 9 adds executable action conditioning while holding the item-8 scene
encoder, visible query inputs, width, grouped splits, normalized MSE objective, and dense
evaluation protocol fixed:

```bash
python scripts/train_micro_q4d.py
```

## Cacheable interface

The 250,883-parameter model separates action-independent and action-dependent work:

```python
scene = model.encode_scene(scene_xyz, scene_rgb)
queries = model.encode_queries(scene, query_xyz)
future = model.predict_candidates(queries, candidate_actions)
```

`candidate_actions` has shape `[B, K, 8, 7]`, and the returned normalized displacement
field has shape `[B, K, Q, 8, 3]`. Scene and query encoding run once; a GRU encodes each
action prefix, and a query decoder fuses query features, action features, and their
elementwise interaction at every future time.

The ordinary `forward(scene_xyz, scene_rgb, actions, query_xyz)` path uses the same
components. Unit tests verify that it agrees with cached decoding and that changing only
the action changes the predicted trajectory.

## Training

- 80 training, 8 validation, and 12 test branches from disjoint state groups;
- 32 sparse queries per training fragment and all 256 points at test time;
- width 128, AdamW, mixed-precision CUDA, and best-validation checkpoint selection;
- 160 epochs in 112.4 seconds on the RTX 4060;
- best epoch 157 and peak training allocation of 32.1 MiB.

## Held-out accuracy

| Predictor | All ADE | Moving ADE | Contact ADE | Object ADE | All FDE |
| --- | ---: | ---: | ---: | ---: | ---: |
| No-action neural | 21.96 mm | 65.22 mm | 94.61 mm | 72.63 mm | 28.58 mm |
| Action KNN | 3.79 mm | 13.84 mm | **36.94 mm** | **15.47 mm** | 5.65 mm |
| **Micro-Q4D** | **3.41 mm** | **11.26 mm** | 39.04 mm | 17.70 mm | **5.37 mm** |

Relative to the controlled no-action network, micro-Q4D reduces ADE by 84.5% overall,
82.7% on moving points, 58.7% near contact, and 75.6% on cube points. It slightly beats
action KNN overall and on moving points, while KNN remains better on the small contact
and object subsets. Those two subsets are the immediate targets for later loss and data
ablations, not grounds for changing the item-9 protocol after seeing the test result.

The mean pairwise trajectory separation among predictions for sibling actions is
31.78 mm, versus 31.93 mm in the targets. The no-action model's sibling separation is
exactly zero. This establishes that the network uses the action chunk to produce
counterfactual futures.

## Candidate benchmark

For 64 action branches with 256 queries per branch, averaged over 30 repetitions:

| Path | Latency | Candidates/s | Peak CUDA allocation |
| --- | ---: | ---: | ---: |
| Re-encode scene/query per branch | 5.98 ms | 10,702 | 351.3 MiB |
| Reuse scene/query cache | **5.16 ms** | **12,405** | **343.1 MiB** |

Caching provides a 1.16× speedup and saves about 8.2 MiB in this fully batched micro
benchmark. This is a valid but deliberately modest result: at width 128, decoding
`64 × 256 × 8` query-times dominates the very small scene encoder. Future experiments
must test whether the advantage grows with a stronger encoder and sparse planning query
sets; the current result does not yet establish a large MPC-level efficiency gain.

The best checkpoint, full training history, metrics, and benchmark are stored under
`artifacts/models/micro_q4d_v1/`.
