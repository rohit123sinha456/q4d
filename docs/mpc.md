# Simple PushCube MPC

Item 13 adds closed-loop model-predictive control using the trained H=8 Q4D and
parameter-matched dense checkpoints. The implementation deliberately starts with batched
random shooting, then applies the same model adapter and evaluation protocol to CEM.

## Control cycle

Each cycle performs the following operations:

1. Observe fresh RGB, metric depth, segmentation, and camera calibration.
2. Backproject 512 sampled visible points into world coordinates.
3. Encode normalized XYZ/RGB once and form one reusable query cache.
4. Evaluate batches of 64 eight-step executable action candidates until the wall-clock
   deadline.
5. Score final predicted cube-surface centroid distance to the simulator target, plus a
   small translation-action penalty.
6. Execute only the first action from the best sequence.
7. Reobserve and repeat for at most 12 cycles.

Segmentation selects the visible cube surface points used by the planning cost. It is
never passed to either predictor. Q4D decodes only those object queries (up to 48), while
the dense baseline decodes all 512 scene points and is scored at the identical object
indices. Both models use the same planner seeds, candidate batch size, target, success
criterion, and time budget.

Random shooting samples bounded, temporally smooth Cartesian translations. CEM refits a
diagonal Gaussian from the lowest-cost 10 percent of each batch. Both maintain the best
candidate seen before the deadline. The deadline includes scene normalization, transfer,
and the single scene encoding; one final batch may produce a small soft-deadline overrun.
CUDA and the renderer-to-model path are warmed before measurement, while genuine control
cycle costs remain included.

## Matched benchmark

The benchmark uses seeds 2601 through 2610 for every condition, for 120 episodes total.
All report integrity checks passed: every requested condition completed, each cycle has
one scene encoding, only one action is executed per plan, metrics are finite, seeds match,
and both checkpoints exist.

| Planner | Budget | Q4D success | Dense success | Q4D candidates/cycle | Dense candidates/cycle | Throughput ratio |
|---|---:|---:|---:|---:|---:|---:|
| Random | 50 ms | 7/10 | 3/10 | 822 | 312 | 2.64x |
| Random | 100 ms | 8/10 | 2/10 | 2,713 | 619 | 4.38x |
| Random | 200 ms | 8/10 | 3/10 | 6,765 | 1,201 | 5.63x |
| CEM | 50 ms | 4/10 | 7/10 | 715 | 320 | 2.23x |
| CEM | 100 ms | 5/10 | 5/10 | 2,073 | 615 | 3.37x |
| CEM | 200 ms | 6/10 | 4/10 | 6,711 | 1,177 | 5.70x |

Random shooting provides the clearest support for queryable prediction under a fixed
planning budget: Q4D evaluates substantially more actions and obtains higher success at
all three budgets. CEM is more variable and does not improve monotonically with budget at
this ten-seed scale. The result should therefore be read as an implementation milestone
and an initial matched comparison, not a statistically final planning claim.

Configuration is in `configs/mpc.toml`. The complete per-cycle and per-episode record is
written to `artifacts/planning/mpc_v1/report.json`; resumable intermediate episodes are in
the adjacent `episodes.json`.

Run the benchmark from the configured WSL environment:

```bash
python scripts/evaluate_mpc.py
```

Useful CLI overrides include `--episodes`, `--models`, `--methods`, `--budgets-ms`, and
`--output` for short pilots or isolated comparisons.

For the item-14 causal control, the no-action network was given the same cached scene and
cube queries but intentionally could not use candidate actions. At 100 ms random
shooting, Q4D succeeded on 7/10 matched seeds while no-action succeeded on 0/10. The
no-action controller's mean final cube-target distance remained 200.3 mm, close to the
initial 200.9 mm, versus 111.1 mm for Q4D.
