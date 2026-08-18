# MVP stop/continue decision

## Decision

**CONTINUE.** All four original MVP hypotheses pass the operational item-14 gate. This
authorizes the next scoped experiment; it does not justify skipping matched controls or
claiming statistical finality from ten planning seeds.

The numeric thresholds were formalized for this engineering decision after the MVP
artifacts existed, rather than preregistered before training. They are now stored in
`configs/stop_gate.toml` and should remain frozen for reruns.

| Gate | Evidence | Result |
|---|---|---|
| Action conditioning | H=8 Q4D reduces ADE versus no-action by 93.9% overall, 94.5% moving, 87.2% contact, and 90.9% object | Pass |
| Scene caching | Exact-index grid speedup is 1.43x minimum, 1.96x median, and 2.53x maximum over nine N/M settings | Pass |
| Dense competitiveness | Q4D/dense ADE is 1.194 overall, 1.018 contact, and 1.001 object; limits are 1.25 overall and 1.10 task-relevant | Pass |
| MPC translation | Matched 100 ms random shooting is 7/10 success for Q4D versus 0/10 for no-action; final distance is 111.1 versus 200.3 mm | Pass |

## Cache discrepancy diagnosis

The older H=8 single cache benchmark reported a 0.48x speedup, contradicting the later
N/M grid. The audit found two measurement issues:

1. Coordinate-nearest query lookup under FP16 could select a different point when
   duplicate or nearly identical coordinates tied.
2. A unitless `1e-3` normalized equality threshold was too strict for different CUDA
   batch shapes, whose FP16 matrix reductions need not be bit-identical.

The corrected benchmark passes identical point indices to both paths and reports the
output discrepancy in metres. Across all nine configurations, the maximum discrepancy is
0.136 mm, below the frozen 0.5 mm mixed-precision tolerance and far below the roughly
1 mm Q4D ADE. Every corrected configuration favors caching. The old result remains in
the source report and is explicitly recorded by the decision artifact rather than being
silently discarded.

## Dense and planning interpretation

Dense prediction retains a modest overall accuracy advantage, so Q4D does not win every
axis. It is nevertheless competitive on the task-relevant cube and contact subsets and
buys substantially more candidate evaluation under fixed planning time. Item 13's random
shooting results favor Q4D over dense at every tested budget.

CEM remains unstable at ten seeds: dense wins at 50 ms, the models tie at 100 ms, and Q4D
wins at 200 ms. Continue with random shooting as the trusted baseline while treating CEM
convergence as a separate diagnosis. Denser queries or explicit contact-event prediction
remain sensible fallbacks if object/contact accuracy or transfer to harder interactions
fails later; they are not required to rescue this PushCube MVP.

Run the deterministic artifact aggregation with:

```bash
python scripts/evaluate_stop_gate.py
```

The machine-readable decision is written to
`artifacts/decisions/mvp_stop_continue/report.json`. The matched no-action planning
control is stored in `artifacts/planning/stop_gate_mpc/report.json`.
