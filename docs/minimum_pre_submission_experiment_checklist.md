## Minimum pre-submission experiment checklist

Run everything in new versioned output directories. Preserve the current results unchanged as the initial study.

### 1. Freeze the submission protocol

- [ ] Write the final research question and allowed claims before rerunning anything.
- [ ] Declare H=8 prediction and 100 ms random-shooting MPC as the primary conditions.
- [ ] Keep H=1/2/4 and other MPC budgets as secondary analyses.
- [ ] Define task and training seed—not individual points—as replicate units.
- [ ] Freeze metrics, thresholds, statistical tests, exclusions, and failure handling.
- [ ] Record the environment, hardware, package versions, and exact commit.
- [ ] Do not change gates after examining new test results.

### 2. Make the MPC action space semantically valid

The current sampler always fixes the gripper command to `-1`, making PlaceSphere and StackCube completion effectively impossible.

- [ ] Add discrete gripper schedules to candidate trajectories:

  - Hold closed.
  - Hold open.
  - Closed → open halfway.
  - Closed → open during the final quarter.
  - Open → closed where relevant.

- [ ] Sample translations and gripper schedules jointly.
- [ ] Use the same candidate schedule library for Q4D, dense, and no-action.
- [ ] Add a final-state stability or settling term for placement and stacking.
- [ ] Verify candidates remain valid executable 7D actions.
- [ ] Add deterministic sampler tests and fixed-seed reproducibility tests.
- [ ] Preserve the existing translation-only MPC results as an ablation.

### 3. Run a planner-validity pilot

Before another full MPC matrix:

- [ ] Run 10 matched seeds per task at 100 ms with Q4D and random shooting.
- [ ] Confirm every task receives both closed and release candidates.
- [ ] Confirm PickCube can remain grasped.
- [ ] Confirm PlaceSphere and StackCube can release and settle.
- [ ] Confirm final task distance generally improves.
- [ ] Confirm success is possible on all four tasks.
- [ ] Inspect several successful and failed trajectories visually.
- [ ] Stop and diagnose if PlaceSphere or StackCube still has zero success.

Pilot gate:

- [ ] No implementation-check failures.
- [ ] No invalid actions.
- [ ] At least one genuine success per task.
- [ ] Task success agrees with recorded gripper/object behavior.

### 4. Run multi-seed H=8 prediction training

Use the existing frozen splits and normalization files.

- [ ] Train three independent seeds for each task and neural model:

  - Micro-Q4D
  - No-action
  - Parameter-matched dense

- [ ] Count the existing seed as seed one if its configuration is unchanged.
- [ ] Run two additional seeds: 24 new training runs.
- [ ] Preserve identical optimizer, batch size, epochs, patience, and evaluation protocol.
- [ ] Run action-shuffle and decoding benchmarks for every Q4D seed.
- [ ] Report every seed rather than only the best one.

Report per task and task-macro:

- [ ] ADE, FDE, contact ADE, object ADE, and p95 error.
- [ ] Q4D improvement over no-action.
- [ ] Q4D/dense accuracy ratio.
- [ ] Mean, standard deviation, and 95% confidence interval.
- [ ] Paired seed-level differences.

### 5. Reconcile the cache benchmarks

- [ ] Define one primary cached-versus-re-encoding protocol.
- [ ] Use exact query indices in both paths.
- [ ] Use identical scenes, queries, actions, batch shapes, AMP settings, and synchronization.
- [ ] Separate warm-up from timed repetitions.
- [ ] Run at least five timing trials of 30 repetitions per configuration.
- [ ] Report latency distributions, not only means.
- [ ] Report output differences in metres with the frozen mixed-precision tolerance.
- [ ] Repeat all four horizons for all four tasks.
- [ ] Repeat the H=8 N/M grid.
- [ ] Preserve the contradictory original benchmark in the appendix.

Compute gate:

- [ ] Numerical equivalence passes.
- [ ] The confidence interval for cache speedup is above 1 under the declared primary protocol.
- [ ] Sparse decoding remains faster than dense decoding.

### 6. Run the definitive closed-loop MPC experiment

Recommended minimum primary matrix:

- Four tasks
- Three models
- Random shooting
- 100 ms
- 30 matched episode seeds

That is 360 primary episodes.

- [ ] Use the corrected gripper-aware action space.
- [ ] Use identical episode and candidate-generation seeds across models.
- [ ] Run Q4D, dense, and no-action.
- [ ] Save each episode immediately for resumability.
- [ ] Add a fixed-candidate-count control to separate accuracy from throughput.
- [ ] Optionally retain 50/200 ms and CEM as secondary results.
- [ ] Explicitly label all planning as `oracle-object-query planning`.

Report:

- [ ] Success rate with confidence intervals.
- [ ] Initial and final task distance.
- [ ] Paired per-seed distance improvement.
- [ ] p50/p95 planning latency.
- [ ] Budget-overrun rate and magnitude.
- [ ] Candidates per second.
- [ ] Control cycles.
- [ ] Object-visibility failures.
- [ ] Gripper schedule selected.
- [ ] Termination reason.

### 7. Run the key planning ablations

- [ ] Translation-only versus gripper-aware MPC on identical seeds.
- [ ] Q4D versus no-action to establish action-conditioning causality.
- [ ] Q4D versus dense under the wall-clock budget.
- [ ] Q4D versus dense at a fixed candidate count.
- [ ] Final-distance-only versus final-distance-plus-settling cost.
- [ ] Record whether improvements come from prediction quality, throughput, or action-space coverage.

### 8. Statistical analysis

- [ ] Use paired tests because models share episode seeds.
- [ ] Bootstrap across tasks and seeds, not points or control cycles.
- [ ] Report confidence intervals and effect sizes.
- [ ] Correct for multiple comparisons or clearly designate one primary comparison.
- [ ] Treat budgets and planners as repeated conditions, not independent replicates.
- [ ] Report failures and zero-success tasks without exclusion.

### 9. Submission decision gate

A positive workshop paper is ready if:

- [ ] Multi-seed Q4D reliably beats no-action.
- [ ] Q4D remains close to dense accuracy under the frozen threshold.
- [ ] The unified compute benchmark supports the efficiency claim.
- [ ] Gripper-aware MPC achieves nontrivial success on more than PullCube.
- [ ] Q4D clearly improves over no-action in success or final distance.
- [ ] Q4D provides higher throughput than dense with comparable planning outcomes.
- [ ] All oracle segmentation and simulation limitations are prominent.

If planning remains weak:

- [ ] Reframe the paper around multi-task action-conditioned prediction and sparse decoding.
- [ ] Present MPC as a diagnostic negative result.
- [ ] Do not claim general-purpose or cross-task planning.

The smallest credible workload is therefore **24 additional neural training runs, a 40-episode planner pilot, and 360 definitive MPC episodes**, plus the unified timing benchmark.
