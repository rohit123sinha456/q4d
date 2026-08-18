# Implementation checklist status

Updated 18 August 2026.

## 1. WSL and GPU environment

- [x] Ubuntu 24.04 WSL 2 distribution identified.
- [x] RTX 4060 visible inside WSL through `nvidia-smi`.
- [x] Isolated Python 3.12 virtual environment created.
- [x] CUDA-enabled PyTorch installed.
- [x] CUDA allocation, matrix multiplication, and backward pass verified.
- [x] Exact resolved packages recorded in `requirements/lock-wsl-cu128.txt`.

## 2. Minimal repository

- [x] Package, configuration, scripts, tests, and documentation scaffolded.
- [x] Eight-GB micro-model limits recorded in configuration.
- [x] Editable installation verified in WSL.
- [x] Unit tests pass.
- [x] Ruff lint checks pass.

## 3. ManiSkill PushCube

- [x] ManiSkill 3.0.1 and SAPIEN 3.0.3 installed.
- [x] `PushCube-v1` CPU-state episode runs.
- [x] Episode actions and outcomes are saved.
- [x] Saved CPU-state episode replays deterministically.
- [x] RGB-D observation runs through WSL software Vulkan rendering.
- [x] RGB image, depth array, camera intrinsics, and extrinsics are saved/verified.
- [ ] ManiSkill PhysX-CUDA simulation and CUDA rendering in WSL.

The final unchecked item is an upstream platform limitation, not a project-code failure.
ManiSkill's support table lists WSL GPU simulation/rendering as unsupported. The local probe
confirmed this by hanging during renderer initialization after no NVIDIA Vulkan ICD was
found. Model training is nevertheless CUDA-enabled; early RGB-D collection can use the
working CPU/software-rendering path.

## 4. Metric depth backprojection and coordinate frames

- [x] Convert ManiSkill signed 16-bit millimetre depth to metres.
- [x] Backproject through `intrinsic_cv` at SAPIEN pixel centres.
- [x] Transform OpenCV-camera XYZ to world coordinates through inverse `extrinsic_cv`.
- [x] Independently transform through OpenGL axes and `cam2world_gl`.
- [x] Verify both world-coordinate routes agree for every valid pixel.
- [x] Compare reconstructed XYZ against the native SAPIEN position texture.
- [x] Verify segmented cube points against the simulator pose and 4 cm dimensions.
- [x] Save full NPZ, workspace PLY, JSON metrics, and a point-cloud plot.
- [x] Cover the geometry implementation with synthetic unit tests.

## 5. Persistent simulator trajectory labels

- [x] Enumerate rendered scene bodies and robot links from the live simulator.
- [x] Map visible segmentation IDs to robot, object, goal, and static categories.
- [x] Use stratified sampling so the small cube is not lost among background pixels.
- [x] Attach each selected world point to its simulator body.
- [x] Convert attached points to body-local coordinates.
- [x] Execute an eight-step low-level action chunk that produces cube motion.
- [x] Reconstruct future world coordinates from future body poses.
- [x] Mark object, robot, goal, static, and approximate contact-region points.
- [x] Separate model-facing training fields from privileged audit fields.
- [x] Validate time-zero reconstruction, rigidity, static points, motion, and finiteness.
- [x] Generate and validate 100 compact PushCube trajectory fragments.
- [x] Save a 3D and top-view trajectory visualization.

## 6. Training dataset pipeline

- [x] Discover model-facing archives without exposing privileged audit fields.
- [x] Validate a fixed schema and consistent point, horizon, and action dimensions.
- [x] Freeze deterministic, disjoint 80/10/10 initial-state-group splits.
- [x] Fit streaming normalization statistics on the training split only.
- [x] Produce deterministic visible-point queries with fixed batch shapes.
- [x] Preserve small-object queries without using segmentation or body identities.
- [x] Configure cached, multi-worker, prefetched, pinned-memory batching.
- [x] Verify non-blocking transfer of every batch tensor to the RTX 4060.
- [x] Save split, normalization, and loader verification artifacts.
- [x] Cover splits, normalization, query sampling, and batch shapes with unit tests.

## 7. Non-neural baselines and evaluation metrics

- [x] Freeze a dense held-out query protocol over all 256 visible points.
- [x] Implement static and training-mean trajectory controls.
- [x] Implement matched scene-only and action-conditioned KNN retrieval controls.
- [x] Report metric 3D ADE, FDE, per-horizon error, and the 95th percentile.
- [x] Stratify moving, contact, robot, object, static, and goal points.
- [x] Measure acceleration, pairwise geometry, and same-body rigidity errors.
- [x] Keep privileged audit fields outside every predictor.
- [x] Record the exact split digest, fitting scope, runtimes, and full-precision results.
- [x] Verify metric identities and baseline behavior with unit tests.
- [x] Detect that the current one-action-per-scene corpus confounds scene and action.

The original one-action-per-scene corpus correctly triggered a data warning when
scene-only KNN matched action KNN. The grouped counterfactual corpus resolves that
confound: action KNN now reduces overall ADE by 82% relative to matched scene KNN.

## 8. No-action neural trajectory baseline

- [x] Collect success, perturbed, no-op, and failure branches from shared initial states.
- [x] Verify byte-identical model inputs and four distinct action chunks per state.
- [x] Keep all sibling branches together in deterministic splits.
- [x] Recompute training-only normalization and non-neural references.
- [x] Implement a query trajectory network whose forward interface has no action input.
- [x] Train with mixed precision on the RTX 4060 and select by validation loss.
- [x] Evaluate all 256 visible points with item-7 metrics and subgroup diagnostics.
- [x] Verify identical predictions across counterfactual siblings.
- [x] Save the best checkpoint, training history, and held-out report.

## 9. Micro-Q4D action-conditioned neural baseline

- [x] Match the no-action scene/query inputs, width, loss, and grouped data protocol.
- [x] Encode normalized eight-step executable action chunks with a temporal GRU.
- [x] Fuse action and query features at every predicted future time.
- [x] Expose separate scene, query, and multi-candidate decoding stages.
- [x] Verify direct and cached predictions agree within mixed-precision tolerance.
- [x] Train with mixed precision on the RTX 4060 and select by validation loss.
- [x] Evaluate dense held-out trajectories with all item-7 metrics and subgroups.
- [x] Verify predictions diverge between counterfactual sibling actions.
- [x] Beat the no-action neural baseline on overall, moving, contact, and object ADE.
- [x] Benchmark 64 cached candidate branches against matched batched re-encoding.
- [x] Save the checkpoint, history, evaluation, and cache benchmark report.

## 10. Eight-GB GPU training budget

- [x] Reserve explicit VRAM headroom and cap project allocations at 6 GiB.
- [x] Validate the memory contract against detected physical GPU capacity at startup.
- [x] Use mixed-precision forward and backward passes.
- [x] Use pinned host batches and non-blocking device transfers.
- [x] Add micro-batch training with correctly scaled gradient accumulation.
- [x] Produce effective batch size 32 from four micro-batches of eight.
- [x] Unscale and clip gradients only at optimizer-update boundaries.
- [x] Fail training if allocated or reserved CUDA memory exceeds the budget.
- [x] Exercise real model, loss, backward, AdamW state, and optimizer updates in the audit.
- [x] Verify steady-state memory does not grow between optimizer steps.
- [x] Preserve the item-9 configuration and checkpoint separately.
- [x] Save a machine-readable RTX 4060 memory report.

## 11. Matched dense point-future baseline

- [x] Reuse the exact micro-Q4D scene, query, action, fusion, and decoder modules.
- [x] Match width, horizon, optimizer, seed, grouped split, and metric protocol.
- [x] Verify exact parameter-count and state-dict parity with micro-Q4D.
- [x] Predict and supervise all 256 visible scene points on every action branch.
- [x] Preserve point correspondence when converting scene order to dataset query order.
- [x] Train with mixed precision on the RTX 4060 under the 6 GiB allocation cap.
- [x] Evaluate all held-out points with item-7 metrics and privileged audit stratification.
- [x] Compare dense and sparse accuracy on all, moving, contact, and object points.
- [x] Benchmark matched cached decoding over the same 64 candidate action branches.
- [x] Save the checkpoint, history, metrics, parameter audit, and throughput report.

The dense model has exactly 250,883 parameters, matching micro-Q4D. It lowers overall ADE
from 3.413 mm to 3.072 mm, while sparse-32 candidate decoding is 7.76 times faster and uses
about 5.4 times less peak CUDA memory than dense-256. This isolates the expected
accuracy-throughput tradeoff. It does not test early versus late action fusion; that is the
next separate controlled baseline.

## 12. Scaled PushCube experiment

- [x] Define a resumable 2,000-state, five-branch collection protocol.
- [x] Separate physical-integrity checks from measured policy outcomes.
- [x] Validate successful, weak, off-target, failure, and no-op branches in a pilot.
- [x] Increase stored visible points from 256 to 512.
- [x] Add exact prefix slicing for H=1, 2, 4, and 8 from the same archives.
- [x] Add a resumable static/no-action/dense/Q4D training matrix.
- [x] Add action-shuffling and cache/no-cache ablations.
- [x] Add N={128,256,512}, M={32,64,128} latency/VRAM benchmarking.
- [x] Generate and validate all 10,000 trajectory fragments.
- [x] Freeze the 8,000/1,000/1,000 state-group split and training-only statistics.
- [x] Train and evaluate every requested model and horizon.
- [ ] Evaluate the fixed prediction and computational gate.

The full corpus, split, and H=1/2/4/8 training matrix are complete. The final gate remains
open because the N/M scaling audit's strict cache-equivalence check did not pass; the
underlying accuracy, latency, and VRAM artifacts are preserved for diagnosis.

## 13. Simple model-predictive control

- [x] Implement batched random shooting over executable eight-step action sequences.
- [x] Score final predicted visible-cube centroid distance to the goal.
- [x] Encode each observed scene exactly once per control cycle.
- [x] Evaluate candidate batches from the shared scene/query cache.
- [x] Execute only the first selected action.
- [x] Reobserve RGB-D, reconstruct the scene, and replan after every action.
- [x] Add diagonal-Gaussian CEM after validating random shooting.
- [x] Measure closed-loop success at fixed 50, 100, and 200 ms budgets.
- [x] Compare Q4D and dense with identical planners, budgets, and episode seeds.
- [x] Treat a cube leaving the camera view as an explicit failed rollout.
- [x] Complete 120 matched closed-loop episodes and pass protocol integrity checks.

At 100 ms, Q4D random shooting succeeds on 8/10 episodes versus 2/10 for dense and
evaluates 4.38 times as many candidates per cycle. At 200 ms the comparison is 8/10
versus 3/10 at 5.63 times candidate throughput. CEM is not monotonic at ten seeds: dense
wins at 50 ms, the models tie at 100 ms, and Q4D wins at 200 ms. These are descriptive
results from a small matched benchmark, not confidence-bounded estimates.

## 14. MVP stop/continue decision

**Decision: CONTINUE**, with random shooting as the trusted planning baseline and no
claim of statistical finality from ten planning seeds.

- [x] Action conditioning beats the no-action predictor.
- [x] Scene caching improves candidate evaluation throughput.
- [x] Q4D is competitive with the parameter-matched dense baseline.
- [x] Prediction improvements translate into MPC success.
- [x] Diagnose the contradictory legacy cache result before making the decision.
- [x] Keep denser queries and explicit contact-event prediction as fallbacks rather than
  silently changing the successful MVP protocol.
- [x] Make and record the gate decision before starting another larger benchmark.

At H=8, Q4D reduces overall ADE by 93.9% and contact ADE by 87.2% relative to the
no-action model. The corrected exact-index N/M grid shows 1.43x--2.53x cache speedup in
all nine configurations. Q4D is 19.4% worse than dense overall, but only 1.8% worse near
contact and 0.1% worse on object points. In matched 100 ms random-shooting MPC, Q4D
succeeds on 7/10 seeds versus 0/10 for no-action.
