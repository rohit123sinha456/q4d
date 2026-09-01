# Submission-v1 pilot implementation amendment 1

Date: 1 September 2026. No statistical threshold, seed, task, model, planner, budget,
schedule, or scientific gate changed.

The first PullCube pilot attempt stopped before executing its first MPC action because
the candidate evaluator returned a non-finite FP16 cost. PullCube's frozen training data
contains only gripper command `-1`, so normalization correctly records action channel 6
as constant with scale `1e-6`. Applying that training scale to the newly required open
command `+1` produced a normalized value near two million and overflowed the predictor.

The correction sets every training-constant normalized action channel to zero after
normalization. Zero is the only normalized value observed for those channels during
training. The executable candidate remains unchanged, including its open/closed gripper
command, and is still passed to the simulator. PullCube therefore receives the complete
candidate schedule library, while its predictor makes no unsupported claim about a
gripper effect absent from its training corpus.

For PickCube, PlaceSphere, and StackCube, the gripper channel is not constant and this
correction makes no numerical change. Their completed pilot reports remain the results
from clean commit `6a6d5864a4f6f9e4bf163b6e2a44f3b3b0af7b33`. Only the affected PullCube
condition is rerun, using the same frozen seeds, under the new versioned output directory
`pull_cube_retry1`. The failed attempt and its original log remain under `pull_cube/`.
