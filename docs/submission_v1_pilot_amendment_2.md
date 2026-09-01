# Submission-v1 pilot analysis amendment 2

Date: 1 September 2026. No episode, model, seed, planning condition, statistical
threshold, or task outcome changed.

The first aggregate-gate implementation applied the 0.05 m/s terminal settling
threshold to all four tasks. This contradicted the checklist's task-specific behavioral
requirements and the pre-run description: PickCube must remain grasped, while
PlaceSphere and StackCube must release and settle. PullCube has no settling requirement.

The evaluator is corrected so that:

- PullCube success requires the simulator success flag, success termination, and positive
  task-distance improvement.
- PickCube additionally requires the primary object to remain grasped.
- PlaceSphere and StackCube additionally require an executed release command, no final
  grasp, and terminal object speed at or below 0.05 m/s.

Re-evaluating the unchanged episode reports still fails the pilot: PickCube's only raw
success, seed 14610, ends with the cube released rather than grasped. The correction
therefore improves the accuracy of the diagnostic without converting the pilot into a
pass.
