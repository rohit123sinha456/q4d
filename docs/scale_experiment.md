# Scaled PushCube experiment

Checklist item 12 is a gated, resumable experiment over 10,000 trajectory fragments. The
collection contains 2,000 initial simulator states with five counterfactual branches per
state: successful push, weak push, off-target push, failure action, and no-op. Sibling
branches share a byte-identical RGB-D/point observation and remain in one train,
validation, or test split.

The scaled archives store 512 visible points and eight future steps. Dataset horizon
slicing reuses the exact same fragments and splits for H=1, 2, 4, and 8. The sparse models
train on M=64 deterministic visible queries, while dense training and held-out metrics use
all N=512 points.

## Collection

The collector snapshots the simulator once after approaching the cube and restores that
state for every sibling branch. It writes a record beside every training/audit pair and an
atomic manifest every ten state groups. An interrupted job resumes without regenerating
complete groups:

```bash
python -u scripts/generate_point_tracks.py \
  --config configs/scale_pushcube.toml \
  --profile scaled \
  --states 2000 \
  --resume \
  --checkpoint-every 10
```

Physical label integrity and policy outcome are audited separately. A policy miss remains
a valid supervised trajectory. Measured cube displacement and final task state classify
each fragment as successful, weak, off-target, or no-motion, so corpus composition is not
inferred from the requested action name.

The 100-fragment validation pilot passed every physical and sibling-identity check and
contained 20 successful, 20 weak, 20 off-target, and 40 no-motion outcomes.

## Training and evaluation matrix

After collection completes, this command prepares grouped 80/10/10 splits and runs the
resumable H=1/2/4/8 matrix:

```bash
python scripts/run_scale_experiment.py
```

For each horizon it evaluates the static control and trains action-free, dense-512, and
sparse-M=64 micro-Q4D models with the same width, optimizer, seed, effective batch size 32,
and 6 GiB allocation cap. Reports include ADE, FDE, contact ADE, latency, and VRAM.

The Q4D evaluation also performs a deterministic action shuffle with no fixed points and
benchmarks 64 candidate actions with and without cached scene/query encoding. At H=8, the
benchmark grid tests N in {128, 256, 512} and M in {32, 64, 128}. Quadratic all-pairs
geometry diagnostics are disabled for this phase; the requested pointwise and contact
metrics still use every held-out point.

Persistent data-loader workers cache deterministic, processed samples after their first
epoch. The complete 8,000-fragment training cache is comfortably below host-memory limits
and avoids repeating NPZ decompression and farthest-point sampling for every one of the 30
epochs.

`scripts/summarize_scale_experiment.py` produces `gate_report.json`. The prediction gate
requires Q4D to beat the no-action model and degrade under action shuffling. The compute
gate requires caching to beat re-encoding and sparse decoding to beat dense decoding at
every horizon. These are scientific checks: the summarizer reports a failed gate rather
than rewriting the criteria after seeing results.
