# Matched dense point-future baseline

Checklist item 11 is a controlled sparse-query versus dense-output ablation. The dense
model uses the exact micro-Q4D scene encoder, query encoder, action GRU, time embedding,
fusion decoder, width, horizon, optimizer, grouped split, and evaluation metrics. It has
the same 250,883 trainable parameters and state-dict structure. The controlled difference
is its output contract: every action branch predicts all 256 visible scene points instead
of an arbitrary sparse query set.

The model first encodes the observed scene, then treats every scene point as a query. The
dataset's all-point FPS order is a permutation of scene order, so the training script
explicitly gathers dense predictions into that order before computing loss. This keeps
point-to-target correspondence exact without giving the predictor audit-only identities or
segmentation.

## Reproduce

From the WSL project environment:

```bash
python scripts/train_dense_baseline.py
```

The configuration is `configs/dense_baseline.toml`. The best checkpoint, complete training
history, metrics, comparisons, memory audit, and candidate benchmark are written to
`artifacts/models/dense_baseline_v1/`.

## Held-out result

Both learned models were evaluated over the same 12 held-out counterfactual fragments and
all 256 visible points per fragment.

| Point group | Dense ADE (mm) | Sparse-trained micro-Q4D ADE (mm) | Dense change |
| --- | ---: | ---: | ---: |
| All | 3.072 | 3.413 | -10.0% |
| Moving | 10.536 | 11.256 | -6.4% |
| Contact | 38.381 | 39.043 | -1.7% |
| Object | 17.411 | 17.701 | -1.6% |

Dense overall FDE is 4.927 mm, compared with 5.373 mm for micro-Q4D. This small-data
experiment therefore shows an accuracy benefit from supervising all visible points. It
does not establish that dense prediction is preferable for planning, because candidate
action throughput is the main hypothesis under test.

## Matched 64-branch benchmark

Both paths cache the same scene and decode the same normalized candidate action tensor on
the RTX 4060 using mixed precision. Dense output contains 256 trajectories per branch;
sparse output contains 32.

| Cached decoder | Time (ms) | Candidates/s | Peak CUDA memory (MiB) |
| --- | ---: | ---: | ---: |
| Dense-256 | 5.152 | 12,423 | 343.0 |
| Sparse-32 | 0.664 | 96,366 | 63.0 |

Sparse decoding is 7.76 times faster for an eightfold reduction in requested trajectories.
The comparison is parameter-matched rather than FLOP-matched: the purpose of the ablation
is to measure the output-dependent work avoided by sparse querying. Training the dense
model took 130.3 seconds, and its peak reserved training memory was 136 MiB, safely below
the 6 GiB project allocation cap.

This baseline isolates output sparsity only. The next controlled baseline must move action
fusion before scene encoding and recompute those features per candidate; that separate
experiment tests the encode-once/cache advantage.
