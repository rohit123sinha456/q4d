# Eight-GB training contract

Checklist item 10 makes the laptop GPU constraint executable rather than relying on an
informal claim that the model is small. Run the live audit with:

```bash
python scripts/verify_8gb_training.py
```

For a full training run under the accumulation profile:

```bash
python scripts/train_micro_q4d.py --config configs/micro_q4d_8gb.toml
```

## Resource policy

The RTX 4060 exposes 8,187.5 MiB of physical VRAM. Project training is capped at
6,144 MiB, leaving 2,043.5 MiB outside the configured budget and requiring at least
1,024 MiB of explicit headroom. `CudaMemoryBudget` rejects an invalid configuration at
startup and raises immediately if either peak PyTorch allocation or reservation exceeds
the cap during training.

The item-10 profile uses:

- mixed-precision forward and backward passes;
- pinned CPU batches and non-blocking CUDA transfers;
- micro-batch size 8;
- four accumulated micro-batches per optimizer update;
- effective batch size 32;
- gradient unscaling and clipping only at optimizer-update boundaries;
- correct loss normalization for a partial final accumulation window;
- `zero_grad(set_to_none=True)` to avoid retaining gradient buffers unnecessarily.

The original item-9 configuration remains separate with accumulation set to one, so its
published checkpoint is reproducible. The item-10 profile writes to a different model
directory and cannot overwrite that checkpoint accidentally.

## Measured RTX 4060 result

The audit loads the trained 250,883-parameter micro-Q4D checkpoint, uses real dataset
batches, creates full AdamW state, performs warm-up updates, then measures six optimizer
updates comprising 24 micro-batches and 192 samples.

| Quantity | Result |
| --- | ---: |
| Physical GPU memory | 8,187.5 MiB |
| Project allocation cap | 6,144 MiB |
| Peak PyTorch allocated | 32.90 MiB |
| Peak PyTorch reserved | 52.00 MiB |
| Budget utilization | 0.85% |
| Steady-state memory growth | 0.00 MiB |
| Effective-batch throughput | 104.7 samples/s |
| Model parameters | 0.96 MiB |
| AdamW state | 1.91 MiB |

The remaining physical memory figure is not a promise that every byte is available to
PyTorch: CUDA context, driver allocations, display use, and other processes are outside
the allocator telemetry. That is why the contract uses both a conservative project cap
and explicit configured headroom.

This result proves the current micro-Q4D training path fits easily on the 8 GB GPU. It
does not automatically certify future 20–50M-parameter variants; the same audit and hard
budget check must pass after every material increase in width, point count, query count,
or horizon.

The machine-readable result is saved to
`artifacts/environment/memory_8gb_report.json`.
