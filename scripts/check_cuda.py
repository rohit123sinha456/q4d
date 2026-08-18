#!/usr/bin/env python3
"""Verify CUDA allocation, matrix multiplication, and autograd."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import torch


def run_check() -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot access CUDA")

    device = torch.device("cuda:0")
    x = torch.randn(512, 512, device=device, requires_grad=True)
    loss = (x @ x.T).square().mean()
    loss.backward()
    torch.cuda.synchronize()

    properties = torch.cuda.get_device_properties(device)
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "cuda_available": True,
        "device": torch.cuda.get_device_name(device),
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "vram_bytes": properties.total_memory,
        "vram_gib": round(properties.total_memory / 2**30, 3),
        "backward_ok": x.grad is not None,
        "loss": float(loss.detach()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = run_check()
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

