"""Explicit CUDA memory contracts for single-GPU experiments."""

from __future__ import annotations

from dataclasses import dataclass


def bytes_to_mib(value: int) -> float:
    return value / 2**20


@dataclass(frozen=True)
class CudaMemoryBudget:
    """Validate a project allocation cap against physical GPU capacity."""

    total_mib: float
    budget_mib: float
    minimum_headroom_mib: float

    def __post_init__(self) -> None:
        if min(self.total_mib, self.budget_mib, self.minimum_headroom_mib) <= 0:
            raise ValueError("memory sizes must be positive")
        if self.budget_mib + self.minimum_headroom_mib > self.total_mib:
            raise ValueError(
                "memory budget plus required headroom exceeds physical GPU memory"
            )

    @property
    def configured_headroom_mib(self) -> float:
        return self.total_mib - self.budget_mib

    def assert_peak(self, *, allocated_mib: float, reserved_mib: float) -> None:
        peak = max(allocated_mib, reserved_mib)
        if peak > self.budget_mib:
            raise RuntimeError(
                f"CUDA peak {peak:.1f} MiB exceeds budget {self.budget_mib:.1f} MiB"
            )

    def report(self, *, allocated_mib: float, reserved_mib: float) -> dict[str, float]:
        self.assert_peak(allocated_mib=allocated_mib, reserved_mib=reserved_mib)
        peak = max(allocated_mib, reserved_mib)
        return {
            "physical_total_mib": self.total_mib,
            "allocation_budget_mib": self.budget_mib,
            "minimum_headroom_mib": self.minimum_headroom_mib,
            "configured_headroom_mib": self.configured_headroom_mib,
            "peak_allocated_mib": allocated_mib,
            "peak_reserved_mib": reserved_mib,
            "budget_utilization_percent": 100.0 * peak / self.budget_mib,
            "unallocated_by_pytorch_at_peak_mib": self.total_mib - peak,
        }
