import pytest

from q4d_wam.training import CudaMemoryBudget, bytes_to_mib


def test_memory_budget_reports_headroom_and_utilization() -> None:
    budget = CudaMemoryBudget(
        total_mib=8192, budget_mib=6144, minimum_headroom_mib=1024
    )

    report = budget.report(allocated_mib=1000, reserved_mib=1200)

    assert report["configured_headroom_mib"] == 2048
    assert report["unallocated_by_pytorch_at_peak_mib"] == 6992
    assert report["budget_utilization_percent"] == pytest.approx(19.53125)
    assert bytes_to_mib(2**20) == 1.0


def test_memory_budget_rejects_invalid_contract_and_peak() -> None:
    with pytest.raises(ValueError, match="exceeds physical"):
        CudaMemoryBudget(total_mib=8192, budget_mib=7500, minimum_headroom_mib=1024)

    budget = CudaMemoryBudget(
        total_mib=8192, budget_mib=6144, minimum_headroom_mib=1024
    )
    with pytest.raises(RuntimeError, match="exceeds budget"):
        budget.assert_peak(allocated_mib=6100, reserved_mib=6200)
