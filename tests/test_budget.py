import pytest
from greentensor.core.budget import CarbonBudget, CarbonBudgetExceeded


def test_within_budget():
    budget = CarbonBudget(max_kg_per_run=0.001)
    budget.check(emissions_kg=0.0005, energy_kwh=0.002)  # should not raise


def test_budget_exceeded_raises():
    budget = CarbonBudget(max_kg_per_run=0.001, raise_on_exceed=True)
    with pytest.raises(CarbonBudgetExceeded) as exc:
        budget.check(emissions_kg=0.002, energy_kwh=0.005)
    assert exc.value.budget_kg == 0.001
    assert exc.value.actual_kg == 0.002
    assert exc.value.overage_pct == 100.0


def test_budget_exceeded_no_raise():
    budget = CarbonBudget(max_kg_per_run=0.001, raise_on_exceed=False)
    budget.check(emissions_kg=0.002, energy_kwh=0.005)  # should warn but not raise


def test_energy_budget():
    budget = CarbonBudget(max_kwh_per_run=0.001, raise_on_exceed=True)
    with pytest.raises(CarbonBudgetExceeded):
        budget.check(emissions_kg=0.0, energy_kwh=0.002)


def test_warn_threshold(capsys):
    budget = CarbonBudget(max_kg_per_run=0.001, warn_at_pct=80.0, raise_on_exceed=False)
    budget.check(emissions_kg=0.00085, energy_kwh=0.001)  # 85% — should warn
