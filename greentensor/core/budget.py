# -*- coding: utf-8 -*-
"""
Carbon Budget Enforcement for GreenTensor.

Allows teams to set hard carbon budgets per job, per project, or per period.
Raises CarbonBudgetExceeded when a job goes over budget.
Integrates with RunHistory for cumulative budget tracking.
"""

from dataclasses import dataclass
from typing import Optional
from greentensor.utils.logger import logger


class CarbonBudgetExceeded(Exception):
    """Raised when a training job exceeds its carbon budget."""
    def __init__(self, budget_kg: float, actual_kg: float, job_name: str = ""):
        self.budget_kg = budget_kg
        self.actual_kg = actual_kg
        self.overage_kg = actual_kg - budget_kg
        self.overage_pct = (self.overage_kg / budget_kg * 100) if budget_kg > 0 else 0
        msg = (
            f"Carbon budget exceeded{' for ' + job_name if job_name else ''}. "
            f"Budget: {budget_kg:.6f} kg CO2. "
            f"Actual: {actual_kg:.6f} kg CO2. "
            f"Overage: {self.overage_kg:.6f} kg ({self.overage_pct:.1f}%)."
        )
        super().__init__(msg)


@dataclass
class CarbonBudget:
    """
    Carbon budget configuration for a training job or project.

    Parameters
    ----------
    max_kg_per_run    : maximum CO2 per single run (hard limit)
    max_kwh_per_run   : maximum energy per single run (hard limit)
    warn_at_pct       : emit warning when this % of budget is consumed (default 80%)
    raise_on_exceed   : if True, raises CarbonBudgetExceeded; if False, just warns
    job_name          : optional label for error messages
    """
    max_kg_per_run: Optional[float] = None
    max_kwh_per_run: Optional[float] = None
    warn_at_pct: float = 80.0
    raise_on_exceed: bool = True
    job_name: str = ""

    def check(self, emissions_kg: float, energy_kwh: float):
        """Check current measurements against budget. Call during or after a run."""
        if self.max_kg_per_run:
            pct = emissions_kg / self.max_kg_per_run * 100
            if pct >= 100:
                msg = (f"Carbon budget exceeded: {emissions_kg:.6f} kg / "
                       f"{self.max_kg_per_run:.6f} kg budget")
                logger.warning(f"[BUDGET] {msg}")
                if self.raise_on_exceed:
                    raise CarbonBudgetExceeded(self.max_kg_per_run, emissions_kg, self.job_name)
            elif pct >= self.warn_at_pct:
                logger.warning(
                    f"[BUDGET] {pct:.1f}% of carbon budget consumed "
                    f"({emissions_kg:.6f} / {self.max_kg_per_run:.6f} kg CO2)"
                )

        if self.max_kwh_per_run:
            pct = energy_kwh / self.max_kwh_per_run * 100
            if pct >= 100:
                msg = (f"Energy budget exceeded: {energy_kwh:.6f} kWh / "
                       f"{self.max_kwh_per_run:.6f} kWh budget")
                logger.warning(f"[BUDGET] {msg}")
                if self.raise_on_exceed:
                    raise CarbonBudgetExceeded(
                        self.max_kwh_per_run * 0.000233,
                        energy_kwh * 0.000233,
                        self.job_name
                    )
            elif pct >= self.warn_at_pct:
                logger.warning(
                    f"[BUDGET] {pct:.1f}% of energy budget consumed "
                    f"({energy_kwh:.6f} / {self.max_kwh_per_run:.6f} kWh)"
                )
