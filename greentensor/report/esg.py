# -*- coding: utf-8 -*-
"""
ESG Report Generator for GreenTensor.

Produces structured ESG (Environmental, Social, Governance) reports
covering Scope 2 carbon emissions from AI/ML workloads.

Aligned with:
- GHG Protocol Scope 2 Guidance
- SEC Climate Disclosure Rules (17 CFR Parts 210, 229, 232, 239, 249)
- EU CSRD (Corporate Sustainability Reporting Directive)
- ISO 14064-1 (GHG quantification and reporting)
"""

import json
import time
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict
from greentensor.report.metrics import RunMetrics, DatacenterConfig


@dataclass
class ESGOrganization:
    name: str
    reporting_period: str          # e.g. "2025-Q1" or "FY2025"
    region: str                    # e.g. "US-East", "EU-West", "India"
    carbon_intensity_kg_per_kwh: float = 0.000233
    reporting_standard: str = "GHG Protocol Scope 2"
    contact_email: str = ""


@dataclass
class ESGRunRecord:
    """A single training/inference run contributing to the ESG report."""
    run_id: str
    timestamp: float
    model_name: str
    stage: str                     # "training" | "inference" | "fine_tuning"
    duration_s: float
    energy_kwh: float
    emissions_kg: float
    energy_kwh_dc: Optional[float] = None
    emissions_kg_dc: Optional[float] = None
    pue: float = 1.0
    num_nodes: int = 1
    security_incidents: int = 0
    optimized: bool = True


@dataclass
class ESGReport:
    organization: ESGOrganization
    generated_at: str
    runs: List[ESGRunRecord] = field(default_factory=list)

    # Aggregated totals
    total_runs: int = 0
    total_energy_kwh: float = 0.0
    total_emissions_kg: float = 0.0
    total_energy_kwh_dc: float = 0.0
    total_emissions_kg_dc: float = 0.0
    total_duration_hours: float = 0.0
    energy_saved_kwh: float = 0.0
    emissions_saved_kg: float = 0.0
    security_incidents_total: int = 0

    # Equivalencies (for human context)
    emissions_equiv_km_driven: float = 0.0
    emissions_equiv_trees_needed: float = 0.0
    emissions_equiv_flights_nyc_la: float = 0.0

    def compute(self):
        """Aggregate all run records into totals and equivalencies."""
        self.total_runs = len(self.runs)
        self.total_energy_kwh = sum(r.energy_kwh for r in self.runs)
        self.total_emissions_kg = sum(r.emissions_kg for r in self.runs)
        self.total_energy_kwh_dc = sum(r.energy_kwh_dc or r.energy_kwh for r in self.runs)
        self.total_emissions_kg_dc = sum(r.emissions_kg_dc or r.emissions_kg for r in self.runs)
        self.total_duration_hours = sum(r.duration_s for r in self.runs) / 3600
        self.security_incidents_total = sum(r.security_incidents for r in self.runs)

        # Equivalencies
        # Average car: 0.000192 kg CO2 per meter = 0.192 kg/km
        self.emissions_equiv_km_driven = self.total_emissions_kg_dc / 0.000192
        # One tree absorbs ~21 kg CO2/year
        self.emissions_equiv_trees_needed = self.total_emissions_kg_dc / 21.0
        # NYC-LA flight: ~163 kg CO2 per passenger
        self.emissions_equiv_flights_nyc_la = self.total_emissions_kg_dc / 163.0

    def to_dict(self) -> dict:
        self.compute()
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def to_text(self) -> str:
        self.compute()
        lines = [
            "",
            "  ╔══════════════════════════════════════════════╗",
            "  ║         GreenTensor ESG Report               ║",
            "  ╚══════════════════════════════════════════════╝",
            f"  Organization    : {self.organization.name}",
            f"  Reporting Period: {self.organization.reporting_period}",
            f"  Region          : {self.organization.region}",
            f"  Standard        : {self.organization.reporting_standard}",
            f"  Generated       : {self.generated_at}",
            "",
            "  ── Scope 2 Emissions Summary ──────────────────",
            f"  Total ML Runs        : {self.total_runs}",
            f"  Total Compute Time   : {self.total_duration_hours:.2f} hours",
            f"  Total Energy (GPU)   : {self.total_energy_kwh:.4f} kWh",
            f"  Total Energy (DC)    : {self.total_energy_kwh_dc:.4f} kWh",
            f"  Total CO2 (GPU)      : {self.total_emissions_kg:.4f} kg",
            f"  Total CO2 (DC adj.)  : {self.total_emissions_kg_dc:.4f} kg",
            "",
            "  ── Avoided Emissions (GreenTensor savings) ────",
            f"  Energy Saved         : {self.energy_saved_kwh:.4f} kWh",
            f"  CO2 Avoided          : {self.emissions_saved_kg:.4f} kg",
            f"  Reduction vs baseline: {(self.emissions_saved_kg / (self.total_emissions_kg_dc + self.emissions_saved_kg) * 100) if (self.total_emissions_kg_dc + self.emissions_saved_kg) > 0 else 0:.1f}%",
            "",
            "  ── Real-World Equivalencies ───────────────────",
            f"  Equivalent km driven : {self.emissions_equiv_km_driven:.1f} km",
            f"  Trees needed (1 yr)  : {self.emissions_equiv_trees_needed:.2f} trees",
            f"  NYC-LA flights       : {self.emissions_equiv_flights_nyc_la:.3f} flights",
            "",
            "  ── Security Summary ───────────────────────────",
            f"  Security Incidents   : {self.security_incidents_total}",
            f"  Runs with incidents  : {sum(1 for r in self.runs if r.security_incidents > 0)}",
            "",
            "  ── Compliance Notes ───────────────────────────",
            "  This report covers Scope 2 (indirect) GHG emissions",
            "  from electricity consumed by AI/ML workloads.",
            "  Datacenter PUE overhead is included in DC-adjusted figures.",
            "  Methodology: GHG Protocol Scope 2 Market-Based approach.",
            "  ══════════════════════════════════════════════\n",
        ]
        return "\n".join(lines)


class ESGReporter:
    """
    Accumulates run metrics across multiple training/inference sessions
    and generates ESG-compliant reports.

    Usage:
        reporter = ESGReporter(org, history_path="esg_history.json")

        # After each GreenTensor run:
        reporter.record_run(gt.metrics, model_name="ResNet50", stage="training")

        # Generate report:
        report = reporter.generate_report()
        print(report.to_text())
        reporter.save_json("esg_report_Q1_2025.json")
    """

    def __init__(self, organization: ESGOrganization,
                 history_path: str = "greentensor_esg_history.json"):
        self.org = organization
        self.history_path = history_path
        self._runs: List[ESGRunRecord] = []
        self._load_history()

    def record_run(
        self,
        metrics: RunMetrics,
        model_name: str = "unknown",
        stage: str = "training",
        dc_config: Optional[DatacenterConfig] = None,
        security_incidents: int = 0,
        optimized: bool = True,
        run_id: Optional[str] = None,
    ):
        """Record a completed run into the ESG history."""
        if dc_config:
            adjusted = metrics.apply_datacenter_config(dc_config)
            energy_dc = adjusted.energy_kwh_dc
            emissions_dc = adjusted.emissions_kg_dc
            pue = dc_config.pue
            nodes = dc_config.num_nodes
        else:
            energy_dc = metrics.energy_kwh
            emissions_dc = metrics.emissions_kg
            pue = 1.0
            nodes = 1

        record = ESGRunRecord(
            run_id=run_id or f"run_{int(time.time())}",
            timestamp=time.time(),
            model_name=model_name,
            stage=stage,
            duration_s=metrics.duration_s,
            energy_kwh=metrics.energy_kwh,
            emissions_kg=metrics.emissions_kg,
            energy_kwh_dc=energy_dc,
            emissions_kg_dc=emissions_dc,
            pue=pue,
            num_nodes=nodes,
            security_incidents=security_incidents,
            optimized=optimized,
        )
        self._runs.append(record)
        self._save_history()
        return record

    def generate_report(
        self,
        baseline_energy_kwh: float = 0.0,
        baseline_emissions_kg: float = 0.0,
    ) -> ESGReport:
        """Generate the full ESG report from all recorded runs."""
        report = ESGReport(
            organization=self.org,
            generated_at=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            runs=list(self._runs),
        )
        report.energy_saved_kwh = max(0, baseline_energy_kwh - sum(r.energy_kwh for r in self._runs))
        report.emissions_saved_kg = max(0, baseline_emissions_kg - sum(r.emissions_kg for r in self._runs))
        report.compute()
        return report

    def save_json(self, path: str, baseline_energy_kwh: float = 0.0,
                  baseline_emissions_kg: float = 0.0):
        report = self.generate_report(baseline_energy_kwh, baseline_emissions_kg)
        with open(path, "w", encoding="utf-8") as f:
            f.write(report.to_json())
        return path

    def _save_history(self):
        try:
            data = [asdict(r) for r in self._runs]
            with open(self.history_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    def _load_history(self):
        try:
            with open(self.history_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._runs = [ESGRunRecord(**r) for r in data]
        except Exception:
            self._runs = []

    @property
    def run_count(self) -> int:
        return len(self._runs)
