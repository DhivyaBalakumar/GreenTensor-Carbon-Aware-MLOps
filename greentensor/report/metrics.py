from dataclasses import dataclass, field
from typing import Optional
from greentensor.water.aquatensor import AquaTensorBridge, AquaTensorConfig, WaterMetrics

# Industry PUE benchmarks
PUE_PRESETS = {
    "local_workstation": 1.0,   # no datacenter overhead
    "hyperscale":        1.1,   # Google/Microsoft/Meta flagship DCs
    "cloud_average":     1.2,   # AWS/GCP/Azure average
    "enterprise_dc":     1.5,   # typical on-premise enterprise DC
    "legacy_dc":         1.8,   # older/inefficient datacenter
    "custom":            None,  # user-supplied value
}

@dataclass
class DatacenterConfig:
    """
    Datacenter-level parameters that affect real energy consumption.

    PUE (Power Usage Effectiveness) = total DC power / IT equipment power.
    A PUE of 1.5 means for every 1W your GPU uses, the DC uses 1.5W total
    (the extra 0.5W goes to cooling, lighting, UPS losses, etc).

    carbon_intensity_kg_per_kwh: CO2 per kWh for the DC's grid region.
    Use https://app.electricitymap.org to find your region's value.
    Default is world average (0.000233 kg/Wh = 0.233 kg/kWh).

    num_nodes: for distributed training, multiply single-node metrics by this.
    """
    pue: float = 1.0
    carbon_intensity_kg_per_kwh: float = 0.000233
    num_nodes: int = 1
    dc_name: str = "local"


@dataclass
class RunMetrics:
    duration_s: float
    energy_kwh: float
    emissions_kg: float
    idle_seconds: float = 0.0
    # Datacenter-adjusted values (populated by apply_datacenter_config)
    energy_kwh_dc: Optional[float] = None
    emissions_kg_dc: Optional[float] = None
    dc_config: Optional[DatacenterConfig] = None
    # Water metrics (populated by apply_aquatensor_config)
    water: Optional[WaterMetrics] = None

    def apply_datacenter_config(self, dc: DatacenterConfig) -> "RunMetrics":
        """
        Returns a new RunMetrics with datacenter overhead applied.
        - Multiplies energy by PUE (cooling + infrastructure overhead)
        - Multiplies by num_nodes (distributed training)
        - Recalculates emissions using the DC's regional carbon intensity
        """
        total_energy = self.energy_kwh * dc.pue * dc.num_nodes
        total_emissions = total_energy * dc.carbon_intensity_kg_per_kwh
        return RunMetrics(
            duration_s=self.duration_s,
            energy_kwh=self.energy_kwh,
            emissions_kg=self.emissions_kg,
            idle_seconds=self.idle_seconds,
            energy_kwh_dc=total_energy,
            emissions_kg_dc=total_emissions,
            dc_config=dc,
            water=self.water,
        )

    def apply_aquatensor_config(self, config: AquaTensorConfig) -> "RunMetrics":
        """
        Returns a new RunMetrics with water metrics populated.
        Uses real measured energy to calculate water consumption and production.
        """
        bridge = AquaTensorBridge(config)
        water_metrics = bridge.calculate_water_metrics(self.energy_kwh, self.duration_s)
        return RunMetrics(
            duration_s=self.duration_s,
            energy_kwh=self.energy_kwh,
            emissions_kg=self.emissions_kg,
            idle_seconds=self.idle_seconds,
            energy_kwh_dc=self.energy_kwh_dc,
            emissions_kg_dc=self.emissions_kg_dc,
            dc_config=self.dc_config,
            water=water_metrics,
        )


def calculate_savings(baseline: RunMetrics, optimized: RunMetrics,
                      use_dc_adjusted: bool = False) -> dict:
    """
    Compare two RunMetrics and return savings.
    If use_dc_adjusted=True and DC config was applied, uses the
    datacenter-adjusted energy/emissions values.
    """
    b_energy = (baseline.energy_kwh_dc if use_dc_adjusted and baseline.energy_kwh_dc
                else baseline.energy_kwh)
    o_energy = (optimized.energy_kwh_dc if use_dc_adjusted and optimized.energy_kwh_dc
                else optimized.energy_kwh)
    b_emissions = (baseline.emissions_kg_dc if use_dc_adjusted and baseline.emissions_kg_dc
                   else baseline.emissions_kg)
    o_emissions = (optimized.emissions_kg_dc if use_dc_adjusted and optimized.emissions_kg_dc
                   else optimized.emissions_kg)

    energy_saved = b_energy - o_energy
    emissions_saved = b_emissions - o_emissions
    energy_reduction_pct = (energy_saved / b_energy * 100) if b_energy else 0.0
    time_saved = baseline.duration_s - optimized.duration_s

    return {
        "energy_saved_kwh": energy_saved,
        "emissions_saved_kg": emissions_saved,
        "energy_reduction_pct": energy_reduction_pct,
        "time_saved_s": time_saved,
        "dc_adjusted": use_dc_adjusted,
    }
