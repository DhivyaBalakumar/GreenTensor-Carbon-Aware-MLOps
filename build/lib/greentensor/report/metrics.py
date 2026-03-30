from dataclasses import dataclass

@dataclass
class RunMetrics:
    duration_s: float
    energy_kwh: float
    emissions_kg: float
    idle_seconds: float = 0.0

def calculate_savings(baseline: RunMetrics, optimized: RunMetrics) -> dict:
    """Compare two real RunMetrics objects and return absolute + relative savings."""
    energy_saved = baseline.energy_kwh - optimized.energy_kwh
    emissions_saved = baseline.emissions_kg - optimized.emissions_kg
    energy_reduction_pct = (energy_saved / baseline.energy_kwh * 100) if baseline.energy_kwh else 0.0
    time_saved = baseline.duration_s - optimized.duration_s

    return {
        "energy_saved_kwh": energy_saved,
        "emissions_saved_kg": emissions_saved,
        "energy_reduction_pct": energy_reduction_pct,
        "time_saved_s": time_saved,
    }
