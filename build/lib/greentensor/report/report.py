# -*- coding: utf-8 -*-
from greentensor.report.metrics import RunMetrics

def generate_report(duration, emissions_kg, energy_kwh,
                    idle_seconds=0.0, baseline=None):
    """
    Generate a human-readable GreenTensor report.
    If a baseline RunMetrics is provided, savings are computed against real data.
    """
    lines = [
        "",
        "  +======================================+",
        "  |        GreenTensor Report            |",
        "  +======================================+",
        "  Runtime          : {:.2f} s".format(duration),
        "  Energy Used      : {:.6f} kWh".format(energy_kwh),
        "  CO2 Emissions    : {:.6f} kg".format(emissions_kg),
    ]

    if idle_seconds > 0:
        lines.append("  Idle Time        : {:.2f} s".format(idle_seconds))

    if baseline:
        energy_saved = baseline.energy_kwh - energy_kwh
        emissions_saved = baseline.emissions_kg - emissions_kg
        reduction_pct = (energy_saved / baseline.energy_kwh * 100) if baseline.energy_kwh else 0.0
        time_saved = baseline.duration_s - duration

        lines += [
            "",
            "  -- Savings vs Baseline --",
            "  Baseline Energy  : {:.6f} kWh".format(baseline.energy_kwh),
            "  Energy Saved     : {:.6f} kWh  ({:.1f}% reduction)".format(energy_saved, reduction_pct),
            "  Emissions Saved  : {:.6f} kg CO2".format(emissions_saved),
            "  Time Saved       : {:.2f} s".format(time_saved),
        ]

    lines.append("  ======================================\n")
    return "\n".join(lines)
