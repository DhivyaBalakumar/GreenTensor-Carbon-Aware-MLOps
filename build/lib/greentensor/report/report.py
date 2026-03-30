# -*- coding: utf-8 -*-
import time
from greentensor.report.metrics import RunMetrics

_SEVERITY_ICON = {
    "low": "[LOW]",
    "medium": "[MED]",
    "high": "[HIGH]",
    "critical": "[CRIT]",
}

def generate_report(duration, emissions_kg, energy_kwh,
                    idle_seconds=0.0, baseline=None, alerts=None):
    lines = [
        "",
        "  +======================================+",
        "  |        GreenTensor Report  v0.3.0    |",
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
            "  -- Efficiency vs Baseline -----------",
            "  Baseline Energy  : {:.6f} kWh".format(baseline.energy_kwh),
            "  Energy Saved     : {:.6f} kWh  ({:.1f}% reduction)".format(energy_saved, reduction_pct),
            "  Emissions Saved  : {:.6f} kg CO2".format(emissions_saved),
            "  Time Saved       : {:.2f} s".format(time_saved),
        ]

    # Security section
    lines += ["", "  -- Security Report ------------------"]
    if not alerts:
        lines.append("  Status           : CLEAN — No threats detected")
    else:
        critical = [a for a in alerts if a.severity == "critical"]
        high     = [a for a in alerts if a.severity == "high"]
        medium   = [a for a in alerts if a.severity == "medium"]
        low      = [a for a in alerts if a.severity == "low"]

        lines.append("  Status           : THREATS DETECTED")
        lines.append("  Total Alerts     : {}  (critical={}, high={}, medium={}, low={})".format(
            len(alerts), len(critical), len(high), len(medium), len(low)
        ))
        lines.append("")

        for a in alerts:
            icon = _SEVERITY_ICON.get(a.severity, "[???]")
            ts = time.strftime("%H:%M:%S", time.localtime(a.timestamp))
            lines.append("  {} [{}] [{}] {}".format(icon, ts, a.source, a.alert_type.upper()))
            lines.append("       {}".format(a.message))

        # Summary by type
        types = {}
        for a in alerts:
            types[a.alert_type] = types.get(a.alert_type, 0) + 1
        lines.append("")
        lines.append("  Alert breakdown  : " + ", ".join(
            "{}: {}".format(k, v) for k, v in types.items()
        ))

    lines.append("  ======================================\n")
    return "\n".join(lines)
