# -*- coding: utf-8 -*-
import time

_SEV = {"low": "[LOW]", "medium": "[MED]", "high": "[HIGH]", "critical": "[CRIT]"}

def generate_report(duration, emissions_kg, energy_kwh,
                    idle_seconds=0.0, baseline=None, alerts=None, footprint=None):
    lines = [
        "",
        "  +======================================+",
        "  |        GreenTensor Report  v0.4.0    |",
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

    lines += ["", "  -- Carbon Anomaly Detection ---------"]
    if not alerts:
        lines.append("  Status           : CLEAN")
    else:
        lines.append("  Status           : {} alert(s) detected".format(len(alerts)))
        for a in alerts:
            ts = time.strftime("%H:%M:%S", time.localtime(a.timestamp))
            lines.append("  {} [{}] [{}] {}".format(
                _SEV.get(a.severity, "[?]"), ts, a.source, a.message))

    lines += ["", "  -- Digital Footprint Report ---------"]
    if not footprint or footprint.is_clean:
        lines.append("  Status           : CLEAN -- No digital attack footprint detected")
    else:
        events = footprint.events
        lines.append("  Status           : {} footprint event(s)".format(len(events)))
        lines.append("  Stage            : {}".format(footprint.stage))
        by_category = {}
        for e in events:
            by_category.setdefault(e.category, []).append(e)
        for cat, evts in by_category.items():
            lines.append("")
            lines.append("  [{}]".format(cat.upper().replace("_", " ")))
            for e in evts:
                ts = time.strftime("%H:%M:%S", time.localtime(e.timestamp))
                lines.append("  {} [{}] [MITRE:{}] {}".format(
                    _SEV.get(e.severity, "[?]"), ts, e.mitre_technique, e.message))
        if footprint.network_connections:
            lines.append("")
            lines.append("  Network connections: {}".format(
                ", ".join(footprint.network_connections[:5])))
        if footprint.child_processes:
            lines.append("  Child processes: {}".format(
                ", ".join(footprint.child_processes[:5])))

    lines.append("  ======================================\n")
    return "\n".join(lines)