# -*- coding: utf-8 -*-
import time

_SEV = {"low": "[LOW]", "medium": "[MED]", "high": "[HIGH]", "critical": "[CRIT]"}

def generate_report(duration, emissions_kg, energy_kwh,
                    idle_seconds=0.0, baseline=None, alerts=None,
                    footprint=None, water=None):
    lines = [
        "",
        "  +======================================+",
        "  |        GreenTensor Report  v0.6.0    |",
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

    # Water section
    if water:
        lines += ["", "  -- AquaTensor Water Intelligence ----"]
        lines.append("  Water Consumed   : {:.3f} L  (cooling, WUE={:.2f})".format(
            water.water_consumed_liters, water.wue))
        if water.aquatensor_installed:
            lines.append("  Heat Recovered   : {:.6f} kWh  (WHR={:.0f}%)".format(
                water.heat_recovered_kwh, water.heat_recovered_kwh / water.heat_generated_kwh * 100))
            lines.append("  Water Produced   : {:.3f} L  (membrane distillation @ {:.0f}C)".format(
                water.water_produced_liters, water.feed_temperature_c))
            net = water.net_water_impact_liters
            sign = "NET POSITIVE" if net < 0 else "net negative"
            lines.append("  Net Water Impact : {:.3f} L  ({})".format(net, sign))
            lines.append("  Drinking Water   : {:.1f} person-days of fresh water generated".format(
                water.drinking_water_days))
        lines.append("  Region Stress    : {} (index {:.1f}/5.0)".format(
            water.water_stress_label, water.water_stress_index))

    # Carbon anomaly alerts
    lines += ["", "  -- Carbon Anomaly Detection ---------"]
    if not alerts:
        lines.append("  Status           : CLEAN")
    else:
        lines.append("  Status           : {} alert(s) detected".format(len(alerts)))
        for a in alerts:
            ts = time.strftime("%H:%M:%S", time.localtime(a.timestamp))
            lines.append("  {} [{}] [{}] {}".format(
                _SEV.get(a.severity, "[?]"), ts, a.source, a.message))

    # Digital footprint
    lines += ["", "  -- Digital Footprint Report ---------"]
    if not footprint or footprint.is_clean:
        lines.append("  Status           : CLEAN -- No digital attack footprint detected")
    else:
        events = footprint.events
        lines.append("  Status           : {} footprint event(s)".format(len(events)))
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

    lines.append("  ======================================\n")
    return "\n".join(lines)