from greentensor.report.metrics import RunMetrics, calculate_savings

def test_calculate_savings_basic():
    baseline = RunMetrics(duration_s=10.0, energy_kwh=0.002, emissions_kg=0.001)
    optimized = RunMetrics(duration_s=7.0, energy_kwh=0.0012, emissions_kg=0.0006)
    s = calculate_savings(baseline, optimized)

    assert abs(s["energy_saved_kwh"] - 0.0008) < 1e-9
    assert abs(s["emissions_saved_kg"] - 0.0004) < 1e-9
    assert abs(s["energy_reduction_pct"] - 40.0) < 1e-6
    assert abs(s["time_saved_s"] - 3.0) < 1e-9

def test_calculate_savings_zero_baseline():
    baseline = RunMetrics(duration_s=0.0, energy_kwh=0.0, emissions_kg=0.0)
    optimized = RunMetrics(duration_s=1.0, energy_kwh=0.001, emissions_kg=0.0)
    s = calculate_savings(baseline, optimized)
    assert s["energy_reduction_pct"] == 0.0  # no div-by-zero

def test_runmetrics_defaults():
    m = RunMetrics(duration_s=5.0, energy_kwh=0.001, emissions_kg=0.0002)
    assert m.idle_seconds == 0.0
