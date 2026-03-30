from greentensor.report.report import generate_report
from greentensor.report.metrics import RunMetrics

def test_report_no_baseline():
    r = generate_report(5.0, 0.0003, 0.0013)
    assert "5.00 s" in r
    assert "0.001300 kWh" in r
    assert "0.000300 kg" in r
    assert "Savings" not in r

def test_report_with_baseline():
    baseline = RunMetrics(duration_s=10.0, energy_kwh=0.002, emissions_kg=0.001)
    r = generate_report(7.0, 0.0006, 0.0013, baseline=baseline)
    assert "Efficiency vs Baseline" in r
    assert "reduction" in r

def test_report_with_idle():
    r = generate_report(5.0, 0.0003, 0.0013, idle_seconds=2.5)
    assert "2.50 s" in r
