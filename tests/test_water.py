from greentensor.water.aquatensor import AquaTensorBridge, AquaTensorConfig, PROVIDER_WUE, MD_YIELD_BY_TEMP
from greentensor.report.metrics import RunMetrics


def _metrics():
    return RunMetrics(duration_s=120.0, energy_kwh=0.0025, emissions_kg=0.00058)


def test_water_consumed_no_aquatensor():
    config = AquaTensorConfig(provider="google", region="US-West", aquatensor_installed=False)
    bridge = AquaTensorBridge(config)
    w = bridge.calculate_water_metrics(0.0025, 120.0)
    assert w.water_consumed_liters == 0.0025 * PROVIDER_WUE["google"]
    assert w.water_produced_liters == 0.0
    assert w.net_water_impact_liters == w.water_consumed_liters


def test_water_produced_with_aquatensor():
    config = AquaTensorConfig(
        provider="google", region="India",
        aquatensor_installed=True,
        whr_ratio=0.65, feed_temperature_c=60.0
    )
    bridge = AquaTensorBridge(config)
    w = bridge.calculate_water_metrics(0.0025, 120.0)
    assert w.water_produced_liters > 0
    assert w.heat_recovered_kwh == 0.0025 * 0.65
    expected_yield = 5.5  # MD_YIELD at 60C
    assert abs(w.md_yield_liters_per_kwh - expected_yield) < 0.01


def test_net_water_positive():
    # High WHR + high MD yield should produce more water than consumed
    config = AquaTensorConfig(
        provider="aws", region="India",
        aquatensor_installed=True,
        whr_ratio=0.95, feed_temperature_c=80.0
    )
    bridge = AquaTensorBridge(config)
    w = bridge.calculate_water_metrics(1.0, 3600.0)
    # aws WUE=1.8, so consumed=1.8L. Produced = 1.0 * 0.95 * 8.5 = 8.075L
    assert w.is_net_water_positive


def test_md_yield_interpolation():
    config = AquaTensorConfig(aquatensor_installed=True, feed_temperature_c=55.0)
    bridge = AquaTensorBridge(config)
    yield_55 = bridge._get_md_yield(55.0)
    # Should be between 40C yield (4.0) and 60C yield (5.5)
    assert 4.0 < yield_55 < 5.5


def test_apply_aquatensor_config():
    metrics = _metrics()
    config = AquaTensorConfig(aquatensor_installed=True, feed_temperature_c=60.0)
    m2 = metrics.apply_aquatensor_config(config)
    assert m2.water is not None
    assert m2.water.energy_kwh == metrics.energy_kwh
    assert m2.energy_kwh == metrics.energy_kwh  # original unchanged


def test_heat_forecast():
    config = AquaTensorConfig(aquatensor_installed=True, feed_temperature_c=65.0, whr_ratio=0.7)
    bridge = AquaTensorBridge(config)
    jobs = [
        {"name": "job1", "estimated_duration_s": 3600, "estimated_power_w": 300},
        {"name": "job2", "estimated_duration_s": 7200, "estimated_power_w": 250},
    ]
    forecast = bridge.forecast_heat(jobs)
    assert forecast.predicted_energy_kwh > 0
    assert forecast.predicted_water_liters > 0
    assert forecast.temperature_sustained is True


def test_water_stress_labels():
    config_low = AquaTensorConfig(region="Norway")
    bridge = AquaTensorBridge(config_low)
    w = bridge.calculate_water_metrics(0.001, 60.0)
    assert w.water_stress_label == "Low"

    config_high = AquaTensorConfig(region="India")
    bridge2 = AquaTensorBridge(config_high)
    w2 = bridge2.calculate_water_metrics(0.001, 60.0)
    assert w2.water_stress_label == "Extremely High"


def test_drinking_water_equivalency():
    config = AquaTensorConfig(aquatensor_installed=True, whr_ratio=1.0, feed_temperature_c=60.0)
    bridge = AquaTensorBridge(config)
    # 1 kWh * 1.0 WHR * 5.5 L/kWh = 5.5L produced / 2L per day = 2.75 person-days
    w = bridge.calculate_water_metrics(1.0, 3600.0)
    assert abs(w.drinking_water_days - 5.5 / 2.0) < 0.01
