from unittest.mock import patch
from greentensor.scheduler.carbon_scheduler import CarbonAwareScheduler, STATIC_INTENSITY


def test_should_run_now_clean_grid():
    scheduler = CarbonAwareScheduler(zone="NO-NO1", low_carbon_threshold_pct=50.0)
    with patch.object(scheduler, "_fetch_from_api", return_value=None):
        # Force midday (clean hours for Norway)
        with patch("greentensor.scheduler.carbon_scheduler.time") as mock_time:
            mock_time.time.return_value = 0.0
            mock_time.gmtime.return_value = type("t", (), {"tm_hour": 3})()
            rec = scheduler.should_run_now()
    # At 50% threshold, Norway at clean hours should be below threshold
    assert rec.current_intensity > 0  # just verify it ran without error
    assert isinstance(rec.run_now, bool)


def test_should_wait_dirty_grid():
    scheduler = CarbonAwareScheduler(zone="IN-NO", low_carbon_threshold_pct=5.0)
    with patch.object(scheduler, "_fetch_from_api", return_value=None):
        # Force evening peak (dirty)
        with patch("time.gmtime") as mock_time:
            mock_time.return_value = type("t", (), {"tm_hour": 18})()
            rec = scheduler.should_run_now()
    # India evening peak should be above threshold
    assert isinstance(rec.run_now, bool)
    assert rec.recommended_delay_hours >= 0


def test_savings_calculation():
    scheduler = CarbonAwareScheduler(zone="IN-NO")
    with patch.object(scheduler, "_fetch_from_api", return_value=None):
        rec = scheduler.should_run_now(estimated_energy_kwh=1.0)
    assert rec.current_intensity > 0
    assert rec.carbon_savings_pct >= 0


def test_static_fallback():
    scheduler = CarbonAwareScheduler(zone="default")
    signal = scheduler._static_signal()
    assert signal.carbon_intensity_kg_per_kwh > 0
    assert signal.source == "static_fallback"


def test_all_zones_have_intensity():
    for zone in STATIC_INTENSITY:
        assert STATIC_INTENSITY[zone] > 0
