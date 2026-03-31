import time
from unittest.mock import patch
from greentensor.security.anomaly_detector import (
    AnomalyDetector, AnomalyDetectorConfig, AnomalyAlert
)

def _make_detector(on_alert=None, **kwargs):
    config = AnomalyDetectorConfig(
        sample_interval_s=0.05,
        consecutive_anomalies_required=2,
        spike_threshold_pct=50.0,
        idle_drain_threshold_w=30.0,
        idle_util_threshold_pct=5.0,
        baseline_window=10,
        use_alibi_detect=False,   # unit tests use threshold path
        use_llm_guard=False,
        **kwargs
    )
    return AnomalyDetector(config=config, on_alert=on_alert or (lambda a: None))

def test_no_alerts_on_normal_power():
    detector = _make_detector()
    normal = {"util_%": 80.0, "power_W": 100.0}
    with patch("greentensor.core.profiler.Profiler.get_gpu_metrics", return_value=normal):
        detector.start()
        time.sleep(0.4)
        alerts = detector.stop()
    assert len(alerts) == 0

def test_power_spike_triggers_alert():
    alerts_received = []
    detector = _make_detector(on_alert=alerts_received.append)
    call_count = [0]

    def mock_metrics():
        call_count[0] += 1
        if call_count[0] <= 5:
            return {"util_%": 80.0, "power_W": 100.0}
        return {"util_%": 90.0, "power_W": 300.0}

    with patch("greentensor.core.profiler.Profiler.get_gpu_metrics", side_effect=mock_metrics):
        detector.start()
        time.sleep(0.8)
        detector.stop()

    assert any(a.alert_type == "power_spike" for a in alerts_received)
    assert any(a.source == "threshold" for a in alerts_received)

def test_idle_drain_triggers_alert():
    alerts_received = []
    detector = _make_detector(on_alert=alerts_received.append)
    suspicious = {"util_%": 2.0, "power_W": 80.0}
    with patch("greentensor.core.profiler.Profiler.get_gpu_metrics", return_value=suspicious):
        detector.start()
        time.sleep(0.4)
        detector.stop()
    assert any(a.alert_type == "idle_drain" for a in alerts_received)

def test_alert_has_severity_and_source():
    alerts_received = []
    detector = _make_detector(on_alert=alerts_received.append)
    call_count = [0]

    def mock_metrics():
        call_count[0] += 1
        return {"util_%": 80.0, "power_W": 100.0} if call_count[0] <= 5 else {"util_%": 90.0, "power_W": 300.0}

    with patch("greentensor.core.profiler.Profiler.get_gpu_metrics", side_effect=mock_metrics):
        detector.start()
        time.sleep(0.8)
        detector.stop()

    for a in alerts_received:
        assert a.severity in ("low", "medium", "high", "critical")
        assert a.source in ("threshold", "alibi-detect", "llm-guard")

def test_context_manager():
    detector = _make_detector()
    normal = {"util_%": 70.0, "power_W": 90.0}
    with patch("greentensor.core.profiler.Profiler.get_gpu_metrics", return_value=normal):
        with detector:
            time.sleep(0.2)
    assert isinstance(detector.alerts, list)

def test_alert_dataclass():
    a = AnomalyAlert(
        timestamp=1234.0,
        alert_type="power_spike",
        current_value=300.0,
        baseline_value=100.0,
        deviation_pct=200.0,
        severity="critical",
        source="alibi-detect",
        message="test alert"
    )
    assert a.severity == "critical"
    assert a.source == "alibi-detect"

def test_report_shows_security_section():
    from greentensor.report.report import generate_report
    from greentensor.security.anomaly_detector import AnomalyAlert
    import time
    alerts = [
        AnomalyAlert(
            timestamp=time.time(),
            alert_type="power_spike",
            current_value=250.0,
            baseline_value=100.0,
            deviation_pct=150.0,
            severity="critical",
            source="alibi-detect",
            message="Test spike alert"
        )
    ]
    report = generate_report(5.0, 0.0003, 0.0013, alerts=alerts)
    assert "alert(s) detected" in report or "THREATS DETECTED" in report
    assert "CRIT" in report
    assert "alibi-detect" in report
