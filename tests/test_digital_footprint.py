import time
from greentensor.security.digital_footprint import (
    DigitalFootprintScanner, FootprintReport
)


def _make_scanner(on_event=None, **kwargs):
    return DigitalFootprintScanner(
        stage="pre_deployment",
        on_event=on_event or (lambda e: None),
        monitor_network=False,
        monitor_processes=False,
        **kwargs
    )


def test_clean_session():
    scanner = _make_scanner()
    scanner.start()
    time.sleep(0.05)
    report = scanner.stop()
    assert isinstance(report, FootprintReport)
    assert report.is_clean


def test_model_file_tampering(tmp_path):
    model_file = tmp_path / "model.pt"
    model_file.write_bytes(b"original weights")
    scanner = _make_scanner()
    scanner.register_model_file(str(model_file))
    model_file.write_bytes(b"tampered weights")
    events = scanner.verify_model_files()
    assert len(events) == 1
    assert events[0].category == "model_tampering"
    assert events[0].signal == "model_weight_modified"
    assert events[0].severity == "critical"
    assert events[0].mitre_technique == "AML.T0018"


def test_model_file_deleted(tmp_path):
    model_file = tmp_path / "model.pt"
    model_file.write_bytes(b"weights")
    scanner = _make_scanner()
    scanner.register_model_file(str(model_file))
    model_file.unlink()
    events = scanner.verify_model_files()
    assert any(e.signal == "model_file_deleted" for e in events)


def test_inference_latency_spike():
    events_received = []
    scanner = _make_scanner(on_event=events_received.append)
    for _ in range(12):
        scanner.record_inference(0.01)
    scanner.record_inference(0.5)
    assert any(e.signal == "latency_spike" for e in events_received)


def test_model_extraction_detection():
    events_received = []
    scanner = _make_scanner(on_event=events_received.append)
    # Pre-populate 25 calls spaced 1ms apart = 1000 calls/sec
    t0 = time.time() - 0.025  # 25ms ago
    for i in range(25):
        scanner._api_call_times.append(t0 + i * 0.001)
        scanner._inference_latencies.append(0.001)
    # This 26th call triggers the check — last 20 calls span 19ms = ~1052/sec
    scanner.record_inference(0.001)
    assert any(e.signal == "high_frequency_probing" for e in events_received)


def test_dependency_scan_no_crash():
    scanner = _make_scanner()
    events = scanner.scan_dependencies()
    assert isinstance(events, list)


def test_footprint_event_has_mitre():
    scanner = _make_scanner()
    e = scanner._make_event(
        category="supply_chain", signal="test", severity="high",
        evidence={}, message="test", mitre="AML.T0010"
    )
    assert e.mitre_technique == "AML.T0010"
    assert e.stage == "pre_deployment"


def test_context_manager():
    scanner = _make_scanner()
    with scanner:
        time.sleep(0.05)
    assert scanner.report.session_end > 0


def test_post_deployment_stage():
    scanner = DigitalFootprintScanner(
        stage="post_deployment",
        on_event=lambda e: None,
        monitor_network=False,
        monitor_processes=False,
    )
    assert scanner.stage == "post_deployment"
