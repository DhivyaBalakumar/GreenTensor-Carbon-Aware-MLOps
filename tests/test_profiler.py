from unittest.mock import patch
from greentensor.core.profiler import Profiler

def test_get_gpu_metrics_no_nvidia_smi():
    # When nvidia-smi is not available, should return zeros gracefully
    with patch("subprocess.run", side_effect=FileNotFoundError):
        m = Profiler.get_gpu_metrics()
    assert m == {"util_%": 0.0, "power_W": 0.0}

def test_get_gpu_metrics_parses_output():
    mock = type("R", (), {"stdout": "45, 120.5\n", "returncode": 0})()
    with patch("subprocess.run", return_value=mock):
        m = Profiler.get_gpu_metrics()
    assert m["util_%"] == 45.0
    assert m["power_W"] == 120.5

def test_estimate_energy_kwh():
    # 100W for 3600s = 0.1 kWh
    kwh = Profiler.estimate_energy_kwh(100.0, 3600.0)
    assert abs(kwh - 0.1) < 1e-9

def test_track_gpu_decorator():
    @Profiler.track_gpu
    def dummy():
        return 42

    with patch.object(Profiler, "get_gpu_metrics", return_value={"util_%": 0.0, "power_W": 0.0}):
        result, profile = dummy()

    assert result == 42
    assert "duration_s" in profile
    assert "avg_power_W" in profile
