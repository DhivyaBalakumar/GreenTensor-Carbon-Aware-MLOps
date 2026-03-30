import time
from unittest.mock import patch, MagicMock
from greentensor.core.context import GreenTensor
from greentensor.report.metrics import RunMetrics

def _mock_tracker():
    t = MagicMock()
    t.start.return_value = None
    t.stop.return_value = (0.00001, 0.00005)  # (emissions_kg, energy_kwh)
    return t

def test_context_manager_runs():
    with patch("greentensor.core.context.Tracker", return_value=_mock_tracker()), \
         patch("greentensor.core.context.GPUOptimizer") as mock_gpu, \
         patch("greentensor.core.context.IdleOptimizer") as mock_idle:

        mock_idle.return_value.idle_seconds = 0.0
        with GreenTensor(verbose=False) as gt:
            time.sleep(0.01)

        assert gt.metrics is not None
        assert gt.metrics.duration_s > 0
        assert gt.metrics.energy_kwh == 0.00005
        assert gt.metrics.emissions_kg == 0.00001

def test_context_manager_does_not_suppress_exceptions():
    with patch("greentensor.core.context.Tracker", return_value=_mock_tracker()), \
         patch("greentensor.core.context.GPUOptimizer"), \
         patch("greentensor.core.context.IdleOptimizer") as mock_idle:

        mock_idle.return_value.idle_seconds = 0.0
        try:
            with GreenTensor(verbose=False):
                raise ValueError("test error")
        except ValueError as e:
            assert str(e) == "test error"
        else:
            assert False, "Exception should not have been suppressed"

def test_profile_decorator():
    with patch("greentensor.core.context.Tracker", return_value=_mock_tracker()), \
         patch("greentensor.core.context.GPUOptimizer"), \
         patch("greentensor.core.context.IdleOptimizer") as mock_idle:

        mock_idle.return_value.idle_seconds = 0.0

        @GreenTensor.profile
        def my_func():
            return 99

        result = my_func()
        assert result == 99
