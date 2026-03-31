import time
import pickle
import functools
from greentensor.core.tracker import Tracker
from greentensor.optimizers.gpu_optimizer import GPUOptimizer
from greentensor.optimizers.idle_optimizer import IdleOptimizer
from greentensor.security.anomaly_detector import AnomalyDetector, AnomalyDetectorConfig
from greentensor.report.report import generate_report
from greentensor.report.metrics import RunMetrics
from greentensor.utils.config import Config
from greentensor.utils.logger import logger


class GreenTensor:
    def __init__(self, config=None, baseline=None, verbose=True,
                 security=True, security_config=None, on_alert=None,
                 save_path="greentensor_metrics.pkl"):
        self.config = config or Config()
        self.baseline = baseline
        self.verbose = verbose
        self.save_path = save_path
        self.tracker = Tracker(self.config)
        self.gpu_optimizer = GPUOptimizer(self.config)
        self.idle_optimizer = IdleOptimizer(self.config)
        self.metrics = None
        self._security_enabled = security
        self.anomaly_detector = AnomalyDetector(
            config=security_config or AnomalyDetectorConfig(),
            on_alert=on_alert,
        ) if security else None

    def __enter__(self):
        logger.info("GreenTensor session starting...")
        self.gpu_optimizer.apply()
        self.idle_optimizer.apply()
        if self._security_enabled:
            self.anomaly_detector.start()
        self.tracker.start()
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.perf_counter() - self._start
        emissions_kg, energy_kwh = self.tracker.stop()
        self.idle_optimizer.revert()
        self.gpu_optimizer.revert()
        alerts = self.anomaly_detector.stop() if self._security_enabled else []

        self.metrics = RunMetrics(
            duration_s=duration,
            energy_kwh=energy_kwh,
            emissions_kg=emissions_kg,
            idle_seconds=self.idle_optimizer.idle_seconds,
        )

        if self.save_path:
            try:
                with open(self.save_path, "wb") as f:
                    pickle.dump(self.metrics, f)
                logger.info(f"Metrics saved to {self.save_path}")
            except Exception as e:
                logger.warning(f"Could not save metrics: {e}")

        if self.verbose:
            report = generate_report(
                duration=duration,
                emissions_kg=emissions_kg,
                energy_kwh=energy_kwh,
                idle_seconds=self.idle_optimizer.idle_seconds,
                baseline=self.baseline,
                alerts=alerts,
            )
            print(report)

        return False

    def mixed_precision(self):
        return self.gpu_optimizer.mixed_precision()

    @property
    def security_alerts(self):
        if self.anomaly_detector:
            return self.anomaly_detector.alerts
        return []

    @staticmethod
    def profile(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with GreenTensor():
                return func(*args, **kwargs)
        return wrapper