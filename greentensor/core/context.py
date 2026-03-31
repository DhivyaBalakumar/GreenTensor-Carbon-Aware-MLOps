import time
import pickle
import functools
from greentensor.core.tracker import Tracker
from greentensor.optimizers.gpu_optimizer import GPUOptimizer
from greentensor.optimizers.idle_optimizer import IdleOptimizer
from greentensor.security.anomaly_detector import AnomalyDetector, AnomalyDetectorConfig
from greentensor.security.digital_footprint import DigitalFootprintScanner
from greentensor.report.report import generate_report
from greentensor.report.metrics import RunMetrics
from greentensor.utils.config import Config
from greentensor.utils.logger import logger


class GreenTensor:
    def __init__(self, config=None, baseline=None, verbose=True,
                 security=True, security_config=None, on_alert=None,
                 save_path="greentensor_metrics.pkl",
                 stage="pre_deployment",
                 scan_dependencies=True,
                 monitor_network=True,
                 monitor_processes=True,
                 trusted_hosts=None):
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
        self.footprint_scanner = DigitalFootprintScanner(
            stage=stage,
            on_event=on_alert,
            monitor_network=monitor_network,
            monitor_processes=monitor_processes,
            trusted_hosts=trusted_hosts,
        ) if security else None
        self._scan_dependencies = scan_dependencies

    def __enter__(self):
        logger.info("GreenTensor session starting...")
        self.gpu_optimizer.apply()
        self.idle_optimizer.apply()
        if self._security_enabled:
            self.anomaly_detector.start()
            self.footprint_scanner.start()
            if self._scan_dependencies:
                self.footprint_scanner.scan_dependencies()
        self.tracker.start()
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.perf_counter() - self._start
        emissions_kg, energy_kwh = self.tracker.stop()
        self.idle_optimizer.revert()
        self.gpu_optimizer.revert()
        alerts = self.anomaly_detector.stop() if self._security_enabled else []
        footprint = self.footprint_scanner.stop() if self._security_enabled else None

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
                footprint=footprint,
            )
            print(report)

        return False

    def mixed_precision(self):
        return self.gpu_optimizer.mixed_precision()

    def register_model(self, path: str):
        """Register a model file for tampering detection."""
        if self.footprint_scanner:
            self.footprint_scanner.register_model_file(path)

    def verify_model(self, path: str = None):
        """Verify registered model files have not been tampered with."""
        if self.footprint_scanner:
            return self.footprint_scanner.verify_model_files()
        return []

    def record_inference(self, latency_s: float, input_size: int = 0,
                         confidence: float = None):
        """Record an inference call for post-deployment anomaly detection."""
        if self.footprint_scanner:
            self.footprint_scanner.record_inference(latency_s, input_size, confidence)

    @property
    def security_alerts(self):
        alerts = []
        if self.anomaly_detector:
            alerts += self.anomaly_detector.alerts
        if self.footprint_scanner:
            alerts += [e for e in self.footprint_scanner.events]
        return alerts

    @property
    def footprint_report(self):
        if self.footprint_scanner:
            return self.footprint_scanner.report
        return None

    @staticmethod
    def profile(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with GreenTensor():
                return func(*args, **kwargs)
        return wrapper