import time
import functools
from greentensor.core.tracker import Tracker
from greentensor.optimizers.gpu_optimizer import GPUOptimizer
from greentensor.optimizers.idle_optimizer import IdleOptimizer
from greentensor.report.report import generate_report
from greentensor.report.metrics import RunMetrics
from greentensor.utils.config import Config
from greentensor.utils.logger import logger


class GreenTensor:
    def __init__(self, config=None, baseline=None, verbose=True):
        self.config = config or Config()
        self.baseline = baseline
        self.verbose = verbose
        self.tracker = Tracker(self.config)
        self.gpu_optimizer = GPUOptimizer(self.config)
        self.idle_optimizer = IdleOptimizer(self.config)
        self.metrics = None

    def __enter__(self):
        logger.info("GreenTensor session starting...")
        self.gpu_optimizer.apply()
        self.idle_optimizer.apply()
        self.tracker.start()
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.perf_counter() - self._start
        emissions_kg, energy_kwh = self.tracker.stop()
        self.idle_optimizer.revert()
        self.gpu_optimizer.revert()

        self.metrics = RunMetrics(
            duration_s=duration,
            energy_kwh=energy_kwh,
            emissions_kg=emissions_kg,
            idle_seconds=self.idle_optimizer.idle_seconds,
        )

        if self.verbose:
            report = generate_report(
                duration=duration,
                emissions_kg=emissions_kg,
                energy_kwh=energy_kwh,
                idle_seconds=self.idle_optimizer.idle_seconds,
                baseline=self.baseline,
            )
            print(report)

        return False

    def mixed_precision(self):
        return self.gpu_optimizer.mixed_precision()

    @staticmethod
    def profile(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with GreenTensor():
                return func(*args, **kwargs)
        return wrapper