import time
import threading
from .base import BaseOptimizer
from greentensor.utils.config import Config
from greentensor.utils.logger import logger

class IdleOptimizer(BaseOptimizer):
    """
    Monitors GPU utilization in a background thread.
    When the GPU is idle (below threshold), it throttles the process
    to reduce wasted energy during idle periods.
    """

    def __init__(self, config: Config = None):
        self.config = config or Config()
        self._stop_event = threading.Event()
        self._thread = None
        self.idle_seconds = 0.0
        self._lock = threading.Lock()

    def apply(self):
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._monitor, daemon=True)
        self._thread.start()
        logger.info("Idle optimizer started.")

    def revert(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2)
        logger.info(f"Idle optimizer stopped. Total idle time: {self.idle_seconds:.2f}s")

    def _monitor(self):
        from greentensor.core.profiler import Profiler
        while not self._stop_event.is_set():
            metrics = Profiler.get_gpu_metrics()
            if metrics["util_%"] < self.config.idle_threshold_pct:
                time.sleep(self.config.idle_sleep_s)
                with self._lock:
                    self.idle_seconds += self.config.idle_sleep_s
            else:
                time.sleep(0.1)
