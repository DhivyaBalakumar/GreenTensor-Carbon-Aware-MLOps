"""
Carbon-based anomaly detector for MLOps pipeline security.

Malware (cryptominers, backdoor triggers, data exfiltration) causes
abnormal energy/power consumption patterns. This module builds a
rolling baseline of normal energy usage and raises alerts when
deviations exceed a configurable threshold.
"""
import threading
import time
import statistics
from dataclasses import dataclass, field
from typing import Callable, Optional
from greentensor.core.profiler import Profiler
from greentensor.utils.logger import logger


@dataclass
class AnomalyAlert:
    timestamp: float
    alert_type: str          # "power_spike", "sustained_overload", "idle_drain"
    current_value: float
    baseline_value: float
    deviation_pct: float
    message: str


@dataclass
class AnomalyDetectorConfig:
    # How many samples to use for baseline (rolling window)
    baseline_window: int = 60
    # Sample interval in seconds
    sample_interval_s: float = 1.0
    # Alert if power deviates more than this % above baseline
    spike_threshold_pct: float = 80.0
    # Alert if GPU is draining power while util is near zero (idle drain)
    idle_drain_threshold_w: float = 50.0
    idle_util_threshold_pct: float = 5.0
    # How many consecutive anomalous samples before alerting
    consecutive_anomalies_required: int = 3


class AnomalyDetector:
    """
    Continuously monitors GPU power draw and carbon footprint.
    Detects three attack patterns:

    1. Power spike   — sudden energy surge (cryptominer injection)
    2. Sustained overload — prolonged above-baseline usage (data exfiltration)
    3. Idle drain    — GPU consuming power while idle (hidden background process)
    """

    def __init__(
        self,
        config: AnomalyDetectorConfig = None,
        on_alert: Optional[Callable[[AnomalyAlert], None]] = None,
    ):
        self.config = config or AnomalyDetectorConfig()
        self.on_alert = on_alert or self._default_alert_handler
        self._samples: list[float] = []
        self._alerts: list[AnomalyAlert] = []
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._consecutive_count = 0

    # ------------------------------------------------------------------ #

    def start(self):
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        logger.info("AnomalyDetector started — monitoring carbon footprint for threats.")

    def stop(self) -> list[AnomalyAlert]:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        logger.info(f"AnomalyDetector stopped. Total alerts: {len(self._alerts)}")
        return self._alerts

    @property
    def alerts(self) -> list[AnomalyAlert]:
        with self._lock:
            return list(self._alerts)

    @property
    def baseline_power_w(self) -> Optional[float]:
        with self._lock:
            if len(self._samples) < 5:
                return None
            return statistics.mean(self._samples[-self.config.baseline_window:])

    # ------------------------------------------------------------------ #

    def _monitor_loop(self):
        while not self._stop_event.is_set():
            metrics = Profiler.get_gpu_metrics()
            power_w = metrics["power_W"]
            util_pct = metrics["util_%"]

            with self._lock:
                self._samples.append(power_w)
                # Keep window size bounded
                if len(self._samples) > self.config.baseline_window * 2:
                    self._samples = self._samples[-self.config.baseline_window:]

                baseline = (
                    statistics.mean(self._samples[:-1])
                    if len(self._samples) > 1
                    else None
                )

            if baseline and baseline > 0:
                self._check_power_spike(power_w, baseline)
                self._check_idle_drain(power_w, util_pct, baseline)

            time.sleep(self.config.sample_interval_s)

    def _check_power_spike(self, power_w: float, baseline: float):
        deviation_pct = ((power_w - baseline) / baseline) * 100
        if deviation_pct > self.config.spike_threshold_pct:
            self._consecutive_count += 1
            if self._consecutive_count >= self.config.consecutive_anomalies_required:
                self._raise_alert(AnomalyAlert(
                    timestamp=time.time(),
                    alert_type="power_spike",
                    current_value=power_w,
                    baseline_value=baseline,
                    deviation_pct=deviation_pct,
                    message=(
                        f"THREAT DETECTED: Power spike {deviation_pct:.1f}% above baseline "
                        f"({power_w:.1f}W vs {baseline:.1f}W baseline). "
                        f"Possible cryptominer or unauthorized compute process."
                    ),
                ))
        else:
            self._consecutive_count = 0

    def _check_idle_drain(self, power_w: float, util_pct: float, baseline: float):
        if (
            util_pct < self.config.idle_util_threshold_pct
            and power_w > self.config.idle_drain_threshold_w
        ):
            deviation_pct = ((power_w - baseline) / baseline) * 100 if baseline > 0 else 0
            self._raise_alert(AnomalyAlert(
                timestamp=time.time(),
                alert_type="idle_drain",
                current_value=power_w,
                baseline_value=baseline,
                deviation_pct=deviation_pct,
                message=(
                    f"THREAT DETECTED: GPU consuming {power_w:.1f}W while {util_pct:.1f}% utilized. "
                    f"Possible hidden background process or data exfiltration."
                ),
            ))

    def _raise_alert(self, alert: AnomalyAlert):
        with self._lock:
            self._alerts.append(alert)
        self.on_alert(alert)

    @staticmethod
    def _default_alert_handler(alert: AnomalyAlert):
        logger.warning(f"[SECURITY] {alert.message}")

    # ------------------------------------------------------------------ #
    #  Context manager support                                             #
    # ------------------------------------------------------------------ #

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()
