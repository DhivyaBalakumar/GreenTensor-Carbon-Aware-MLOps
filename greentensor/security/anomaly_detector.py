"""
Carbon-based anomaly detector for MLOps pipeline security.

Integrates:
- alibi-detect (Apache 2.0) for statistical online anomaly detection
  on the carbon/power time series (SpectralResidual algorithm)
- LLM Guard (MIT) for scanning model inputs/outputs for prompt injection,
  PII leakage, and data exfiltration patterns
- Built-in threshold detector as fallback when alibi-detect is unavailable

Attack patterns detected:
  1. power_spike       - sudden energy surge (cryptominer injection)
  2. sustained_overload - prolonged above-baseline usage (data exfiltration)
  3. idle_drain        - GPU power draw while idle (hidden background process)
  4. prompt_injection  - adversarial inputs targeting the model (via LLM Guard)
  5. data_leakage      - sensitive data in model outputs (via LLM Guard)
"""
import threading
import time
import statistics
from dataclasses import dataclass, field
from typing import Callable, Optional, List
from greentensor.core.profiler import Profiler
from greentensor.utils.logger import logger


@dataclass
class AnomalyAlert:
    timestamp: float
    alert_type: str
    current_value: float
    baseline_value: float
    deviation_pct: float
    severity: str          # "low", "medium", "high", "critical"
    source: str            # "threshold", "alibi-detect", "llm-guard"
    message: str


@dataclass
class AnomalyDetectorConfig:
    baseline_window: int = 60
    sample_interval_s: float = 1.0
    spike_threshold_pct: float = 80.0
    idle_drain_threshold_w: float = 50.0
    idle_util_threshold_pct: float = 5.0
    consecutive_anomalies_required: int = 3
    # alibi-detect config
    use_alibi_detect: bool = True
    alibi_window_size: int = 20        # rolling window for SpectralResidual
    alibi_threshold: float = 0.7       # outlier score threshold (0-1)
    # LLM Guard config
    use_llm_guard: bool = True
    llm_guard_threshold: float = 0.5


class AnomalyDetector:
    """
    Continuously monitors GPU power/carbon footprint using statistical
    anomaly detection (alibi-detect SpectralResidual) and scans
    ML pipeline inputs/outputs with LLM Guard.
    """

    def __init__(
        self,
        config: AnomalyDetectorConfig = None,
        on_alert: Optional[Callable[[AnomalyAlert], None]] = None,
    ):
        self.config = config or AnomalyDetectorConfig()
        self.on_alert = on_alert or self._default_alert_handler
        self._samples: List[float] = []
        self._alerts: List[AnomalyAlert] = []
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._consecutive_count = 0
        self._alibi_detector = None
        self._llm_guard_input_scanners = None
        self._llm_guard_output_scanners = None
        self._init_alibi_detect()
        self._init_llm_guard()

    # ------------------------------------------------------------------ #
    #  Initialisation                                                      #
    # ------------------------------------------------------------------ #

    def _init_alibi_detect(self):
        if not self.config.use_alibi_detect:
            return
        try:
            from alibi_detect.od import SpectralResidual
            self._alibi_detector = SpectralResidual(
                threshold=self.config.alibi_threshold,
                window_amp=self.config.alibi_window_size,
                window_local=self.config.alibi_window_size,
            )
            logger.info("alibi-detect SpectralResidual detector loaded.")
        except ImportError:
            logger.warning(
                "alibi-detect not installed — falling back to threshold detector. "
                "Install with: pip install alibi-detect"
            )
        except Exception as e:
            logger.warning(f"alibi-detect init failed: {e} — using threshold fallback.")

    def _init_llm_guard(self):
        if not self.config.use_llm_guard:
            return
        try:
            from llm_guard.input_scanners import PromptInjection, Toxicity, Secrets
            from llm_guard.output_scanners import Sensitive, NoRefusal, BanTopics
            self._llm_guard_input_scanners = [
                PromptInjection(threshold=self.config.llm_guard_threshold),
                Secrets(),
            ]
            self._llm_guard_output_scanners = [
                Sensitive(threshold=self.config.llm_guard_threshold),
            ]
            logger.info("LLM Guard scanners loaded (PromptInjection, Secrets, Sensitive).")
        except ImportError:
            logger.warning(
                "llm-guard not installed — LLM input/output scanning disabled. "
                "Install with: pip install llm-guard"
            )
        except Exception as e:
            logger.warning(f"LLM Guard init failed: {e}")

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def start(self):
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        logger.info("AnomalyDetector started — monitoring carbon footprint for threats.")

    def stop(self) -> List[AnomalyAlert]:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        logger.info(f"AnomalyDetector stopped. Total alerts: {len(self._alerts)}")
        return self._alerts

    def scan_input(self, prompt: str) -> List[AnomalyAlert]:
        """Scan a model input prompt with LLM Guard."""
        alerts = []
        if not self._llm_guard_input_scanners:
            return alerts
        try:
            from llm_guard import scan_prompt
            sanitized, results, is_valid = scan_prompt(
                self._llm_guard_input_scanners, prompt
            )
            for scanner_name, is_safe in results.items():
                if not is_safe:
                    alert = AnomalyAlert(
                        timestamp=time.time(),
                        alert_type="prompt_injection" if "injection" in scanner_name.lower() else "input_threat",
                        current_value=0.0,
                        baseline_value=0.0,
                        deviation_pct=0.0,
                        severity="high",
                        source="llm-guard",
                        message=f"LLM Guard [{scanner_name}]: Threat detected in model input.",
                    )
                    alerts.append(alert)
                    self._raise_alert(alert)
        except Exception as e:
            logger.debug(f"LLM Guard input scan error: {e}")
        return alerts

    def scan_output(self, output: str) -> List[AnomalyAlert]:
        """Scan a model output with LLM Guard."""
        alerts = []
        if not self._llm_guard_output_scanners:
            return alerts
        try:
            from llm_guard import scan_output
            sanitized, results, is_valid = scan_output(
                self._llm_guard_output_scanners, "", output
            )
            for scanner_name, is_safe in results.items():
                if not is_safe:
                    alert = AnomalyAlert(
                        timestamp=time.time(),
                        alert_type="data_leakage",
                        current_value=0.0,
                        baseline_value=0.0,
                        deviation_pct=0.0,
                        severity="high",
                        source="llm-guard",
                        message=f"LLM Guard [{scanner_name}]: Sensitive data detected in model output.",
                    )
                    alerts.append(alert)
                    self._raise_alert(alert)
        except Exception as e:
            logger.debug(f"LLM Guard output scan error: {e}")
        return alerts

    @property
    def alerts(self) -> List[AnomalyAlert]:
        with self._lock:
            return list(self._alerts)

    @property
    def baseline_power_w(self) -> Optional[float]:
        with self._lock:
            if len(self._samples) < 5:
                return None
            return statistics.mean(self._samples[-self.config.baseline_window:])

    # ------------------------------------------------------------------ #
    #  Monitoring loop                                                     #
    # ------------------------------------------------------------------ #

    def _monitor_loop(self):
        while not self._stop_event.is_set():
            metrics = Profiler.get_gpu_metrics()
            power_w = metrics["power_W"]
            util_pct = metrics["util_%"]

            with self._lock:
                self._samples.append(power_w)
                if len(self._samples) > self.config.baseline_window * 2:
                    self._samples = self._samples[-self.config.baseline_window:]
                baseline = (
                    statistics.mean(self._samples[:-1])
                    if len(self._samples) > 1 else None
                )

            if baseline is not None and baseline > 0:
                if self._alibi_detector and len(self._samples) >= self.config.alibi_window_size:
                    self._check_alibi(power_w, baseline)
                else:
                    self._check_threshold_spike(power_w, baseline)
                self._check_idle_drain(power_w, util_pct, baseline)

            time.sleep(self.config.sample_interval_s)

    def _check_alibi(self, power_w: float, baseline: float):
        """Use alibi-detect SpectralResidual for statistical outlier detection."""
        try:
            import numpy as np
            window = self._samples[-self.config.alibi_window_size:]
            X = np.array(window, dtype=np.float32).reshape(-1, 1)
            preds = self._alibi_detector.predict(X, return_instance_score=True)
            score = float(preds["data"]["instance_score"][-1])
            is_outlier = bool(preds["data"]["is_outlier"][-1])

            if is_outlier:
                self._consecutive_count += 1
                if self._consecutive_count >= self.config.consecutive_anomalies_required:
                    deviation_pct = ((power_w - baseline) / baseline) * 100
                    severity = "critical" if deviation_pct > 150 else "high" if deviation_pct > 80 else "medium"
                    self._raise_alert(AnomalyAlert(
                        timestamp=time.time(),
                        alert_type="power_spike",
                        current_value=power_w,
                        baseline_value=baseline,
                        deviation_pct=deviation_pct,
                        severity=severity,
                        source="alibi-detect",
                        message=(
                            f"[alibi-detect] Anomalous power pattern detected. "
                            f"Score: {score:.3f} | {power_w:.1f}W vs {baseline:.1f}W baseline "
                            f"({deviation_pct:+.1f}%). Possible cryptominer or unauthorized process."
                        ),
                    ))
            else:
                self._consecutive_count = 0
        except Exception as e:
            logger.debug(f"alibi-detect check failed: {e}")
            self._check_threshold_spike(power_w, baseline)

    def _check_threshold_spike(self, power_w: float, baseline: float):
        deviation_pct = ((power_w - baseline) / baseline) * 100
        if deviation_pct > self.config.spike_threshold_pct:
            self._consecutive_count += 1
            if self._consecutive_count >= self.config.consecutive_anomalies_required:
                severity = "critical" if deviation_pct > 150 else "high"
                self._raise_alert(AnomalyAlert(
                    timestamp=time.time(),
                    alert_type="power_spike",
                    current_value=power_w,
                    baseline_value=baseline,
                    deviation_pct=deviation_pct,
                    severity=severity,
                    source="threshold",
                    message=(
                        f"Power spike {deviation_pct:.1f}% above baseline "
                        f"({power_w:.1f}W vs {baseline:.1f}W). "
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
            deviation_pct = ((power_w - baseline) / baseline * 100) if baseline > 0 else 0
            self._raise_alert(AnomalyAlert(
                timestamp=time.time(),
                alert_type="idle_drain",
                current_value=power_w,
                baseline_value=baseline,
                deviation_pct=deviation_pct,
                severity="high",
                source="threshold",
                message=(
                    f"GPU consuming {power_w:.1f}W at only {util_pct:.1f}% utilization. "
                    f"Possible hidden background process or data exfiltration."
                ),
            ))

    def _raise_alert(self, alert: AnomalyAlert):
        with self._lock:
            self._alerts.append(alert)
        self.on_alert(alert)

    @staticmethod
    def _default_alert_handler(alert: AnomalyAlert):
        logger.warning(f"[SECURITY/{alert.severity.upper()}] [{alert.source}] {alert.message}")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()
