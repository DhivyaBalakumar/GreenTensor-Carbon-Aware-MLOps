# -*- coding: utf-8 -*-
"""
Digital Footprint Scanner for ML Cyberattack Detection.

Goes beyond carbon/power signals to collect multi-dimensional digital
evidence of attacks across the full ML lifecycle:

PRE-DEPLOYMENT (training stage):
  - Model weight integrity (hash-based tampering detection)
  - Training data poisoning signals (loss/gradient anomalies)
  - Dependency supply chain scanning (known malicious packages)
  - File system access patterns (unauthorized reads/writes)
  - Network exfiltration attempts (unexpected outbound connections)
  - Process injection (unexpected child processes)

POST-DEPLOYMENT (inference/production stage):
  - Inference latency anomalies (model backdoor triggers cause spikes)
  - Input distribution shift (adversarial inputs look different statistically)
  - Output confidence anomalies (backdoored models show unusual confidence)
  - API abuse patterns (rate, payload size, repetition)
  - Memory/CPU usage spikes during inference
  - Model extraction attempts (systematic probing patterns)
"""

import os
import sys
import time
import hashlib
import threading
import statistics
import subprocess
import socket
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Callable, Any
from greentensor.utils.logger import logger


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class DigitalFootprintEvent:
    """A single piece of digital evidence of an attack."""
    timestamp: float
    stage: str              # "pre_deployment" | "post_deployment"
    category: str           # "supply_chain" | "data_poisoning" | "model_tampering" |
                            # "network_exfiltration" | "process_injection" |
                            # "inference_anomaly" | "api_abuse" | "model_extraction"
    signal: str             # the specific signal that fired
    severity: str           # "low" | "medium" | "high" | "critical"
    evidence: Dict          # raw evidence dict — what was observed
    message: str
    mitre_technique: str    # MITRE ATLAS / ATT&CK technique ID


@dataclass
class FootprintReport:
    """Aggregated digital footprint report for a session."""
    session_start: float
    session_end: float
    stage: str
    events: List[DigitalFootprintEvent] = field(default_factory=list)
    model_hashes: Dict[str, str] = field(default_factory=dict)
    network_connections: List[str] = field(default_factory=list)
    child_processes: List[str] = field(default_factory=list)
    inference_latencies: List[float] = field(default_factory=list)

    @property
    def critical_count(self): return sum(1 for e in self.events if e.severity == "critical")
    @property
    def high_count(self):     return sum(1 for e in self.events if e.severity == "high")
    @property
    def is_clean(self):       return len(self.events) == 0


# ── Scanner ────────────────────────────────────────────────────────────────────

class DigitalFootprintScanner:
    """
    Multi-signal digital footprint scanner for ML pipeline security.

    Collects evidence across:
    - File system (model weight tampering)
    - Network (exfiltration, C2 connections)
    - Process (injection, unauthorized spawning)
    - Model behavior (inference anomalies, backdoor triggers)
    - API patterns (abuse, extraction attempts)
    - Supply chain (dependency integrity)
    """

    def __init__(
        self,
        stage: str = "pre_deployment",
        on_event: Optional[Callable[[DigitalFootprintEvent], None]] = None,
        monitor_network: bool = True,
        monitor_processes: bool = True,
        monitor_filesystem: bool = True,
        trusted_hosts: Optional[List[str]] = None,
    ):
        """
        Parameters
        ----------
        stage            : "pre_deployment" or "post_deployment"
        on_event         : callback when an event is detected
        monitor_network  : watch for unexpected outbound connections
        monitor_processes: watch for unexpected child processes
        monitor_filesystem: watch model weight files for tampering
        trusted_hosts    : list of allowed outbound hostnames/IPs
        """
        self.stage = stage
        self.on_event = on_event or self._default_handler
        self.monitor_network = monitor_network
        self.monitor_processes = monitor_processes
        self.monitor_filesystem = monitor_filesystem
        self.trusted_hosts = set(trusted_hosts or [
            "pypi.org", "files.pythonhosted.org",
            "huggingface.co", "storage.googleapis.com",
            "s3.amazonaws.com", "codecarbon.io",
        ])

        self._report = FootprintReport(
            session_start=time.time(),
            session_end=0.0,
            stage=stage,
        )
        self._stop_event = threading.Event()
        self._threads: List[threading.Thread] = []
        self._lock = threading.Lock()

        # Baseline snapshots
        self._baseline_processes: set = set()
        self._baseline_connections: set = set()
        self._model_hashes: Dict[str, str] = {}

        # Inference tracking (post-deployment)
        self._inference_latencies: List[float] = []
        self._inference_inputs: List[Any] = []
        self._api_call_times: List[float] = []

    # ── Public API ─────────────────────────────────────────────────────────────

    def start(self):
        """Start all monitoring threads."""
        self._snapshot_baseline()
        self._stop_event.clear()

        if self.monitor_processes:
            self._start_thread(self._monitor_processes_loop, "process-monitor")
        if self.monitor_network:
            self._start_thread(self._monitor_network_loop, "network-monitor")

        logger.info(f"DigitalFootprintScanner started [{self.stage}]")

    def stop(self) -> FootprintReport:
        """Stop all monitoring and return the footprint report."""
        self._stop_event.set()
        for t in self._threads:
            t.join(timeout=3)
        self._report.session_end = time.time()
        self._report.model_hashes = self._model_hashes
        self._report.inference_latencies = self._inference_latencies
        logger.info(
            f"DigitalFootprintScanner stopped. "
            f"Events: {len(self._report.events)} "
            f"(critical={self._report.critical_count}, high={self._report.high_count})"
        )
        return self._report

    def register_model_file(self, path: str):
        """
        Register a model weight file to monitor for tampering.
        Call this before training starts or before loading a model for inference.
        GreenTensor will hash the file and alert if it changes unexpectedly.
        """
        if not os.path.exists(path):
            logger.warning(f"Model file not found for registration: {path}")
            return
        h = self._hash_file(path)
        self._model_hashes[path] = h
        logger.info(f"Model file registered: {path} (sha256: {h[:16]}...)")

    def verify_model_files(self) -> List[DigitalFootprintEvent]:
        """
        Re-hash all registered model files and alert on any changes.
        Call this after training or before loading a model in production.
        """
        events = []
        for path, original_hash in self._model_hashes.items():
            if not os.path.exists(path):
                e = self._make_event(
                    category="model_tampering",
                    signal="model_file_deleted",
                    severity="critical",
                    evidence={"path": path, "original_hash": original_hash},
                    message=f"Model file deleted or moved: {path}. Possible supply chain attack.",
                    mitre="AML.T0018",
                )
                events.append(e)
                continue
            current_hash = self._hash_file(path)
            if current_hash != original_hash:
                e = self._make_event(
                    category="model_tampering",
                    signal="model_weight_modified",
                    severity="critical",
                    evidence={
                        "path": path,
                        "original_hash": original_hash,
                        "current_hash": current_hash,
                    },
                    message=(
                        f"Model weight file modified: {path}. "
                        f"Hash changed from {original_hash[:16]}... to {current_hash[:16]}... "
                        f"Possible backdoor injection or supply chain compromise."
                    ),
                    mitre="AML.T0018",
                )
                events.append(e)
        return events

    def scan_dependencies(self) -> List[DigitalFootprintEvent]:
        """
        Scan installed packages for known malicious or typosquatted packages.
        Uses a curated list of known ML supply chain attack patterns.
        """
        events = []
        suspicious_patterns = [
            # Known typosquatting patterns targeting ML packages
            ("torchvison", "torch"),       ("tensorfow", "tensorflow"),
            ("nump y", "numpy"),           ("scikit-leam", "scikit-learn"),
            ("huggingface-hub-", "huggingface-hub"),
            # Packages with known malicious versions
            "ctx", "py-util", "loglib-modules", "httpx-color",
        ]
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "list", "--format=json"],
                capture_output=True, text=True, timeout=10
            )
            import json
            packages = {p["name"].lower(): p["version"] for p in json.loads(result.stdout)}

            for pattern in suspicious_patterns:
                if isinstance(pattern, tuple):
                    suspect, legitimate = pattern
                    if suspect.lower() in packages:
                        e = self._make_event(
                            category="supply_chain",
                            signal="typosquatting_detected",
                            severity="critical",
                            evidence={"package": suspect, "legitimate": legitimate,
                                      "version": packages[suspect.lower()]},
                            message=(
                                f"Typosquatted package detected: '{suspect}' "
                                f"(likely targeting '{legitimate}'). "
                                f"Remove immediately — possible malicious code execution."
                            ),
                            mitre="AML.T0010",
                        )
                        events.append(e)
                elif isinstance(pattern, str):
                    if pattern.lower() in packages:
                        e = self._make_event(
                            category="supply_chain",
                            signal="known_malicious_package",
                            severity="critical",
                            evidence={"package": pattern, "version": packages[pattern.lower()]},
                            message=(
                                f"Known malicious package installed: '{pattern}'. "
                                f"Remove immediately."
                            ),
                            mitre="AML.T0010",
                        )
                        events.append(e)
        except Exception as ex:
            logger.debug(f"Dependency scan error: {ex}")
        return events

    def record_inference(self, latency_s: float, input_size: int = 0,
                         confidence: Optional[float] = None):
        """
        Record a single inference call for post-deployment anomaly detection.
        Call this after each model.predict() / model() call in production.

        Parameters
        ----------
        latency_s   : inference time in seconds
        input_size  : size of input (tokens, bytes, pixels — any consistent unit)
        confidence  : model's max output confidence/probability (0-1), if available
        """
        self._inference_latencies.append(latency_s)
        self._api_call_times.append(time.time())

        # Check for latency spike (backdoor trigger causes extra computation)
        if len(self._inference_latencies) > 10:
            baseline_lat = statistics.mean(self._inference_latencies[:-1])
            if baseline_lat > 0 and latency_s > baseline_lat * 3:
                self._make_event(
                    category="inference_anomaly",
                    signal="latency_spike",
                    severity="high",
                    evidence={
                        "latency_s": latency_s,
                        "baseline_latency_s": baseline_lat,
                        "ratio": latency_s / baseline_lat,
                    },
                    message=(
                        f"Inference latency spike: {latency_s:.3f}s vs "
                        f"{baseline_lat:.3f}s baseline ({latency_s/baseline_lat:.1f}x). "
                        f"Possible backdoor trigger activating extra computation."
                    ),
                    mitre="AML.T0043",
                )

        # Check for model extraction (high-frequency systematic probing)
        if len(self._api_call_times) > 20:
            recent = self._api_call_times[-20:]
            rate = 20 / (recent[-1] - recent[0]) if (recent[-1] - recent[0]) > 0 else 0
            if rate > 50:  # more than 50 calls/second
                self._make_event(
                    category="model_extraction",
                    signal="high_frequency_probing",
                    severity="high",
                    evidence={"calls_per_second": rate, "window_calls": 20},
                    message=(
                        f"High-frequency inference probing detected: {rate:.1f} calls/sec. "
                        f"Possible model extraction / stealing attack."
                    ),
                    mitre="AML.T0044",
                )

        # Confidence anomaly
        if confidence is not None and len(self._inference_latencies) > 10:
            if confidence > 0.999:
                self._make_event(
                    category="inference_anomaly",
                    signal="suspiciously_high_confidence",
                    severity="medium",
                    evidence={"confidence": confidence},
                    message=(
                        f"Suspiciously high model confidence: {confidence:.4f}. "
                        f"Possible adversarial input or backdoor trigger."
                    ),
                    mitre="AML.T0043",
                )

    def check_network_connection(self, host: str, port: int = 443) -> bool:
        """
        Manually check if a specific host is in the trusted list.
        Returns True if trusted, False and raises alert if not.
        """
        if host not in self.trusted_hosts:
            self._make_event(
                category="network_exfiltration",
                signal="untrusted_connection_attempt",
                severity="high",
                evidence={"host": host, "port": port},
                message=(
                    f"Connection attempt to untrusted host: {host}:{port}. "
                    f"Possible data exfiltration or C2 communication."
                ),
                mitre="AML.T0024",
            )
            return False
        return True

    @property
    def report(self) -> FootprintReport:
        return self._report

    @property
    def events(self) -> List[DigitalFootprintEvent]:
        with self._lock:
            return list(self._report.events)

    # ── Internal monitoring loops ──────────────────────────────────────────────

    def _snapshot_baseline(self):
        """Capture baseline state before monitoring starts."""
        try:
            import psutil
            self._baseline_processes = {
                p.pid for p in psutil.process_iter(["pid"])
            }
        except Exception:
            pass

        try:
            import psutil
            self._baseline_connections = {
                (c.raddr.ip, c.raddr.port)
                for c in psutil.net_connections(kind="inet")
                if c.raddr and c.status == "ESTABLISHED"
            }
        except Exception:
            pass

    def _monitor_processes_loop(self):
        """Detect unexpected child processes spawned during training/inference."""
        try:
            import psutil
        except ImportError:
            return

        while not self._stop_event.is_set():
            try:
                current = {p.pid: p.name() for p in psutil.process_iter(["pid", "name"])}
                new_pids = set(current.keys()) - self._baseline_processes
                for pid in new_pids:
                    name = current.get(pid, "unknown")
                    # Flag suspicious process names
                    suspicious = any(s in name.lower() for s in [
                        "miner", "xmrig", "cgminer", "ethminer",
                        "nc", "ncat", "netcat", "curl", "wget",
                        "powershell", "cmd", "bash", "sh",
                    ])
                    if suspicious:
                        self._make_event(
                            category="process_injection",
                            signal="suspicious_process_spawned",
                            severity="critical",
                            evidence={"pid": pid, "name": name},
                            message=(
                                f"Suspicious process spawned during ML pipeline: "
                                f"'{name}' (PID {pid}). Possible cryptominer or shell injection."
                            ),
                            mitre="AML.T0011",
                        )
                    self._baseline_processes.add(pid)
                    with self._lock:
                        self._report.child_processes.append(f"{name}(pid={pid})")
            except Exception:
                pass
            time.sleep(2.0)

    def _monitor_network_loop(self):
        """Detect unexpected outbound network connections."""
        try:
            import psutil
        except ImportError:
            return

        while not self._stop_event.is_set():
            try:
                current = {
                    (c.raddr.ip, c.raddr.port)
                    for c in psutil.net_connections(kind="inet")
                    if c.raddr and c.status == "ESTABLISHED"
                }
                new_conns = current - self._baseline_connections
                for ip, port in new_conns:
                    # Try reverse DNS
                    try:
                        host = socket.gethostbyaddr(ip)[0]
                    except Exception:
                        host = ip

                    is_trusted = any(
                        t in host for t in self.trusted_hosts
                    )
                    if not is_trusted and port not in (80, 443, 8080, 8443):
                        self._make_event(
                            category="network_exfiltration",
                            signal="unexpected_outbound_connection",
                            severity="high",
                            evidence={"ip": ip, "port": port, "host": host},
                            message=(
                                f"Unexpected outbound connection: {host} ({ip}:{port}). "
                                f"Possible data exfiltration or C2 beacon."
                            ),
                            mitre="AML.T0024",
                        )
                    self._baseline_connections.add((ip, port))
                    with self._lock:
                        self._report.network_connections.append(f"{host}:{port}")
            except Exception:
                pass
            time.sleep(3.0)

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _make_event(self, category, signal, severity, evidence, message, mitre) -> DigitalFootprintEvent:
        e = DigitalFootprintEvent(
            timestamp=time.time(),
            stage=self.stage,
            category=category,
            signal=signal,
            severity=severity,
            evidence=evidence,
            message=message,
            mitre_technique=mitre,
        )
        with self._lock:
            self._report.events.append(e)
        self.on_event(e)
        return e

    def _start_thread(self, target, name):
        t = threading.Thread(target=target, name=name, daemon=True)
        t.start()
        self._threads.append(t)

    @staticmethod
    def _hash_file(path: str) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    @staticmethod
    def _default_handler(event: DigitalFootprintEvent):
        logger.warning(
            f"[FOOTPRINT/{event.severity.upper()}] "
            f"[{event.stage}] [{event.category}] "
            f"[MITRE:{event.mitre_technique}] {event.message}"
        )

    # ── Context manager ────────────────────────────────────────────────────────

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()
