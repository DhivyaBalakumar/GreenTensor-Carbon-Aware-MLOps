"""
GreenTensor — energy-efficient, security-aware middleware for PyTorch ML workloads.
"""
from greentensor.core.context import GreenTensor
from greentensor.report.metrics import RunMetrics, calculate_savings
from greentensor.optimizers.batch_optimizer import optimize_batch_size
from greentensor.utils.config import Config
from greentensor.security.anomaly_detector import AnomalyDetector, AnomalyDetectorConfig, AnomalyAlert

__all__ = [
    "GreenTensor",
    "RunMetrics",
    "calculate_savings",
    "optimize_batch_size",
    "Config",
    "AnomalyDetector",
    "AnomalyDetectorConfig",
    "AnomalyAlert",
]
