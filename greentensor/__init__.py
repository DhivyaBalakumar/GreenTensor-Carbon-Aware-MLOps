"""
GreenTensor — Sustainable ML Training Library
Carbon tracking, water intelligence, and ML security in one context manager.

Author: Dhivya Balakumar <dhivyabalakumar28@gmail.com>
License: MIT
"""

from greentensor.core.context import GreenTensor
from greentensor.core.budget import CarbonBudget, CarbonBudgetExceeded
from greentensor.core.tracker import Tracker
from greentensor.core.profiler import Profiler
from greentensor.core.history import RunHistory
from greentensor.report.metrics import RunMetrics, DatacenterConfig, calculate_savings, PUE_PRESETS
from greentensor.report.esg import ESGReporter, ESGOrganization, ESGReport, ESGRunRecord
from greentensor.report.report import generate_report
from greentensor.optimizers.gpu_optimizer import GPUOptimizer
from greentensor.optimizers.batch_optimizer import BatchOptimizer, optimize_batch_size
from greentensor.optimizers.idle_optimizer import IdleOptimizer
from greentensor.optimizers.recommender import EfficiencyRecommender, Recommendation
from greentensor.security import (
    AnomalyDetector, AnomalyDetectorConfig, AnomalyAlert,
    DigitalFootprintScanner, DigitalFootprintEvent, FootprintReport,
    PatternMatcher, PatternMatchResult,
)
from greentensor.alerts.webhooks import SlackWebhook, PagerDutyAlert, GenericWebhook, MultiAlert
from greentensor.water.aquatensor import (
    AquaTensorBridge, AquaTensorConfig, WaterMetrics,
    PROVIDER_WUE, REGIONAL_WATER_STRESS,
)
from greentensor.scheduler.carbon_scheduler import (
    CarbonAwareScheduler, GridSignal, ScheduleRecommendation, STATIC_INTENSITY,
)
from greentensor.utils.config import Config
from greentensor.utils.logger import get_logger

__version__ = "0.7.1"
__author__ = "Dhivya Balakumar"

__all__ = [
    # Core
    "GreenTensor",
    "CarbonBudget",
    "CarbonBudgetExceeded",
    "Tracker",
    "Profiler",
    "RunHistory",
    # Report
    "RunMetrics",
    "DatacenterConfig",
    "calculate_savings",
    "PUE_PRESETS",
    "ESGReporter",
    "ESGOrganization",
    "ESGReport",
    "ESGRunRecord",
    "generate_report",
    # Optimizers
    "GPUOptimizer",
    "BatchOptimizer",
    "optimize_batch_size",
    "IdleOptimizer",
    "EfficiencyRecommender",
    "Recommendation",
    # Security
    "AnomalyDetector",
    "AnomalyDetectorConfig",
    "AnomalyAlert",
    "DigitalFootprintScanner",
    "DigitalFootprintEvent",
    "FootprintReport",
    "PatternMatcher",
    "PatternMatchResult",
    # Alerts
    "SlackWebhook",
    "PagerDutyAlert",
    "GenericWebhook",
    "MultiAlert",
    # Water
    "AquaTensorBridge",
    "AquaTensorConfig",
    "WaterMetrics",
    "PROVIDER_WUE",
    "REGIONAL_WATER_STRESS",
    # Scheduler
    "CarbonAwareScheduler",
    "GridSignal",
    "ScheduleRecommendation",
    "STATIC_INTENSITY",
    # Utils
    "Config",
    "get_logger",
]
