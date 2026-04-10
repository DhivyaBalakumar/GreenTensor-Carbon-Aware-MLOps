"""
GreenTensor -- Carbon-Secure MLOps + AquaTensor Water Intelligence.
"""
from greentensor.core.context import GreenTensor
from greentensor.core.history import RunHistory
from greentensor.core.budget import CarbonBudget, CarbonBudgetExceeded
from greentensor.report.metrics import RunMetrics, calculate_savings, DatacenterConfig, PUE_PRESETS
from greentensor.report.esg import ESGReporter, ESGOrganization, ESGReport, ESGRunRecord
from greentensor.optimizers.batch_optimizer import optimize_batch_size
from greentensor.optimizers.recommender import EfficiencyRecommender, Recommendation
from greentensor.utils.config import Config
from greentensor.security.anomaly_detector import AnomalyDetector, AnomalyDetectorConfig, AnomalyAlert
from greentensor.security.digital_footprint import DigitalFootprintScanner, DigitalFootprintEvent, FootprintReport
from greentensor.security.pattern_matcher import PatternMatcher, PatternMatchResult
from greentensor.water.aquatensor import AquaTensorBridge, AquaTensorConfig, WaterMetrics, PROVIDER_WUE, REGIONAL_WATER_STRESS
from greentensor.scheduler.carbon_scheduler import CarbonAwareScheduler, GridSignal, ScheduleRecommendation, STATIC_INTENSITY
from greentensor.alerts.webhooks import SlackWebhook, PagerDutyAlert, GenericWebhook, MultiAlert

__all__ = [
    "GreenTensor",
    "RunHistory",
    "CarbonBudget",
    "CarbonBudgetExceeded",
    "RunMetrics",
    "calculate_savings",
    "optimize_batch_size",
    "EfficiencyRecommender",
    "Recommendation",
    "Config",
    "DatacenterConfig",
    "PUE_PRESETS",
    "ESGReporter",
    "ESGOrganization",
    "ESGReport",
    "ESGRunRecord",
    "AnomalyDetector",
    "AnomalyDetectorConfig",
    "AnomalyAlert",
    "DigitalFootprintScanner",
    "DigitalFootprintEvent",
    "FootprintReport",
    "PatternMatcher",
    "PatternMatchResult",
    "AquaTensorBridge",
    "AquaTensorConfig",
    "WaterMetrics",
    "PROVIDER_WUE",
    "REGIONAL_WATER_STRESS",
    "CarbonAwareScheduler",
    "GridSignal",
    "ScheduleRecommendation",
    "STATIC_INTENSITY",
    "SlackWebhook",
    "PagerDutyAlert",
    "GenericWebhook",
    "MultiAlert",
]