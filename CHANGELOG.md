# Changelog

## [0.7.1] — 2026-05-06

### Fixed
- **Critical**: Fixed `SyntaxError` in `report.py` that caused `import greentensor` to fail entirely on install
- **Critical**: `security/__init__.py` was empty on disk — `AnomalyDetector`, `DigitalFootprintScanner`, and `PatternMatcher` were not importable from the security module
- **Critical**: `greentensor/__init__.py` was missing `__version__`, `Tracker`, `Profiler`, `GPUOptimizer`, `BatchOptimizer`, `IdleOptimizer`, `generate_report`, and `get_logger` exports
- Fixed `test_context.py` — tests were hanging because `AnomalyDetector` and `DigitalFootprintScanner` background threads were not mocked

### Improved
- All 58 unit tests now pass
- PyPI metadata updated: author, classifiers, development status, documentation URL

---

## [0.7.0] — 2026-04-28

### Added
- `AquaTensor` water intelligence layer — physics-based membrane distillation calculations
- `CarbonAwareScheduler` — queries electricityMap API for real-time grid carbon intensity
- `DigitalFootprintScanner` — multi-signal ML cyberattack detection with MITRE ATLAS tagging
- `ESGReporter` — GHG Protocol Scope 2 compliant reports with real-world equivalencies
- `PatternMatcher` — correlates carbon spikes with digital footprint signals
- Slack, PagerDuty, and generic SIEM webhook alerting
- Carbon budget enforcement with `CarbonBudgetExceeded` exception
- `EfficiencyRecommender` — post-run actionable optimization recommendations

### Core
- `GreenTensor` context manager wraps existing training code with zero changes
- CodeCarbon integration for hardware-level energy measurement
- Fallback to nvidia-smi power sampling when CodeCarbon unavailable
- cuDNN benchmark mode + mixed precision auto-enablement
- Idle GPU detection via background thread monitoring
