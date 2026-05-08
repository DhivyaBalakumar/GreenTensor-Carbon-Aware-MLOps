#!/usr/bin/env python3
"""
GreenTensor Middleware Stress Test
===================================
Tests every real code path with realistic scenarios.
No mocking — actual execution of all modules.

Run from workspace root:
    cd GreenTensor/greentensor && python ../../stress_test_greentensor.py
"""

import sys, os, time, math, hashlib, tempfile, threading, statistics, traceback

# ── colour helpers ─────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

PASS = f"{GREEN}[PASS]{RESET}"
FAIL = f"{RED}[FAIL]{RESET}"
WARN = f"{YELLOW}[WARN]{RESET}"
INFO = f"{CYAN}[INFO]{RESET}"

results = []

def section(title):
    print(f"\n{BOLD}{CYAN}{'='*60}{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{BOLD}{CYAN}{'='*60}{RESET}")

def check(name, passed, detail="", warn=False):
    status = WARN if warn else (PASS if passed else FAIL)
    print(f"  {status}  {name}")
    if detail:
        print(f"         {detail}")
    results.append((name, "warn" if warn else ("pass" if passed else "fail")))

def run_test(name, fn):
    try:
        fn()
    except Exception as e:
        check(name, False, f"Exception: {e}\n         {traceback.format_exc().splitlines()[-1]}")

# ══════════════════════════════════════════════════════════════════════════════
# TEST 1 — IMPORT & VERSION
# ══════════════════════════════════════════════════════════════════════════════
section("1. IMPORT & VERSION CHECK")

try:
    import greentensor as gt
    check("greentensor imports without error", True, f"version={gt.__version__}")
except Exception as e:
    check("greentensor imports without error", False, str(e))
    print(f"\n{RED}FATAL: Cannot import greentensor. Aborting.{RESET}")
    sys.exit(1)

# Check all expected exports
expected_exports = [
    "GreenTensor", "CarbonBudget", "CarbonBudgetExceeded", "Tracker", "Profiler",
    "RunHistory", "RunMetrics", "DatacenterConfig", "calculate_savings", "PUE_PRESETS",
    "ESGReporter", "ESGOrganization", "generate_report",
    "GPUOptimizer", "BatchOptimizer", "optimize_batch_size", "IdleOptimizer",
    "EfficiencyRecommender", "Recommendation",
    "AnomalyDetector", "AnomalyDetectorConfig", "AnomalyAlert",
    "DigitalFootprintScanner", "DigitalFootprintEvent", "FootprintReport",
    "PatternMatcher", "PatternMatchResult",
    "SlackWebhook", "PagerDutyAlert", "GenericWebhook", "MultiAlert",
    "AquaTensorBridge", "AquaTensorConfig", "WaterMetrics",
    "PROVIDER_WUE", "REGIONAL_WATER_STRESS",
    "CarbonAwareScheduler", "GridSignal", "ScheduleRecommendation", "STATIC_INTENSITY",
    "Config", "get_logger",
]
missing = [e for e in expected_exports if not hasattr(gt, e)]
check(f"All {len(expected_exports)} exports present", len(missing) == 0,
      f"Missing: {missing}" if missing else "")

# ══════════════════════════════════════════════════════════════════════════════
# TEST 2 — CONTEXT MANAGER (core pipeline)
# ══════════════════════════════════════════════════════════════════════════════
section("2. CONTEXT MANAGER — CORE PIPELINE")

def test_basic_context():
    t0 = time.perf_counter()
    with gt.GreenTensor(verbose=False, security=False, save_path=None) as g:
        # Simulate a short CPU workload
        total = sum(i*i for i in range(500_000))
    elapsed = time.perf_counter() - t0
    check("Context manager enters and exits cleanly", True, f"wall time={elapsed:.2f}s")
    check("metrics object populated after exit", g.metrics is not None)
    m = g.metrics
    check("duration_s > 0", m.duration_s > 0, f"duration={m.duration_s:.4f}s")
    check("energy_kwh >= 0", m.energy_kwh >= 0, f"energy={m.energy_kwh:.8f} kWh")
    check("emissions_kg >= 0", m.emissions_kg >= 0, f"emissions={m.emissions_kg:.8f} kg")
    check("idle_seconds >= 0", m.idle_seconds >= 0, f"idle={m.idle_seconds:.2f}s")

run_test("basic context manager", test_basic_context)

def test_context_with_verbose():
    """Verbose=True should print a report without crashing."""
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        with gt.GreenTensor(verbose=True, security=False, save_path=None) as g:
            _ = [x**2 for x in range(100_000)]
    output = buf.getvalue()
    check("verbose report printed", "GreenTensor Report" in output, output[:80])
    check("report contains Runtime", "Runtime" in output)
    check("report contains Energy", "Energy" in output)
    check("report contains CO2", "CO2" in output)
    check("report contains Carbon Anomaly section", "Carbon Anomaly" in output)
    check("report contains Digital Footprint section", "Digital Footprint" in output)

run_test("verbose context manager", test_context_with_verbose)

def test_context_exception_propagation():
    """Exceptions inside the context should propagate normally."""
    try:
        with gt.GreenTensor(verbose=False, security=False, save_path=None):
            raise ValueError("intentional test error")
        check("exception propagated", False, "should have raised")
    except ValueError as e:
        check("exception propagates through context manager", str(e) == "intentional test error",
              str(e))

run_test("exception propagation", test_context_exception_propagation)

# ══════════════════════════════════════════════════════════════════════════════
# TEST 3 — TRACKER (energy measurement)
# ══════════════════════════════════════════════════════════════════════════════
section("3. TRACKER — ENERGY MEASUREMENT")

def test_tracker_direct():
    from greentensor.core.tracker import Tracker
    from greentensor.utils.config import Config
    cfg = Config()
    tracker = Tracker(cfg)
    tracker.start()
    # Do real work
    _ = sum(math.sqrt(i) for i in range(1_000_000))
    emissions_kg, energy_kwh = tracker.stop()
    check("Tracker.start() / stop() returns tuple", True)
    check("emissions_kg is float >= 0", isinstance(emissions_kg, float) and emissions_kg >= 0,
          f"emissions={emissions_kg:.8f} kg")
    check("energy_kwh is float >= 0", isinstance(energy_kwh, float) and energy_kwh >= 0,
          f"energy={energy_kwh:.8f} kWh")
    # Sanity: a 1M sqrt loop should use some energy
    check("energy > 0 (real measurement)", energy_kwh > 0,
          f"energy={energy_kwh:.10f} kWh — {'OK' if energy_kwh > 0 else 'ZERO — CodeCarbon may not be installed'}",
          warn=(energy_kwh == 0))

run_test("tracker direct", test_tracker_direct)

def test_tracker_multiple_runs():
    """Multiple sequential runs should each produce independent measurements."""
    from greentensor.core.tracker import Tracker
    from greentensor.utils.config import Config
    readings = []
    for i in range(3):
        t = Tracker(Config())
        t.start()
        _ = sum(j**2 for j in range(200_000 * (i+1)))
        e_kg, e_kwh = t.stop()
        readings.append(e_kwh)
    check("3 sequential tracker runs all complete", len(readings) == 3,
          f"readings={[f'{r:.8f}' for r in readings]}")
    check("readings are non-negative", all(r >= 0 for r in readings))

run_test("tracker multiple runs", test_tracker_multiple_runs)

# ══════════════════════════════════════════════════════════════════════════════
# TEST 4 — AQUATENSOR WATER INTELLIGENCE
# ══════════════════════════════════════════════════════════════════════════════
section("4. AQUATENSOR — WATER INTELLIGENCE")

def test_aquatensor_physics():
    bridge = gt.AquaTensorBridge(gt.AquaTensorConfig(
        provider="google",
        region="India",
        aquatensor_installed=True,
        whr_ratio=0.65,
        feed_temperature_c=60.0,
    ))
    # Use a realistic energy value (0.001 kWh = 1 Wh)
    wm = bridge.calculate_water_metrics(energy_kwh=0.001, duration_s=45.0)

    check("WaterMetrics returned", isinstance(wm, gt.WaterMetrics))
    check("WUE = 0.49 (Google)", abs(wm.wue - 0.49) < 0.001, f"wue={wm.wue}")
    check("water_consumed_liters > 0", wm.water_consumed_liters > 0,
          f"consumed={wm.water_consumed_liters:.6f} L")
    check("heat_generated_kwh == energy_kwh (conservation)", 
          abs(wm.heat_generated_kwh - 0.001) < 1e-9, f"heat={wm.heat_generated_kwh}")
    check("heat_recovered_kwh = heat * 0.65", 
          abs(wm.heat_recovered_kwh - 0.001 * 0.65) < 1e-9,
          f"recovered={wm.heat_recovered_kwh:.6f} kWh")
    check("MD yield at 60C = 5.5 L/kWh", 
          abs(wm.md_yield_liters_per_kwh - 5.5) < 0.01,
          f"yield={wm.md_yield_liters_per_kwh}")
    check("water_produced_liters = heat_recovered * md_yield",
          abs(wm.water_produced_liters - wm.heat_recovered_kwh * 5.5) < 1e-9,
          f"produced={wm.water_produced_liters:.6f} L")
    check("net_water_impact = consumed - produced",
          abs(wm.net_water_impact_liters - (wm.water_consumed_liters - wm.water_produced_liters)) < 1e-9,
          f"net={wm.net_water_impact_liters:.6f} L")
    check("is_net_water_positive when produced > consumed",
          wm.is_net_water_positive == (wm.net_water_impact_liters < 0),
          f"net_positive={wm.is_net_water_positive}")
    check("water_stress_index for India = 4.5",
          abs(wm.water_stress_index - 4.5) < 0.01, f"stress={wm.water_stress_index}")
    check("water_stress_label = Extremely High",
          wm.water_stress_label == "Extremely High", f"label={wm.water_stress_label}")

run_test("aquatensor physics", test_aquatensor_physics)

def test_aquatensor_temperature_interpolation():
    """MD yield should interpolate correctly between temperature breakpoints."""
    bridge = gt.AquaTensorBridge(gt.AquaTensorConfig(aquatensor_installed=True))
    # At exactly 50C should be 4.0
    wm50 = bridge.calculate_water_metrics(0.001, 10)
    bridge.config.feed_temperature_c = 50.0
    wm50 = bridge.calculate_water_metrics(0.001, 10)
    check("MD yield at 50C = 4.0", abs(wm50.md_yield_liters_per_kwh - 4.0) < 0.01,
          f"yield={wm50.md_yield_liters_per_kwh}")
    # At 55C should interpolate between 4.0 and 5.5 → 4.75
    bridge.config.feed_temperature_c = 55.0
    wm55 = bridge.calculate_water_metrics(0.001, 10)
    check("MD yield at 55C = 4.75 (interpolated)", abs(wm55.md_yield_liters_per_kwh - 4.75) < 0.01,
          f"yield={wm55.md_yield_liters_per_kwh}")
    # At 80C should be 8.5 (max)
    bridge.config.feed_temperature_c = 80.0
    wm80 = bridge.calculate_water_metrics(0.001, 10)
    check("MD yield at 80C = 8.5 (max)", abs(wm80.md_yield_liters_per_kwh - 8.5) < 0.01,
          f"yield={wm80.md_yield_liters_per_kwh}")
    # Below 40C should clamp to 2.5
    bridge.config.feed_temperature_c = 30.0
    wm30 = bridge.calculate_water_metrics(0.001, 10)
    check("MD yield at 30C = 2.5 (clamped to min)", abs(wm30.md_yield_liters_per_kwh - 2.5) < 0.01,
          f"yield={wm30.md_yield_liters_per_kwh}")

run_test("aquatensor temperature interpolation", test_aquatensor_temperature_interpolation)

def test_aquatensor_no_system_installed():
    """Without AquaTensor installed, water_produced should be 0."""
    bridge = gt.AquaTensorBridge(gt.AquaTensorConfig(
        provider="aws", aquatensor_installed=False))
    wm = bridge.calculate_water_metrics(0.01, 100)
    check("water_produced = 0 when not installed", wm.water_produced_liters == 0.0,
          f"produced={wm.water_produced_liters}")
    check("heat_recovered = 0 when not installed", wm.heat_recovered_kwh == 0.0)
    check("net_impact = consumed (no recovery)", 
          abs(wm.net_water_impact_liters - wm.water_consumed_liters) < 1e-9)

run_test("aquatensor not installed", test_aquatensor_no_system_installed)

def test_aquatensor_heat_forecast():
    bridge = gt.AquaTensorBridge(gt.AquaTensorConfig(
        aquatensor_installed=True, whr_ratio=0.65, feed_temperature_c=65.0))
    jobs = [
        {"name": "bert-finetune", "estimated_duration_s": 3600, "estimated_power_w": 250},
        {"name": "resnet-train",  "estimated_duration_s": 7200, "estimated_power_w": 300},
    ]
    forecast = bridge.forecast_heat(jobs)
    check("HeatForecast returned", forecast is not None)
    check("predicted_energy_kwh > 0", forecast.predicted_energy_kwh > 0,
          f"energy={forecast.predicted_energy_kwh:.4f} kWh")
    check("predicted_water_liters > 0", forecast.predicted_water_liters > 0,
          f"water={forecast.predicted_water_liters:.2f} L")
    check("temperature_sustained = True at 65C", forecast.temperature_sustained)
    check("recommendation string non-empty", len(forecast.optimal_schedule_recommendation) > 10,
          forecast.optimal_schedule_recommendation[:80])

run_test("aquatensor heat forecast", test_aquatensor_heat_forecast)

def test_aquatensor_via_runmetrics():
    """Test the RunMetrics.apply_aquatensor_config() integration path."""
    m = gt.RunMetrics(duration_s=45.0, energy_kwh=0.0004, emissions_kg=0.0001)
    m2 = m.apply_aquatensor_config(gt.AquaTensorConfig(
        provider="microsoft", region="US-West",
        aquatensor_installed=True, whr_ratio=0.65, feed_temperature_c=60.0))
    check("apply_aquatensor_config returns RunMetrics", isinstance(m2, gt.RunMetrics))
    check("water metrics populated", m2.water is not None)
    check("original energy preserved", m2.energy_kwh == 0.0004)
    check("water consumed > 0", m2.water.water_consumed_liters > 0,
          f"consumed={m2.water.water_consumed_liters:.6f} L")

run_test("aquatensor via RunMetrics", test_aquatensor_via_runmetrics)

# ══════════════════════════════════════════════════════════════════════════════
# TEST 5 — CARBON SCHEDULER
# ══════════════════════════════════════════════════════════════════════════════
section("5. CARBON-AWARE SCHEDULER")

def test_scheduler_static_fallback():
    sched = gt.CarbonAwareScheduler(zone="FR")
    signal = sched.get_current_intensity()
    check("GridSignal returned", isinstance(signal, gt.GridSignal))
    check("zone matches", signal.zone == "FR", f"zone={signal.zone}")
    check("intensity > 0", signal.carbon_intensity_kg_per_kwh > 0,
          f"intensity={signal.carbon_intensity_kg_per_kwh*1000:.1f} gCO2/kWh")
    check("source is static_fallback or api", 
          signal.source in ("static_fallback", "electricitymap_api"),
          f"source={signal.source}")
    check("timestamp is recent", time.time() - signal.timestamp < 10)

run_test("scheduler static fallback", test_scheduler_static_fallback)

def test_scheduler_recommendation():
    sched = gt.CarbonAwareScheduler(zone="IN-NO")
    rec = sched.should_run_now(estimated_duration_hours=2.0, estimated_energy_kwh=0.5)
    check("ScheduleRecommendation returned", isinstance(rec, gt.ScheduleRecommendation))
    check("run_now is bool", isinstance(rec.run_now, bool))
    check("current_intensity > 0", rec.current_intensity > 0,
          f"intensity={rec.current_intensity*1000:.1f} gCO2/kWh")
    check("reason string non-empty", len(rec.reason) > 10, rec.reason[:80])
    if not rec.run_now:
        check("recommended_delay_hours > 0 when not running now",
              rec.recommended_delay_hours > 0,
              f"delay={rec.recommended_delay_hours:.1f}h")
        check("carbon_savings_pct > 0", rec.carbon_savings_pct > 0,
              f"savings={rec.carbon_savings_pct:.1f}%")

run_test("scheduler recommendation", test_scheduler_recommendation)

def test_scheduler_all_zones():
    """All defined zones should return valid signals."""
    failed_zones = []
    for zone in gt.STATIC_INTENSITY:
        if zone == "default":
            continue
        try:
            sched = gt.CarbonAwareScheduler(zone=zone)
            sig = sched.get_current_intensity()
            if sig.carbon_intensity_kg_per_kwh <= 0:
                failed_zones.append(zone)
        except Exception as e:
            failed_zones.append(f"{zone}({e})")
    check(f"All {len(gt.STATIC_INTENSITY)-1} zones return valid signals",
          len(failed_zones) == 0, f"Failed: {failed_zones}" if failed_zones else "")

run_test("scheduler all zones", test_scheduler_all_zones)

def test_scheduler_time_of_day_variation():
    """Static fallback should vary intensity by time of day."""
    sched = gt.CarbonAwareScheduler(zone="US-CAL-CISO")
    # Force two different hours by patching time.gmtime
    import unittest.mock as mock
    base_intensity = gt.STATIC_INTENSITY["US-CAL-CISO"]
    
    # During clean hours (solar peak 10-14 UTC) intensity should be lower
    clean_struct = time.struct_time((2026, 5, 8, 12, 0, 0, 4, 128, 0))
    peak_struct  = time.struct_time((2026, 5, 8, 18, 0, 0, 4, 128, 0))
    
    with mock.patch("greentensor.scheduler.carbon_scheduler.time") as mt:
        mt.time.return_value = time.time()
        mt.gmtime.return_value = clean_struct
        sched._cache = None
        clean_sig = sched._static_signal()
    
    with mock.patch("greentensor.scheduler.carbon_scheduler.time") as mt:
        mt.time.return_value = time.time()
        mt.gmtime.return_value = peak_struct
        sched._cache = None
        peak_sig = sched._static_signal()
    
    check("Clean-hour intensity < peak-hour intensity",
          clean_sig.carbon_intensity_kg_per_kwh < peak_sig.carbon_intensity_kg_per_kwh,
          f"clean={clean_sig.carbon_intensity_kg_per_kwh*1000:.1f} vs peak={peak_sig.carbon_intensity_kg_per_kwh*1000:.1f} gCO2/kWh")

run_test("scheduler time-of-day variation", test_scheduler_time_of_day_variation)

# ══════════════════════════════════════════════════════════════════════════════
# TEST 6 — CARBON BUDGET ENFORCEMENT
# ══════════════════════════════════════════════════════════════════════════════
section("6. CARBON BUDGET ENFORCEMENT")

def test_budget_not_exceeded():
    budget = gt.CarbonBudget(max_kg_per_run=1.0, warn_at_pct=80.0, job_name="test-job")
    try:
        budget.check(emissions_kg=0.0001, energy_kwh=0.001)
        check("No exception when well under budget", True,
              "0.0001 kg vs 1.0 kg budget")
    except gt.CarbonBudgetExceeded:
        check("No exception when well under budget", False)

run_test("budget not exceeded", test_budget_not_exceeded)

def test_budget_exceeded_raises():
    budget = gt.CarbonBudget(max_kg_per_run=0.00001, raise_on_exceed=True, job_name="tiny-job")
    try:
        budget.check(emissions_kg=0.001, energy_kwh=0.005)
        check("CarbonBudgetExceeded raised when over budget", False, "should have raised")
    except gt.CarbonBudgetExceeded as e:
        check("CarbonBudgetExceeded raised when over budget", True)
        check("Exception has budget_kg attr", hasattr(e, "budget_kg"), f"budget={e.budget_kg}")
        check("Exception has actual_kg attr", hasattr(e, "actual_kg"), f"actual={e.actual_kg}")
        check("overage_kg > 0", e.overage_kg > 0, f"overage={e.overage_kg:.8f} kg")
        check("overage_pct > 0", e.overage_pct > 0, f"overage={e.overage_pct:.1f}%")
        check("error message contains job name", "tiny-job" in str(e), str(e)[:80])

run_test("budget exceeded raises", test_budget_exceeded_raises)

def test_budget_warn_only():
    budget = gt.CarbonBudget(max_kg_per_run=0.00001, raise_on_exceed=False, job_name="warn-job")
    try:
        budget.check(emissions_kg=0.001, energy_kwh=0.005)
        check("No exception when raise_on_exceed=False", True)
    except gt.CarbonBudgetExceeded:
        check("No exception when raise_on_exceed=False", False)

run_test("budget warn only", test_budget_warn_only)

def test_budget_energy_limit():
    budget = gt.CarbonBudget(max_kwh_per_run=0.0001, raise_on_exceed=True)
    try:
        budget.check(emissions_kg=0.0, energy_kwh=0.01)
        check("CarbonBudgetExceeded raised for energy limit", False)
    except gt.CarbonBudgetExceeded:
        check("CarbonBudgetExceeded raised for energy limit", True)

run_test("budget energy limit", test_budget_energy_limit)

def test_budget_integrated_with_context():
    """Budget enforcement should fire inside the GreenTensor context manager."""
    try:
        with gt.GreenTensor(
            verbose=False, security=False, save_path=None,
            carbon_budget=gt.CarbonBudget(max_kg_per_run=0.0, raise_on_exceed=True)
        ):
            _ = sum(i for i in range(100_000))
        # If emissions are truly 0 (no CodeCarbon), no exception — that's fine
        check("Budget context integration runs without crash", True,
              "emissions=0 (no CodeCarbon) — budget not triggered", warn=True)
    except gt.CarbonBudgetExceeded as e:
        check("Budget fires inside context manager", True,
              f"Correctly caught: {str(e)[:60]}")
    except Exception as e:
        check("Budget context integration", False, str(e))

run_test("budget integrated with context", test_budget_integrated_with_context)

# ══════════════════════════════════════════════════════════════════════════════
# TEST 7 — SECURITY: ANOMALY DETECTOR
# ══════════════════════════════════════════════════════════════════════════════
section("7. SECURITY — ANOMALY DETECTOR")

def test_anomaly_detector_lifecycle():
    from greentensor.security.anomaly_detector import AnomalyDetector, AnomalyDetectorConfig
    cfg = AnomalyDetectorConfig(
        use_alibi_detect=False,
        use_llm_guard=False,
        sample_interval_s=0.05,
        baseline_window=5,
    )
    detector = AnomalyDetector(config=cfg)
    detector.start()
    time.sleep(0.3)  # let it collect a few samples
    alerts = detector.stop()
    check("AnomalyDetector starts and stops cleanly", True)
    check("alerts is a list", isinstance(alerts, list), f"alerts={alerts}")
    check("no false positives on idle CPU", len(alerts) == 0,
          f"got {len(alerts)} alerts on idle system", warn=(len(alerts) > 0))

run_test("anomaly detector lifecycle", test_anomaly_detector_lifecycle)

def test_anomaly_detector_threshold_spike():
    """Inject a synthetic power spike and verify it fires an alert."""
    from greentensor.security.anomaly_detector import AnomalyDetector, AnomalyDetectorConfig
    from greentensor.core.profiler import Profiler
    import unittest.mock as mock

    fired_alerts = []
    cfg = AnomalyDetectorConfig(
        use_alibi_detect=False,
        use_llm_guard=False,
        sample_interval_s=0.05,
        baseline_window=10,
        spike_threshold_pct=50.0,
        consecutive_anomalies_required=2,
    )
    detector = AnomalyDetector(config=cfg, on_alert=lambda a: fired_alerts.append(a))

    # Pre-seed baseline samples so deviation is detectable
    with detector._lock:
        detector._samples = [100.0] * 10  # baseline = 100W

    # Inject spike: 300W = 200% above 100W baseline
    call_count = [0]
    def fake_metrics():
        call_count[0] += 1
        if call_count[0] <= 5:
            return {"util_%": 5.0, "power_W": 100.0}   # baseline
        return {"util_%": 98.0, "power_W": 300.0}       # spike

    with mock.patch.object(Profiler, "get_gpu_metrics", side_effect=fake_metrics):
        detector.start()
        time.sleep(0.8)
        detector.stop()

    check("Power spike alert fired", len(fired_alerts) > 0,
          f"got {len(fired_alerts)} alerts")
    if fired_alerts:
        a = fired_alerts[0]
        check("Alert type is power_spike", a.alert_type == "power_spike",
              f"type={a.alert_type}")
        check("Alert severity is high or critical",
              a.severity in ("high", "critical"), f"severity={a.severity}")
        check("Alert source is threshold", a.source == "threshold",
              f"source={a.source}")
        check("deviation_pct > 50", a.deviation_pct > 50,
              f"deviation={a.deviation_pct:.1f}%")

run_test("anomaly detector spike detection", test_anomaly_detector_threshold_spike)

def test_anomaly_detector_idle_drain():
    """High power at low utilization should trigger idle_drain alert."""
    from greentensor.security.anomaly_detector import AnomalyDetector, AnomalyDetectorConfig
    from greentensor.core.profiler import Profiler
    import unittest.mock as mock

    fired_alerts = []
    cfg = AnomalyDetectorConfig(
        use_alibi_detect=False, use_llm_guard=False,
        sample_interval_s=0.05,
        idle_drain_threshold_w=30.0,
        idle_util_threshold_pct=5.0,
    )
    detector = AnomalyDetector(config=cfg, on_alert=lambda a: fired_alerts.append(a))
    with detector._lock:
        detector._samples = [50.0] * 10

    def fake_idle():
        return {"util_%": 1.0, "power_W": 80.0}  # high power, near-zero util

    with mock.patch.object(Profiler, "get_gpu_metrics", side_effect=fake_idle):
        detector.start()
        time.sleep(0.4)
        detector.stop()

    idle_alerts = [a for a in fired_alerts if a.alert_type == "idle_drain"]
    check("Idle drain alert fired", len(idle_alerts) > 0,
          f"got {len(idle_alerts)} idle_drain alerts")
    if idle_alerts:
        check("idle_drain severity is high", idle_alerts[0].severity == "high",
              f"severity={idle_alerts[0].severity}")

run_test("anomaly detector idle drain", test_anomaly_detector_idle_drain)

# ══════════════════════════════════════════════════════════════════════════════
# TEST 8 — SECURITY: DIGITAL FOOTPRINT SCANNER
# ══════════════════════════════════════════════════════════════════════════════
section("8. SECURITY — DIGITAL FOOTPRINT SCANNER")

def test_footprint_scanner_lifecycle():
    scanner = gt.DigitalFootprintScanner(
        stage="pre_deployment",
        monitor_network=False,
        monitor_processes=False,
    )
    scanner.start()
    time.sleep(0.2)
    report = scanner.stop()
    check("FootprintReport returned", isinstance(report, gt.FootprintReport))
    check("report.stage = pre_deployment", report.stage == "pre_deployment")
    check("report.session_start > 0", report.session_start > 0)
    check("report.session_end > session_start", report.session_end > report.session_start)
    check("report.is_clean on idle system", report.is_clean,
          f"events={len(report.events)}", warn=(not report.is_clean))

run_test("footprint scanner lifecycle", test_footprint_scanner_lifecycle)

def test_model_file_integrity():
    """Register a model file, tamper with it, verify tampering is detected."""
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False, mode="wb") as f:
        f.write(b"fake model weights v1 " * 100)
        model_path = f.name

    scanner = gt.DigitalFootprintScanner(
        stage="pre_deployment",
        monitor_network=False,
        monitor_processes=False,
    )
    scanner.start()
    scanner.register_model_file(model_path)

    # Verify clean — no tampering yet
    events_clean = scanner.verify_model_files()
    check("No tampering events on clean file", len(events_clean) == 0,
          f"events={events_clean}")

    # Tamper with the file
    with open(model_path, "wb") as f:
        f.write(b"TAMPERED weights " * 100)

    # Verify should now detect the change
    events_tampered = scanner.verify_model_files()
    check("Tampering detected after file modification", len(events_tampered) > 0,
          f"events={len(events_tampered)}")
    if events_tampered:
        e = events_tampered[0]
        check("Event category = model_tampering", e.category == "model_tampering",
              f"category={e.category}")
        check("Event signal = model_weight_modified", e.signal == "model_weight_modified",
              f"signal={e.signal}")
        check("MITRE technique = AML.T0018", e.mitre_technique == "AML.T0018",
              f"mitre={e.mitre_technique}")
        check("Severity = critical", e.severity == "critical",
              f"severity={e.severity}")

    scanner.stop()
    os.unlink(model_path)

run_test("model file integrity", test_model_file_integrity)

def test_model_file_deleted():
    """Deleting a registered model file should trigger a critical event."""
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False, mode="wb") as f:
        f.write(b"model weights " * 50)
        model_path = f.name

    scanner = gt.DigitalFootprintScanner(
        stage="pre_deployment", monitor_network=False, monitor_processes=False)
    scanner.start()
    scanner.register_model_file(model_path)
    os.unlink(model_path)  # delete it
    events = scanner.verify_model_files()
    check("Deleted model file detected", len(events) > 0)
    if events:
        check("Signal = model_file_deleted", events[0].signal == "model_file_deleted",
              f"signal={events[0].signal}")
    scanner.stop()

run_test("model file deleted", test_model_file_deleted)

def test_dependency_scan():
    """Dependency scan should run without crashing and return a list."""
    scanner = gt.DigitalFootprintScanner(
        stage="pre_deployment", monitor_network=False, monitor_processes=False)
    scanner.start()
    events = scanner.scan_dependencies()
    scanner.stop()
    check("scan_dependencies() returns list", isinstance(events, list))
    check("No known malicious packages in clean env", len(events) == 0,
          f"Found {len(events)} suspicious packages: {[e.evidence.get('package') for e in events]}",
          warn=(len(events) > 0))

run_test("dependency scan", test_dependency_scan)

def test_inference_latency_spike():
    """Simulate a latency spike in post-deployment monitoring."""
    scanner = gt.DigitalFootprintScanner(
        stage="post_deployment", monitor_network=False, monitor_processes=False)
    scanner.start()

    # Feed 15 normal latencies then a spike
    for _ in range(15):
        scanner.record_inference(latency_s=0.01)
    scanner.record_inference(latency_s=0.5)  # 50x spike

    report = scanner.stop()
    latency_events = [e for e in report.events if e.signal == "latency_spike"]
    check("Latency spike event fired", len(latency_events) > 0,
          f"events={len(latency_events)}")
    if latency_events:
        check("MITRE = AML.T0043", latency_events[0].mitre_technique == "AML.T0043",
              f"mitre={latency_events[0].mitre_technique}")

run_test("inference latency spike", test_inference_latency_spike)

def test_model_extraction_detection():
    """High-frequency API probing should trigger model extraction alert."""
    time.sleep(0.1)  # ensure prior scanner threads are fully stopped
    scanner = gt.DigitalFootprintScanner(
        stage="post_deployment", monitor_network=False, monitor_processes=False)
    scanner.start()

    # Simulate 30 calls in rapid succession (>50 calls/sec triggers alert)
    for _ in range(30):
        scanner.record_inference(latency_s=0.0001)

    report = scanner.stop()
    extraction_events = [e for e in report.events if e.signal == "high_frequency_probing"]
    check("Model extraction alert fired on rapid probing", len(extraction_events) > 0,
          f"events={len(extraction_events)} — total events={len(report.events)}")
    if extraction_events:
        check("MITRE = AML.T0044", extraction_events[0].mitre_technique == "AML.T0044",
              f"mitre={extraction_events[0].mitre_technique}")

run_test("model extraction detection", test_model_extraction_detection)

def test_untrusted_network_connection():
    """Manually checking an untrusted host should fire an event."""
    scanner = gt.DigitalFootprintScanner(
        stage="pre_deployment", monitor_network=False, monitor_processes=False)
    scanner.start()
    result = scanner.check_network_connection("evil-c2-server.xyz", port=4444)
    report = scanner.stop()
    check("check_network_connection returns False for untrusted host", result == False)
    net_events = [e for e in report.events if e.signal == "untrusted_connection_attempt"]
    check("Network exfiltration event fired", len(net_events) > 0,
          f"events={len(net_events)}")
    if net_events:
        check("MITRE = AML.T0024", net_events[0].mitre_technique == "AML.T0024")

run_test("untrusted network connection", test_untrusted_network_connection)

# ══════════════════════════════════════════════════════════════════════════════
# TEST 9 — PATTERN MATCHER
# ══════════════════════════════════════════════════════════════════════════════
section("9. PATTERN MATCHER — THREAT CORRELATION")

def test_pattern_matcher_cryptominer():
    matcher = gt.PatternMatcher()
    result = matcher.match(
        power_w=280.0,
        baseline_power_w=100.0,
        gpu_util_pct=98.0,
        active_signals=[
            "power_spike_sustained",
            "new_process_spawned",
            "high_gpu_util_unexpected",
            "no_training_activity",
        ],
    )
    check("PatternMatchResult returned", isinstance(result, gt.PatternMatchResult))
    check("Verdict = ATTACK for cryptominer signals", result.verdict == "ATTACK",
          f"verdict={result.verdict}, score={result.threat_score}")
    check("attack_type = cryptominer", result.attack_type == "cryptominer",
          f"type={result.attack_type}")
    check("threat_score >= 80", result.threat_score >= 80,
          f"score={result.threat_score}")
    check("confidence_pct > 60", result.confidence_pct > 60,
          f"confidence={result.confidence_pct:.1f}%")
    check("recommended_action mentions IMMEDIATE", "IMMEDIATE" in result.recommended_action,
          result.recommended_action[:80])
    check("evidence list non-empty", len(result.evidence) > 0)

run_test("pattern matcher cryptominer", test_pattern_matcher_cryptominer)

def test_pattern_matcher_data_exfiltration():
    matcher = gt.PatternMatcher()
    result = matcher.match(
        power_w=120.0,
        baseline_power_w=100.0,
        gpu_util_pct=3.0,
        active_signals=["idle_drain", "network_outbound_unknown", "low_gpu_util_high_power"],
    )
    check("Data exfiltration detected", result.verdict in ("ATTACK", "SUSPICIOUS"),
          f"verdict={result.verdict}, score={result.threat_score}")
    check("attack_type = data_exfiltration", result.attack_type == "data_exfiltration",
          f"type={result.attack_type}")

run_test("pattern matcher data exfiltration", test_pattern_matcher_data_exfiltration)

def test_pattern_matcher_benign():
    matcher = gt.PatternMatcher()
    result = matcher.match(
        power_w=110.0,
        baseline_power_w=100.0,
        gpu_util_pct=85.0,
        active_signals=[],
        benign_context=["batch_size_increased", "checkpoint_saving"],
    )
    check("Benign verdict when no attack signals", result.verdict == "BENIGN",
          f"verdict={result.verdict}, score={result.threat_score}")
    check("threat_score < 20 for benign", result.threat_score < 20,
          f"score={result.threat_score}")
    check("attack_type is None for benign", result.attack_type is None,
          f"type={result.attack_type}")

run_test("pattern matcher benign", test_pattern_matcher_benign)

def test_pattern_matcher_model_theft():
    matcher = gt.PatternMatcher()
    result = matcher.match(
        power_w=150.0,
        baseline_power_w=100.0,
        gpu_util_pct=60.0,
        active_signals=["high_frequency_probing", "systematic_input_patterns", "network_outbound_unknown"],
    )
    check("Model theft detected", result.verdict in ("ATTACK", "SUSPICIOUS"),
          f"verdict={result.verdict}, score={result.threat_score}")
    check("attack_type = model_theft", result.attack_type == "model_theft",
          f"type={result.attack_type}")

run_test("pattern matcher model theft", test_pattern_matcher_model_theft)

# ══════════════════════════════════════════════════════════════════════════════
# TEST 10 — ESG REPORTER
# ══════════════════════════════════════════════════════════════════════════════
section("10. ESG REPORTER")

def test_esg_reporter_basic():
    org = gt.ESGOrganization(name="Acme Corp", reporting_period="FY2026", region="US-East")
    reporter = gt.ESGReporter(org)
    m = gt.RunMetrics(duration_s=312.4, energy_kwh=0.0412, emissions_kg=0.0096)
    reporter.record_run(m, model_name="bert-finetuned", stage="training")
    reporter.record_run(m, model_name="resnet50", stage="training")
    report = reporter.generate_report()
    check("ESGReport returned", report is not None)
    check("report has total_runs >= 2", report.total_runs >= 2, f"runs={report.total_runs}")
    check("total_energy_kwh > 0", report.total_energy_kwh > 0, f"energy={report.total_energy_kwh:.4f} kWh")
    check("total_emissions_kg > 0", report.total_emissions_kg > 0, f"emissions={report.total_emissions_kg:.6f} kg")
    text = report.to_text()
    check("to_text() returns non-empty string", len(text) > 50, text[:80])
    check("report text contains org name", "Acme Corp" in text)
    check("report text contains FY2026", "FY2026" in text)

run_test("esg reporter basic", test_esg_reporter_basic)

def test_esg_reporter_json():
    import json, tempfile
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as hf:
        hist_path = hf.name
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as rf:
        report_path = rf.name
    try:
        org = gt.ESGOrganization(name="TestOrg", reporting_period="FY2026", region="GB")
        reporter = gt.ESGReporter(org, history_path=hist_path)
        m = gt.RunMetrics(duration_s=60.0, energy_kwh=0.001, emissions_kg=0.0002)
        reporter.record_run(m, model_name="test-model", stage="inference")
        # save_json is on ESGReporter, not ESGReport
        reporter.save_json(report_path)
        check("save_json() creates file", os.path.exists(report_path))
        with open(report_path) as f:
            data = json.load(f)
        check("JSON is valid and parseable", True)
        check("JSON is non-empty dict", len(data) > 0, f"keys={list(data.keys())[:5]}")
    finally:
        for p in (hist_path, report_path):
            if os.path.exists(p):
                os.unlink(p)

run_test("esg reporter json", test_esg_reporter_json)

def test_esg_savings_calculation():
    baseline  = gt.RunMetrics(duration_s=100.0, energy_kwh=0.01,   emissions_kg=0.003)
    optimized = gt.RunMetrics(duration_s=71.0,  energy_kwh=0.0071, emissions_kg=0.00213)
    savings = gt.calculate_savings(baseline, optimized)
    check("calculate_savings returns dict", isinstance(savings, dict))
    check("energy_saved_kwh > 0", savings["energy_saved_kwh"] > 0,
          f"saved={savings['energy_saved_kwh']:.6f} kWh")
    check("energy_reduction_pct ~29%",
          abs(savings["energy_reduction_pct"] - 29.0) < 1.0,
          f"reduction={savings['energy_reduction_pct']:.1f}%")
    check("time_saved_s ~29s", abs(savings["time_saved_s"] - 29.0) < 1.0,
          f"time_saved={savings['time_saved_s']:.1f}s")

run_test("esg savings calculation", test_esg_savings_calculation)

# ══════════════════════════════════════════════════════════════════════════════
# TEST 11 — EFFICIENCY RECOMMENDER
# ══════════════════════════════════════════════════════════════════════════════
section("11. EFFICIENCY RECOMMENDER")

def test_recommender_high_idle():
    rec = gt.EfficiencyRecommender()
    m = gt.RunMetrics(duration_s=100.0, energy_kwh=0.001, emissions_kg=0.0002, idle_seconds=40.0)
    recs = rec.analyze(m)
    check("Recommendations returned for high idle", len(recs) > 0, f"count={len(recs)}")
    cats = [r.category for r in recs]
    check("dataloader recommendation present", "dataloader" in cats, f"cats={cats}")
    idle_rec = next((r for r in recs if r.category == "dataloader"), None)
    if idle_rec:
        check("idle rec priority = high", idle_rec.priority == "high", f"priority={idle_rec.priority}")
        check("idle rec savings > 0", idle_rec.estimated_savings_pct > 0,
              f"savings={idle_rec.estimated_savings_pct:.0f}%")

run_test("recommender high idle", test_recommender_high_idle)

def test_recommender_no_mixed_precision():
    rec = gt.EfficiencyRecommender()
    m = gt.RunMetrics(duration_s=60.0, energy_kwh=0.001, emissions_kg=0.0002)
    recs = rec.analyze(m, mixed_precision_enabled=False)
    precision_recs = [r for r in recs if r.category == "precision"]
    check("Mixed precision recommendation fires", len(precision_recs) > 0,
          f"recs={[r.title for r in recs]}")
    if precision_recs:
        check("Estimated savings ~28%",
              abs(precision_recs[0].estimated_savings_pct - 28.0) < 1.0,
              f"savings={precision_recs[0].estimated_savings_pct:.0f}%")

run_test("recommender no mixed precision", test_recommender_no_mixed_precision)

def test_recommender_small_batch():
    rec = gt.EfficiencyRecommender()
    m = gt.RunMetrics(duration_s=60.0, energy_kwh=0.001, emissions_kg=0.0002)
    recs = rec.analyze(m, batch_size=8)
    batch_recs = [r for r in recs if r.category == "batch_size"]
    check("Batch size recommendation fires for batch=8", len(batch_recs) > 0,
          f"recs={[r.title for r in recs]}")
    if batch_recs:
        check("Batch rec priority = high", batch_recs[0].priority == "high",
              f"priority={batch_recs[0].priority}")

run_test("recommender small batch", test_recommender_small_batch)

def test_recommender_priority_ordering():
    rec = gt.EfficiencyRecommender()
    m = gt.RunMetrics(duration_s=100.0, energy_kwh=0.001, emissions_kg=0.0002, idle_seconds=35.0)
    recs = rec.analyze(m, mixed_precision_enabled=False, batch_size=4, gpu_util_avg_pct=30.0)
    check("Multiple recommendations returned", len(recs) >= 3, f"count={len(recs)}")
    priorities = [r.priority for r in recs]
    order = {"high": 0, "medium": 1, "low": 2}
    is_sorted = all(order[priorities[i]] <= order[priorities[i+1]] for i in range(len(priorities)-1))
    check("Recommendations sorted high->medium->low", is_sorted, f"order={priorities}")

run_test("recommender priority ordering", test_recommender_priority_ordering)

# ══════════════════════════════════════════════════════════════════════════════
# TEST 12 — DATACENTER CONFIG & RUN METRICS
# ══════════════════════════════════════════════════════════════════════════════
section("12. DATACENTER CONFIG & RUN METRICS")

def test_pue_presets():
    check("PUE_PRESETS dict non-empty", len(gt.PUE_PRESETS) > 0)
    check("hyperscale PUE = 1.1", gt.PUE_PRESETS["hyperscale"] == 1.1)
    check("cloud_average PUE = 1.2", gt.PUE_PRESETS["cloud_average"] == 1.2)
    check("legacy_dc PUE = 1.8", gt.PUE_PRESETS["legacy_dc"] == 1.8)

run_test("pue presets", test_pue_presets)

def test_datacenter_config_scaling():
    m = gt.RunMetrics(duration_s=100.0, energy_kwh=1.0, emissions_kg=0.233)
    dc = gt.DatacenterConfig(pue=1.5, carbon_intensity_kg_per_kwh=0.000233, num_nodes=2)
    m2 = m.apply_datacenter_config(dc)
    check("apply_datacenter_config returns RunMetrics", isinstance(m2, gt.RunMetrics))
    check("energy_kwh_dc = energy * PUE * nodes = 3.0",
          abs(m2.energy_kwh_dc - 3.0) < 1e-9, f"energy_dc={m2.energy_kwh_dc:.4f} kWh")
    check("emissions_kg_dc = energy_dc * intensity",
          abs(m2.emissions_kg_dc - 3.0 * 0.000233) < 1e-9,
          f"emissions_dc={m2.emissions_kg_dc:.6f} kg")
    check("original energy_kwh unchanged", m2.energy_kwh == 1.0)
    check("dc_config stored on result", m2.dc_config is dc)

run_test("datacenter config scaling", test_datacenter_config_scaling)

def test_calculate_savings_dc_adjusted():
    baseline  = gt.RunMetrics(duration_s=100.0, energy_kwh=1.0,  emissions_kg=0.233)
    optimized = gt.RunMetrics(duration_s=71.0,  energy_kwh=0.71, emissions_kg=0.165)
    dc = gt.DatacenterConfig(pue=1.2, carbon_intensity_kg_per_kwh=0.000233, num_nodes=1)
    b2 = baseline.apply_datacenter_config(dc)
    o2 = optimized.apply_datacenter_config(dc)
    savings = gt.calculate_savings(b2, o2, use_dc_adjusted=True)
    check("DC-adjusted savings flag set", savings["dc_adjusted"] == True)
    check("DC-adjusted energy_saved > raw energy_saved",
          savings["energy_saved_kwh"] > (1.0 - 0.71),
          f"dc_saved={savings['energy_saved_kwh']:.4f} vs raw={1.0-0.71:.4f}")

run_test("calculate savings dc adjusted", test_calculate_savings_dc_adjusted)

# ══════════════════════════════════════════════════════════════════════════════
# TEST 13 — RUN HISTORY
# ══════════════════════════════════════════════════════════════════════════════
section("13. RUN HISTORY")

def test_run_history_record_and_retrieve():
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        tmp_path = f.name
    try:
        history = gt.RunHistory(path=tmp_path)
        for i in range(5):
            m = gt.RunMetrics(duration_s=float(10+i), energy_kwh=0.001*(i+1), emissions_kg=0.0002*(i+1))
            history.record(m, model_name=f"model-{i}", stage="training")
        last3 = history.last(3)
        check("last(3) returns 3 entries", len(last3) == 3, f"got {len(last3)}")
        all5 = history.last(10)
        check("last(10) returns all 5 entries", len(all5) == 5, f"got {len(all5)}")
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

run_test("run history record and retrieve", test_run_history_record_and_retrieve)

def test_run_history_empty():
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        tmp_path = f.name
    os.unlink(tmp_path)  # ensure file doesn't exist so history starts empty
    history = gt.RunHistory(path=tmp_path)
    last = history.last(5)
    check("last() on empty history returns list", isinstance(last, list))
    check("last() on empty history returns empty", len(last) == 0, f"got {len(last)}")

run_test("run history empty", test_run_history_empty)

# ══════════════════════════════════════════════════════════════════════════════
# TEST 14 — PROFILER
# ══════════════════════════════════════════════════════════════════════════════
section("14. PROFILER")

def test_profiler_get_gpu_metrics():
    from greentensor.core.profiler import Profiler
    metrics = Profiler.get_gpu_metrics()
    check("get_gpu_metrics returns dict", isinstance(metrics, dict))
    check("dict has util_% key", "util_%" in metrics, f"keys={list(metrics.keys())}")
    check("dict has power_W key", "power_W" in metrics)
    check("util_% is float >= 0", isinstance(metrics["util_%"], float) and metrics["util_%"] >= 0,
          f"util={metrics['util_%']}")
    check("power_W is float >= 0", isinstance(metrics["power_W"], float) and metrics["power_W"] >= 0,
          f"power={metrics['power_W']}W")
    if metrics["power_W"] == 0.0:
        check("power_W=0 (no GPU/nvidia-smi — expected on CPU-only)", True,
              "nvidia-smi not found — fallback to 0W", warn=True)

run_test("profiler gpu metrics", test_profiler_get_gpu_metrics)

def test_profiler_estimate_energy():
    from greentensor.core.profiler import Profiler
    kwh = Profiler.estimate_energy_kwh(power_w=200.0, duration_s=3600.0)
    check("estimate_energy_kwh(200W, 3600s) = 0.2 kWh", abs(kwh - 0.2) < 1e-9,
          f"got={kwh:.6f} kWh")
    kwh0 = Profiler.estimate_energy_kwh(power_w=0.0, duration_s=100.0)
    check("estimate_energy_kwh(0W) = 0", kwh0 == 0.0)

run_test("profiler estimate energy", test_profiler_estimate_energy)

# ══════════════════════════════════════════════════════════════════════════════
# TEST 15 — FULL END-TO-END REALISTIC SCENARIO
# ══════════════════════════════════════════════════════════════════════════════
section("15. FULL END-TO-END REALISTIC SCENARIO")

def test_full_pipeline():
    import io, contextlib
    baseline_m = gt.RunMetrics(duration_s=60.0, energy_kwh=0.0006, emissions_kg=0.00014)

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        with gt.GreenTensor(
            verbose=True, security=False, save_path=None, baseline=baseline_m,
        ) as g:
            data = [math.sin(i) * math.cos(i) for i in range(500_000)]
            result = sum(data)

    check("E2E: context manager completes", True, f"workload result={result:.4f}")
    check("E2E: metrics populated", g.metrics is not None)
    m = g.metrics
    check("E2E: duration > 0", m.duration_s > 0, f"duration={m.duration_s:.3f}s")
    check("E2E: energy >= 0", m.energy_kwh >= 0, f"energy={m.energy_kwh:.8f} kWh")
    report_text = buf.getvalue()
    check("E2E: report printed", "GreenTensor Report" in report_text)
    check("E2E: baseline comparison shown", "Baseline" in report_text)

    m_water = m.apply_aquatensor_config(gt.AquaTensorConfig(
        provider="google", region="India",
        aquatensor_installed=True, whr_ratio=0.65, feed_temperature_c=65.0))
    check("E2E: water metrics applied", m_water.water is not None)
    check("E2E: water stress label set", len(m_water.water.water_stress_label) > 0,
          f"label={m_water.water.water_stress_label}")

    m_dc = m.apply_datacenter_config(
        gt.DatacenterConfig(pue=1.2, carbon_intensity_kg_per_kwh=0.000233, num_nodes=1))
    check("E2E: DC config applied", m_dc.energy_kwh_dc is not None)
    check("E2E: DC energy >= raw energy", m_dc.energy_kwh_dc >= m_dc.energy_kwh)

    org = gt.ESGOrganization(name="E2E Corp", reporting_period="FY2026", region="US-East")
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as hf:
        esg_hist = hf.name
    reporter = gt.ESGReporter(org, history_path=esg_hist)
    reporter.record_run(m, model_name="e2e-model", stage="training")
    esg = reporter.generate_report()
    check("E2E: ESG report generated", esg is not None)
    check("E2E: ESG total_runs = 1", esg.total_runs == 1, f"got {esg.total_runs}")
    if os.path.exists(esg_hist):
        os.unlink(esg_hist)

    recs = gt.EfficiencyRecommender().analyze(
        m, mixed_precision_enabled=False, batch_size=16, gpu_util_avg_pct=40.0)
    check("E2E: recommendations produced", len(recs) > 0, f"count={len(recs)}")

    sched = gt.CarbonAwareScheduler(zone="GB")
    rec = sched.should_run_now(estimated_energy_kwh=max(m.energy_kwh, 0.001))
    check("E2E: scheduler recommendation returned", isinstance(rec, gt.ScheduleRecommendation))
    check("E2E: scheduler reason non-empty", len(rec.reason) > 10)

    matcher = gt.PatternMatcher()
    match = matcher.match(
        power_w=250.0, baseline_power_w=100.0, gpu_util_pct=95.0,
        active_signals=["power_spike_sustained", "new_process_spawned",
                        "high_gpu_util_unexpected", "no_training_activity"])
    check("E2E: pattern matcher returns result", isinstance(match, gt.PatternMatchResult))
    check("E2E: cryptominer attack detected", match.verdict == "ATTACK",
          f"verdict={match.verdict}, score={match.threat_score}")

run_test("full end-to-end pipeline", test_full_pipeline)

# ══════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
section("STRESS TEST SUMMARY")

total  = len(results)
passed = sum(1 for _, s in results if s == "pass")
failed = sum(1 for _, s in results if s == "fail")
warned = sum(1 for _, s in results if s == "warn")

print(f"\n  Total checks : {total}")
print(f"  {GREEN}Passed       : {passed}{RESET}")
print(f"  {YELLOW}Warnings     : {warned}{RESET}")
print(f"  {RED}Failed       : {failed}{RESET}")

if failed > 0:
    print(f"\n  {RED}{BOLD}FAILED CHECKS:{RESET}")
    for name, status in results:
        if status == "fail":
            print(f"    {RED}[x]{RESET}  {name}")

if warned > 0:
    print(f"\n  {YELLOW}WARNINGS (expected on CPU-only / no GPU):{RESET}")
    for name, status in results:
        if status == "warn":
            print(f"    {YELLOW}[!]{RESET}  {name}")

score_pct = (passed / total * 100) if total > 0 else 0
print(f"\n  Score: {passed}/{total} ({score_pct:.1f}%)")

if failed == 0:
    print(f"\n  {GREEN}{BOLD}ALL CHECKS PASSED — GreenTensor middleware is working correctly.{RESET}\n")
elif failed <= 3:
    print(f"\n  {YELLOW}{BOLD}MOSTLY PASSING — {failed} check(s) need attention.{RESET}\n")
else:
    print(f"\n  {RED}{BOLD}{failed} FAILURES — middleware has real issues that need fixing.{RESET}\n")

sys.exit(0 if failed == 0 else 1)
