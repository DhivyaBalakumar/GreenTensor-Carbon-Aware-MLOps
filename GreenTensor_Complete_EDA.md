# GreenTensor Package - Complete EDA (Exploratory Data Analysis)

**Version:** 0.7.1  
**Author:** Dhivya Balakumar  
**License:** MIT  
**Purpose:** Carbon-Secure MLOps + Water Intelligence for Sustainable AI

---

## 📋 Executive Summary

GreenTensor is a **comprehensive middleware platform** that wraps around existing ML training code to provide:
1. **Energy & Carbon Tracking** - Real-time measurement of GPU power consumption and CO2 emissions
2. **Security Monitoring** - ML-specific cyberattack detection (cryptominers, backdoors, data exfiltration)
3. **Water Intelligence** - Tracks water consumption and production via membrane distillation
4. **Automatic Optimization** - GPU efficiency improvements (mixed precision, batch size, idle detection)
5. **ESG Compliance** - Automated Scope 2 emissions reporting (SEC, EU CSRD, GHG Protocol)
6. **Carbon-Aware Scheduling** - Recommends optimal training times based on grid carbon intensity

---

## 🏗️ Package Architecture

```
greentensor/
├── core/                    # Core functionality
│   ├── context.py          # Main GreenTensor context manager
│   ├── tracker.py          # Energy & carbon tracking (CodeCarbon + nvidia-smi)
│   ├── profiler.py         # GPU metrics collection
│   ├── budget.py           # Carbon budget enforcement
│   └── history.py          # Run history persistence
│
├── security/               # ML Security Suite
│   ├── anomaly_detector.py    # Power-based anomaly detection (alibi-detect)
│   ├── digital_footprint.py   # Multi-signal attack detection (MITRE ATLAS)
│   └── pattern_matcher.py     # Correlates carbon spikes with threats
│
├── optimizers/             # Automatic Optimizations
│   ├── gpu_optimizer.py       # Mixed precision, cuDNN tuning
│   ├── batch_optimizer.py     # Optimal batch size calculation
│   ├── idle_optimizer.py      # Idle GPU detection & throttling
│   └── recommender.py         # Post-run efficiency recommendations
│
├── water/                  # Water Intelligence
│   └── aquatensor.py          # Membrane distillation calculations
│
├── scheduler/              # Carbon-Aware Scheduling
│   └── carbon_scheduler.py    # Grid carbon intensity API + scheduling
│
├── report/                 # Reporting & Metrics
│   ├── metrics.py             # RunMetrics, DatacenterConfig, PUE
│   ├── esg.py                 # ESG report generation (GHG Protocol)
│   └── report.py              # Text report formatting
│
├── alerts/                 # Alerting & Webhooks
│   └── webhooks.py            # Slack, PagerDuty, generic webhooks
│
├── integrations/           # Framework Integrations
│   ├── huggingface.py         # HuggingFace Transformers callback
│   └── lightning.py           # PyTorch Lightning callback
│
└── utils/                  # Utilities
    ├── config.py              # Configuration management
    └── logger.py              # Logging setup
```

---

## 🎯 Core Capabilities

### 1. **Energy & Carbon Tracking**

**What it does:**
- Measures real GPU power consumption using CodeCarbon (hardware counters)
- Falls back to nvidia-smi power sampling when CodeCarbon unavailable
- Calculates CO2 emissions from energy × regional carbon intensity
- Tracks idle GPU time to identify data pipeline bottlenecks

**Key Components:**
- `Tracker` - Energy measurement engine
- `Profiler` - GPU metrics collection (utilization %, power W)
- `RunMetrics` - Stores duration, energy, emissions, idle time

**Usage:**
```python
from greentensor import GreenTensor

with GreenTensor() as gt:
    train()  # Your existing code

print(f"Energy: {gt.metrics.energy_kwh} kWh")
print(f"CO2: {gt.metrics.emissions_kg} kg")
print(f"Idle time: {gt.metrics.idle_seconds}s")
```

**Output:**
- Energy consumption (kWh)
- Carbon emissions (kg CO2)
- Idle time (seconds)
- Duration (seconds)

---

### 2. **Security Monitoring**

**What it does:**
- Detects ML-specific cyberattacks using carbon footprint + digital signals
- Monitors for cryptominers, backdoors, data exfiltration, model theft
- Tags threats with MITRE ATLAS technique IDs
- Provides threat scores and recommended actions

**Key Components:**

#### **A. Anomaly Detector**
- Uses **alibi-detect SpectralResidual** algorithm for statistical anomaly detection
- Monitors GPU power time series for unusual patterns
- Integrates **LLM Guard** for prompt injection, secrets, PII detection
- Detects: power spikes, idle drain, sustained overload

**Attack Patterns Detected:**
- `power_spike` - Sudden energy surge (cryptominer injection)
- `sustained_overload` - Prolonged high usage (data exfiltration)
- `idle_drain` - GPU power draw while idle (hidden process)
- `prompt_injection` - Adversarial inputs (via LLM Guard)
- `data_leakage` - Sensitive data in outputs (via LLM Guard)

#### **B. Digital Footprint Scanner**
Multi-signal attack detection across ML lifecycle:

**Pre-Deployment (Training):**
- Model weight tampering (SHA-256 hash verification) - `AML.T0018`
- Supply chain attacks (typosquatting detection) - `AML.T0010`
- Process injection (unexpected child processes) - `AML.T0011`
- Network exfiltration (untrusted connections) - `AML.T0024`

**Post-Deployment (Inference):**
- Inference latency spikes (backdoor triggers) - `AML.T0043`
- Model extraction (high-frequency probing) - `AML.T0044`
- Confidence anomalies (adversarial inputs)

#### **C. Pattern Matcher**
- Correlates carbon spikes with digital footprint signals
- Calculates threat scores (0-100)
- Provides verdicts: BENIGN, SUSPICIOUS, ATTACK
- Recommends actions (monitor, investigate, terminate)

**Usage:**
```python
with GreenTensor(
    security=True,
    scan_dependencies=True,
    monitor_network=True,
    stage="production"
) as gt:
    gt.register_model("model.pt")  # Track integrity
    train()
    gt.verify_model("model.pt")    # Check for tampering

alerts = gt.security_alerts
footprint = gt.footprint_report
```

**Output:**
- Security alerts with severity (low/medium/high/critical)
- Digital footprint report (events, network connections, processes)
- Model integrity verification results
- Threat scores and recommended actions

---

### 3. **Water Intelligence (AquaTensor)**

**What it does:**
- Calculates water consumed by datacenter cooling (using WUE coefficients)
- Calculates water produced by membrane distillation from waste heat
- Provides net water impact (consumed - produced)
- Regional water stress context (WRI Aqueduct index)

**Physics:**
- Every watt of GPU compute becomes waste heat
- Membrane distillation uses heat (40-80°C) to produce fresh water
- Yield: 2.5-8.5 liters per kWh of thermal energy

**Key Metrics:**
- **WUE** - Water Usage Effectiveness (liters per kWh)
- **WHR** - Waste Heat Recovery ratio (fraction captured)
- **MD Yield** - Membrane Distillation efficiency (liters/kWh)
- **WPI** - Water Productivity Index (liters produced per kWh)

**Provider WUE Values:**
- Google: 0.49 L/kWh
- Microsoft: 0.49 L/kWh
- AWS: 1.80 L/kWh (estimated)
- Meta: 0.26 L/kWh
- On-premise: 1.80 L/kWh (ASHRAE average)

**Usage:**
```python
from greentensor import AquaTensorConfig

with GreenTensor() as gt:
    train()

water = gt.metrics.apply_aquatensor_config(AquaTensorConfig(
    provider="aws",
    region="India",
    aquatensor_installed=True,
    whr_ratio=0.65,
    feed_temperature_c=65.0,
))

print(f"Water consumed: {water.water.water_consumed_liters:.3f} L")
print(f"Water produced: {water.water.water_produced_liters:.3f} L")
print(f"Net impact: {water.water.net_water_impact_liters:.3f} L")
print(f"Drinking water: {water.water.drinking_water_days:.1f} person-days")
```

**Output:**
- Water consumed (liters)
- Water produced (liters)
- Net water impact (negative = net positive)
- Drinking water equivalency (person-days)
- Regional water stress index

---

### 4. **Automatic Optimizations**

**What it does:**
- Applies GPU optimizations automatically on context entry
- Reverts settings on exit (non-destructive)
- Monitors for inefficiencies and provides recommendations

**Key Components:**

#### **A. GPU Optimizer**
- **Mixed Precision (FP16)** - 20-40% energy reduction on Tensor Core GPUs
- **cuDNN Benchmark Mode** - Finds optimal kernels for input shapes
- Automatic enable/disable on context entry/exit

#### **B. Batch Optimizer**
- Queries available GPU memory
- Recommends optimal batch size
- Calculates memory requirements per sample

#### **C. Idle Optimizer**
- Background thread monitoring GPU utilization
- Throttles process when GPU idle (<5% utilization)
- Tracks total idle time

#### **D. Efficiency Recommender**
- Analyzes completed runs
- Generates ranked recommendations
- Estimates savings percentage

**Recommendations Generated:**
- DataLoader bottlenecks (add num_workers)
- Mixed precision not enabled
- Batch size too small
- Low GPU utilization
- Model distillation opportunities
- Experiment batching for short runs

**Usage:**
```python
with GreenTensor() as gt:
    with gt.mixed_precision():  # Auto FP16
        train()

# Get recommendations
gt.recommend(
    batch_size=32,
    mixed_precision_enabled=False,
    gpu_util_avg_pct=45.0
)
```

**Output:**
- Priority-ranked recommendations (high/medium/low)
- Estimated savings percentage
- Actionable implementation details

---

### 5. **Carbon-Aware Scheduling**

**What it does:**
- Queries real-time grid carbon intensity via electricityMap API
- Recommends optimal time windows to run training jobs
- Can wait for clean grid before starting (context manager)
- Falls back to static regional averages with time-of-day variation

**Supported Zones:**
- US: CAL-CISO, TEX-ERCO, MIDA-PJM, NW-PACW
- Europe: GB, DE, FR, NO-NO1
- Asia: IN-NO, IN-SO, SG
- Australia: AU-NSW

**Usage:**
```python
from greentensor import CarbonAwareScheduler

scheduler = CarbonAwareScheduler(zone="US-CAL-CISO")

# Check before running
rec = scheduler.should_run_now(estimated_energy_kwh=0.5)
if rec.run_now:
    train()
else:
    print(rec.reason)
    # Wait rec.recommended_delay_hours

# Or wait automatically
with scheduler.wait_for_clean_grid(max_wait_hours=6):
    train()
```

**Output:**
- Current grid carbon intensity (gCO2/kWh)
- Recommendation (RUN NOW or WAIT)
- Optimal time window (UTC)
- Estimated carbon savings (%)
- CO2 avoided (kg)

---

### 6. **ESG Compliance & Reporting**

**What it does:**
- Generates regulatory-compliant ESG reports
- Tracks Scope 2 emissions across multiple runs
- Provides real-world equivalencies (km driven, trees, flights)
- Aligned with GHG Protocol, SEC, EU CSRD, ISO 14064-1

**Key Components:**

#### **A. ESG Reporter**
- Accumulates run metrics across sessions
- Generates structured reports (JSON, text)
- Includes datacenter PUE overhead
- Tracks security incidents

#### **B. Run History**
- Persists metrics to JSON file
- Used by dashboard and ESG reporter
- Tracks model name, stage, timestamp

**Usage:**
```python
from greentensor import ESGReporter, ESGOrganization

reporter = ESGReporter(ESGOrganization(
    name="Acme Corp",
    reporting_period="FY2025",
    region="US-East",
))

# After each run
reporter.record_run(
    gt.metrics,
    model_name="bert-finetuned",
    stage="training"
)

# Generate report
report = reporter.generate_report()
print(report.to_text())
report.save_json("scope2_emissions_FY2025.json")
```

**Output:**
- Total runs, compute time, energy, CO2
- Datacenter-adjusted values (PUE)
- Avoided emissions (GreenTensor savings)
- Real-world equivalencies
- Security incident summary
- Compliance notes

---

### 7. **Carbon Budget Enforcement**

**What it does:**
- Sets hard limits on carbon emissions per run
- Raises exception when budget exceeded
- Warns at configurable threshold (default 80%)

**Usage:**
```python
from greentensor import CarbonBudget, CarbonBudgetExceeded

try:
    with GreenTensor(
        carbon_budget=CarbonBudget(
            max_kg_per_run=0.005,
            warn_at_pct=80.0,
            job_name="bert-v3"
        )
    ) as gt:
        train()
except CarbonBudgetExceeded as e:
    print(f"Budget exceeded: {e.overage_kg:.6f} kg over")
```

**Output:**
- Warning at 80% of budget
- Exception raised when exceeded
- Overage amount and percentage

---

### 8. **Alerting & Webhooks**

**What it does:**
- Sends security alerts and budget violations to external systems
- Supports multiple destinations simultaneously

**Supported Integrations:**
- **Slack** - Webhook with formatted messages
- **PagerDuty** - Incident creation
- **Generic Webhook** - Any HTTP endpoint
- **MultiAlert** - Send to multiple destinations

**Usage:**
```python
from greentensor import SlackWebhook, PagerDutyAlert, MultiAlert

with GreenTensor(on_alert=MultiAlert(
    SlackWebhook("https://hooks.slack.com/services/YOUR/WEBHOOK"),
    PagerDutyAlert("your-integration-key"),
)) as gt:
    train()
```

**Output:**
- Real-time alerts to configured destinations
- Formatted with severity, type, message, MITRE technique

---

### 9. **Framework Integrations**

**What it does:**
- Seamless integration with popular ML frameworks
- Automatic metric collection via callbacks

**Supported Frameworks:**

#### **A. HuggingFace Transformers**
```python
from transformers import Trainer
from greentensor.integrations.huggingface import GreenTensorCallback

trainer = Trainer(
    model=model,
    callbacks=[GreenTensorCallback(model_name="bert-finetuned")],
)
trainer.train()
```

#### **B. PyTorch Lightning**
```python
from pytorch_lightning import Trainer
from greentensor.integrations.lightning import GreenTensorCallback

trainer = Trainer(
    callbacks=[GreenTensorCallback(model_name="my_model")],
)
trainer.fit(model)
```

---

### 10. **Datacenter Configuration**

**What it does:**
- Applies datacenter overhead (PUE) to GPU measurements
- Adjusts for regional carbon intensity
- Supports multi-node distributed training

**PUE Presets:**
- Local workstation: 1.0 (no overhead)
- Hyperscale: 1.1 (Google/Microsoft/Meta)
- Cloud average: 1.2 (AWS/GCP/Azure)
- Enterprise DC: 1.5 (typical on-premise)
- Legacy DC: 1.8 (older/inefficient)

**Usage:**
```python
from greentensor import DatacenterConfig

dc = DatacenterConfig(
    pue=1.2,
    carbon_intensity_kg_per_kwh=0.000320,
    num_nodes=4,
    dc_name="AWS-US-East"
)

adjusted = gt.metrics.apply_datacenter_config(dc)
print(f"DC-adjusted energy: {adjusted.energy_kwh_dc} kWh")
print(f"DC-adjusted CO2: {adjusted.emissions_kg_dc} kg")
```

---

## 📊 Data Flow

```
User Code (train())
    ↓
GreenTensor Context Manager (__enter__)
    ↓
├─→ GPU Optimizer (apply mixed precision, cuDNN)
├─→ Idle Optimizer (start monitoring thread)
├─→ Anomaly Detector (start monitoring thread)
├─→ Digital Footprint Scanner (start monitoring threads)
└─→ Tracker (start energy measurement)
    ↓
[Training happens]
    ↓
GreenTensor Context Manager (__exit__)
    ↓
├─→ Tracker (stop, calculate energy & CO2)
├─→ Idle Optimizer (stop, report idle time)
├─→ GPU Optimizer (revert settings)
├─→ Anomaly Detector (stop, return alerts)
├─→ Digital Footprint Scanner (stop, return report)
├─→ Carbon Budget (check limits)
├─→ RunHistory (auto-record)
└─→ Generate Report (if verbose=True)
    ↓
RunMetrics Object
    ↓
├─→ Apply Datacenter Config (PUE, multi-node)
├─→ Apply AquaTensor Config (water metrics)
├─→ ESG Reporter (accumulate for compliance)
├─→ Efficiency Recommender (generate recommendations)
└─→ Save to .pkl file
```

---

## 🔧 Configuration Options

### **GreenTensor Constructor Parameters:**

```python
GreenTensor(
    config=None,                    # Config object (carbon intensity, thresholds)
    baseline=None,                  # Baseline metrics for comparison
    verbose=True,                   # Print report on exit
    security=True,                  # Enable security monitoring
    security_config=None,           # AnomalyDetectorConfig
    on_alert=None,                  # Alert handler (webhook, etc.)
    save_path="greentensor_metrics.pkl",  # Metrics save path
    stage="pre_deployment",         # "pre_deployment" or "post_deployment"
    scan_dependencies=True,         # Scan for malicious packages
    monitor_network=True,           # Monitor network connections
    monitor_processes=True,         # Monitor child processes
    trusted_hosts=None,             # List of allowed hosts
    carbon_budget=None,             # CarbonBudget object
    model_name="unknown",           # Model name for tracking
)
```

---

## 📈 Metrics & Outputs

### **RunMetrics Object:**
```python
{
    "duration_s": 45.23,
    "energy_kwh": 0.000412,
    "emissions_kg": 0.000096,
    "idle_seconds": 2.10,
    "energy_kwh_dc": 0.000494,      # With PUE applied
    "emissions_kg_dc": 0.000115,    # With DC carbon intensity
    "water": {
        "water_consumed_liters": 0.000202,
        "water_produced_liters": 0.001474,
        "net_water_impact_liters": -0.001272,
        "drinking_water_days": 0.737,
        "water_stress_index": 4.5,
        "water_stress_label": "Extremely High"
    }
}
```

### **Security Alerts:**
```python
{
    "timestamp": 1715234567.89,
    "alert_type": "power_spike",
    "severity": "high",
    "source": "alibi-detect",
    "message": "Power spike +140% above baseline...",
    "current_value": 250.5,
    "baseline_value": 104.2,
    "deviation_pct": 140.3
}
```

### **Digital Footprint Report:**
```python
{
    "session_start": 1715234500.0,
    "session_end": 1715234545.23,
    "stage": "pre_deployment",
    "events": [...],
    "model_hashes": {"model.pt": "sha256:abc123..."},
    "network_connections": ["pypi.org:443", "huggingface.co:443"],
    "child_processes": ["python(pid=12345)"],
    "critical_count": 0,
    "high_count": 2,
    "is_clean": False
}
```

---

## 🎯 Use Cases

### **1. Cost Reduction**
- Identify inefficiencies (idle time, small batches)
- Optimize GPU utilization
- Reduce energy consumption by 20-40%

### **2. ESG Compliance**
- Automated Scope 2 emissions reporting
- SEC Climate Disclosure compliance
- EU CSRD compliance
- GHG Protocol alignment

### **3. Security**
- Detect cryptominer injection
- Prevent data exfiltration
- Identify backdoor triggers
- Monitor model integrity

### **4. Sustainability**
- Track carbon footprint
- Optimize for clean grid windows
- Calculate water impact
- Generate ESG reports

### **5. Research & Development**
- Compare model architectures
- Benchmark efficiency
- Track improvements over time
- Generate reproducible metrics

---

## 🚀 Quick Start Examples

### **Basic Usage:**
```python
from greentensor import GreenTensor

with GreenTensor() as gt:
    train()  # Your existing code

print(f"Energy: {gt.metrics.energy_kwh} kWh")
print(f"CO2: {gt.metrics.emissions_kg} kg")
```

### **With Security:**
```python
with GreenTensor(security=True, stage="production") as gt:
    gt.register_model("model.pt")
    train()
    gt.verify_model("model.pt")

print(f"Security alerts: {len(gt.security_alerts)}")
```

### **With Water Intelligence:**
```python
from greentensor import AquaTensorConfig

with GreenTensor() as gt:
    train()

water = gt.metrics.apply_aquatensor_config(AquaTensorConfig(
    aquatensor_installed=True,
    feed_temperature_c=65.0,
))
print(f"Net water: {water.water.net_water_impact_liters:.3f} L")
```

### **With Carbon Scheduling:**
```python
from greentensor import CarbonAwareScheduler

scheduler = CarbonAwareScheduler(zone="US-CAL-CISO")
with scheduler.wait_for_clean_grid(max_wait_hours=6):
    with GreenTensor() as gt:
        train()
```

### **With ESG Reporting:**
```python
from greentensor import ESGReporter, ESGOrganization

reporter = ESGReporter(ESGOrganization(
    name="Acme Corp",
    reporting_period="FY2025",
))

with GreenTensor() as gt:
    train()

reporter.record_run(gt.metrics, model_name="bert")
report = reporter.generate_report()
print(report.to_text())
```

---

## 📦 Dependencies

### **Core:**
- `numpy` - Numerical operations
- `psutil` - System monitoring
- `codecarbon` - Energy tracking (optional)

### **Security (Optional):**
- `alibi-detect` - Anomaly detection
- `llm-guard` - LLM input/output scanning

### **Framework Integrations:**
- `torch` - PyTorch support
- `transformers` - HuggingFace support
- `pytorch-lightning` - Lightning support

### **Dashboards:**
- `streamlit` - Web dashboards
- `pandas` - Data analysis
- `plotly` - Visualizations

---

## 🎓 Key Innovations

1. **Carbon-Based Security** - First tool to use power consumption patterns for ML attack detection
2. **Water Intelligence** - Only tool tracking water impact of AI compute
3. **Unified Platform** - Combines sustainability, security, and optimization in one package
4. **Zero Code Changes** - Wraps existing training code with context manager
5. **MITRE ATLAS Mapping** - Tags threats with standardized technique IDs
6. **ESG Automation** - Generates compliance reports automatically
7. **Carbon-Aware Scheduling** - Optimizes for grid carbon intensity
8. **Pattern Matching** - Correlates carbon spikes with digital footprint signals

---

## 📊 Performance Impact

- **Overhead:** <2% runtime overhead from monitoring
- **Energy Savings:** 20-40% with mixed precision
- **Idle Detection:** Identifies 10-50% wasted GPU time
- **Security:** Real-time threat detection with <1s latency
- **Storage:** ~1KB per run for metrics persistence

---

## 🔮 Future Roadmap

Based on CHANGELOG.md, potential future features:
- Real-time dashboard streaming
- Multi-GPU distributed training support
- Cloud provider integrations (AWS, GCP, Azure)
- Model compression recommendations
- Automated hyperparameter tuning for efficiency
- Blockchain-based carbon credit tracking

---

## 📝 Summary

**GreenTensor is a comprehensive MLOps platform that provides:**

✅ **Visibility** - Know exactly how much energy and carbon your models consume  
✅ **Security** - Detect ML-specific attacks in real-time  
✅ **Optimization** - Automatic efficiency improvements  
✅ **Compliance** - Automated ESG reporting  
✅ **Sustainability** - Water impact tracking and carbon-aware scheduling  
✅ **Simplicity** - Zero code changes required  

**One line of code. Complete visibility. Sustainable AI.**

```python
with GreenTensor() as gt:
    train()  # That's it!
```

---

**Generated:** 2026-05-09  
**Package Version:** 0.7.1  
**Analysis Type:** Complete EDA
