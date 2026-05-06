# GreenTensor

**Carbon-Secure MLOps + AquaTensor Water Intelligence**

GreenTensor is the first middleware platform that simultaneously reduces AI compute costs, measures carbon and water impact, detects ML-specific cyberattacks, generates regulatory-compliant ESG reports, and schedules jobs for the cleanest grid window — all from a single Python wrapper around your existing training code.

Built by Dhivya Balakumar | v0.7.1 | MIT License

PyPI: https://pypi.org/project/greentensor | GitHub: https://github.com/DhivyaBalakumar/GreenTensor-Carbon-Aware-MLOps

---

## The Problem

AI systems are not sustainable. Training GPT-4 consumed ~700,000 liters of fresh water and emitted ~500 tonnes of CO2. Global AI compute demand doubles every 6 months. By 2030, AI is projected to consume 3-4% of global electricity.

Companies face three simultaneous pressures:
- **Cost**: GPU compute bills growing faster than revenue
- **Compliance**: SEC climate disclosure and EU CSRD require Scope 2 emissions reporting from AI workloads
- **Security**: ML-specific attacks (cryptominer injection, model theft, backdoor triggers) are growing and undetected by existing tools

No single tool addresses all three. GreenTensor does.

---

## Install

```bash
pip install greentensor

# With full security stack (alibi-detect + LLM Guard)
pip install greentensor[security]
```

---

## Quickstart

```python
from greentensor import GreenTensor

with GreenTensor() as gt:
    with gt.mixed_precision():
        train()   # your existing code, unchanged
```

That single wrapper gives you everything below.

---

## Sample Report Output

```
  +======================================+
  |        GreenTensor Report  v0.6.0    |
  +======================================+
  Runtime          : 45.23 s
  Energy Used      : 0.000412 kWh
  CO2 Emissions    : 0.000096 kg
  Idle Time        : 2.10 s

  -- Efficiency vs Baseline -----------
  Baseline Energy  : 0.000580 kWh
  Energy Saved     : 0.000168 kWh  (29.0% reduction)
  Emissions Saved  : 0.000039 kg CO2
  Time Saved       : 12.40 s

  -- AquaTensor Water Intelligence ----
  Water Consumed   : 0.000202 L  (cooling, WUE=0.49)
  Heat Recovered   : 0.000268 kWh  (WHR=65%)
  Water Produced   : 0.001474 L  (membrane distillation @ 60C)
  Net Water Impact : -0.001272 L  (NET POSITIVE)
  Drinking Water   : 0.0 person-days of fresh water generated
  Region Stress    : Extremely High (index 4.5/5.0)

  -- Carbon Anomaly Detection ---------
  Status           : CLEAN

  -- Digital Footprint Report ---------
  Status           : CLEAN -- No digital attack footprint detected
  ======================================
```

---

## Features

### Energy and Carbon Tracking
- Real energy measurement via CodeCarbon (hardware counters, not estimates)
- Falls back to nvidia-smi power sampling when CodeCarbon unavailable
- CO2 calculated from real energy x regional grid carbon intensity
- Idle GPU time tracking — surfaces data pipeline bottlenecks
- Auto-saves `RunMetrics` to `.pkl` after every run

### GPU Optimization
- Mixed precision (FP16) via `torch.cuda.amp.autocast()` — 20-40% energy reduction on Tensor Core GPUs
- cuDNN benchmark mode — finds optimal kernel for your input shapes, 10-30% speedup on CNN workloads
- Batch size optimizer — queries available GPU memory and recommends optimal batch size
- All optimizations are reversible — original settings restored on exit

### AquaTensor Water Intelligence
- Calculates water consumed by datacenter cooling per workload (using published WUE coefficients)
- Calculates water produced by AquaTensor membrane distillation from waste heat
- Interpolated MD yield by feed temperature (2.5-8.5 L/kWh, 40-80C range)
- Net water impact — shows whether a workload is net water positive
- Regional water stress context (WRI Aqueduct index)
- Heat forecasting for upcoming job queues
- Drinking water equivalency in person-days

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
print(f"Water produced: {water.water.water_produced_liters:.3f} L")
print(f"Net impact: {water.water.net_water_impact_liters:.3f} L")
```

### Datacenter PUE Impact
- Applies PUE (Power Usage Effectiveness) to scale GPU measurements to real DC energy
- Presets: local (1.0), hyperscale (1.1), cloud average (1.2), enterprise (1.5), legacy (1.8)
- Multi-node support for distributed training
- Regional carbon intensity lookup

### Carbon-Based Security Monitoring
- **alibi-detect SpectralResidual** — statistical frequency-domain anomaly detection on GPU power time series (same algorithm as Microsoft Azure Metrics Advisor)
- **Threshold detector** — fallback when alibi-detect unavailable
- **LLM Guard** — scans model inputs for prompt injection and secrets, outputs for data leakage
- Detects: power spikes (cryptominer), idle drain (data exfiltration), prompt injection, data leakage

### Digital Footprint Scanner
Multi-signal attack detection across the full ML lifecycle, tagged with MITRE ATLAS technique IDs:

**Pre-deployment:**
- Model weight tampering (SHA-256 hash verification) — MITRE AML.T0018
- Supply chain attacks (typosquatting, known malicious packages) — MITRE AML.T0010
- Process injection (unexpected child processes) — MITRE AML.T0011
- Network exfiltration (unexpected outbound connections) — MITRE AML.T0024

**Post-deployment:**
- Inference latency spikes (backdoor trigger activation) — MITRE AML.T0043
- Model extraction (high-frequency API probing) — MITRE AML.T0044
- Confidence anomalies (adversarial inputs)

```python
# Pre-deployment
with GreenTensor(stage="pre_deployment") as gt:
    gt.register_model("model.pt")
    train()
    gt.verify_model("model.pt")

# Post-deployment
with GreenTensor(stage="post_deployment") as gt:
    for request in production_requests:
        t0 = time.perf_counter()
        output = model(request)
        gt.record_inference(
            latency_s=time.perf_counter() - t0,
            confidence=output.softmax(-1).max().item()
        )
```

### Pattern Matching — Carbon Spike Correlation
When a carbon spike is detected, GreenTensor immediately runs pattern matching to determine if it is a cyberattack or a benign event:

```
Power spike detected (+140% above baseline)
  + new process spawned: xmrig
  + GPU util 98% during data loading phase
  + outbound connection to unknown IP
  = Threat score: 94/100 | Verdict: ATTACK | Type: cryptominer
  = Action: Terminate pipeline. Preserve logs for forensics.
```

### ESG Reporting
Automated Scope 2 emissions reports aligned with:
- GHG Protocol Scope 2 Guidance
- SEC Climate Disclosure Rules (17 CFR Parts 210, 229, 232, 239, 249)
- EU CSRD (Corporate Sustainability Reporting Directive)
- ISO 14064-1

```python
from greentensor import ESGReporter, ESGOrganization

reporter = ESGReporter(ESGOrganization(
    name="Acme Corp",
    reporting_period="FY2025",
    region="US-East",
))

# Records automatically after each run
reporter.record_run(gt.metrics, model_name="bert-finetuned", stage="training")

# Generate report
report = reporter.generate_report()
print(report.to_text())
report.save_json("scope2_emissions_FY2025.json")
```

Sample ESG report output:
```
  GreenTensor ESG Report
  Organization    : Acme Corp
  Reporting Period: FY2025
  Standard        : GHG Protocol Scope 2

  Scope 2 Emissions Summary
  Total ML Runs        : 47
  Total Compute Time   : 312.40 hours
  Total Energy (GPU)   : 1.2847 kWh
  Total Energy (DC)    : 1.5416 kWh
  Total CO2 (DC adj.)  : 0.3592 kg

  Avoided Emissions (GreenTensor savings)
  Energy Saved         : 0.4230 kWh
  CO2 Avoided          : 0.0985 kg
  Reduction vs baseline: 24.7%

  Real-World Equivalencies
  Equivalent km driven : 512.5 km
  Trees needed (1 yr)  : 0.02 trees
  NYC-LA flights       : 0.001 flights
```

### Framework Integrations

**HuggingFace Transformers:**
```python
from transformers import Trainer
from greentensor.integrations.huggingface import GreenTensorCallback

trainer = Trainer(
    model=model,
    args=training_args,
    callbacks=[GreenTensorCallback(model_name="bert-finetuned")],
)
trainer.train()
```

**PyTorch Lightning:**
```python
from pytorch_lightning import Trainer
from greentensor.integrations.lightning import GreenTensorCallback

trainer = Trainer(
    max_epochs=10,
    callbacks=[GreenTensorCallback(model_name="my_model")],
)
trainer.fit(model)
```

### Carbon-Aware Scheduling

Check real-time grid carbon intensity before running a job. Same compute, less CO2 — just by choosing the right time.

```python
from greentensor import CarbonAwareScheduler

scheduler = CarbonAwareScheduler(zone="IN-NO")
rec = scheduler.should_run_now(estimated_energy_kwh=0.5)
# "Grid is carbon-heavy (680 gCO2/kWh). Wait 3.0h for clean window (10:00-14:00 UTC). Estimated 35% reduction."

# Block until grid is clean (up to 6 hours)
with scheduler.wait_for_clean_grid(max_wait_hours=6):
    train()
```

Supported zones: US-CAL-CISO, US-TEX-ERCO, GB, DE, FR, IN-NO, IN-SO, SG, AU-NSW, NO-NO1 and more. Uses electricityMap API when available, falls back to static regional averages with time-of-day variation.

### Carbon Budget Enforcement

```python
from greentensor import GreenTensor, CarbonBudget

with GreenTensor(
    carbon_budget=CarbonBudget(max_kg_per_run=0.005, warn_at_pct=80.0, job_name="bert-v3")
) as gt:
    train()
# Raises CarbonBudgetExceeded if job emits more than 0.005 kg CO2
```

### Efficiency Recommendations

```python
with GreenTensor() as gt:
    train()

gt.recommend(batch_size=32, mixed_precision_enabled=False, gpu_util_avg_pct=45.0)
# [HIGH] Mixed precision not enabled — 28% estimated savings
# [HIGH] GPU idle 50% — add num_workers=4 to DataLoader — 35% estimated savings
# [HIGH] Batch size too small — increase to 64+ — 15% estimated savings
```

### Webhook Alerting

```python
from greentensor import GreenTensor, SlackWebhook, PagerDutyAlert, MultiAlert

with GreenTensor(on_alert=MultiAlert(
    SlackWebhook("https://hooks.slack.com/services/YOUR/WEBHOOK"),
    PagerDutyAlert("your-pagerduty-key"),
)) as gt:
    train()
```

### Run History
Persists metrics across sessions for trend analysis:
```python
from greentensor.core.history import RunHistory

history = RunHistory()
history.record(gt.metrics, model_name="ResNet50", stage="training")
print(history.last(10))
```

---

## Dashboards

**Analysis dashboard** (energy, water, ESG, security, history):
```bash
streamlit run dashboard/app.py
```

**Live observability dashboard** (real-time carbon + security correlation):
```bash
streamlit run dashboard/observability.py
```

The observability dashboard shows carbon footprint and security signals side by side. When a spike is detected, the pattern matcher fires instantly and shows the verdict with threat score, evidence, and recommended action.

---

## Run Tests

```bash
pytest tests/ -v
```

58 tests across: config, metrics, report, profiler, batch optimizer, context manager, anomaly detector, digital footprint scanner, water intelligence, carbon scheduler, budget enforcement, efficiency recommender.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| ML framework | PyTorch (cuDNN, AMP) |
| Carbon tracking | CodeCarbon, nvidia-smi |
| Anomaly detection | alibi-detect SpectralResidual |
| LLM security | LLM Guard (Protect AI) |
| System monitoring | psutil, subprocess |
| Model integrity | hashlib SHA-256 |
| Water intelligence | Membrane distillation physics (Khayet & Matsuura) |
| Water stress data | WRI Aqueduct index |
| Carbon scheduling | electricityMap API + static regional fallback |
| Budget enforcement | CarbonBudget / CarbonBudgetExceeded |
| Efficiency recommendations | EfficiencyRecommender |
| Webhook alerting | Slack, PagerDuty, Generic HTTP, MultiAlert |
| Dashboard | Streamlit, Pandas |
| Packaging | PyPI, setuptools, GitHub Actions |
| Security taxonomy | MITRE ATLAS |
| ESG standards | GHG Protocol, SEC, EU CSRD, ISO 14064-1 |

---

## License

MIT License — Copyright (c) 2026 Dhivya Balakumar
