# GreenTensor Code-Based EDA

## Overview
This is a **complete, executable Python script** that explores **ALL capabilities** of the GreenTensor package through code examples. No theory - just working code.

## What's Included

### 20 Comprehensive Sections:

1. **Basic Energy & Carbon Tracking** - Core functionality
2. **Configuration Options** - Custom configs
3. **Datacenter Configuration & PUE** - Infrastructure overhead
4. **Water Intelligence (AquaTensor)** - Water consumption & production
5. **GPU Optimization** - Mixed precision, batch size, idle detection
6. **Efficiency Recommendations** - Actionable optimization suggestions
7. **Carbon Budget Enforcement** - Hard limits on emissions
8. **Run History & Persistence** - Track metrics across sessions
9. **ESG Reporting** - Compliance reports (GHG Protocol, SEC, EU CSRD)
10. **Carbon-Aware Scheduling** - Optimize for grid carbon intensity
11. **Security - Anomaly Detection** - Power-based threat detection
12. **Security - Digital Footprint Scanner** - Multi-signal attack detection
13. **Security - Pattern Matching** - Correlate carbon spikes with threats
14. **Alerting & Webhooks** - Slack, PagerDuty, custom webhooks
15. **Profiler & Metrics** - GPU metrics collection
16. **Calculate Savings** - Compare baseline vs optimized
17. **Framework Integrations** - HuggingFace, PyTorch Lightning
18. **Advanced Features** - Decorators, manual tracking
19. **Complete Workflow Example** - Full production pipeline
20. **Summary** - Recap of all capabilities

## How to Run

```bash
# Install GreenTensor
pip install greentensor

# Run the EDA
python GreenTensor_Code_EDA.py
```

## What You'll See

- **Live code execution** for every feature
- **Real metrics** from actual workloads
- **Security scanning** results
- **Water impact** calculations
- **ESG reports** generation
- **Optimization recommendations**
- **Carbon scheduling** decisions

## Output

The script will:
- ✅ Execute 100+ code examples
- ✅ Generate test files (metrics, reports, models)
- ✅ Print results for every capability
- ✅ Demonstrate real-world workflows

## Files Created

- `test_metrics.pkl` - Saved metrics
- `test_history.json` - Run history
- `test_esg_history.json` - ESG tracking
- `test_esg_report.json` - ESG report
- `test_model.pkl` - Test model for integrity checks
- `production_esg.json` - Production ESG data

## Key Features Demonstrated

### Energy & Sustainability
- Real-time energy tracking
- Carbon emissions calculation
- Water consumption & production
- Datacenter PUE overhead
- Carbon-aware scheduling

### Security
- Anomaly detection (alibi-detect)
- LLM Guard scanning
- Digital footprint monitoring
- Model integrity verification
- Dependency scanning
- Pattern matching

### Optimization
- Mixed precision (FP16)
- Batch size optimization
- Idle GPU detection
- Efficiency recommendations

### Compliance
- ESG report generation
- GHG Protocol Scope 2
- SEC Climate Disclosure
- EU CSRD alignment

### Integration
- HuggingFace Transformers
- PyTorch Lightning
- Custom callbacks
- Decorator patterns

## Requirements

```
numpy
greentensor
```

Optional for full security features:
```
alibi-detect
llm-guard
```

## Duration

The complete EDA takes approximately **2-3 minutes** to run all examples.

## Notes

- All examples use simulated workloads (numpy operations)
- Security features may require additional dependencies
- Some features (like electricityMap API) may fall back to static data
- Test files are created in the current directory

## Cleanup

To remove test files after running:

```python
import os
files = [
    "test_metrics.pkl",
    "test_history.json",
    "test_esg_history.json",
    "test_esg_report.json",
    "test_model.pkl",
    "production_esg.json"
]
for f in files:
    os.remove(f) if os.path.exists(f) else None
```

## Author

Dhivya Balakumar  
GreenTensor v0.7.1

## License

MIT
