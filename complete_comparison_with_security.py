"""
COMPLETE COMPARISON: WITH vs WITHOUT GreenTensor
=================================================
Shows ALL features including SECURITY monitoring
"""

import numpy as np
import time
import os

# Set environment variable to avoid OpenMP conflict
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Import GreenTensor
from greentensor import GreenTensor

print("="*70)
print("GREENTENSOR COMPLETE COMPARISON - WITH SECURITY")
print("="*70)

# Simulate a training workload
def simulate_training(iterations=50):
    """Simulate ML training with matrix operations"""
    X = np.random.randn(1000, 500)
    W = np.random.randn(500, 100)
    
    for i in range(iterations):
        # Forward pass
        hidden = np.dot(X, W)
        output = 1 / (1 + np.exp(-hidden))
        
        # Backward pass
        grad = output * (1 - output)
        W_grad = np.dot(X.T, grad)
        
        # Update weights
        W -= 0.01 * W_grad
        
        # Simulate data loading delay
        if i % 10 == 0:
            time.sleep(0.05)

print("\n📊 Setup: 1000 samples, 500 features, 50 iterations")

# ============================================================================
# PART 1: WITHOUT GREENTENSOR
# ============================================================================
print("\n" + "="*70)
print("🔴 TRAINING WITHOUT GREENTENSOR")
print("="*70)

start_time = time.time()
simulate_training(iterations=50)
training_time_without = time.time() - start_time

print(f"\n⏱️  Training Time: {training_time_without:.2f} seconds")
print("❌ Energy Consumption: UNKNOWN")
print("❌ Carbon Emissions: UNKNOWN")
print("❌ Water Usage: UNKNOWN")
print("❌ Idle Time: UNKNOWN")
print("❌ GPU Efficiency: UNKNOWN")
print("\n🔒 SECURITY:")
print("   ❌ Anomaly Detection: DISABLED")
print("   ❌ Digital Footprint Scanning: DISABLED")
print("   ❌ Prompt Injection Detection: DISABLED")
print("   ❌ Secrets Scanning: DISABLED")
print("   ❌ PII Detection: DISABLED")
print("   ❌ Model Integrity Verification: DISABLED")
print("   ❌ Network Monitoring: DISABLED")
print("\n❌ Optimization Recommendations: NONE")

# ============================================================================
# PART 2: WITH GREENTENSOR - FULL SECURITY ENABLED
# ============================================================================
print("\n" + "="*70)
print("🟢 TRAINING WITH GREENTENSOR - SECURITY ENABLED")
print("="*70)
print("\n⏳ Initializing security features (this may take a moment)...")
print("   • Loading LLM Guard scanners...")
print("   • Initializing anomaly detection...")
print("   • Starting digital footprint scanner...")

# Use GreenTensor with FULL security features
with GreenTensor(
    model_name="Secure_Training_Demo",
    verbose=False,
    security=True,  # Enable security
    scan_dependencies=True,  # Scan dependencies for vulnerabilities
    monitor_network=True,  # Monitor network activity
    monitor_processes=True,  # Monitor processes
    stage="production"  # Production stage for stricter security
) as gt:
    print("✅ Security features initialized!\n")
    simulate_training(iterations=50)

# Get metrics
metrics = gt.metrics
security_alerts = gt.security_alerts
footprint_report = gt.footprint_report

print(f"\n⏱️  Training Time: {metrics.duration_s:.2f} seconds")
print(f"✅ Energy Consumption: {metrics.energy_kwh:.6f} kWh")
print(f"✅ Carbon Emissions: {metrics.emissions_kg:.6f} kg CO2")
print(f"✅ Idle Time Detected: {metrics.idle_seconds:.2f} seconds")

print("\n🔒 SECURITY MONITORING RESULTS:")
print(f"   ✅ Total Security Alerts: {len(security_alerts)}")
print(f"   ✅ Anomaly Detection: ACTIVE")
print(f"   ✅ Digital Footprint Scanning: ACTIVE")
print(f"   ✅ Prompt Injection Detection: ACTIVE")
print(f"   ✅ Secrets Scanning: ACTIVE")
print(f"   ✅ PII Detection: ACTIVE")

if footprint_report:
    print(f"\n📊 Digital Footprint Report:")
    print(f"   • Total Events: {len(footprint_report.events)}")
    print(f"   • Critical Events: {footprint_report.critical_count}")
    print(f"   • High Priority Events: {footprint_report.high_count}")
    print(f"   • Network Connections: {len(footprint_report.network_connections)}")
    print(f"   • Child Processes: {len(footprint_report.child_processes)}")
    print(f"   • Model Hashes Tracked: {len(footprint_report.model_hashes)}")
    print(f"   • Status: {'✅ CLEAN' if footprint_report.is_clean else '⚠️ THREATS DETECTED'}")

# Get recommendations
print(f"\n💡 Optimization Recommendations:")
recommendations = gt.recommend()

# ============================================================================
# DETAILED COMPARISON TABLE
# ============================================================================
print("\n" + "="*70)
print("📊 DETAILED COMPARISON: THE GREENTENSOR DIFFERENCE")
print("="*70)

print("\n┌─────────────────────────────────────────────────────────────────┐")
print("│ FEATURE                WITHOUT GT          WITH GT              │")
print("├─────────────────────────────────────────────────────────────────┤")
print(f"│ Training Time          {training_time_without:>8.2f}s          {metrics.duration_s:>8.2f}s              │")
print(f"│ Energy Tracked         {'UNKNOWN':>15}  {metrics.energy_kwh:>10.6f} kWh     │")
print(f"│ Carbon Tracked         {'UNKNOWN':>15}  {metrics.emissions_kg:>10.6f} kg CO2  │")
print(f"│ Idle Time              {'UNKNOWN':>15}  {metrics.idle_seconds:>10.2f}s        │")
print("├─────────────────────────────────────────────────────────────────┤")
print("│ SECURITY FEATURES                                               │")
print("├─────────────────────────────────────────────────────────────────┤")
print(f"│ Anomaly Detection      {'NO':>15}  {'YES':>15}              │")
print(f"│ Digital Footprint      {'NO':>15}  {'YES':>15}              │")
print(f"│ Prompt Injection       {'NO':>15}  {'YES':>15}              │")
print(f"│ Secrets Scanning       {'NO':>15}  {'YES':>15}              │")
print(f"│ PII Detection          {'NO':>15}  {'YES':>15}              │")
print(f"│ Network Monitoring     {'NO':>15}  {'YES':>15}              │")
print(f"│ Process Monitoring     {'NO':>15}  {'YES':>15}              │")
print(f"│ Security Alerts        {'0':>15}  {len(security_alerts):>3} detected        │")
print("├─────────────────────────────────────────────────────────────────┤")
print(f"│ Recommendations        {'NO':>15}  {'YES':>15}              │")
print("└─────────────────────────────────────────────────────────────────┘")

print("\n🎯 KEY TAKEAWAYS:")
print("\n   WITHOUT GreenTensor:")
print("   • Train completely blind - no metrics")
print("   • Zero security monitoring")
print("   • No anomaly detection")
print("   • No optimization insights")
print("   • Vulnerable to threats")

print("\n   WITH GreenTensor:")
print(f"   • Complete visibility: {metrics.energy_kwh:.6f} kWh, {metrics.emissions_kg:.6f} kg CO2")
print(f"   • Full security suite: {len(security_alerts)} threats monitored")
print(f"   • Anomaly detection: ACTIVE")
print(f"   • Digital footprint: {len(footprint_report.events) if footprint_report else 0} events tracked")
print(f"   • Optimization: Actionable recommendations")
print(f"   • Efficiency: {metrics.idle_seconds:.2f}s idle time detected")

print("\n💰 BUSINESS VALUE:")
print("   ✓ ESG Compliance: Track carbon & energy automatically")
print("   ✓ Security: Detect threats, anomalies, and vulnerabilities")
print("   ✓ Cost Savings: Identify inefficiencies and optimize")
print("   ✓ Risk Management: Monitor digital footprint")
print("   ✓ Audit Trail: Complete tracking for compliance")

print("\n🔒 SECURITY VALUE:")
print("   ✓ Prompt Injection Protection: Detect malicious inputs")
print("   ✓ Secrets Detection: Find exposed credentials")
print("   ✓ PII Protection: Identify sensitive data leaks")
print("   ✓ Anomaly Detection: Catch unusual behavior")
print("   ✓ Dependency Scanning: Check for vulnerabilities")
print("   ✓ Network Monitoring: Track suspicious connections")

print("\n" + "="*70)
print("✅ THIS IS THE GREENTENSOR DIFFERENCE!")
print("   Complete visibility + Security + Optimization")
print("="*70)
