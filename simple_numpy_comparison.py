"""
SIMPLE COMPARISON: Training WITH vs WITHOUT GreenTensor
=========================================================
Using only NumPy - no PyTorch required
"""

import numpy as np
import time

# Import GreenTensor
from greentensor import GreenTensor

print("="*70)
print("GREENTENSOR COMPARISON DEMO (NumPy Version)")
print("="*70)

# Simulate a training workload with NumPy
def simulate_training(iterations=100):
    """Simulate ML training with matrix operations"""
    X = np.random.randn(1000, 500)
    W = np.random.randn(500, 100)
    
    for i in range(iterations):
        # Forward pass
        hidden = np.dot(X, W)
        output = 1 / (1 + np.exp(-hidden))  # sigmoid
        
        # Backward pass (gradient computation)
        grad = output * (1 - output)
        W_grad = np.dot(X.T, grad)
        
        # Update weights
        W -= 0.01 * W_grad
        
        # Add some idle time to simulate data loading
        if i % 10 == 0:
            time.sleep(0.1)

print("\n📊 Setup: 1000 samples, 500 features, 100 iterations")

# ============================================================================
# PART 1: WITHOUT GREENTENSOR
# ============================================================================
print("\n" + "="*70)
print("🔴 TRAINING WITHOUT GREENTENSOR")
print("="*70)

start_time = time.time()
simulate_training(iterations=100)
training_time_without = time.time() - start_time

print(f"\n⏱️  Training Time: {training_time_without:.2f} seconds")
print("❌ Energy Consumption: UNKNOWN")
print("❌ Carbon Emissions: UNKNOWN")
print("❌ Water Usage: UNKNOWN")
print("❌ Idle Time: UNKNOWN")
print("❌ Security Monitoring: NONE")
print("❌ Optimization Recommendations: NONE")

# ============================================================================
# PART 2: WITH GREENTENSOR
# ============================================================================
print("\n" + "="*70)
print("🟢 TRAINING WITH GREENTENSOR")
print("="*70)

# Use GreenTensor context manager
with GreenTensor(
    model_name="NumPy_Training_Demo",
    verbose=False,
    security=True,
    scan_dependencies=False,
    monitor_network=False,
    monitor_processes=False
) as gt:
    simulate_training(iterations=100)

# Get metrics
metrics = gt.metrics
security_alerts = gt.security_alerts

print(f"\n⏱️  Training Time: {metrics.duration_s:.2f} seconds")
print(f"✅ Energy Consumption: {metrics.energy_kwh:.6f} kWh")
print(f"✅ Carbon Emissions: {metrics.emissions_kg:.6f} kg CO2")
print(f"✅ Idle Time Detected: {metrics.idle_seconds:.2f} seconds")
print(f"✅ Security Alerts: {len(security_alerts)} detected")

# Get recommendations
print(f"\n💡 Optimization Recommendations:")
recommendations = gt.recommend()

# ============================================================================
# SUMMARY COMPARISON
# ============================================================================
print("\n" + "="*70)
print("📊 SUMMARY: THE GREENTENSOR DIFFERENCE")
print("="*70)

print("\n┌─────────────────────────────────────────────────────────────────┐")
print("│                    WITHOUT GT          WITH GT                  │")
print("├─────────────────────────────────────────────────────────────────┤")
print(f"│ Training Time      {training_time_without:>8.2f}s          {metrics.duration_s:>8.2f}s                │")
print(f"│ Energy Tracked     {'UNKNOWN':>15}  {metrics.energy_kwh:>10.6f} kWh       │")
print(f"│ Carbon Tracked     {'UNKNOWN':>15}  {metrics.emissions_kg:>10.6f} kg CO2    │")
print(f"│ Idle Time          {'UNKNOWN':>15}  {metrics.idle_seconds:>10.2f}s          │")
print(f"│ Security           {'NO':>15}  {len(security_alerts):>3} alerts detected  │")
print(f"│ Recommendations    {'NO':>15}  {'YES':>15}                │")
print("└─────────────────────────────────────────────────────────────────┘")

print("\n🎯 KEY TAKEAWAY:")
print("   WITHOUT GreenTensor: You train blindly with ZERO visibility")
print("   WITH GreenTensor: You get COMPLETE environmental insights")

print("\n💰 BUSINESS VALUE WITH GREENTENSOR:")
print(f"   ✓ Energy: {metrics.energy_kwh:.6f} kWh measured")
print(f"   ✓ Carbon: {metrics.emissions_kg:.6f} kg CO2 tracked")
print(f"   ✓ Efficiency: {metrics.idle_seconds:.2f}s idle time detected")
print(f"   ✓ Security: {len(security_alerts)} security events monitored")
print(f"   ✓ Insights: Actionable recommendations provided")

print("\n📈 WHAT YOU GET:")
print("   • Track and reduce energy costs")
print("   • Meet ESG compliance requirements")
print("   • Detect inefficiencies automatically")
print("   • Monitor security threats")
print("   • Get optimization recommendations")

print("\n" + "="*70)
print("✅ This is the GreenTensor difference - visibility vs blindness!")
print("="*70)
