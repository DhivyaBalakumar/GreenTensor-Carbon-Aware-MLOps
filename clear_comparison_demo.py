"""
Clear Comparison: Training WITH vs WITHOUT GreenTensor
========================================================
This script shows the exact difference in metrics when using GreenTensor
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time

# Import GreenTensor
from greentensor import GreenTensor

print("="*70)
print("GREENTENSOR COMPARISON DEMO")
print("="*70)

# Create a simple dataset (MNIST-like)
print("\n📊 Creating dataset...")
X_train = torch.randn(1000, 784)  # 1000 samples, 784 features
y_train = torch.randint(0, 10, (1000,))  # 10 classes
dataset = TensorDataset(X_train, y_train)

# Simple Neural Network
class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

print("✅ Dataset ready: 1000 samples, 10 classes")
print("✅ Model: 3-layer Neural Network")

# ============================================================================
# PART 1: WITHOUT GREENTENSOR
# ============================================================================
print("\n" + "="*70)
print("🔴 TRAINING WITHOUT GREENTENSOR")
print("="*70)

model_without = SimpleNN()
dataloader_without = DataLoader(dataset, batch_size=32, shuffle=True)
criterion = nn.CrossEntropyLoss()
optimizer_without = optim.Adam(model_without.parameters(), lr=0.001)

start_time = time.time()
for epoch in range(5):
    for batch_x, batch_y in dataloader_without:
        optimizer_without.zero_grad()
        outputs = model_without(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer_without.step()

training_time_without = time.time() - start_time

print(f"\n⏱️  Training Time: {training_time_without:.2f} seconds")
print("❌ Energy Consumption: UNKNOWN")
print("❌ Carbon Emissions: UNKNOWN")
print("❌ Water Usage: UNKNOWN")
print("❌ GPU Efficiency: UNKNOWN")
print("❌ Security Monitoring: NONE")
print("❌ Optimization Recommendations: NONE")

# ============================================================================
# PART 2: WITH GREENTENSOR
# ============================================================================
print("\n" + "="*70)
print("🟢 TRAINING WITH GREENTENSOR")
print("="*70)

model_with = SimpleNN()
dataloader_with = DataLoader(dataset, batch_size=32, shuffle=True)
optimizer_with = optim.Adam(model_with.parameters(), lr=0.001)

# Use GreenTensor context manager
with GreenTensor(
    model_name="SimpleNN_Demo",
    verbose=False,  # We'll print our own summary
    security=True,
    scan_dependencies=False,  # Skip for demo speed
    monitor_network=False,
    monitor_processes=False
) as gt:
    for epoch in range(5):
        for batch_x, batch_y in dataloader_with:
            optimizer_with.zero_grad()
            outputs = model_with(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer_with.step()

# Get metrics from GreenTensor
metrics = gt.metrics
security_alerts = gt.security_alerts

print(f"\n⏱️  Training Time: {metrics.duration_s:.2f} seconds")
print(f"✅ Energy Consumption: {metrics.energy_kwh:.6f} kWh")
print(f"✅ Carbon Emissions: {metrics.emissions_kg:.6f} kg CO2")
print(f"✅ Idle Time Detected: {metrics.idle_seconds:.2f} seconds")
print(f"✅ Security Alerts: {len(security_alerts)} detected")

# Calculate water usage (if available)
try:
    from greentensor.water.aquatensor import AquaTensorBridge, AquaTensorConfig
    water_config = AquaTensorConfig(
        provider="aws",
        region="us-east-1"
    )
    metrics_with_water = metrics.apply_aquatensor_config(water_config)
    if metrics_with_water.water:
        print(f"✅ Water Consumption: {metrics_with_water.water.consumption_liters:.4f} liters")
        print(f"✅ Water Production: {metrics_with_water.water.production_liters:.4f} liters")
except Exception as e:
    print(f"⚠️  Water metrics: Not available ({e})")

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
print("   Without GreenTensor: You train blindly with no visibility")
print("   With GreenTensor: You get complete environmental & performance insights")

print("\n💰 BUSINESS VALUE:")
print(f"   ✓ Track and reduce energy costs")
print(f"   ✓ Meet ESG compliance requirements")
print(f"   ✓ Optimize GPU utilization")
print(f"   ✓ Detect security anomalies")
print(f"   ✓ Get actionable recommendations")
print(f"   ✓ Monitor water consumption")

print("\n📈 WHAT YOU GET WITH GREENTENSOR:")
print(f"   • Energy: {metrics.energy_kwh:.6f} kWh measured")
print(f"   • Carbon: {metrics.emissions_kg:.6f} kg CO2 tracked")
print(f"   • Efficiency: {metrics.idle_seconds:.2f}s idle time detected")
print(f"   • Security: {len(security_alerts)} security events monitored")
print(f"   • Insights: Actionable recommendations provided")

print("\n" + "="*70)
print("Demo complete! This is the GreenTensor difference.")
print("="*70)
