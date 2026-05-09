"""
Basic GreenTensor test with a small model
"""
import warnings
warnings.filterwarnings('ignore')

import torch
from greentensor import GreenTensor
import time

print("="*80)
print("GREENTENSOR BASIC TEST")
print("="*80)

# Check device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nDevice: {device}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Test 1: Baseline
print("\n" + "="*80)
print("TEST 1: BASELINE (No GreenTensor)")
print("="*80)

if device == "cuda":
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

start_time = time.time()

# Simple tensor operations
x = torch.randn(1000, 1000, device=device)
for i in range(100):
    y = torch.matmul(x, x)
    if (i + 1) % 20 == 0:
        print(f"Step {i+1}/100")

baseline_time = time.time() - start_time

if device == "cuda":
    baseline_memory = torch.cuda.max_memory_allocated() / 1e6
else:
    import psutil
    import os
    process = psutil.Process(os.getpid())
    baseline_memory = process.memory_info().rss / 1e6

print(f"\nBaseline Time: {baseline_time:.2f}s")
print(f"Baseline Memory: {baseline_memory:.2f} MB")

# Test 2: With GreenTensor
print("\n" + "="*80)
print("TEST 2: WITH GREENTENSOR")
print("="*80)

if device == "cuda":
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

gt = GreenTensor(
    energy_budget=100.0,
    enable_optimizations=True,
    track_water=True,
    track_carbon=True
)

start_time = time.time()

with gt.track():
    x = torch.randn(1000, 1000, device=device)
    for i in range(100):
        y = torch.matmul(x, x)
        if (i + 1) % 20 == 0:
            print(f"Step {i+1}/100")

greentensor_time = time.time() - start_time

if device == "cuda":
    greentensor_memory = torch.cuda.max_memory_allocated() / 1e6
else:
    process = psutil.Process(os.getpid())
    greentensor_memory = process.memory_info().rss / 1e6

gt_metrics = gt.get_metrics()

print(f"\nGreenTensor Time: {greentensor_time:.2f}s")
print(f"GreenTensor Memory: {greentensor_memory:.2f} MB")
print(f"Energy Consumed: {gt_metrics.get('energy_consumed', 0):.4f} Wh")
print(f"Carbon Emissions: {gt_metrics.get('carbon_emissions', 0):.4f} gCO2")
print(f"Water Usage: {gt_metrics.get('water_usage', 0):.4f} L")

# Comparison
print("\n" + "="*80)
print("COMPARISON")
print("="*80)
time_diff = ((greentensor_time - baseline_time) / baseline_time) * 100
memory_diff = ((greentensor_memory - baseline_memory) / baseline_memory) * 100

print(f"Time difference: {time_diff:+.2f}%")
print(f"Memory difference: {memory_diff:+.2f}%")
print(f"\n✓ GreenTensor tracked {gt_metrics.get('energy_consumed', 0):.4f} Wh energy")
print(f"✓ GreenTensor tracked {gt_metrics.get('carbon_emissions', 0):.4f} gCO2 emissions")
print(f"✓ GreenTensor tracked {gt_metrics.get('water_usage', 0):.4f} L water")
print("="*80)

print("\n✓ Test completed successfully!")
