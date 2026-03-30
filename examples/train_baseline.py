"""
Baseline training run — no GreenTensor optimizations.
Run this first to capture a RunMetrics baseline, then compare with train_optimized.py.
"""
import torch
import time
import pickle
from greentensor.core.tracker import Tracker
from greentensor.report.metrics import RunMetrics

def train():
    x = torch.randn(10000, 1000)
    for _ in range(50):
        y = x @ x.T

if __name__ == "__main__":
    tracker = Tracker()
    tracker.start()
    t0 = time.perf_counter()

    train()

    duration = time.perf_counter() - t0
    emissions_kg, energy_kwh = tracker.stop()

    metrics = RunMetrics(duration_s=duration, energy_kwh=energy_kwh, emissions_kg=emissions_kg)
    print(f"Baseline — duration: {duration:.2f}s | energy: {energy_kwh:.6f} kWh | emissions: {emissions_kg:.6f} kg CO2")

    # Save for comparison in train_optimized.py
    with open("baseline_metrics.pkl", "wb") as f:
        pickle.dump(metrics, f)
    print("Baseline metrics saved to baseline_metrics.pkl")
