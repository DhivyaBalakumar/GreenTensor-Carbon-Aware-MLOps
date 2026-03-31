"""
Optimized training run — uses GreenTensor.
Loads baseline_metrics.pkl if available to show real savings.
"""
import torch
import pickle
import os
from greentensor.core.context import GreenTensor
from greentensor.report.metrics import RunMetrics

def train(gt: GreenTensor):
    x = torch.randn(10000, 1000)
    with gt.mixed_precision():
        for _ in range(50):
            y = x @ x.T

if __name__ == "__main__":
    baseline = None
    pkl_path = "baseline_metrics.pkl"
    if os.path.exists(pkl_path):
        with open(pkl_path, "rb") as f:
            baseline = pickle.load(f)
        print("Loaded baseline metrics for comparison.")

    with GreenTensor(baseline=baseline, save_path="optimized_metrics.pkl") as gt:
        train(gt)
