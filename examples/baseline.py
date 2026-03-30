"""
GPU baseline — raw matrix ops, no optimizations.
"""
import torch
import time

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn(5000, 5000).to(device)
    for _ in range(20):
        y = torch.mm(x, x)

if __name__ == "__main__":
    start = time.perf_counter()
    train()
    print(f"Baseline runtime: {time.perf_counter() - start:.2f}s")
