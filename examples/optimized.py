"""
GPU optimized — same ops wrapped with GreenTensor.
"""
import torch
from greentensor.core.context import GreenTensor

def train(gt: GreenTensor):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn(5000, 5000).to(device)
    with gt.mixed_precision():
        for _ in range(20):
            y = torch.mm(x, x)

if __name__ == "__main__":
    with GreenTensor() as gt:
        train(gt)
