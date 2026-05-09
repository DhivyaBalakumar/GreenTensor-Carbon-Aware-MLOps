#!/usr/bin/env python
"""Simple import test without early exits"""
import sys
import os

# Set UTF-8 encoding for Windows console
if os.name == 'nt':
    os.system('chcp 65001 > nul')

print("Python version:", sys.version)
print("\nTesting imports...\n")

# Test 1: torch
try:
    import torch
    print(f"[OK] torch: {torch.__version__}")
    torch_ok = True
except Exception as e:
    print(f"[FAIL] torch: {e}")
    torch_ok = False

# Test 2: transformers
try:
    from transformers import AutoModelForCausalLM
    print("[OK] transformers")
    transformers_ok = True
except Exception as e:
    print(f"[FAIL] transformers: {e}")
    transformers_ok = False

# Test 3: datasets
try:
    from datasets import load_dataset
    print("[OK] datasets")
    datasets_ok = True
except Exception as e:
    print(f"[FAIL] datasets: {e}")
    datasets_ok = False

# Test 4: greentensor
try:
    from greentensor import GreenTensor
    print("[OK] greentensor")
    greentensor_ok = True
except Exception as e:
    print(f"[FAIL] greentensor: {e}")
    greentensor_ok = False

# Summary
print("\n" + "="*50)
if all([torch_ok, transformers_ok, datasets_ok, greentensor_ok]):
    print("SUCCESS: All imports successful!")
    
    if torch_ok:
        print("\nDevice check:")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("Running on CPU")
    sys.exit(0)
else:
    print("ERROR: Some imports failed")
    sys.exit(1)
