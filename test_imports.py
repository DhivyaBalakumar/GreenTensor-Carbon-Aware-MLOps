import sys
print("Python version:", sys.version)
print("\nTesting imports...")

try:
    import torch
    print("✓ torch:", torch.__version__)
except Exception as e:
    print("✗ torch:", e)
    sys.exit(1)

try:
    from transformers import AutoModelForCausalLM
    print("✓ transformers")
except Exception as e:
    print("✗ transformers:", e)
    sys.exit(1)

try:
    from datasets import load_dataset
    print("✓ datasets")
except Exception as e:
    print("✗ datasets:", e)
    print("Full error:", type(e).__name__, str(e))
    sys.exit(1)

try:
    from greentensor import GreenTensor
    print("✓ greentensor")
except Exception as e:
    print("✗ greentensor:", e)
    sys.exit(1)

print("\n✅ All imports successful!")
print("\nDevice check:")
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("Running on CPU")
