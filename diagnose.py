import sys

print("Starting tests...")
tests = []

# Test 1
print("Testing torch...")
try:
    import torch
    tests.append(("torch", True, torch.__version__))
    print("torch imported successfully")
except Exception as e:
    tests.append(("torch", False, str(e)))
    print(f"torch failed: {e}")

# Test 2
print("Testing transformers...")
try:
    from transformers import AutoModelForCausalLM
    tests.append(("transformers", True, "OK"))
    print("transformers imported successfully")
except Exception as e:
    tests.append(("transformers", False, str(e)))
    print(f"transformers failed: {e}")

# Test 3
print("Testing datasets...")
try:
    from datasets import load_dataset
    tests.append(("datasets", True, "OK"))
    print("datasets imported successfully")
except Exception as e:
    tests.append(("datasets", False, str(e)))
    print(f"datasets failed: {e}")

# Test 4
print("Testing greentensor...")
try:
    from greentensor import GreenTensor
    tests.append(("greentensor", True, "OK"))
    print("greentensor imported successfully")
except Exception as e:
    tests.append(("greentensor", False, str(e)))
    print(f"greentensor failed: {e}")

# Print results
print("\n=== RESULTS ===")
for name, success, info in tests:
    status = "PASS" if success else "FAIL"
    print(f"{name}: {status} - {info}")

# Check if all passed
all_passed = all(t[1] for t in tests)
print(f"\nAll tests passed: {all_passed}")

if all_passed and tests[0][1]:  # If torch loaded
    import torch
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

print("Exiting...")
sys.exit(0 if all_passed else 1)
