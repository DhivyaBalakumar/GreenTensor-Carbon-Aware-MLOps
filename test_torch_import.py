"""Quick test to verify torch imports correctly"""
import sys

print("Python:", sys.version)
print("\nImporting torch...")

try:
    import torch
    print(f"SUCCESS: torch {torch.__version__} imported")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    else:
        print("Running on CPU only")
        
    # Test basic tensor operation
    x = torch.tensor([1.0, 2.0, 3.0])
    print(f"\nTest tensor: {x}")
    print("Torch is working correctly!")
    
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
