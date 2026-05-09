import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

try:
    print("Starting imports...")

    import torch
    print("✓ torch imported")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    print("✓ transformers imported")

    from datasets import load_dataset
    print("✓ datasets imported")

    from greentensor import GreenTensor
    print("✓ greentensor imported")

    print("\nAll imports successful!")

    # Check device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        
except Exception as e:
    print(f"\nERROR: {e}")
    import traceback
    traceback.print_exc()
