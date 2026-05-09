"""
OPT-1.3B Training Test: Baseline vs GreenTensor
Side-by-side training comparison with real metrics.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import time
import psutil
import os
from greentensor import GreenTensor
import json
import numpy as np

print("="*80)
print("OPT-1.3B TRAINING COMPARISON: BASELINE vs GREENTENSOR")
print("="*80)

# Check GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nUsing device: {device}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("WARNING: Running on CPU - this will be very slow!")

# Load model and tokenizer
print("\nLoading OPT-1.3B model...")
model_name = "facebook/opt-1.3b"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Set padding token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Model loaded successfully!")

# Prepare training data
print("\nLoading training dataset...")
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

# Take a small subset for training
train_texts = [text for text in dataset['text'][:500] if len(text.strip()) > 50]
print(f"Using {len(train_texts)} training samples")

# Training parameters
batch_size = 2
num_epochs = 1
learning_rate = 5e-5
max_length = 128
num_training_steps = 50  # Limit to 50 steps for comparison

print(f"\nTraining Configuration:")
print(f"  Batch size: {batch_size}")
print(f"  Learning rate: {learning_rate}")
print(f"  Max sequence length: {max_length}")
print(f"  Training steps: {num_training_steps}")

# Prepare data batches
def prepare_batch(texts, tokenizer, max_length, device):
    encodings = tokenizer(
        texts,
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors='pt'
    )
    input_ids = encodings['input_ids'].to(device)
    attention_mask = encodings['attention_mask'].to(device)
    return input_ids, attention_mask

# Create batches
batches = []
for i in range(0, min(len(train_texts), num_training_steps * batch_size), batch_size):
    batch_texts = train_texts[i:i+batch_size]
    if len(batch_texts) == batch_size:
        batches.append(batch_texts)

print(f"Created {len(batches)} batches for training")

# ============================================================================
# 1. BASELINE TRAINING (No GreenTensor)
# ============================================================================

print("\n" + "="*80)
print("BASELINE TRAINING - Pure PyTorch (No Optimization)")
print("="*80)

# Load fresh model for baseline
model_baseline = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
)
model_baseline = model_baseline.to(device)
model_baseline.train()

# Optimizer
optimizer_baseline = torch.optim.AdamW(model_baseline.parameters(), lr=learning_rate)

baseline_results = {
    "total_time": 0,
    "total_loss": 0,
    "num_steps": 0,
    "peak_memory_mb": 0,
    "losses": []
}

# Clear cache
if device == "cuda":
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

start_time = time.time()

for step, batch_texts in enumerate(batches[:num_training_steps]):
    input_ids, attention_mask = prepare_batch(batch_texts, tokenizer, max_length, device)
    
    # Forward pass
    outputs = model_baseline(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
    loss = outputs.loss
    
    # Backward pass
    optimizer_baseline.zero_grad()
    loss.backward()
    optimizer_baseline.step()
    
    baseline_results["total_loss"] += loss.item()
    baseline_results["losses"].append(loss.item())
    baseline_results["num_steps"] += 1
    
    if (step + 1) % 10 == 0:
        print(f"Step {step+1}/{num_training_steps} | Loss: {loss.item():.4f}")

end_time = time.time()
baseline_results["total_time"] = end_time - start_time
baseline_results["avg_loss"] = baseline_results["total_loss"] / baseline_results["num_steps"]

if device == "cuda":
    baseline_results["peak_memory_mb"] = torch.cuda.max_memory_allocated() / 1e6
    baseline_results["peak_memory_gb"] = baseline_results["peak_memory_mb"] / 1000
else:
    process = psutil.Process(os.getpid())
    baseline_results["peak_memory_mb"] = process.memory_info().rss / 1e6
    baseline_results["peak_memory_gb"] = baseline_results["peak_memory_mb"] / 1000

baseline_results["steps_per_second"] = baseline_results["num_steps"] / baseline_results["total_time"]

print("\n" + "="*80)
print("BASELINE TRAINING RESULTS")
print("="*80)
print(f"Total Time: {baseline_results['total_time']:.2f} seconds")
print(f"Total Steps: {baseline_results['num_steps']}")
print(f"Steps/Second: {baseline_results['steps_per_second']:.2f}")
print(f"Average Loss: {baseline_results['avg_loss']:.4f}")
print(f"Final Loss: {baseline_results['losses'][-1]:.4f}")
print(f"Peak Memory: {baseline_results['peak_memory_gb']:.2f} GB")
print("="*80)

# Clean up
del model_baseline, optimizer_baseline
if device == "cuda":
    torch.cuda.empty_cache()

# ============================================================================
# 2. GREENTENSOR TRAINING (With Optimization)
# ============================================================================

print("\n" + "="*80)
print("GREENTENSOR TRAINING - With Optimization")
print("="*80)

# Load fresh model for GreenTensor
model_greentensor = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
)
model_greentensor = model_greentensor.to(device)
model_greentensor.train()

# Optimizer
optimizer_greentensor = torch.optim.AdamW(model_greentensor.parameters(), lr=learning_rate)

# Clear cache
if device == "cuda":
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

# Initialize GreenTensor
gt = GreenTensor(
    energy_budget=1000.0,  # High budget for large model
    enable_optimizations=True,
    track_water=True,
    track_carbon=True
)

greentensor_results = {
    "total_time": 0,
    "total_loss": 0,
    "num_steps": 0,
    "peak_memory_mb": 0,
    "losses": []
}

start_time = time.time()

with gt.track():
    for step, batch_texts in enumerate(batches[:num_training_steps]):
        input_ids, attention_mask = prepare_batch(batch_texts, tokenizer, max_length, device)
        
        # Forward pass
        outputs = model_greentensor(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = outputs.loss
        
        # Backward pass
        optimizer_greentensor.zero_grad()
        loss.backward()
        optimizer_greentensor.step()
        
        greentensor_results["total_loss"] += loss.item()
        greentensor_results["losses"].append(loss.item())
        greentensor_results["num_steps"] += 1
        
        if (step + 1) % 10 == 0:
            print(f"Step {step+1}/{num_training_steps} | Loss: {loss.item():.4f}")

end_time = time.time()
greentensor_results["total_time"] = end_time - start_time
greentensor_results["avg_loss"] = greentensor_results["total_loss"] / greentensor_results["num_steps"]

if device == "cuda":
    greentensor_results["peak_memory_mb"] = torch.cuda.max_memory_allocated() / 1e6
    greentensor_results["peak_memory_gb"] = greentensor_results["peak_memory_mb"] / 1000
else:
    process = psutil.Process(os.getpid())
    greentensor_results["peak_memory_mb"] = process.memory_info().rss / 1e6
    greentensor_results["peak_memory_gb"] = greentensor_results["peak_memory_mb"] / 1000

greentensor_results["steps_per_second"] = greentensor_results["num_steps"] / greentensor_results["total_time"]

# Get GreenTensor metrics
gt_metrics = gt.get_metrics()

print("\n" + "="*80)
print("GREENTENSOR TRAINING RESULTS")
print("="*80)
print(f"Total Time: {greentensor_results['total_time']:.2f} seconds")
print(f"Total Steps: {greentensor_results['num_steps']}")
print(f"Steps/Second: {greentensor_results['steps_per_second']:.2f}")
print(f"Average Loss: {greentensor_results['avg_loss']:.4f}")
print(f"Final Loss: {greentensor_results['losses'][-1]:.4f}")
print(f"Peak Memory: {greentensor_results['peak_memory_gb']:.2f} GB")
print("\n--- GreenTensor Specific Metrics ---")
print(f"Energy Consumed: {gt_metrics.get('energy_consumed', 0):.4f} Wh")
print(f"Carbon Emissions: {gt_metrics.get('carbon_emissions', 0):.4f} gCO2")
print(f"Water Usage: {gt_metrics.get('water_usage', 0):.4f} L")
print(f"Energy Budget Remaining: {gt_metrics.get('energy_budget_remaining', 0):.2f} Wh")
print(f"Optimizations Applied: {gt_metrics.get('optimizations_applied', 0)}")
print("="*80)

# Clean up
del model_greentensor, optimizer_greentensor
if device == "cuda":
    torch.cuda.empty_cache()

# ============================================================================
# 3. SIDE-BY-SIDE COMPARISON
# ============================================================================

print("\n" + "="*80)
print("SIDE-BY-SIDE COMPARISON: BASELINE vs GREENTENSOR")
print("="*80)
print(f"\n{'Metric':<30} {'Baseline':<20} {'GreenTensor':<20} {'Difference':<20}")
print("-"*90)

# Time comparison
time_diff = ((greentensor_results['total_time'] - baseline_results['total_time']) / baseline_results['total_time']) * 100
print(f"{'Total Time (s)':<30} {baseline_results['total_time']:<20.2f} {greentensor_results['total_time']:<20.2f} {time_diff:+.2f}%")

# Steps comparison
print(f"{'Total Steps':<30} {baseline_results['num_steps']:<20} {greentensor_results['num_steps']:<20} {'Same'}")

# Throughput comparison
throughput_diff = ((greentensor_results['steps_per_second'] - baseline_results['steps_per_second']) / baseline_results['steps_per_second']) * 100
print(f"{'Steps/Second':<30} {baseline_results['steps_per_second']:<20.2f} {greentensor_results['steps_per_second']:<20.2f} {throughput_diff:+.2f}%")

# Loss comparison
loss_diff = ((greentensor_results['avg_loss'] - baseline_results['avg_loss']) / baseline_results['avg_loss']) * 100
print(f"{'Average Loss':<30} {baseline_results['avg_loss']:<20.4f} {greentensor_results['avg_loss']:<20.4f} {loss_diff:+.2f}%")

final_loss_diff = ((greentensor_results['losses'][-1] - baseline_results['losses'][-1]) / baseline_results['losses'][-1]) * 100
print(f"{'Final Loss':<30} {baseline_results['losses'][-1]:<20.4f} {greentensor_results['losses'][-1]:<20.4f} {final_loss_diff:+.2f}%")

# Memory comparison
memory_diff = ((greentensor_results['peak_memory_gb'] - baseline_results['peak_memory_gb']) / baseline_results['peak_memory_gb']) * 100
print(f"{'Peak Memory (GB)':<30} {baseline_results['peak_memory_gb']:<20.2f} {greentensor_results['peak_memory_gb']:<20.2f} {memory_diff:+.2f}%")

print("\n" + "="*80)
print("GREENTENSOR EXCLUSIVE METRICS")
print("="*80)
print(f"Energy Consumed: {gt_metrics.get('energy_consumed', 0):.4f} Wh")
print(f"Carbon Emissions: {gt_metrics.get('carbon_emissions', 0):.4f} gCO2")
print(f"Water Usage: {gt_metrics.get('water_usage', 0):.4f} L")
print(f"Optimizations Applied: {gt_metrics.get('optimizations_applied', 0)}")
print("="*80)

# Summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
if time_diff < 0:
    print(f"✓ GreenTensor training was {abs(time_diff):.2f}% FASTER")
else:
    print(f"✗ GreenTensor training was {time_diff:.2f}% SLOWER")

if memory_diff < 0:
    print(f"✓ GreenTensor used {abs(memory_diff):.2f}% LESS memory")
else:
    print(f"✗ GreenTensor used {memory_diff:.2f}% MORE memory")

if loss_diff < 0:
    print(f"✓ GreenTensor achieved {abs(loss_diff):.2f}% LOWER average loss")
else:
    print(f"✗ GreenTensor had {loss_diff:.2f}% HIGHER average loss")

print(f"\n✓ GreenTensor tracked {gt_metrics.get('energy_consumed', 0):.4f} Wh energy consumption")
print(f"✓ GreenTensor tracked {gt_metrics.get('carbon_emissions', 0):.4f} gCO2 emissions")
print(f"✓ GreenTensor tracked {gt_metrics.get('water_usage', 0):.4f} L water usage")
print(f"✓ GreenTensor applied {gt_metrics.get('optimizations_applied', 0)} optimizations")
print("="*80)

# Save results to JSON
results = {
    "model": model_name,
    "device": device,
    "baseline": baseline_results,
    "greentensor": greentensor_results,
    "greentensor_metrics": gt_metrics,
    "comparison": {
        "time_difference_percent": time_diff,
        "memory_difference_percent": memory_diff,
        "throughput_difference_percent": throughput_diff,
        "loss_difference_percent": loss_diff,
        "final_loss_difference_percent": final_loss_diff
    }
}

with open('opt_1.3b_training_comparison_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n✓ Results saved to opt_1.3b_training_comparison_results.json")
print("\nTest completed!")
