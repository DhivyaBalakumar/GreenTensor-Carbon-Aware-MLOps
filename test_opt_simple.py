"""
Simple OPT-1.3B Training Test: Baseline vs GreenTensor
"""
import warnings
warnings.filterwarnings('ignore')

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import time
import psutil
import os
from greentensor import GreenTensor

print("="*80)
print("OPT-1.3B TRAINING TEST: BASELINE vs GREENTENSOR")
print("="*80)

# Check device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nDevice: {device}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Load model
print("\nLoading OPT-1.3B model...")
model_name = "facebook/opt-1.3b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load dataset
print("Loading dataset...")
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
train_texts = [text for text in dataset['text'][:100] if len(text.strip()) > 50]
print(f"Using {len(train_texts)} training samples")

# Training params
batch_size = 2
learning_rate = 5e-5
max_length = 128
num_steps = 20  # Reduced for faster testing

def prepare_batch(texts, tokenizer, max_length, device):
    encodings = tokenizer(
        texts,
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors='pt'
    )
    return encodings['input_ids'].to(device), encodings['attention_mask'].to(device)

# Create batches
batches = []
for i in range(0, min(len(train_texts), num_steps * batch_size), batch_size):
    batch_texts = train_texts[i:i+batch_size]
    if len(batch_texts) == batch_size:
        batches.append(batch_texts)

print(f"Created {len(batches)} batches")

# ============================================================================
# BASELINE TRAINING
# ============================================================================
print("\n" + "="*80)
print("BASELINE TRAINING")
print("="*80)

model_baseline = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
)
model_baseline = model_baseline.to(device)
model_baseline.train()

optimizer_baseline = torch.optim.AdamW(model_baseline.parameters(), lr=learning_rate)

if device == "cuda":
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

baseline_losses = []
start_time = time.time()

for step, batch_texts in enumerate(batches[:num_steps]):
    input_ids, attention_mask = prepare_batch(batch_texts, tokenizer, max_length, device)
    outputs = model_baseline(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
    loss = outputs.loss
    
    optimizer_baseline.zero_grad()
    loss.backward()
    optimizer_baseline.step()
    
    baseline_losses.append(loss.item())
    if (step + 1) % 5 == 0:
        print(f"Step {step+1}/{num_steps} | Loss: {loss.item():.4f}")

baseline_time = time.time() - start_time
baseline_avg_loss = sum(baseline_losses) / len(baseline_losses)

if device == "cuda":
    baseline_memory = torch.cuda.max_memory_allocated() / 1e9
else:
    process = psutil.Process(os.getpid())
    baseline_memory = process.memory_info().rss / 1e9

print(f"\nBaseline Results:")
print(f"  Time: {baseline_time:.2f}s")
print(f"  Avg Loss: {baseline_avg_loss:.4f}")
print(f"  Final Loss: {baseline_losses[-1]:.4f}")
print(f"  Peak Memory: {baseline_memory:.2f} GB")

del model_baseline, optimizer_baseline
if device == "cuda":
    torch.cuda.empty_cache()

# ============================================================================
# GREENTENSOR TRAINING
# ============================================================================
print("\n" + "="*80)
print("GREENTENSOR TRAINING")
print("="*80)

model_gt = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
)
model_gt = model_gt.to(device)
model_gt.train()

optimizer_gt = torch.optim.AdamW(model_gt.parameters(), lr=learning_rate)

if device == "cuda":
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

gt_losses = []
start_time = time.time()

with GreenTensor(model_name="opt-1.3b", verbose=True) as gt:
    for step, batch_texts in enumerate(batches[:num_steps]):
        input_ids, attention_mask = prepare_batch(batch_texts, tokenizer, max_length, device)
        outputs = model_gt(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = outputs.loss
        
        optimizer_gt.zero_grad()
        loss.backward()
        optimizer_gt.step()
        
        gt_losses.append(loss.item())
        if (step + 1) % 5 == 0:
            print(f"Step {step+1}/{num_steps} | Loss: {loss.item():.4f}")

gt_time = time.time() - start_time
gt_avg_loss = sum(gt_losses) / len(gt_losses)

if device == "cuda":
    gt_memory = torch.cuda.max_memory_allocated() / 1e9
else:
    process = psutil.Process(os.getpid())
    gt_memory = process.memory_info().rss / 1e9

print(f"\nGreenTensor Results:")
print(f"  Time: {gt_time:.2f}s")
print(f"  Avg Loss: {gt_avg_loss:.4f}")
print(f"  Final Loss: {gt_losses[-1]:.4f}")
print(f"  Peak Memory: {gt_memory:.2f} GB")

if gt.metrics:
    print(f"  Energy: {gt.metrics.energy_kwh * 1000:.4f} Wh")
    print(f"  Carbon: {gt.metrics.emissions_kg * 1000:.4f} gCO2")

del model_gt, optimizer_gt
if device == "cuda":
    torch.cuda.empty_cache()

# ============================================================================
# COMPARISON
# ============================================================================
print("\n" + "="*80)
print("SIDE-BY-SIDE COMPARISON")
print("="*80)
print(f"\n{'Metric':<25} {'Baseline':<15} {'GreenTensor':<15} {'Difference'}")
print("-"*70)

time_diff = ((gt_time - baseline_time) / baseline_time) * 100
print(f"{'Time (s)':<25} {baseline_time:<15.2f} {gt_time:<15.2f} {time_diff:+.2f}%")

loss_diff = ((gt_avg_loss - baseline_avg_loss) / baseline_avg_loss) * 100
print(f"{'Avg Loss':<25} {baseline_avg_loss:<15.4f} {gt_avg_loss:<15.4f} {loss_diff:+.2f}%")

memory_diff = ((gt_memory - baseline_memory) / baseline_memory) * 100
print(f"{'Peak Memory (GB)':<25} {baseline_memory:<15.2f} {gt_memory:<15.2f} {memory_diff:+.2f}%")

if gt.metrics:
    print(f"\n{'GreenTensor Exclusive Metrics:'}")
    print(f"  Energy Consumed: {gt.metrics.energy_kwh * 1000:.4f} Wh")
    print(f"  Carbon Emissions: {gt.metrics.emissions_kg * 1000:.4f} gCO2")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
if time_diff < 0:
    print(f"✓ GreenTensor was {abs(time_diff):.2f}% FASTER")
else:
    print(f"✗ GreenTensor was {time_diff:.2f}% SLOWER")

if gt.metrics:
    print(f"✓ Tracked {gt.metrics.energy_kwh * 1000:.4f} Wh energy consumption")
    print(f"✓ Tracked {gt.metrics.emissions_kg * 1000:.4f} gCO2 emissions")

print("="*80)
print("\n✓ Test completed!")
