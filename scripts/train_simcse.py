"""
train_simcse.py — Contrastive fine-tuning using code-level pairs.

Fixes the 'checkpoints only' bug by explicitly saving the underlying
transformer in HuggingFace format after SimCSE training.
"""
import sys
sys.path.insert(0, '.')
import pandas as pd
import shutil
from pathlib import Path
from sentence_transformers import SentenceTransformer, losses, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from torch.utils.data import DataLoader
import random

print("📥 Loading pairs...")
pairs_df = pd.read_parquet('data/gold/simcse_pairs.parquet')
pairs_df = pairs_df.sample(frac=1, random_state=42).reset_index(drop=True)
split = int(0.9 * len(pairs_df))
train_df, val_df = pairs_df[:split], pairs_df[split:]

train_examples = [InputExample(texts=[r.text1, r.text2], label=float(r.label)) for _, r in train_df.iterrows()]
val_examples = [InputExample(texts=[r.text1, r.text2], label=float(r.label)) for _, r in val_df.iterrows()]

print(f" Train: {len(train_examples)}, Val: {len(val_examples)}")

model_path = 'outputs/evaluations/E-005c_Merged_ZO/stage2/Z/model/model'
model = SentenceTransformer(model_path, device='cpu')

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
# Use MultipleNegativesRankingLoss for better learning
train_loss = losses.MultipleNegativesRankingLoss(model)

evaluator = EmbeddingSimilarityEvaluator.from_input_examples(val_examples, name='val')

# --- FIXED OUTPUT PATH to match evaluate.py expectations ---
experiment = 'E-005c_simcse_Z'
output_base = Path(f'outputs/evaluations/{experiment}/stage2/Z/model')
output_base.mkdir(parents=True, exist_ok=True)

# Temporary SimCSE output (checkpoints go here)
temp_output = Path('outputs/models/simcse_E005c_cluster_temp')
if temp_output.exists():
    shutil.rmtree(temp_output)
temp_output.mkdir(parents=True, exist_ok=True)

print("🚀 Training SimCSE (3 epochs)...")
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    evaluator=evaluator,
    epochs=3,
    warmup_steps=100,
    optimizer_params={'lr': 2e-5},
    output_path=str(temp_output),
    evaluation_steps=500,
    save_best_model=True,
    show_progress_bar=True
)

print("\n💾 Saving final model in HuggingFace format...")

# 1. Save SentenceTransformer format (for future SimCSE use)
model.save(str(output_base))

# 2. CRITICAL FIX: Save underlying transformer for AutoModel loading
# This creates config.json with model_type, pytorch_model.bin/safetensors
transformer = model._first_module().auto_model
tokenizer = model.tokenizer

transformer.save_pretrained(str(output_base))
tokenizer.save_pretrained(str(output_base))

# 3. Copy label_map from source Z model (needed by inference.py)
src_label_map = Path(model_path).parent.parent / 'label_map.json'
if src_label_map.exists():
    shutil.copy2(src_label_map, output_base.parent / 'label_map.json')

# 4. Clean up temp checkpoints
if temp_output.exists():
    shutil.rmtree(temp_output)

# 5. Verify
required_files = ['config.json', 'model.safetensors', 'tokenizer.json']
# model.safetensors might be pytorch_model.bin depending on transformers version
has_weights = (output_base / 'model.safetensors').exists() or (output_base / 'pytorch_model.bin').exists()
has_config = (output_base / 'config.json').exists()

print(f"\n✅ Model saved to {output_base}")
print(f"   - config.json: {has_config}")
print(f"   - weights: {has_weights}")
print(f"   - tokenizer: {(output_base / 'tokenizer.json').exists()}")

if has_config:
    import json
    with open(output_base / 'config.json') as f:
        cfg = json.load(f)
    print(f"   - model_type: {cfg.get('model_type', 'MISSING')}")   