"""
extract_embeddings.py — Extract embeddings using SimCSE encoder
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--encoder', required=True)
parser.add_argument('--output', required=True)
parser.add_argument('--data', default='data/gold/medsynth_gold_augmented.parquet')
args = parser.parse_args()

print(f"📥 Loading data from {args.data}")
df = pd.read_parquet(args.data)
texts = df['apso_note'].tolist()

print(f"🔧 Loading encoder from {args.encoder}")
model = SentenceTransformer(args.encoder, device='cpu')

print(f"🧠 Extracting {len(texts)} embeddings...")
embeddings = model.encode(
    texts,
    batch_size=32,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True
)

Path(args.output).parent.mkdir(parents=True, exist_ok=True)
np.save(args.output, embeddings)
print(f"✅ Saved {embeddings.shape} to {args.output}")
