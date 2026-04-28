"""
generate_simcse_pairs.py — Create contrastive pairs from HDBSCAN clusters.

Uses the clustering output to generate:
- Positive pairs: notes with SAME ICD-10 CODE (not cluster)
- Hard negatives: notes from different clusters but same chapter (esp. Z)

Output: data/gold/simcse_pairs.parquet
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
from pathlib import Path
from itertools import combinations
from tqdm import tqdm
import random

random.seed(42)
np.random.seed(42)

# Load clustering results
print("📥 Loading clustering outputs...")
labels = np.load('outputs/clustering/hdbscan_labels.npy')
embeddings = np.load('outputs/clustering/embeddings.npy')
df = pd.read_parquet('data/gold/medsynth_gold_augmented.parquet')

# Align (clustering used sample of 11214)
if len(df) > len(labels):
    df = df.iloc[:len(labels)].reset_index(drop=True)

df['cluster'] = labels
df['chapter'] = df['standard_icd10'].str[0]

print(f" {len(df)} notes, {len(set(labels)) - (1 if -1 in labels else 0)} clusters")

pairs = []

# 1. Positive pairs from SAME ICD-10 CODE (not cluster)
print("🔗 Generating positive pairs (same code)...")
code_groups = df.groupby('standard_icd10')

for code, group in tqdm(code_groups, total=len(code_groups)):
    if len(group) < 2:
        continue

    # Sample up to 15 pairs per code to balance frequent codes
    max_pairs = min(15, len(group) * (len(group) - 1) // 2)
    idx_pairs = list(combinations(group.index, 2))

    if len(idx_pairs) > max_pairs:
        idx_pairs = random.sample(idx_pairs, max_pairs)

    for i, j in idx_pairs:
        pairs.append({
            'text1': df.loc[i, 'apso_note'],
            'text2': df.loc[j, 'apso_note'],
            'label': 1,
            'type': 'positive_same_code',
            'chapter1': df.loc[i, 'chapter'],
            'chapter2': df.loc[j, 'chapter'],
            'code1': code,
            'code2': code,
            'cluster1': df.loc[i, 'cluster'],
            'cluster2': df.loc[j, 'cluster']
        })

        if len([p for p in pairs if p['label'] == 1]) >= 3000:
            break
    if len([p for p in pairs if p['label'] == 1]) >= 3000:
        break

print(f" Generated {len([p for p in pairs if p['label']==1])} positives")

# 2. Hard negatives: same chapter, different clusters (focus on Z)
print("⚔️ Generating hard negatives...")
for chapter in ['Z', 'R', 'M', 'I']:
    chap_df = df[(df['chapter'] == chapter) & (df['cluster']!= -1)]
    clusters = chap_df['cluster'].unique()

    n_pairs_target = 1500 if chapter == 'Z' else 500

    count = 0
    for c1, c2 in combinations(clusters, 2):
        n1 = chap_df[chap_df['cluster'] == c1]
        n2 = chap_df[chap_df['cluster'] == c2]

        if len(n1) == 0 or len(n2) == 0:
            continue

        # Sample 1-2 pairs per cluster pair
        for _ in range(min(2, len(n1), len(n2))):
            i = np.random.choice(n1.index)
            j = np.random.choice(n2.index)
            pairs.append({
                'text1': df.loc[i, 'apso_note'],
                'text2': df.loc[j, 'apso_note'],
                'label': 0,
                'type': f'hard_negative_{chapter}',
                'chapter1': chapter,
                'chapter2': chapter,
                'code1': df.loc[i, 'standard_icd10'],
                'code2': df.loc[j, 'standard_icd10'],
                'cluster1': c1,
                'cluster2': c2
            })
            count += 1
            if count >= n_pairs_target:
                break
        if count >= n_pairs_target:
            break

pairs_df = pd.DataFrame(pairs)
print(f"\n✅ Generated {len(pairs_df)} pairs:")
print(pairs_df['type'].value_counts())

# Balance check
pos_count = len(pairs_df[pairs_df['label'] == 1])
neg_count = len(pairs_df[pairs_df['label'] == 0])
print(f"\nBalance: {pos_count} positives, {neg_count} negatives ({pos_count/neg_count:.2f} ratio)")

# Save
out_path = Path('data/gold/simcse_pairs.parquet')
out_path.parent.mkdir(parents=True, exist_ok=True)
pairs_df.to_parquet(out_path, index=False)
print(f"\n💾 Saved to {out_path}")

# Stats for Z specifically
z_pairs = pairs_df[pairs_df['chapter1'] == 'Z']
print(f"\nZ-chapter pairs: {len(z_pairs)} ({len(z_pairs[z_pairs['label']==1])} pos, {len(z_pairs[z_pairs['label']==0])} neg)")