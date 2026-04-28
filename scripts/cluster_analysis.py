"""
cluster_analysis.py — Unsupervised clustering of clinical note embeddings.

Purpose:
    Embed all Gold-layer clinical notes using the trained E-005c encoder,
    then cluster to discover latent structure, measure ICD-10 code purity,
    and visualize why Chapter Z is difficult.

Outputs:
    - outputs/clustering/embeddings.npy (10240 x 768)
    - outputs/clustering/hdbscan_labels.npy
    - outputs/clustering/cluster_report.csv (purity per code/chapter)
    - outputs/clustering/umap_visualization.html (interactive plot)
    - outputs/clustering/tsne_visualization.html

Usage:
    uv run python scripts/cluster_analysis.py \
        --gold-path data/gold/medsynth_gold_augmented.parquet \
        --experiment E-005c_Merged_ZO \
        --sample 10240
"""

import sys
sys.path.insert(0, '.')

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import hdbscan
import umap
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances
import plotly.express as px
import plotly.graph_objects as go

from src.config import config

def load_gold_notes(gold_path: Path, sample: int = None):
    """Load apso_note and icd10_code from Gold layer."""
    print(f"📥 Loading Gold data from {gold_path}")
    df = pd.read_parquet(gold_path)
    
    # Use apso_note (preprocessed, redacted, Assessment-first)
    notes = df['apso_note'].tolist()
    codes = df['standard_icd10'].tolist()
    chapters = [c[0] if c else 'U' for c in codes]
    
    if sample and sample < len(notes):
        idx = np.random.choice(len(notes), sample, replace=False)
        notes = [notes[i] for i in idx]
        codes = [codes[i] for i in idx]
        chapters = [chapters[i] for i in idx]
    
    print(f"   Loaded {len(notes)} notes")
    print(f"   Unique codes: {len(set(codes))}")
    print(f"   Chapters: {sorted(set(chapters))}")
    return notes, codes, chapters, df

def extract_embeddings(notes, model_path, batch_size=32, device='mps'):
    """Extract [CLS] embeddings using trained encoder."""
    print(f"🔧 Loading encoder from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModel.from_pretrained(str(model_path)).to(device)
    model.eval()
    
    embeddings = []
    print(f"🧠 Extracting embeddings (batch_size={batch_size})...")
    for i in tqdm(range(0, len(notes), batch_size)):
        batch = notes[i:i+batch_size]
        inputs = tokenizer(
            batch,
            truncation=True,
            padding='max_length',
            max_length=512,
            return_tensors='pt'
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            # Use [CLS] token (first token)
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(cls_embeddings)
    
    embeddings = np.vstack(embeddings)
    print(f"   Embeddings shape: {embeddings.shape}")
    return embeddings

def run_hdbscan(embeddings, min_cluster_size=25):
    """Cluster embeddings with HDBSCAN."""
    print(f"🎯 Running HDBSCAN (min_cluster_size={min_cluster_size})...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=5,
        metric='euclidean',
        cluster_selection_method='eom'
    )
    labels = clusterer.fit_predict(embeddings)
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    print(f"   Found {n_clusters} clusters")
    print(f"   Noise points: {n_noise} ({100*n_noise/len(labels):.1f}%)")
    return labels, clusterer

def compute_purity(codes, chapters, cluster_labels):
    """Compute cluster purity per ICD-10 code and chapter."""
    print("📊 Computing cluster purity...")
    df = pd.DataFrame({
        'code': codes,
        'chapter': chapters,
        'cluster': cluster_labels
    })
    
    # Per-code purity
    code_stats = []
    for code, group in df.groupby('code'):
        n = len(group)
        n_clusters = group['cluster'].nunique()
        # Purity = 1 / (number of clusters this code spans)
        # Perfect purity = 1.0 (all in one cluster)
        purity = 1.0 / n_clusters if n_clusters > 0 else 0
        dominant_cluster = group['cluster'].mode()[0]
        dominant_pct = (group['cluster'] == dominant_cluster).mean()
        
        code_stats.append({
            'code': code,
            'chapter': group['chapter'].iloc[0],
            'n_samples': n,
            'n_clusters': n_clusters,
            'purity': purity,
            'dominant_cluster': int(dominant_cluster),
            'dominant_pct': dominant_pct
        })
    
    code_df = pd.DataFrame(code_stats).sort_values('purity')
    
    # Per-chapter summary
    chapter_stats = code_df.groupby('chapter').agg({
        'code': 'count',
        'n_samples': 'sum',
        'purity': 'mean',
        'n_clusters': 'mean',
        'dominant_pct': 'mean'
    }).rename(columns={'code': 'n_codes'}).reset_index()
    
    chapter_stats = chapter_stats.sort_values('purity')
    
    print("=== CHAPTER PURITY (lowest = hardest) ===")
    print(chapter_stats.to_string(index=False, float_format='%.3f'))
    
    return code_df, chapter_stats

def create_visualizations(embeddings, codes, chapters, cluster_labels, output_dir):
    """Create UMAP and t-SNE interactive plots."""
    print("🎨 Creating visualizations...")
    
    # UMAP (faster, better global structure)
    print("   Running UMAP...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    umap_2d = reducer.fit_transform(embeddings)
    
    # t-SNE (slower, better local structure)
    print("   Running t-SNE (this takes ~3-5 min)...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca')
    tsne_2d = tsne.fit_transform(embeddings)
    
    # Create DataFrames for plotting
    plot_df = pd.DataFrame({
        'x_umap': umap_2d[:, 0],
        'y_umap': umap_2d[:, 1],
        'x_tsne': tsne_2d[:, 0],
        'y_tsne': tsne_2d[:, 1],
        'code': codes,
        'chapter': chapters,
        'cluster': cluster_labels,
        'cluster_str': [f'Cluster {c}' if c != -1 else 'Noise' for c in cluster_labels]
    })
    
    # UMAP colored by chapter
    fig_umap_chapter = px.scatter(
        plot_df, x='x_umap', y='y_umap',
        color='chapter',
        hover_data=['code', 'cluster'],
        title='UMAP: Clinical Note Embeddings Colored by ICD-10 Chapter',
        width=1200, height=800
    )
    fig_umap_chapter.write_html(output_dir / 'umap_by_chapter.html')
    
    # UMAP colored by cluster
    fig_umap_cluster = px.scatter(
        plot_df, x='x_umap', y='y_umap',
        color='cluster_str',
        hover_data=['code', 'chapter'],
        title='UMAP: HDBSCAN Clusters',
        width=1200, height=800
    )
    fig_umap_cluster.write_html(output_dir / 'umap_by_cluster.html')
    
    # t-SNE by chapter
    fig_tsne = px.scatter(
        plot_df, x='x_tsne', y='y_tsne',
        color='chapter',
        hover_data=['code', 'cluster'],
        title='t-SNE: Clinical Note Embeddings Colored by ICD-10 Chapter',
        width=1200, height=800
    )
    fig_tsne.write_html(output_dir / 'tsne_by_chapter.html')
    
    # Focus on Chapter Z
    z_df = plot_df[plot_df['chapter'] == 'Z']
    if len(z_df) > 0:
        fig_z = px.scatter(
            z_df, x='x_umap', y='y_umap',
            color='code',
            hover_data=['cluster'],
            title=f'UMAP: Chapter Z Only (n={len(z_df)}, {z_df["code"].nunique()} codes)',
            width=1200, height=800
        )
        fig_z.write_html(output_dir / 'umap_chapter_Z.html')
    
    print(f"   Saved 4 interactive HTML plots to {output_dir}")
    return plot_df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gold-path', type=str, 
                       default='data/gold/medsynth_gold_augmented.parquet')
    parser.add_argument('--experiment', type=str, default='E-005c_Merged_ZO')
    parser.add_argument('--sample', type=int, default=None,
                       help='Random sample size (default: all)')
    parser.add_argument('--min-cluster-size', type=int, default=25)
    args = parser.parse_args()
    
    # Setup paths
    gold_path = Path(args.gold_path)
    model_path = Path('outputs/evaluations/E-005c_Merged_ZO/stage2/Z/model/model')
    output_dir = Path('outputs/clustering')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load data
    notes, codes, chapters, df = load_gold_notes(gold_path, args.sample)
    
    # 2. Extract embeddings
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    embeddings = extract_embeddings(notes, model_path, device=device)
    np.save(output_dir / 'embeddings.npy', embeddings)
    
    # 3. Cluster
    cluster_labels, clusterer = run_hdbscan(embeddings, args.min_cluster_size)
    np.save(output_dir / 'hdbscan_labels.npy', cluster_labels)
    
    # 4. Compute purity
    code_df, chapter_df = compute_purity(codes, chapters, cluster_labels)
    code_df.to_csv(output_dir / 'code_purity.csv', index=False)
    chapter_df.to_csv(output_dir / 'chapter_purity.csv', index=False)
    
    # 5. Visualize
    plot_df = create_visualizations(embeddings, codes, chapters, cluster_labels, output_dir)
    plot_df.to_csv(output_dir / 'embeddings_2d.csv', index=False)
    
    # 6. Summary report
    print("" + "="*60)
    print("CLUSTERING SUMMARY")
    print("="*60)
    print(f"Total notes: {len(notes)}")
    print(f"Embedding dim: {embeddings.shape[1]}")
    print(f"HDBSCAN clusters: {len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)}")
    print(f"Noise points: {list(cluster_labels).count(-1)}")
    print(f"Hardest chapters (lowest purity):")
    print(chapter_df.head(5)[['chapter', 'n_codes', 'purity', 'n_clusters']].to_string(index=False))
    print(f"Easiest chapters (highest purity):")
    print(chapter_df.tail(5)[['chapter', 'n_codes', 'purity', 'n_clusters']].to_string(index=False))
    
    print(f"✅ Outputs saved to {output_dir}/")
    print("   - embeddings.npy")
    print("   - hdbscan_labels.npy")
    print("   - code_purity.csv")
    print("   - chapter_purity.csv")
    print("   - umap_by_chapter.html (open in browser)")
    print("   - umap_by_cluster.html")
    print("   - umap_chapter_Z.html")
    print("   - tsne_by_chapter.html")

if __name__ == '__main__':
    main()
