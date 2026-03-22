# src/data_loader.py
import polars as pl
import json
from pathlib import Path
from typing import Tuple, Dict, Optional
from src.config import config

def load_medsynth_raw() -> pl.DataFrame:
    """
    Standard loader for the raw MedSynth dataset from Hugging Face.
    Converts to Polars immediately for high-performance EDA.
    """
    from datasets import load_dataset
    
    print("📥 Loading raw MedSynth from Hugging Face...")
    ds = load_dataset("Ahmad0067/MedSynth", split="train")
    
    # Convert to Polars and clean column names
    df = pl.from_arrow(ds.data.table)
    df = df.rename({c: c.strip() for c in df.columns})
    
    return df

def load_surgical_splits() -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, Dict]:
    """
    Loads the Pydantic-validated and SOAP-extracted splits from the GOLD layer.
    These are the final artifacts used for E-002 training.
    """
    gold_dir = config.resolve_path("data", "gold")
    map_dir = config.resolve_path("outputs", "evaluations") / "label_mappings"
    
    # Locate the latest mapping (finds the most recent experiment mapping)
    mappings = list(map_dir.glob("*_label_mapping.json"))
    if not mappings:
        raise FileNotFoundError(f"⚠️ No label mapping found in {map_dir}")
    
    latest_map = max(mappings, key=lambda p: p.stat().st_mtime)

    # Load using Polars Parquet reader
    train_df = pl.read_parquet(gold_dir / "train.parquet")
    val_df = pl.read_parquet(gold_dir / "val.parquet")
    test_df = pl.read_parquet(gold_dir / "test.parquet")
    
    with open(latest_map, "r") as f:
        mapping_data = json.load(f)
    
    print(f"✅ Loaded Gold Splits: Train({len(train_df)}), Val({len(val_df)}), Test({len(test_df)})")
    print(f"📖 Using mapping: {latest_map.name}")
    
    return train_df, val_df, test_df, mapping_data['label2id']