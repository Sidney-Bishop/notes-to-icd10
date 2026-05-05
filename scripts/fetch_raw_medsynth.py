#!/usr/bin/env python3
"""
scripts/fetch_raw_medsynth.py
-----------------------------
Reproducible fetcher for the raw MedSynth source.

Source: Hugging Face dataset Ahmad0067/MedSynth (split=train)
Output: data/raw/medsynth_notes_v2024.parquet

This is the canonical Phase 0 ingestion step. It replaces the
ad-hoc notebook cell and ensures the raw anchor is reproducible
across machines and CI.

Usage:
    uv run python scripts/fetch_raw_medsynth.py
    # or
    make fetch-raw
"""
from pathlib import Path
import polars as pl
from datasets import load_dataset

DATASET_ID = "Ahmad0067/MedSynth"
SPLIT = "train"
# Pin the dataset revision for reproducibility — update after first successful run
# To get the commit hash: run once, then check ds.info or HF page
REVISION = "main"

OUTPUT_PATH = Path("data/raw/medsynth_notes_v2024.parquet")

def main():
    print(f"📥 Loading {DATASET_ID}@{REVISION} [{SPLIT}]...")
    ds = load_dataset(DATASET_ID, split=SPLIT, revision=REVISION)

    df = pl.from_arrow(ds.data.table)
    # Strip whitespace from column names (present in HF release)
    df = df.rename({c: c.strip() for c in df.columns})

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(OUTPUT_PATH, compression="snappy")

    print(f"✅ Wrote {len(df):,} rows × {len(df.columns)} cols → {OUTPUT_PATH}")
    print(f" Columns: {list(df.columns)}")
    print(f"Next steps:")
    print(f"  1. dvc add {OUTPUT_PATH}  (if tracking locally)")
    print(f"  2. hf upload SidneyBishop/medsynth-icd10 {OUTPUT_PATH} raw/medsynth_notes_v2024.parquet")

if __name__ == "__main__":
    main()
