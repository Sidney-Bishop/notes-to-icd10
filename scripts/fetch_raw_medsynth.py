#!/usr/bin/env python3
"""
fetch_raw_medsynth.py — Reproducible Phase 0 raw fetcher

Downloads Ahmad0067/MedSynth from HuggingFace and writes the canonical
DVC-tracked parquet with the EXACT schema the pipeline expects.

This preserves:
- ICD10 as List[str] (not flattened to string)
- Original column casing: ID, Note, Dialogue, ICD10
- Optional ICD10_desc dropped (not used downstream)
"""
from pathlib import Path
from datasets import load_dataset
import polars as pl

HF_REPO = "Ahmad0067/MedSynth"
OUT_PATH = Path("data/raw/medsynth_notes_v2024.parquet")

def main():
    print(f"📥 Fetching {HF_REPO} from HuggingFace...")
    ds = load_dataset(HF_REPO, split="train")
    df = pl.from_arrow(ds.data.table)

    # Strip whitespace from HF column names
    df = df.rename({c: c.strip() for c in df.columns})
    print(f" Raw columns: {df.columns}")

    # Ensure ICD10 is List[str] — Parquet was flattening it to Utf8
    if "ICD10" in df.columns:
        # HF gives list already, but be defensive
        if df.schema["ICD10"]!= pl.List(pl.Utf8):
            df = df.with_columns(
                pl.col("ICD10").map_elements(
                    lambda x: [x] if isinstance(x, str) else (x if x is not None else []),
                    return_dtype=pl.List(pl.Utf8)
                )
            )

    # Drop helper column if present (not used by pipeline)
    if "ICD10_desc" in df.columns:
        df = df.drop("ICD10_desc")

    # Keep original casing — pipeline expects ID, Note, Dialogue, ICD10
    # Generate ID if missing (newer HF releases dropped it)
    if "ID" not in df.columns:
        print(" ⚠️ ID missing — generating sequential IDs")
        df = df.with_columns(pl.int_range(0, len(df)).cast(pl.Utf8).alias("ID"))

    # Select canonical columns in order
    df = df.select(["ID", "Note", "Dialogue", "ICD10"])

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(OUT_PATH, compression="zstd")
    print(f"✅ Saved {len(df):,} rows to {OUT_PATH}")
    print(f" Schema: {df.schema}")

if __name__ == "__main__":
    main()