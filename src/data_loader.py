"""
data_loader.py — Gold layer ingestion utilities.

Provides two loaders that mirror the data access patterns established
in the model training notebooks:

    load_medsynth_raw()
        Fetches the raw MedSynth dataset from Hugging Face and returns
        it as a Polars DataFrame. Used in notebook 01 (EDA).

    load_gold_parquet()
        Loads the most recent Gold layer Parquet produced by notebook 01
        Phase 4 (the APSO-flipped, redacted, CDC-validated dataset).
        Mirrors the glob pattern used in notebooks 02-05 Phase 1.

    load_label_mapping(experiment_name)
        Loads the label2id / id2label JSON saved during Phase 10 registry
        promotion. Mirrors the path used in notebooks 02 and 03.

Note on splits
--------------
Train/val/test splits are NOT persisted to disk anywhere in the pipeline —
they are produced in-memory during each notebook's Phase 2 via
sklearn.model_selection.train_test_split. There is therefore no
``load_surgical_splits()`` function: callers must reproduce the split
from the Gold layer using the same seed and parameters as the original
experiment (seed=42, 80/10/10, stratified train/temp, random val/test).
"""

import json
import polars as pl
from pathlib import Path
from typing import Dict, Optional, Tuple

from src.config import config


def load_medsynth_raw() -> pl.DataFrame:
    """
    Fetch the raw MedSynth dataset from Hugging Face.

    Converts to Polars immediately for high-performance EDA and strips
    any whitespace from column names (present in the HuggingFace release).

    Returns
    -------
    pl.DataFrame
        Raw MedSynth records with columns: ID, Note, Dialogue, ICD10.

    Raises
    ------
    ImportError
        If the ``datasets`` package is not installed.
    """
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            "The 'datasets' package is required to load MedSynth from HuggingFace. "
            "Install it with: pip install datasets"
        ) from e

    print("📥 Loading raw MedSynth from Hugging Face...")
    ds = load_dataset("Ahmad0067/MedSynth", split="train")

    df = pl.from_arrow(ds.data.table)
    df = df.rename({c: c.strip() for c in df.columns})

    print(f"   ✅ Loaded {len(df):,} records, {len(df.columns)} columns")
    return df


def load_gold_parquet() -> pl.DataFrame:
    """
    Load the most recent Gold layer Parquet from the data/gold directory.

    Selects the latest file matching the ``medsynth_gold_apso_*.parquet``
    naming convention produced by notebook 01 Phase 4. This mirrors the
    discovery pattern used in the Phase 1 cells of notebooks 02–05:

        gold_dir      = config.resolve_path("data", "gold")
        parquet_files = sorted(gold_dir.glob("medsynth_gold_apso_*.parquet"))
        GOLD_PARQUET_PATH = parquet_files[-1]  # most recent by filename timestamp

    Returns
    -------
    pl.DataFrame
        Gold layer with columns including: id, apso_note, standard_icd10,
        code_status, assessment, plan, subjective, objective, and others.

    Raises
    ------
    FileNotFoundError
        If no matching Parquet file exists in the gold directory.
        Run notebook 01 through Phase 4 first.
    """
    gold_dir = config.resolve_path("data", "gold")
    parquet_files = sorted(gold_dir.glob("medsynth_gold_apso_*.parquet"))

    if not parquet_files:
        raise FileNotFoundError(
            f"No Gold layer Parquet found in {gold_dir}\n"
            f"Run notebook 01-EDA_SOAP.ipynb through Phase 4 first.\n"
            f"Expected filename pattern: medsynth_gold_apso_*.parquet"
        )

    latest = parquet_files[-1]  # most recent by filename timestamp
    print(f"📂 Loading Gold layer: {latest.name}")
    df = pl.read_parquet(latest)
    print(f"   ✅ Loaded {len(df):,} records, {len(df.columns)} columns")
    return df


def load_label_mapping(experiment_name: str) -> Dict:
    """
    Load the label2id / id2label mapping for a completed experiment.

    Looks for ``label_mapping.json`` in the experiment's registry directory —
    the canonical location after Phase 10 promotion:

        outputs/evaluations/registry/{experiment_name}/label_mapping.json

    This mirrors the path used in notebooks 02 and 03 Phase 10, where the
    mapping is copied from the experiment directory to the registry.

    Parameters
    ----------
    experiment_name : str
        The experiment name as used in cfg["experiment_name"], e.g.
        "E-001_Baseline_ICD3" or "E-002_FullICD10_ClinicalBERT".

    Returns
    -------
    dict with keys:
        label2id : dict[str, int]  — ICD code string → integer class index
        id2label : dict[str, str]  — string integer index → ICD code string
                                     (JSON keys are always strings)

    Raises
    ------
    FileNotFoundError
        If the mapping file does not exist in the registry.
        Run the relevant notebook through Phase 10 first.
    """
    registry_base = config.resolve_path("outputs", "evaluations") / "registry"
    mapping_path  = registry_base / experiment_name / "label_mapping.json"

    if not mapping_path.exists():
        raise FileNotFoundError(
            f"Label mapping not found: {mapping_path}\n"
            f"Run notebook for '{experiment_name}' through Phase 10 "
            f"(model registry promotion) first."
        )

    with open(mapping_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)

    n_labels = len(mapping.get("label2id", {}))
    print(f"📖 Loaded label mapping: {experiment_name} ({n_labels:,} classes)")
    return mapping