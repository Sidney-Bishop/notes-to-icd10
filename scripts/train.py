"""
train.py — Model fine-tuning script for the ICD-10 coding pipeline.

Trains a flat (single-stage) or hierarchical (two-stage) ICD-10 classifier
using the EncoderAdapter interface from src/adapters.py. The model is a
config value — change the model_name_or_path in artifacts.yaml or pass
--model on the command line to swap Bio_ClinicalBERT for MedBERT, PubMedBERT,
or any other HuggingFace encoder without touching the training code.

Supported training modes
------------------------
    flat        Train a single classifier over ICD-3 or flat ICD-10 labels.
                Replicates notebooks 02 (ICD-3) and 03 (flat ICD-10).

    hierarchical  Train Stage-1 (chapter router) and/or Stage-2 (per-chapter
                resolvers). Replicates notebooks 04/05.

The default mode is flat. Pass --mode hierarchical for multi-stage training.

Usage examples
--------------
    # Flat ICD-3 baseline (replicates E-001)
    uv run python scripts/train.py \\
        --experiment E-001_Baseline_ICD3 \\
        --label-scheme icd3 \\
        --model emilyalsentzer/Bio_ClinicalBERT

    # Flat ICD-10 baseline (replicates E-002)
    uv run python scripts/train.py \\
        --experiment E-002_FullICD10_ClinicalBERT \\
        --label-scheme icd10

    # Hierarchical Stage-1 only (replicates E-003 Stage-1)
    uv run python scripts/train.py \\
        --experiment E-003_Hierarchical_ICD10 \\
        --mode hierarchical \\
        --stage 1

    # Hierarchical Stage-2 from E-002 init (replicates E-004a)
    uv run python scripts/train.py \\
        --experiment E-004a_Hierarchical_E002Init \\
        --mode hierarchical \\
        --stage 2 \\
        --stage2-init outputs/evaluations/registry/E-002_FullICD10_ClinicalBERT/model

    # Filter to billable codes only (the default for hierarchical experiments)
    uv run python scripts/train.py \\
        --experiment E-004a_Hierarchical_E002Init \\
        --mode hierarchical \\
        --code-filter billable

    # Dry run — validate data loading without training
    uv run python scripts/train.py --dry-run

Output layout
-------------
    outputs/evaluations/{experiment_name}/
        stage1/model/          — Stage-1 model (hierarchical mode)
        stage2/{chapter}/model/ — Per-chapter Stage-2 models (hierarchical mode)
        model/                 — Flat model (flat mode)
        label_map.json         — label2id / id2label mapping
        train_result.json      — TrainingResult summary
"""

import re
import sys
import json
import argparse
import mlflow
import polars as pl
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from datasets import Dataset as HFDataset

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------

def _bootstrap_project_root() -> Path:
    current = Path.cwd()
    while current != current.parent:
        if (current / "artifacts.yaml").exists():
            return current.resolve()
        current = current.parent
    raise FileNotFoundError("Could not find 'artifacts.yaml' — run from within the project tree.")

PROJECT_ROOT = _bootstrap_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import config                                   # noqa: E402
from src.data_loader import load_gold_parquet, load_label_mapping  # noqa: E402
from src.adapters import EncoderAdapter, TrainingResult         # noqa: E402


# ==============================================================================
# Label scheme helpers
# ==============================================================================

def _stem_icd3(code: str) -> str:
    """Collapse ICD-10 code to 3-char ICD-3 stem. E.g. 'M25.562' → 'M25'."""
    return code.split(".")[0][:3].upper()


def _build_label_maps(
    codes: list[str],
    label_scheme: str,
) -> tuple[dict[str, int], dict[int, str]]:
    """
    Build label2id and id2label from a list of ICD code strings.

    Parameters
    ----------
    codes : list[str]
        Raw ICD code strings from the Gold layer (standard_icd10 column).
    label_scheme : str
        'icd3'  — collapse to 3-char stems (E-001 style)
        'icd10' — full canonical codes (E-002/E-004a/E-005a style)
        'chapter' — first character only (Stage-1 router)
    """
    if label_scheme == "icd3":
        labels = sorted(set(_stem_icd3(c) for c in codes if c))
    elif label_scheme == "chapter":
        labels = sorted(set(c[0] for c in codes if c))
    else:  # icd10
        labels = sorted(set(codes))

    label2id = {lbl: i for i, lbl in enumerate(labels)}
    id2label = {i: lbl for i, lbl in enumerate(labels)}
    return label2id, id2label


def _apply_label_scheme(code: str, label_scheme: str) -> str:
    """Apply label scheme to a single code string."""
    if label_scheme == "icd3":
        return _stem_icd3(code)
    if label_scheme == "chapter":
        return code[0] if code else "UNKNOWN"
    return code  # icd10


# ==============================================================================
# Data preparation helpers
# ==============================================================================

def _filter_gold(df: pl.DataFrame, code_filter: str) -> pl.DataFrame:
    """
    Filter Gold layer by code_status.

    'billable'  — 9,660 records (notebooks 02-05 default for hierarchical)
    'all'       — 10,240 records (no filter)
    """
    if code_filter == "all":
        return df
    if code_filter == "billable":
        return df.filter(pl.col("code_status") == "billable")
    raise ValueError(f"Unknown code_filter: '{code_filter}'. Use 'all' or 'billable'.")


def _make_hf_dataset(
    df: pl.DataFrame,
    label_scheme: str,
    label2id: dict[str, int],
    tokenizer,
    max_length: int = 512,
) -> HFDataset:
    """
    Convert a Polars DataFrame slice to a HuggingFace Dataset for the Trainer.

    Applies label scheme transformation and integer encoding, then tokenises.
    The 'labels' column uses the integer index from label2id.
    """
    # Build (text, label_str) pairs
    texts  = df["apso_note"].to_list()
    raw_labels = df["standard_icd10"].to_list()
    label_strs = [_apply_label_scheme(c, label_scheme) for c in raw_labels]

    # Integer encode — drop records whose label is not in label2id
    # (can happen if the val/test split introduces a code unseen in train)
    filtered = [
        (t, label2id[ls])
        for t, ls in zip(texts, label_strs)
        if ls in label2id
    ]
    if len(filtered) < len(texts):
        n_dropped = len(texts) - len(filtered)
        print(f"   ⚠️  {n_dropped} records dropped (label not in label2id)")

    texts_f, labels_f = zip(*filtered) if filtered else ([], [])

    # Tokenise
    encodings = tokenizer(
        list(texts_f),
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors=None,
    )

    dataset = HFDataset.from_dict({
        "input_ids":      encodings["input_ids"],
        "attention_mask": encodings["attention_mask"],
        "labels":         list(labels_f),
    })
    dataset.set_format("torch")
    return dataset


def _split_dataframe(
    df: pl.DataFrame,
    label_col: str,
    seed: int = 42,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Stratified 80/10/10 train/val/test split.

    Mirrors the split logic used in all model notebooks:
        train_test_split(seed=42, stratify=labels, 80/20)
        then split the 20% into equal val and test

    Returns (train_df, val_df, test_df).
    """
    df_pd = df.to_pandas()

    train_pd, temp_pd = train_test_split(
        df_pd,
        test_size=0.2,
        random_state=seed,
        stratify=df_pd[label_col],
    )
    val_pd, test_pd = train_test_split(
        temp_pd,
        test_size=0.5,
        random_state=seed,
    )

    return (
        pl.from_pandas(train_pd),
        pl.from_pandas(val_pd),
        pl.from_pandas(test_pd),
    )


# ==============================================================================
# Flat training
# ==============================================================================

def train_flat(
    df: pl.DataFrame,
    cfg: dict,
    output_base: Path,
) -> TrainingResult:
    """
    Train a flat single-stage classifier.

    Replicates notebooks 02 (ICD-3) and 03 (flat ICD-10):
        - Stratified 80/10/10 split
        - EncoderAdapter fine-tuned via HuggingFace Trainer
        - Best model saved to output_base/model/
        - label_map.json saved to output_base/

    Parameters
    ----------
    df : pl.DataFrame
        Filtered Gold layer (apply _filter_gold before calling).
    cfg : dict
        Training configuration. Expected keys:
            model_name_or_path, label_scheme, num_epochs, learning_rate,
            batch_size, warmup_ratio, weight_decay, max_length,
            experiment_name, seed.
    output_base : Path
        Root output directory for this experiment.

    Returns
    -------
    TrainingResult
    """
    label_scheme = cfg["label_scheme"]
    seed         = cfg.get("seed", 42)
    max_length   = cfg.get("max_length", 512)

    print(f"\n── Flat training: {cfg['experiment_name']} ─────────────────────────")
    print(f"   Model:        {cfg['model_name_or_path']}")
    print(f"   Label scheme: {label_scheme} ({len(df):,} records)")

    # Build label maps from full dataset (not just train split)
    label2id, id2label = _build_label_maps(df["standard_icd10"].to_list(), label_scheme)
    print(f"   Num classes:  {len(label2id)}")

    # Label column for stratification
    label_col = "label_str"
    df = df.with_columns([
        pl.col("standard_icd10")
          .map_elements(lambda c: _apply_label_scheme(c, label_scheme), return_dtype=pl.String)
          .alias(label_col)
    ])

    # Drop any label that appears fewer than 2 times (can't stratify)
    counts = df.group_by(label_col).agg(pl.len().alias("n"))
    rare   = counts.filter(pl.col("n") < 2)[label_col].to_list()
    if rare:
        print(f"   ⚠️  Dropping {len(rare)} rare labels (n<2) for stratification")
        df = df.filter(~pl.col(label_col).is_in(rare))
        # Rebuild label maps after dropping rare
        label2id, id2label = _build_label_maps(
            df["standard_icd10"].to_list(), label_scheme
        )

    train_df, val_df, test_df = _split_dataframe(df, label_col, seed=seed)
    print(f"   Split: {len(train_df):,} / {len(val_df):,} / {len(test_df):,}")

    # Initialise adapter
    adapter = EncoderAdapter.from_pretrained(
        model_name_or_path=cfg["model_name_or_path"],
        label2id=label2id,
        id2label=id2label,
        device=cfg.get("device"),
    )

    # Tokenise datasets
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name_or_path"])

    print("   🔤 Tokenising datasets...")
    train_hf = _make_hf_dataset(train_df, label_scheme, label2id, tokenizer, max_length)
    val_hf   = _make_hf_dataset(val_df,   label_scheme, label2id, tokenizer, max_length)

    # Save test split for evaluate.py
    test_path = output_base / "test_split.parquet"
    test_df.write_parquet(test_path)
    print(f"   💾 Test split saved: {test_path.name} ({len(test_df):,} rows)")

    # Save label map
    output_base.mkdir(parents=True, exist_ok=True)
    label_map = {
        "label2id": label2id,
        "id2label": {str(k): v for k, v in id2label.items()},
        "label_scheme": label_scheme,
    }
    with open(output_base / "label_map.json", "w") as f:
        json.dump(label_map, f, indent=2)

    # Train
    model_dir = output_base / "model"
    result    = adapter.train(train_hf, val_hf, cfg, model_dir)

    # Save best model
    adapter.save(model_dir)

    # Persist result
    result_dict = {
        "experiment_name":   result.experiment_name,
        "best_epoch":        result.best_epoch,
        "best_val_accuracy": result.best_val_accuracy,
        "best_val_f1":       result.best_val_f1,
        "elapsed_seconds":   result.elapsed_seconds,
        "training_history":  result.training_history,
        "num_classes":       len(label2id),
        "label_scheme":      label_scheme,
        "model":             cfg["model_name_or_path"],
        "timestamp":         datetime.now().isoformat(),
    }
    with open(output_base / "train_result.json", "w") as f:
        json.dump(result_dict, f, indent=2)

    print(f"\n   {result.summary()}")
    return result


# ==============================================================================
# Hierarchical training
# ==============================================================================

def train_hierarchical_stage1(
    df: pl.DataFrame,
    cfg: dict,
    output_base: Path,
) -> TrainingResult:
    """
    Train Stage-1 chapter router (22-way ICD-10 chapter classification).

    Initialised from cfg['stage1_init'] if provided (mirrors E-003 which
    initialises from E-001 ICD-3 weights). Falls back to raw pretrained
    weights if not specified.
    """
    print(f"\n── Stage-1 Router: {cfg['experiment_name']} ──────────────────────────")

    init_path    = cfg.get("stage1_init", cfg["model_name_or_path"])
    label2id, id2label = _build_label_maps(df["standard_icd10"].to_list(), "chapter")
    label_col    = "chapter_label"

    df = df.with_columns([
        pl.col("standard_icd10")
          .map_elements(lambda c: c[0] if c else "UNKNOWN", return_dtype=pl.String)
          .alias(label_col)
    ])

    # Drop degenerate chapters
    counts = df.group_by(label_col).agg(pl.len().alias("n"))
    rare   = counts.filter(pl.col("n") < 2)[label_col].to_list()
    if rare:
        print(f"   ⚠️  Dropping chapters with n<2: {rare}")
        df = df.filter(~pl.col(label_col).is_in(rare))
        label2id, id2label = _build_label_maps(df["standard_icd10"].to_list(), "chapter")

    train_df, val_df, test_df = _split_dataframe(df, label_col, seed=cfg.get("seed", 42))
    print(f"   Chapters: {len(label2id)} | Split: {len(train_df):,} / {len(val_df):,} / {len(test_df):,}")

    adapter = EncoderAdapter.from_pretrained(
        model_name_or_path=init_path,
        label2id=label2id,
        id2label=id2label,
        device=cfg.get("device"),
    )

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(init_path)
    max_length = cfg.get("max_length", 512)

    train_hf = _make_hf_dataset(train_df, "chapter", label2id, tokenizer, max_length)
    val_hf   = _make_hf_dataset(val_df,   "chapter", label2id, tokenizer, max_length)

    s1_dir = output_base / "stage1"
    s1_dir.mkdir(parents=True, exist_ok=True)

    # Save test split
    test_df.write_parquet(s1_dir / "test_split.parquet")

    # Save label map
    with open(s1_dir / "label_map.json", "w") as f:
        json.dump({
            "label2id": label2id,
            "id2label": {str(k): v for k, v in id2label.items()},
            "label_scheme": "chapter",
        }, f, indent=2)

    s1_cfg      = dict(cfg)
    s1_cfg["experiment_name"] = f"{cfg['experiment_name']}_Stage1"
    result = adapter.train(train_hf, val_hf, s1_cfg, s1_dir / "model")
    adapter.save(s1_dir / "model")

    with open(s1_dir / "train_result.json", "w") as f:
        json.dump({
            "best_epoch": result.best_epoch,
            "elapsed_seconds": result.elapsed_seconds,
            "training_history": result.training_history,
        }, f, indent=2)

    print(f"   {result.summary()}")
    return result


def train_hierarchical_stage2(
    df: pl.DataFrame,
    cfg: dict,
    output_base: Path,
) -> dict[str, TrainingResult]:
    """
    Train per-chapter Stage-2 resolvers.

    Trains one EncoderAdapter per ICD-10 chapter present in df.
    Mirrors the per-chapter loop in notebooks 04/05 Phase 6.

    Skip chapters (P, Q, U by default) — defined in cfg['skip_chapters']
    or defaulting to the same set as notebooks 04/05 — are not trained.
    They receive majority-class fallback predictions at inference time.

    Initialisation: if cfg['stage2_init'] is set (a path to a saved E-002
    flat model directory), each chapter resolver is initialised from those
    weights rather than raw Bio_ClinicalBERT — this is the key innovation
    of E-004a.
    """
    print(f"\n── Stage-2 Resolvers: {cfg['experiment_name']} ──────────────────────")

    skip_chapters = set(cfg.get("skip_chapters", ["P", "Q", "U"]))
    stage2_init   = cfg.get("stage2_init", cfg["model_name_or_path"])
    max_length    = cfg.get("max_length", 512)
    seed          = cfg.get("seed", 42)

    from transformers import AutoTokenizer
    # Load tokenizer from stage2_init so it matches the initialisation weights
    tokenizer = AutoTokenizer.from_pretrained(stage2_init)

    chapters = sorted(df["standard_icd10"].map_elements(
        lambda c: c[0] if c else None, return_dtype=pl.String
    ).drop_nulls().unique().to_list())

    results: dict[str, TrainingResult] = {}
    skip_defaults: dict[str, str]      = {}

    for chapter in chapters:
        if chapter in skip_chapters:
            # Record majority-class fallback for this chapter
            ch_df        = df.filter(pl.col("standard_icd10").str.starts_with(chapter))
            majority_code = (
                ch_df.group_by("standard_icd10")
                .agg(pl.len().alias("n"))
                .sort("n", descending=True)
                .row(0)[0]
            )
            skip_defaults[chapter] = majority_code
            print(f"   ⏭  Chapter {chapter}: skipped (fallback={majority_code})")
            continue

        ch_df = df.filter(pl.col("standard_icd10").str.starts_with(chapter))

        if len(ch_df) < 4:
            print(f"   ⚠️  Chapter {chapter}: only {len(ch_df)} records — skipping")
            skip_defaults[chapter] = ch_df["standard_icd10"].mode()[0] if len(ch_df) > 0 else "UNKNOWN"
            continue

        label2id, id2label = _build_label_maps(ch_df["standard_icd10"].to_list(), "icd10")
        label_col = "label_str"
        ch_df = ch_df.with_columns([
            pl.col("standard_icd10").alias(label_col)
        ])

        # Drop labels with n<2 for stratification
        counts = ch_df.group_by(label_col).agg(pl.len().alias("n"))
        rare   = counts.filter(pl.col("n") < 2)[label_col].to_list()
        if rare:
            ch_df    = ch_df.filter(~pl.col(label_col).is_in(rare))
            label2id, id2label = _build_label_maps(ch_df["standard_icd10"].to_list(), "icd10")

        if len(ch_df) < 4:
            print(f"   ⚠️  Chapter {chapter}: too few records after rare label drop — skipping")
            continue

        train_df, val_df, test_df = _split_dataframe(ch_df, label_col, seed=seed)

        print(f"   📂 Chapter {chapter}: {len(label2id)} codes | "
              f"{len(train_df):,}/{len(val_df):,}/{len(test_df):,}")

        adapter = EncoderAdapter.from_pretrained(
            model_name_or_path=stage2_init,
            label2id=label2id,
            id2label=id2label,
            device=cfg.get("device"),
        )

        train_hf = _make_hf_dataset(train_df, "icd10", label2id, tokenizer, max_length)
        val_hf   = _make_hf_dataset(val_df,   "icd10", label2id, tokenizer, max_length)

        ch_dir = output_base / "stage2" / chapter
        ch_dir.mkdir(parents=True, exist_ok=True)

        # Save test split and label map per chapter
        test_df.write_parquet(ch_dir / "test_split.parquet")
        with open(ch_dir / "label_map.json", "w") as f:
            json.dump({
                "label2id": label2id,
                "id2label": {str(k): v for k, v in id2label.items()},
                "chapter":  chapter,
            }, f, indent=2)

        ch_cfg = dict(cfg)
        ch_cfg["experiment_name"] = f"{cfg['experiment_name']}_Stage2_{chapter}"
        result = adapter.train(train_hf, val_hf, ch_cfg, ch_dir / "model")
        adapter.save(ch_dir / "model")

        results[chapter] = result
        print(f"      ✅ {chapter}: best epoch {result.best_epoch}")

    # Save stage2 summary including skip chapter defaults
    s2_summary = {
        "skip_chapters": skip_defaults,
        "chapters_trained": list(results.keys()),
        "timestamp": datetime.now().isoformat(),
    }
    stage2_dir = output_base / "stage2"
    stage2_dir.mkdir(parents=True, exist_ok=True)
    with open(stage2_dir / "stage2_results.json", "w") as f:
        json.dump(s2_summary, f, indent=2)

    trained = len(results)
    skipped = len(skip_defaults)
    print(f"\n   ✅ Stage-2 complete: {trained} resolvers trained, {skipped} chapters skipped")
    return results


# ==============================================================================
# MLflow experiment logging
# ==============================================================================

def _log_mlflow(cfg: dict, result: TrainingResult) -> None:
    """
    Log training run to MLflow.

    Mirrors the Phase 9 (MLflow logging) cells in notebooks 02–05.
    Called after training completes. Non-fatal — MLflow failures are
    logged as warnings, not exceptions.
    """
    try:
        mlflow.set_tracking_uri(f"sqlite:///{PROJECT_ROOT}/mlflow.db")
        mlflow.set_experiment(cfg.get("experiment_name", "unknown"))

        with mlflow.start_run(run_name=cfg.get("experiment_name", "train")):
            # Hyperparameters
            mlflow.log_params({
                "model":            cfg.get("model_name_or_path"),
                "label_scheme":     cfg.get("label_scheme"),
                "num_epochs":       cfg.get("num_epochs"),
                "learning_rate":    cfg.get("learning_rate"),
                "batch_size":       cfg.get("batch_size"),
                "warmup_ratio":     cfg.get("warmup_ratio"),
                "weight_decay":     cfg.get("weight_decay"),
                "max_length":       cfg.get("max_length"),
            })
            # Metrics
            for entry in result.training_history:
                step = int(entry.get("epoch", 0))
                if entry.get("eval_loss") is not None:
                    mlflow.log_metric("eval_loss",    entry["eval_loss"],  step=step)
                if entry.get("train_loss") is not None:
                    mlflow.log_metric("train_loss",   entry["train_loss"], step=step)

            mlflow.log_metric("best_epoch",    result.best_epoch)
            mlflow.log_metric("elapsed_s",     result.elapsed_seconds)

        print("   ✅ MLflow run logged")
    except Exception as e:
        print(f"   ⚠️  MLflow logging failed (non-fatal): {e}")


# ==============================================================================
# Entry point
# ==============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fine-tune ICD-10 classifier using the EncoderAdapter interface."
    )
    p.add_argument(
        "--experiment", "-e",
        default="E-train",
        help="Experiment name — used for output directory and MLflow logging.",
    )
    p.add_argument(
        "--mode",
        choices=["flat", "hierarchical"],
        default="flat",
        help="Training mode: flat (single classifier) or hierarchical (two-stage).",
    )
    p.add_argument(
        "--stage",
        type=int,
        choices=[1, 2, 12],
        default=12,
        help="Which hierarchical stage(s) to train. 1=Stage-1, 2=Stage-2, 12=both (default).",
    )
    p.add_argument(
        "--label-scheme",
        choices=["icd3", "icd10"],
        default="icd10",
        help="Label granularity: icd3 (675 classes) or icd10 (1,926 classes).",
    )
    p.add_argument(
        "--model",
        default="emilyalsentzer/Bio_ClinicalBERT",
        help="HuggingFace model name or local path.",
    )
    p.add_argument(
        "--stage1-init",
        default=None,
        help="Path to saved model directory to initialise Stage-1 from.",
    )
    p.add_argument(
        "--stage2-init",
        default=None,
        help="Path to saved E-002 flat model to initialise Stage-2 from (key for E-004a).",
    )
    p.add_argument(
        "--code-filter",
        choices=["all", "billable"],
        default="billable",
        help="Filter Gold layer by code_status: 'billable' (default) or 'all'.",
    )
    p.add_argument("--epochs",        type=int,   default=10)
    p.add_argument("--lr",            type=float, default=2e-5)
    p.add_argument("--batch-size",    type=int,   default=8)
    p.add_argument("--warmup-ratio",  type=float, default=0.1)
    p.add_argument("--weight-decay",  type=float, default=0.01)
    p.add_argument("--max-length",    type=int,   default=512)
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Load data and validate without training.",
    )
    p.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Disable MLflow logging.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    start = datetime.now()
    print(f"\n{'='*70}")
    print(f"  train.py — ICD-10 Classifier Fine-Tuning")
    print(f"  Experiment: {args.experiment}")
    print(f"  Mode:       {args.mode}")
    print(f"  Started:    {start.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")

    # Build training config
    cfg = {
        "experiment_name":    args.experiment,
        "model_name_or_path": args.model,
        "label_scheme":       args.label_scheme,
        "num_epochs":         args.epochs,
        "learning_rate":      args.lr,
        "batch_size":         args.batch_size,
        "warmup_ratio":       args.warmup_ratio,
        "weight_decay":       args.weight_decay,
        "max_length":         args.max_length,
        "seed":               args.seed,
    }
    if args.stage1_init:
        cfg["stage1_init"] = args.stage1_init
    if args.stage2_init:
        cfg["stage2_init"] = args.stage2_init

    # Output directory
    output_base = (
        config.resolve_path("outputs", "evaluations")
        / args.experiment
    )
    output_base.mkdir(parents=True, exist_ok=True)

    # Load Gold layer
    print("\n── Loading Gold layer ───────────────────────────────────────────────")
    df = load_gold_parquet()
    df = _filter_gold(df, args.code_filter)
    print(f"   ✅ {len(df):,} records after filter='{args.code_filter}'")

    if args.dry_run:
        print("\n   ℹ️  --dry-run: exiting before training")
        # Quick schema check
        for col in ("apso_note", "standard_icd10", "code_status"):
            assert col in df.columns, f"❌ Missing column: {col}"
        print("   ✅ Schema OK")
        return

    # Route to appropriate training function
    results = {}

    if args.mode == "flat":
        result  = train_flat(df, cfg, output_base)
        results["flat"] = result
        if not args.no_mlflow:
            _log_mlflow(cfg, result)

    elif args.mode == "hierarchical":
        if args.stage in (1, 12):
            r1 = train_hierarchical_stage1(df, cfg, output_base)
            results["stage1"] = r1
            if not args.no_mlflow:
                _log_mlflow({**cfg, "experiment_name": f"{args.experiment}_Stage1"}, r1)

        if args.stage in (2, 12):
            r2_dict = train_hierarchical_stage2(df, cfg, output_base)
            results["stage2"] = r2_dict
            # Log aggregate Stage-2 metrics (one MLflow run per chapter is too granular)
            # scripts/evaluate.py handles per-chapter evaluation.

    elapsed = (datetime.now() - start).total_seconds()
    print(f"\n{'='*70}")
    print(f"  ✅ Training complete — {elapsed:.0f}s")
    print(f"  Outputs: {output_base}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()