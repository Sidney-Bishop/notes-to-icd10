"""
prepare_splits.py — Generate deterministic per-chapter train/val/test splits

Implements the reproducible split strategy from Prj_Overview.md section
"Reproducibility Crisis - 23 April 2026".

Creates stage2/{CHAPTER}/{train_split,val_split,test_split}.parquet for all
chapters, stratified by chapter with seed=42. This ensures evaluate.py has
the exact test sets it expects and makes the pipeline reproducible.

Usage:
    uv run python scripts/prepare_splits.py
    uv run python scripts/prepare_splits.py --experiment E-006_Hierarchical_Clean
    uv run python scripts/prepare_splits.py --gold-path data/gold/medsynth_gold_augmented.parquet
"""

import sys
import argparse
import polars as pl
from pathlib import Path
from sklearn.model_selection import train_test_split

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

from src.config import config  # noqa: E402

def prepare_splits(
    gold_path: Path,
    experiment_name: str = "E-006_Hierarchical_Clean",
    seed: int = 42,
    test_size: float = 0.2,
    val_size: float = 0.5,  # 0.5 of the 0.2 = 0.1 overall
) -> None:
    """
    Create deterministic per-chapter train/val/test splits.

    Strategy matches notebooks: stratified 80/10/10 split by chapter.
    Chapter = first letter of standard_icd10.

    Parameters

    gold_path : Path
        Path to medsynth_gold_augmented.parquet or medsynth_gold.parquet
    experiment_name : str
        Output directory under outputs/evaluations/
    seed : int
        Random seed for reproducibility. Default 42 per overview doc.
    test_size : float
        Fraction for test+val combined. Default 0.2 gives 80/20 train/rest
    val_size : float
        Fraction of test_size to use for val. Default 0.5 gives 80/10/10 overall
    """
    print(f"\n{'='*70}")
    print(f" prepare_splits.py — Deterministic Split Generation")
    print(f" Gold path: {gold_path}")
    print(f" Experiment: {experiment_name}")
    print(f" Seed: {seed}")
    print(f"{'='*70}")

    if not gold_path.exists():
        raise FileNotFoundError(
            f"Gold layer not found at {gold_path}\n"
            f"Run scripts/prepare_data.py or scripts/augment.py first."
        )

    # Load gold layer
    print(f"\n📥 Loading gold layer...")
    df = pl.read_parquet(gold_path)
    print(f" Total records: {len(df):,}")

    # Extract chapter for stratification
    df = df.with_columns(
        pl.col("standard_icd10").str.slice(0, 1).alias("chapter")
    )

    chapters = sorted(df["chapter"].unique().to_list())
    print(f" Chapters found: {', '.join(chapters)} ({len(chapters)} total)")

    # Output base directory
    base_dir = config.resolve_path("outputs", "evaluations") / experiment_name / "stage2"
    base_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    total_train, total_val, total_test = 0, 0, 0

    # Split per chapter to maintain exact stratification
    for ch in chapters:
        ch_df = df.filter(pl.col("chapter") == ch)
        n_total = len(ch_df)

        if n_total < 3:
            print(f" ⚠️  Warning: Chapter {ch} has only {n_total} records, skipping stratification")
            # Put all in train if too few
            train_df = ch_df
            val_df = ch_df.head(0)
            test_df = ch_df.head(0)
        else:
            # First split: train vs (val+test)
            train_df, temp_df = train_test_split(
                ch_df.to_pandas(),
                test_size=test_size,
                random_state=seed,
                stratify=ch_df["standard_icd10"].to_pandas(),
            )

            # Second split: val vs test
            if len(temp_df) < 2:
                val_df = temp_df
                test_df = temp_df.head(0)
            else:
                # Only stratify if every class has at least 2 members
                # temp_df is pandas here — value_counts() returns a Series
                strat_counts = temp_df["standard_icd10"].value_counts()
                can_stratify = (
                    temp_df["standard_icd10"].nunique() > 1
                    and strat_counts.min() >= 2
                )
                val_df, test_df = train_test_split(
                    temp_df,
                    test_size=val_size,
                    random_state=seed,
                    stratify=temp_df["standard_icd10"] if can_stratify else None,
                )

            # Convert back to polars
            train_df = pl.from_pandas(train_df)
            val_df = pl.from_pandas(val_df)
            test_df = pl.from_pandas(test_df)

        # Write splits
        ch_dir = base_dir / ch
        ch_dir.mkdir(parents=True, exist_ok=True)

        train_df.write_parquet(ch_dir / "train_split.parquet")
        val_df.write_parquet(ch_dir / "val_split.parquet")
        test_df.write_parquet(ch_dir / "test_split.parquet")

        n_train, n_val, n_test = len(train_df), len(val_df), len(test_df)
        total_train += n_train
        total_val += n_val
        total_test += n_test

        summary_rows.append({
            "chapter": ch,
            "n_train": n_train,
            "n_val": n_val,
            "n_test": n_test,
            "n_total": n_total,
        })

    # Print summary table
    print(f"\n{'='*70}")
    print(f" Split Summary")
    print(f"{'='*70}")
    print(f"{'Chapter':<8} {'Train':>8} {'Val':>8} {'Test':>8} {'Total':>8}")
    print(f"{'-'*70}")
    for row in summary_rows:
        print(
            f"{row['chapter']:<8} "
            f"{row['n_train']:>8,} "
            f"{row['n_val']:>8,} "
            f"{row['n_test']:>8,} "
            f"{row['n_total']:>8,}"
        )
    print(f"{'-'*70}")
    print(
        f"{'TOTAL':<8} "
        f"{total_train:>8,} "
        f"{total_val:>8,} "
        f"{total_test:>8,} "
        f"{total_train + total_val + total_test:>8,}"
    )
    print(f"{'='*70}")

    # Verify test count matches expectation
    expected_test = int(len(df) * test_size * val_size)
    if abs(total_test - expected_test) > len(chapters):  # allow ±1 per chapter due to rounding
        print(f"\n⚠️  Warning: Test count {total_test} differs from expected ~{expected_test}")
        print(f" This is normal due to per-chapter stratification and small-n chapters.")
    else:
        print(f"\n✅ Test set size {total_test:,} matches expected distribution.")

    print(f"\n💾 Splits written to: {base_dir}")
    print(f" Next: uv run python scripts/train.py --experiment {experiment_name} --mode hierarchical")

def main():
    parser = argparse.ArgumentParser(description="Generate deterministic per-chapter splits")
    parser.add_argument(
        "--gold-path",
        type=Path,
        default=None,
        help="Path to gold parquet. Defaults to data/gold/medsynth_gold_augmented.parquet if exists, else medsynth_gold.parquet",
    )
    parser.add_argument(
        "--experiment",
        default="E-006_Hierarchical_Clean",
        help="Experiment name for output directory",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()

    # Resolve gold path — always use PROJECT_ROOT-relative paths
    gold_dir = config.resolve_path("data", "gold")
    if args.gold_path is None:
        augmented = gold_dir / "medsynth_gold_augmented.parquet"
        # fallback: find most recent non-augmented parquet
        candidates = sorted([p for p in gold_dir.glob("*.parquet") if "augmented" not in p.name])
        base = candidates[-1] if candidates else gold_dir / "medsynth_gold.parquet"
        args.gold_path = augmented if augmented.exists() else base
    else:
        # If a relative path was given, resolve it against PROJECT_ROOT
        # so it works regardless of the user's current working directory
        if not args.gold_path.is_absolute():
            args.gold_path = PROJECT_ROOT / args.gold_path
        # Also try resolving against gold_dir if the file still isn't found
        if not args.gold_path.exists():
            alt = gold_dir / args.gold_path.name
            if alt.exists():
                args.gold_path = alt

    prepare_splits(
        gold_path=args.gold_path,
        experiment_name=args.experiment,
        seed=args.seed,
    )

if __name__ == "__main__":
    main()