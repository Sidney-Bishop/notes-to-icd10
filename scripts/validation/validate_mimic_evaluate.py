"""
scripts/validation/validate_mimic_evaluate.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MIMIC-IV-Note Validation Pipeline — Evaluation (Gold → Results)

PURPOSE
-------
Loads the Gold layer produced by validate_mimic_prepare.py and runs the
E-010 hierarchical model (optionally with E-014 SupCon Z override), computing:

    - End-to-end accuracy
    - Macro F1
    - ECE (calibration)
    - Coverage@τ=0.7
    - Per-chapter accuracy breakdown
    - Comparison to synthetic MedSynth test set results

Results are written to outputs/evaluations/mimic_iv_validation/ and logged
to the project DuckDB database.

RESTRICTED DATA — DO NOT COMMIT DERIVED DATA
---------------------------------------------
MIMIC-IV is a credentialed dataset. Results that could indirectly reveal
patient information (e.g. raw predictions, note IDs) are NOT committed.
Only aggregate statistics are saved to the git-tracked results file.

USAGE
-----
  # Standard evaluation (E-010 only)
  uv run python scripts/validation/validate_mimic_evaluate.py

  # With SupCon Z override (E-010 + E-014)
  uv run python scripts/validation/validate_mimic_evaluate.py --supcon-z

  # Custom threshold
  uv run python scripts/validation/validate_mimic_evaluate.py --threshold 0.8
"""

import sys
import argparse
import json
import time
import warnings
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import polars as pl
from sklearn.metrics import f1_score

from src.config import config
from src.inference import HierarchicalPredictor
from src.paths import ExperimentPaths


# ── Reference results from MedSynth test set (E-010, 4-run mean) ──────────
MEDSYNTH_REFERENCE = {
    "experiment":       "E-010_40ep_E002Init",
    "e2e_accuracy":     0.858,
    "macro_f1":         0.790,
    "ece":              0.031,
    "coverage_at_0.7":  0.864,
    "n_records":        972,
    "note":             "4-run mean, MedSynth synthetic test set",
}

MEDSYNTH_HYBRID_REFERENCE = {
    "experiment":       "E-010 + E-014 SupCon Z hybrid",
    "e2e_accuracy":     0.867,
    "macro_f1":         None,
    "ece":              0.027,
    "coverage_at_0.7":  0.882,
    "n_records":        972,
    "note":             "E-014 SupCon Z override, MedSynth synthetic test set",
}

MEDSYNTH_REFERENCE_DELEAKED = {
    "experiment":       "E-021_Hier_ClinicalBERT_Deleaked",
    "e2e_accuracy":     0.592,
    "macro_f1":         0.477,
    "ece":              0.088,
    "coverage_at_0.7":  0.553,
    "n_records":        966,
    "note":             "de-leaked hierarchical ClinicalBERT, content-addressed split (run 20260603_200854); supersedes E-016 0.567 on the old position-addressed split",
}


def load_gold_layer() -> pl.DataFrame:
    """Load the prepared MIMIC-IV Gold layer."""
    gold_path = config.project_root / "data" / "mimic" / "gold" / "mimic_gold.parquet"
    if not gold_path.exists():
        raise FileNotFoundError(
            f"Gold layer not found: {gold_path}\n"
            f"Run validate_mimic_prepare.py first."
        )
    df = pl.read_parquet(gold_path)
    print(f" ✅ Gold layer loaded: {len(df):,} records, {df['icd_chapter'].n_unique()} chapters")
    return df


def load_predictor(
    use_supcon_z: bool,
    base_experiment: str = "E-010_40ep_E002Init",
    stage1_experiment: str = "E-003_Hierarchical_ICD10",
) -> HierarchicalPredictor:
    """Load the hierarchical predictor, optionally with SupCon Z override."""
    predictor = HierarchicalPredictor(
        experiment_name=base_experiment,
        stage1_experiment=stage1_experiment,
    )

    if use_supcon_z:
        import json
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        z_paths = ExperimentPaths("E-014_SupCon_Z")
        z_model_dir = z_paths.stage2_base / "Z"
        z_label_map = z_paths.stage2_base / "Z" / "label_map.json"
        z_temp_path = z_paths.stage2_base / "Z" / "temperature.json"

        if not z_model_dir.exists():
            raise FileNotFoundError(
                f"E-014 SupCon Z resolver not found at {z_model_dir}. "
                f"Run train_supcon_z.py first."
            )

        model = AutoModelForSequenceClassification.from_pretrained(
            str(z_model_dir)
        ).to(predictor.device)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(str(z_model_dir))

        with open(z_label_map) as f:
            lmap = json.load(f)
        id2label = {int(k): v for k, v in lmap["id2label"].items()}
        temperature = 1.0
        if z_temp_path.exists():
            with open(z_temp_path) as f:
                temperature = json.load(f).get("temperature", 1.0)

        predictor.stage2_models["Z"]      = model
        predictor.stage2_tokenizers["Z"]  = tokenizer
        predictor.stage2_id2label["Z"]    = id2label
        predictor.stage2_temperatures["Z"] = temperature
        print(f" ✅ SupCon Z override applied (T={temperature:.4f})")

    return predictor


def run_evaluation(
    df: pl.DataFrame,
    predictor: HierarchicalPredictor,
    threshold: float,
) -> dict:
    """
    Run inference on the Gold layer and compute all metrics.

    Parameters
    ----------
    df : pl.DataFrame
        Gold layer with columns: apso_note, icd_code, icd_chapter.
    predictor : HierarchicalPredictor
        Loaded model.
    threshold : float
        Confidence threshold for Coverage@τ.

    Returns
    -------
    dict
        Full results dictionary with aggregate and per-chapter metrics.
    """
    notes      = df["apso_note"].to_list()
    true_codes = df["icd_code"].to_list()
    chapters   = df["icd_chapter"].to_list()
    n          = len(notes)

    pred_codes, confidences = [], []
    print(f" Running inference on {n:,} notes...", flush=True)
    t0 = time.perf_counter()

    for i, note in enumerate(notes):
        if i % 500 == 0 and i > 0:
            elapsed = time.perf_counter() - t0
            rate = i / elapsed
            eta = (n - i) / rate
            print(f"  {i:,}/{n:,} ({i/n:.0%}) — {rate:.0f} notes/sec — ETA {eta:.0f}s",
                  flush=True)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            result = predictor.predict(note, top_k=5, preprocessed=True)

        pred_codes.append(result["codes"][0])
        confidences.append(result["scores"][0])

    elapsed = time.perf_counter() - t0
    print(f" Inference complete: {elapsed:.0f}s ({n/elapsed:.0f} notes/sec)")

    # ── Metrics ────────────────────────────────────────────────────────────
    confidences  = np.array(confidences)
    correct_e2e  = np.array([p == t for p, t in zip(pred_codes, true_codes)])

    e2e_accuracy  = float(correct_e2e.mean())
    macro_f1      = f1_score(true_codes, pred_codes, average="macro", zero_division=0)

    # ECE (15 bins)
    n_bins = 15
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confidences >= bin_edges[i]) & (confidences < bin_edges[i + 1])
        if mask.sum() > 0:
            ece += (mask.sum() / n) * abs(correct_e2e[mask].mean() - confidences[mask].mean())

    # Coverage@τ
    covered          = confidences >= threshold
    coverage         = float(covered.mean())
    acc_on_covered   = float(correct_e2e[covered].mean()) if covered.any() else 0.0

    # Per-chapter accuracy
    chapter_groups = defaultdict(list)
    for ch, correct in zip(chapters, correct_e2e):
        chapter_groups[ch].append(bool(correct))

    chapter_accuracy = {
        ch: {"accuracy": round(float(np.mean(vals)), 4), "n": len(vals)}
        for ch, vals in sorted(chapter_groups.items())
    }

    return {
        "n_records":         n,
        "e2e_accuracy":      round(e2e_accuracy, 4),
        "macro_f1":          round(macro_f1, 4),
        "ece":               round(ece, 4),
        f"coverage_at_{threshold}": round(coverage, 4),
        "accuracy_on_covered": round(acc_on_covered, 4),
        "chapter_accuracy":  chapter_accuracy,
        "inference_seconds": round(elapsed, 1),
    }


def print_comparison(results: dict, reference: dict, threshold: float) -> None:
    """Print a formatted comparison table: MIMIC vs MedSynth."""
    print(f"\n{'='*70}")
    print(f"  MIMIC-IV vs MedSynth Comparison")
    print(f"{'='*70}")
    print(f"  {'Metric':<25} {'MedSynth':>12} {'MIMIC-IV':>12} {'Delta':>10}")
    print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*10}")

    metrics = [
        ("E2E Accuracy",       "e2e_accuracy",          f".1%"),
        ("Macro F1",           "macro_f1",               f".3f"),
        ("ECE",                "ece",                    f".4f"),
        (f"Coverage@{threshold}", f"coverage_at_{threshold}", f".1%"),
    ]

    for label, key, fmt in metrics:
        ref_val  = reference.get(key)
        mimic_val = results.get(key)
        if ref_val is None or mimic_val is None:
            print(f"  {label:<25} {'—':>12} {mimic_val if mimic_val is not None else '—':>12}")
            continue
        delta = mimic_val - ref_val
        sign  = "+" if delta >= 0 else ""
        print(f"  {label:<25} {ref_val:>11{fmt}} {mimic_val:>11{fmt}} {sign}{delta:>8{fmt}}")

    print(f"\n  MIMIC-IV records:  {results['n_records']:,}")
    print(f"  MedSynth records:  {reference['n_records']:,}")
    print(f"\n  Per-chapter accuracy (MIMIC-IV):")
    for ch, stats in results["chapter_accuracy"].items():
        ref_ch = None  # Could add per-chapter MedSynth reference if needed
        print(f"    {ch}: {stats['accuracy']:.3f} (n={stats['n']})")


def save_results(
    results: dict,
    reference: dict,
    use_supcon_z: bool,
    threshold: float,
    base_experiment: str = "E-010_40ep_E002Init",
) -> Path:
    """
    Save aggregate results (no patient-level data) to the evaluation directory.

    Only aggregate statistics are saved — no note IDs, no predictions,
    no text. This ensures no MIMIC data leaks into git-tracked files.

    The output directory is keyed on the experiment id-prefix
    (mimic_iv_validation_<PREFIX>/) so that separate runs (e.g. the leaky and
    de-leaked publication runs) write to DISTINCT directories and do not
    overwrite each other's result. PREFIX = base_experiment.split('_')[0].
    """
    exp_prefix = base_experiment.split("_")[0]
    out_dir = (config.project_root / "outputs" / "evaluations"
               / f"mimic_iv_validation_{exp_prefix}")
    out_dir.mkdir(parents=True, exist_ok=True)

    output = {
        "experiment":         base_experiment + (" + E-014_SupCon_Z" if use_supcon_z else ""),
        "validation_dataset": "MIMIC-IV-Note v2.2 + MIMIC-IV clinical v2.2",
        "threshold":          threshold,
        "mimic_results":      {k: v for k, v in results.items() if k != "chapter_accuracy"},
        "medsynth_reference": reference,
        "chapter_accuracy":   results["chapter_accuracy"],
        "data_access":        "PhysioNet credentialed — DUA v1.5.0 signed",
        "note":               "Aggregate statistics only — no patient-level data",
    }

    out_path = out_dir / ("summary_supcon_z.json" if use_supcon_z else "summary.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n 💾 Results saved: {out_path}")
    print(f"    (Aggregate statistics only — no patient data)")
    return out_path


def main(args) -> None:
    t_start = time.perf_counter()
    experiment_label = args.base_experiment + (" + E-014 SupCon Z" if args.supcon_z else "")
    if args.deleaked_reference:
        reference = MEDSYNTH_REFERENCE_DELEAKED
    else:
        reference = MEDSYNTH_HYBRID_REFERENCE if args.supcon_z else MEDSYNTH_REFERENCE

    print("=" * 70)
    print(f"  validate_mimic_evaluate.py — MIMIC-IV Validation")
    print(f"  Experiment:  {experiment_label}")
    print(f"  Threshold:   τ={args.threshold}")
    print("=" * 70)

    # Load data and model
    df = load_gold_layer()
    print(f"\n📥 Loading predictor ({experiment_label})...")
    predictor = load_predictor(
        use_supcon_z=args.supcon_z,
        base_experiment=args.base_experiment,
        stage1_experiment=args.stage1_experiment,
    )

    # Run evaluation
    print(f"\n── Evaluation ───────────────────────────────────────────────────────")
    results = run_evaluation(df, predictor, threshold=args.threshold)

    # Print comparison
    print_comparison(results, reference, threshold=args.threshold)

    # Save results (aggregate only)
    out_path = save_results(results, reference, args.supcon_z, args.threshold,
                            base_experiment=args.base_experiment)

    # Log to DuckDB
    with config.duckdb_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS mimic_validation_results (
                experiment      VARCHAR,
                n_records       INTEGER,
                e2e_accuracy    DOUBLE,
                macro_f1        DOUBLE,
                ece             DOUBLE,
                coverage        DOUBLE,
                created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        coverage_key = f"coverage_at_{args.threshold}"
        conn.execute("""
            INSERT INTO mimic_validation_results
            (experiment, n_records, e2e_accuracy, macro_f1, ece, coverage)
            VALUES (?, ?, ?, ?, ?, ?)
        """, [
            experiment_label,
            results["n_records"],
            results["e2e_accuracy"],
            results["macro_f1"],
            results["ece"],
            results.get(coverage_key, 0.0),
        ])

    elapsed = time.perf_counter() - t_start
    print(f"\n{'='*70}")
    print(f" ✅ Evaluation complete — {elapsed:.0f}s")
    print(f" E2E: {results['e2e_accuracy']:.1%} | "
          f"F1: {results['macro_f1']:.3f} | "
          f"ECE: {results['ece']:.4f} | "
          f"Coverage@{args.threshold}: {results.get(f'coverage_at_{args.threshold}', 0):.1%}")
    print(f"{'='*70}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="MIMIC-IV validation — evaluation"
    )
    parser.add_argument(
        "--supcon-z",
        action="store_true",
        help="Use E-014 SupCon Z resolver override for chapter Z"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Confidence threshold for Coverage@τ (default: 0.7)"
    )
    parser.add_argument(
        "--base-experiment",
        default="E-010_40ep_E002Init",
        help="Base hierarchical experiment for all resolvers (default: E-010_40ep_E002Init)"
    )
    parser.add_argument(
        "--stage1-experiment",
        default="E-003_Hierarchical_ICD10",
        help="Stage-1 router experiment (default: E-003_Hierarchical_ICD10)"
    )
    parser.add_argument(
        "--deleaked-reference",
        action="store_true",
        help="Compare against the de-leaked E-021 reference (0.592) instead of leaky E-010 (0.858)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
