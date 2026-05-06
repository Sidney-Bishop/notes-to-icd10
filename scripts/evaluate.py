"""
evaluate.py — Evaluation script for trained ICD-10 classifiers.

Computes the full evaluation suite for any saved EncoderAdapter — flat or
hierarchical — including the confidence calibration metrics required by the
Use Case B design (automated coding with human review of exceptions).

Metrics computed

    Standard classification
        Top-1 accuracy, Top-5 accuracy, Macro F1, Weighted F1
        Per-class accuracy and support (saved to CSV)
        Confusion matrix (saved as Parquet for large label spaces)

    Hierarchical-specific
        Chapter routing accuracy (Stage-1)
        Within-chapter accuracy (Stage-2)
        End-to-end accuracy (combined)
        Per-chapter accuracy breakdown

    Confidence calibration (Use Case B)
        Expected Calibration Error (ECE) — how well top-1 confidence
        correlates with empirical accuracy (key for auto-code threshold)
        Calibration curve (saved to JSON for visualisation)
        Auto-code coverage at threshold τ: fraction of records where
        confidence ≥ τ (and accuracy of those auto-coded records)
        Threshold sweep: coverage vs accuracy trade-off table

    Decision support metrics
        Precision-at-k (k=1,3,5): does the correct code appear in top-k?
        Selective accuracy curve: accuracy vs fraction of records covered

Usage examples

    # Evaluate flat model
    uv run python scripts/evaluate.py \\
        --experiment E-001_Baseline_ICD3 \\
        --mode flat

    # Evaluate hierarchical best model (E-005a)
    uv run python scripts/evaluate.py \\
        --experiment E-005a_Hierarchical_ICD10_Extended \\
        --mode hierarchical \\
        --stage1-experiment E-003_Hierarchical_ICD10

    # Evaluate with custom confidence threshold
    uv run python scripts/evaluate.py \\
        --experiment E-005a_Hierarchical_ICD10_Extended \\
        --mode hierarchical \\
        --threshold 0.7

    # Quick validation on a subset
    uv run python scripts/evaluate.py \\
        --experiment E-001_Baseline_ICD3 \\
        --mode flat \\
        --sample 500

Output layout

    outputs/evaluations/{experiment_name}/eval/
        summary.json — all scalar metrics
        calibration.json — calibration curve data
        threshold_sweep.json — coverage/accuracy at each τ
        per_class_metrics.csv — per-label accuracy, F1, support
        predictions.parquet — id, true_code, pred_code, confidence
        chapter_accuracy.json — per-chapter E2E breakdown (hierarchical)
"""

import sys
import json
import argparse
import warnings
import numpy as np
import polars as pl
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
)

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------

# Bootstrap: locate project root before src/ is on sys.path.
# find_project_root() is defined in src/config.py — single source of truth.
def _find_root() -> "Path":
    current = Path.cwd()
    while current != current.parent:
        if (current / "artifacts.yaml").exists():
            return current.resolve()
        current = current.parent
    raise FileNotFoundError("artifacts.yaml not found — run from within the project tree.")

PROJECT_ROOT = _find_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Now src/ is importable — use the canonical implementation going forward.
from src.config import find_project_root  # noqa: E402 (import after sys.path setup)

from src.config import config # noqa: E402
from src.adapters import EncoderAdapter, PredictionResult # noqa: E402
from src.inference import HierarchicalPredictor # noqa: E402

# ==============================================================================
# Confidence calibration utilities
# ==============================================================================

def expected_calibration_error(
    confidences: np.ndarray,
    correct: np.ndarray,
    n_bins: int = 15,
) -> tuple[float, list[dict]]:
    """
    Compute Expected Calibration Error (ECE) and per-bin data.

    ECE measures how well the model's top-1 confidence predicts actual accuracy.
    A perfectly calibrated model has ECE=0: when it says 70% confidence, it is
    correct 70% of the time.

    Parameters

    confidences : np.ndarray shape (N,)
        Top-1 softmax probability for each prediction.
    correct : np.ndarray shape (N,) bool
        Whether the top-1 prediction was correct.
    n_bins : int
        Number of equal-width bins in [0, 1].

    Returns

    ece : float
        Expected Calibration Error.
    bins : list[dict]
        Per-bin data: {bin_lower, bin_upper, accuracy, avg_confidence, count}.
    """
    n = len(confidences)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bins = []
    ece_sum = 0.0

    for i in range(n_bins):
        lower, upper = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (confidences >= lower) & (confidences < upper)
        count = mask.sum()

        if count == 0:
            bins.append({
                "bin_lower": float(lower),
                "bin_upper": float(upper),
                "accuracy": None,
                "avg_confidence": None,
                "count": 0,
            })
            continue

        bin_acc = float(correct[mask].mean())
        bin_conf = float(confidences[mask].mean())
        ece_sum += (count / n) * abs(bin_acc - bin_conf)

        bins.append({
            "bin_lower": float(lower),
            "bin_upper": float(upper),
            "accuracy": bin_acc,
            "avg_confidence": bin_conf,
            "count": int(count),
        })

    return float(ece_sum), bins

def threshold_sweep(
    confidences: np.ndarray,
    correct: np.ndarray,
    thresholds: list[float] | None = None,
) -> list[dict]:
    """
    Coverage and accuracy at each confidence threshold.

    For Use Case B: given a threshold τ, records with confidence ≥ τ are
    auto-coded; records below are sent to human review. This sweep produces
    the coverage–accuracy trade-off table for threshold selection.

    Parameters

    confidences, correct : np.ndarray
        As in expected_calibration_error.
    thresholds : list[float] or None
        Thresholds to evaluate. Defaults to 0.05 steps from 0.5 to 1.0.

    Returns

    list[dict]
        One entry per threshold: {threshold, coverage, accuracy, n_auto, n_review}.
    """
    if thresholds is None:
        thresholds = [round(t, 2) for t in np.arange(0.5, 1.01, 0.05)]

    n = len(confidences)
    rows = []
    for tau in thresholds:
        mask = confidences >= tau
        n_auto = int(mask.sum())
        n_review = n - n_auto
        coverage = n_auto / n
        accuracy = float(correct[mask].mean()) if n_auto > 0 else 0.0
        rows.append({
            "threshold": float(tau),
            "coverage": round(coverage, 4),
            "accuracy": round(accuracy, 4),
            "n_auto": n_auto,
            "n_review": n_review,
        })
    return rows

# ==============================================================================
# Flat evaluation
# ==============================================================================

def evaluate_flat(
    experiment_name: str,
    sample: int | None = None,
    threshold: float = 0.7,
    batch_size: int = 32,
) -> dict:
    """
    Evaluate a flat (single-stage) EncoderAdapter.

    Loads the test split saved by scripts/train.py and runs inference using
    the predict_batch() interface for efficiency.

    Parameters

    experiment_name : str
        Name of the experiment (matches the output directory name).
    sample : int or None
        If set, evaluate on a random sample of this size (for quick validation).
    threshold : float
        Confidence threshold for the threshold sweep.
    batch_size : int
        Batch size for predict_batch().

    Returns

    dict
        All scalar metrics (also written to eval/summary.json).
    """
    exp_dir = config.resolve_path("outputs", "evaluations") / experiment_name
    model_dir = exp_dir / "model"

    if not model_dir.exists():
        raise FileNotFoundError(
            f"Model not found at {model_dir}\n"
            f"Run scripts/train.py --experiment {experiment_name} first."
        )

    test_path = exp_dir / "test_split.parquet"
    if not test_path.exists():
        raise FileNotFoundError(
            f"Test split not found at {test_path}\n"
            f"Re-run training to regenerate the test split."
        )

    print(f"\n── Flat evaluation: {experiment_name} ──────────────────────────────")
    adapter = EncoderAdapter.load(model_dir)
    print(f" ✅ Model loaded: {len(adapter.id2label)} classes | device={adapter.device}")

    # Load test split
    test_df = pl.read_parquet(test_path)
    if sample is not None:
        test_df = test_df.sample(n=min(sample, len(test_df)), seed=42)
    print(f" 📊 Test records: {len(test_df):,}")

    # Load label scheme from label_map.json
    lmap_path = exp_dir / "label_map.json"
    label_scheme = "icd10"
    if lmap_path.exists():
        with open(lmap_path) as f:
            lmap = json.load(f)
        label_scheme = lmap.get("label_scheme", "icd10")

    # True labels (apply same label scheme used during training)
    from scripts.train import _apply_label_scheme
    true_labels = [
        _apply_label_scheme(c, label_scheme)
        for c in test_df["standard_icd10"].to_list()
    ]

    # Batch prediction — tokenise apso_note directly.
    # The test split already contains preprocessed apso_note text (APSO-flipped
    # and ICD-10-redacted by the Gold layer pipeline). We must NOT call
    # prepare_inference_input() again — the SOAP regex extraction would fail
    # on already-reordered text and corrupt the input.
    print(f" 🔮 Running predictions (batch_size={batch_size})...")
    notes = test_df["apso_note"].to_list()
    all_results = []
    for i in range(0, len(notes), batch_size):
        batch = notes[i : i + batch_size]
        inputs = adapter.tokenizer(
            batch,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt",
        )
        inputs = {
            k: v.to(adapter.device)
            for k, v in inputs.items()
            if k in adapter.tokenizer.model_input_names
        }
        import torch
        with torch.no_grad():
            logits = adapter.model(**inputs).logits
            probs_batch = torch.softmax(logits, dim=-1).cpu().numpy()

        for probs in probs_batch:
            top_k_actual = min(5, len(adapter.id2label))
            top_idx = probs.argsort()[::-1][:top_k_actual]
            codes = [adapter.id2label.get(int(j), "UNKNOWN") for j in top_idx]
            scores = [float(probs[j]) for j in top_idx]
            from src.adapters import PredictionResult
            all_results.append(PredictionResult(
                codes=codes, scores=scores, confidence=scores[0]
            ))
    results = all_results

    pred_labels = [r.top1_code for r in results]
    confidences = np.array([r.confidence for r in results])
    top5_lists = [r.codes for r in results]

    # Core metrics
    correct = np.array([t == p for t, p in zip(true_labels, pred_labels)])
    top1_acc = float(correct.mean())
    top5_correct = [t in top5 for t, top5 in zip(true_labels, top5_lists)]
    top5_acc = float(np.mean(top5_correct))
    macro_f1 = f1_score(true_labels, pred_labels, average="macro", zero_division=0)
    weighted_f1 = f1_score(true_labels, pred_labels, average="weighted", zero_division=0)

    print(f"\n 📈 Top-1 Accuracy: {top1_acc:.3f}")
    print(f" 📈 Top-5 Accuracy: {top5_acc:.3f}")
    print(f" 📈 Macro F1: {macro_f1:.3f}")
    print(f" 📈 Weighted F1: {weighted_f1:.3f}")

    # Confidence calibration
    ece, cal_bins = expected_calibration_error(confidences, correct)
    sweep = threshold_sweep(confidences, correct)
    tau_row = next((r for r in sweep if r["threshold"] == threshold), None)

    print(f"\n 📈 ECE: {ece:.4f}")
    if tau_row:
        print(f" 📈 Coverage@τ={threshold}: {tau_row['coverage']:.1%} "
              f"(accuracy={tau_row['accuracy']:.3f})")

    # Per-class metrics
    report = classification_report(
        true_labels, pred_labels, output_dict=True, zero_division=0
    )
    per_class_rows = []
    for label, metrics in report.items():
        if isinstance(metrics, dict):
            per_class_rows.append({
                "label": label,
                "precision": metrics.get("precision"),
                "recall": metrics.get("recall"),
                "f1": metrics.get("f1-score"),
                "support": int(metrics.get("support", 0)),
            })
    per_class_df = pl.DataFrame(per_class_rows)

    # Predictions Parquet (for downstream analysis)
    preds_df = pl.DataFrame({
        "id": test_df["id"].to_list() if "id" in test_df.columns else list(range(len(test_df))),
        "true_code": true_labels,
        "pred_code": pred_labels,
        "confidence": confidences.tolist(),
        "correct": correct.tolist(),
    })

    # Write outputs
    eval_dir = exp_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "experiment_name": experiment_name,
        "label_scheme": label_scheme,
        "num_classes": len(adapter.id2label),
        "n_test": len(test_df),
        "top1_accuracy": round(top1_acc, 4),
        "top5_accuracy": round(top5_acc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "ece": round(ece, 4),
        "threshold": threshold,
        "coverage_at_threshold": tau_row["coverage"] if tau_row else None,
        "accuracy_at_threshold": tau_row["accuracy"] if tau_row else None,
        "timestamp": datetime.now().isoformat(),
    }

    with open(eval_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(eval_dir / "calibration.json", "w") as f:
        json.dump(cal_bins, f, indent=2)
    with open(eval_dir / "threshold_sweep.json", "w") as f:
        json.dump(sweep, f, indent=2)

    per_class_df.write_csv(eval_dir / "per_class_metrics.csv")
    preds_df.write_parquet(eval_dir / "predictions.parquet")

    print(f"\n 💾 Results saved: {eval_dir}")
    return summary

# ==============================================================================
# Hierarchical evaluation
# ==============================================================================

def evaluate_hierarchical(
    experiment_name: str,
    stage1_experiment: str = "E-003_Hierarchical_ICD10",
    sample: int | None = None,
    threshold: float = 0.7,
) -> dict:
    """
    Evaluate the full two-stage hierarchical pipeline.

    Uses HierarchicalPredictor from src/inference.py — the same class used
    for production inference — to ensure evaluation exactly mirrors deployment.

    Computes:
        - Stage-1 (chapter routing) accuracy
        - Stage-2 (within-chapter) accuracy
        - End-to-end accuracy (correct chapter AND correct code)
        - Per-chapter accuracy breakdown
        - Full confidence calibration suite
        - Threshold sweep for Use Case B decision support

    The test set is assembled from the per-chapter test splits saved by
    scripts/train.py Stage-2 training.
    """
    print(f"\n── Hierarchical evaluation: {experiment_name} ──────────────────────")

    # Load HierarchicalPredictor (loads all models)
    predictor = HierarchicalPredictor(
        experiment_name=experiment_name,
        stage1_experiment=stage1_experiment,
    )

    # Assemble test records from per-chapter test splits
    # Also load Stage-1 test split for chapter routing evaluation
    exp_dir = config.resolve_path("outputs", "evaluations") / experiment_name
    s1_dir = (
        config.resolve_path("outputs", "evaluations")
        / stage1_experiment
        / "stage1"
    )
    s2_dir = exp_dir / "stage2"

    records = [] # {id, true_code, true_chapter, apso_note}

    # Stage-2 test splits — one Parquet per chapter
    for ch_dir in sorted(s2_dir.iterdir()):
        if not ch_dir.is_dir():
            continue
        test_path = ch_dir / "test_split.parquet"
        if not test_path.exists():
            raise FileNotFoundError(f"{experiment_name}: missing test_split.parquet for chapter {ch_dir.name} at {test_path}. Run prepare_splits.py first.")
        ch_df = pl.read_parquet(test_path)
        for row in ch_df.iter_rows(named=True):
            records.append({
                "id": row.get("id", ""),
                "true_code": row.get("standard_icd10", ""),
                "true_chapter": row.get("standard_icd10", "X")[0],
                "apso_note": row.get("apso_note", ""),
            })

    if not records:
        raise FileNotFoundError(
            f"No per-chapter test splits found under {s2_dir}.\n"
            f"Run scripts/train.py --mode hierarchical --stage 2 first."
        )

    if sample is not None:
        import random
        random.seed(42)
        records = random.sample(records, min(sample, len(records)))

    print(f" 📊 Test records: {len(records):,}")
    print(f" 🔮 Running hierarchical predictions...")

    # Predict
    pred_codes = []
    pred_chapters = []
    confidences = []
    correct_e2e = []
    correct_ch = []

    for rec in records:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning,message=".*ClinicalNoteInput.*")
            result = predictor.predict(rec["apso_note"], top_k=5, preprocessed=True)
            
        pred_code = result["codes"][0]
        pred_chapter = result.get("chapter", "UNKNOWN")
        confidence = result["scores"][0]

        pred_codes.append(pred_code)
        pred_chapters.append(pred_chapter)
        confidences.append(confidence)
        correct_e2e.append(pred_code == rec["true_code"])
        correct_ch.append(pred_chapter == rec["true_chapter"])

    confidences = np.array(confidences)
    correct_e2e = np.array(correct_e2e)
    correct_ch = np.array(correct_ch)

    # Stage-1 routing accuracy
    ch_accuracy = float(correct_ch.mean())

    # End-to-end accuracy
    e2e_accuracy = float(correct_e2e.mean())

    # Within-chapter accuracy (only for records where Stage-1 was correct)
    within_ch = float(correct_e2e[correct_ch].mean()) if correct_ch.sum() > 0 else 0.0

    true_codes = [r["true_code"] for r in records]
    macro_f1 = f1_score(true_codes, pred_codes, average="macro", zero_division=0)

    # Per-chapter accuracy
    chapter_groups: dict[str, list] = defaultdict(list)
    for rec, correct in zip(records, correct_e2e):
        chapter_groups[rec["true_chapter"]].append(bool(correct))

    chapter_accuracy = {
        ch: {
            "accuracy": round(sum(vals) / len(vals), 4),
            "n": len(vals),
        }
        for ch, vals in sorted(chapter_groups.items())
    }

    print(f"\n 📈 Stage-1 (chapter) accuracy: {ch_accuracy:.3f}")
    print(f" 📈 Stage-2 (within-chapter): {within_ch:.3f}")
    print(f" 📈 End-to-end accuracy: {e2e_accuracy:.3f}")
    print(f" 📈 Macro F1: {macro_f1:.3f}")

    # Confidence calibration
    ece, cal_bins = expected_calibration_error(confidences, correct_e2e)
    sweep = threshold_sweep(confidences, correct_e2e)
    tau_row = next((r for r in sweep if r["threshold"] == threshold), None)

    print(f"\n 📈 ECE: {ece:.4f}")
    if tau_row:
        print(f" 📈 Coverage@τ={threshold}: {tau_row['coverage']:.1%} "
              f"(accuracy={tau_row['accuracy']:.3f})")

    # Per-chapter summary (abbreviated)
    print("\n 📊 Per-chapter accuracy:")
    for ch, stats in chapter_accuracy.items():
        print(f" {ch}: {stats['accuracy']:.3f} (n={stats['n']:,})")

    # Predictions Parquet
    preds_df = pl.DataFrame({
        "id": [r["id"] for r in records],
        "true_code": [r["true_code"] for r in records],
        "true_chapter": [r["true_chapter"] for r in records],
        "pred_code": pred_codes,
        "pred_chapter": pred_chapters,
        "confidence": confidences.tolist(),
        "correct_e2e": correct_e2e.tolist(),
        "correct_ch": correct_ch.tolist(),
    })

    # Write outputs
    eval_dir = exp_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "experiment_name": experiment_name,
        "stage1_experiment": stage1_experiment,
        "n_test": len(records),
        "stage1_accuracy": round(ch_accuracy, 4),
        "within_chapter_accuracy": round(within_ch, 4),
        "e2e_accuracy": round(e2e_accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "ece": round(ece, 4),
        "threshold": threshold,
        "coverage_at_threshold": tau_row["coverage"] if tau_row else None,
        "accuracy_at_threshold": tau_row["accuracy"] if tau_row else None,
        "chapter_accuracy": chapter_accuracy,
        "timestamp": datetime.now().isoformat(),
    }

    with open(eval_dir / "summary.json", "w") as f: json.dump(summary, f, indent=2)
    with open(eval_dir / "calibration.json", "w") as f: json.dump(cal_bins, f, indent=2)
    with open(eval_dir / "threshold_sweep.json", "w") as f: json.dump(sweep, f, indent=2)
    with open(eval_dir / "chapter_accuracy.json", "w") as f: json.dump(chapter_accuracy, f, indent=2)
    preds_df.write_parquet(eval_dir / "predictions.parquet")

    print(f"\n 💾 Results saved: {eval_dir}")
    return summary

# ==============================================================================
# Entry point
# ==============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate a trained ICD-10 classifier with confidence calibration."
    )
    p.add_argument(
        "--experiment", "-e",
        required=True,
        help="Experiment name (matches the output directory).",
    )
    p.add_argument(
        "--mode",
        choices=["flat", "hierarchical"],
        default="flat",
    )
    p.add_argument(
        "--stage1-experiment",
        default="E-003_Hierarchical_ICD10",
        help="Stage-1 experiment name (hierarchical mode only).",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Confidence threshold for Use Case B coverage/accuracy reporting.",
    )
    p.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Evaluate on a random sample of N records (for quick validation).",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for flat model predict_batch() (flat mode only).",
    )
    return p.parse_args()

def main() -> None:
    args = parse_args()

    start = datetime.now()
    print(f"\n{'='*70}")
    print(f" evaluate.py — ICD-10 Classifier Evaluation")
    print(f" Experiment: {args.experiment}")
    print(f" Mode: {args.mode}")
    print(f" Threshold: {args.threshold}")
    print(f" Started: {start.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")

    # Log start to experiment registry
    from src.experiment_logger import ExperimentLogger
    exp_logger = ExperimentLogger(args.experiment, script="evaluate.py")
    exp_logger.log_start("evaluate", params={
        "mode":             args.mode,
        "stage1_experiment": getattr(args, "stage1_experiment", "—"),
        "threshold":        args.threshold,
        "sample":           args.sample or "all",
    })

    try:
        if args.mode == "flat":
            summary = evaluate_flat(
                experiment_name=args.experiment,
                sample=args.sample,
                threshold=args.threshold,
                batch_size=args.batch_size,
            )
        else:
            summary = evaluate_hierarchical(
                experiment_name=args.experiment,
                stage1_experiment=args.stage1_experiment,
                sample=args.sample,
                threshold=args.threshold,
            )
    except Exception as e:
        exp_logger.log_failed("evaluate", reason=str(e))
        raise

    elapsed = (datetime.now() - start).total_seconds()

    # Log results
    eval_dir = config.resolve_path("outputs", "evaluations") / args.experiment / "eval"
    exp_logger.log_complete("evaluate", artifacts={
        "eval_dir": str(eval_dir),
    })
    metrics = {}
    if args.mode == "flat":
        metrics = {
            "e2e_accuracy":   summary.get("top1_accuracy", 0),
            "macro_f1":       summary.get("macro_f1", 0),
            "ece":            summary.get("ece", 0),
            "coverage_07":    summary.get("coverage_at_threshold", 0),
        }
    else:
        metrics = {
            "stage1_accuracy": summary.get("stage1_accuracy", 0),
            "stage2_accuracy": summary.get("within_chapter_accuracy", 0),
            "e2e_accuracy":    summary.get("e2e_accuracy", 0),
            "macro_f1":        summary.get("macro_f1", 0),
            "ece":             summary.get("ece", 0),
            "coverage_07":     summary.get("coverage_at_threshold",
                               summary.get("coverage_07", 0)),
        }
    exp_logger.log_results(metrics)

    print(f"\n{'='*70}")
    if args.mode == "flat":
        print(f" Top-1 Accuracy: {summary['top1_accuracy']:.3f} | "
              f"Macro F1: {summary['macro_f1']:.3f} | "
              f"ECE: {summary['ece']:.4f}")
    else:
        print(f" E2E Accuracy: {summary['e2e_accuracy']:.3f} | "
              f"Macro F1: {summary['macro_f1']:.3f} | "
              f"ECE: {summary['ece']:.4f}")
    print(f" Elapsed: {elapsed:.1f}s")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()