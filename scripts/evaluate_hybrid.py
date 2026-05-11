"""
evaluate_hybrid.py — Hybrid E2E evaluation with per-chapter resolver overrides.

Loads a base hierarchical experiment (default: E-010_40ep_E002Init) and
substitutes specific chapter resolvers from override experiments.

Primary use case: measure the full-system impact of E-014 SupCon Z fine-tuning
by running E-010 for all 21 chapters and E-014 for the Z chapter.

Usage
-----
  uv run python scripts/evaluate_hybrid.py [options]

  --base-experiment     Base experiment for all resolvers (default: E-010_40ep_E002Init)
  --stage1-experiment   Stage-1 router experiment (default: E-003_Hierarchical_ICD10)
  --override            Chapter=Experiment override, repeatable.
                        e.g. --override Z=E-014_SupCon_Z
  --threshold           Confidence threshold for Coverage@τ (default: 0.7)
  --experiment          Output experiment name for saving results (default: auto)

Example
-------
  uv run python scripts/evaluate_hybrid.py \\
      --base-experiment E-010_40ep_E002Init \\
      --stage1-experiment E-003_Hierarchical_ICD10 \\
      --override Z=E-014_SupCon_Z \\
      --threshold 0.7
"""

import sys
sys.path.insert(0, '.')

import argparse
import json
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config import config
from src.inference import HierarchicalPredictor
from src.paths import ExperimentPaths


# ==============================================================================
# Helpers
# ==============================================================================

def load_override_resolver(chapter: str, override_experiment: str, device) -> dict:
    """
    Load a chapter resolver from an override experiment.

    Returns dict with keys: model, tokenizer, id2label, temperature.
    """
    paths = ExperimentPaths(override_experiment)
    hf_dir = paths.stage2_model_dir(chapter)
    label_map_path = paths.stage2_label_map(chapter)

    if hf_dir is None or not hf_dir.exists():
        raise FileNotFoundError(
            f"Override resolver for chapter {chapter} not found "
            f"in {override_experiment}. Expected at: {paths.stage2_base / chapter}"
        )
    if label_map_path is None or not label_map_path.exists():
        raise FileNotFoundError(
            f"label_map.json not found for chapter {chapter} "
            f"in {override_experiment}"
        )

    with open(label_map_path) as f:
        lmap = json.load(f)

    model = AutoModelForSequenceClassification.from_pretrained(str(hf_dir)).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(str(hf_dir))
    id2label = {int(k): v for k, v in lmap["id2label"].items()}

    # Load temperature if calibrated
    temp_path = paths.stage2_temperature_existing(chapter)
    temperature = 1.0
    if temp_path is not None:
        with open(temp_path) as f:
            temperature = json.load(f).get("temperature", 1.0)

    print(f" ✅ Override loaded: chapter {chapter} ← {override_experiment} "
          f"(T={temperature:.4f})")
    return {
        "model":       model,
        "tokenizer":   tokenizer,
        "id2label":    id2label,
        "temperature": temperature,
    }


def assemble_test_records(base_experiment: str) -> list[dict]:
    """
    Assemble the full test set from per-chapter test splits of the base experiment.
    """
    exp_dir = config.resolve_path("outputs", "evaluations") / base_experiment
    s2_dir = exp_dir / "stage2"

    records = []
    for ch_dir in sorted(s2_dir.iterdir()):
        if not ch_dir.is_dir():
            continue
        test_path = ch_dir / "test_split.parquet"
        if not test_path.exists():
            print(f" ⚠️  No test_split.parquet for chapter {ch_dir.name} — skipping")
            continue
        ch_df = pl.read_parquet(test_path)
        for row in ch_df.iter_rows(named=True):
            records.append({
                "id":           row.get("id", ""),
                "true_code":    row.get("standard_icd10", ""),
                "true_chapter": row.get("standard_icd10", "X")[0],
                "apso_note":    row.get("apso_note", ""),
            })

    if not records:
        raise FileNotFoundError(
            f"No per-chapter test splits found under {s2_dir}. "
            f"Ensure {base_experiment} has been fully trained."
        )
    return records


# ==============================================================================
# Main
# ==============================================================================

def main(cfg: dict) -> None:
    print(f"\n{'='*70}")
    print(f"  evaluate_hybrid.py — Hybrid Hierarchical Evaluation")
    print(f"  Base:        {cfg['base_experiment']}")
    print(f"  Stage-1:     {cfg['stage1_experiment']}")
    for ch, exp in cfg["overrides"].items():
        print(f"  Override:    Chapter {ch} ← {exp}")
    print(f"  Threshold:   τ={cfg['threshold']}")
    print(f"{'='*70}\n")

    # ── Load base predictor ────────────────────────────────────────────────────
    predictor = HierarchicalPredictor(
        experiment_name=cfg["base_experiment"],
        stage1_experiment=cfg["stage1_experiment"],
    )

    # ── Apply chapter overrides ────────────────────────────────────────────────
    if cfg["overrides"]:
        print(f"\n📥 Applying chapter overrides...")
        for chapter, override_exp in cfg["overrides"].items():
            override = load_override_resolver(chapter, override_exp, predictor.device)
            predictor.stage2_models[chapter]      = override["model"]
            predictor.stage2_tokenizers[chapter]  = override["tokenizer"]
            predictor.stage2_id2label[chapter]    = override["id2label"]
            predictor.stage2_temperatures[chapter] = override["temperature"]

    # ── Assemble test set ──────────────────────────────────────────────────────
    records = assemble_test_records(cfg["base_experiment"])
    print(f"\n 📊 Test records: {len(records):,}")

    # ── Run predictions ────────────────────────────────────────────────────────
    print(f" 🔮 Running hybrid predictions...")
    pred_codes, pred_chapters, confidences = [], [], []
    correct_e2e, correct_ch = [], []

    for rec in records:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning,
                                    message=".*ClinicalNoteInput.*")
            result = predictor.predict(rec["apso_note"], top_k=5, preprocessed=True)

        pred_code    = result["codes"][0]
        pred_chapter = result.get("chapter", "UNKNOWN")
        confidence   = result["scores"][0]

        pred_codes.append(pred_code)
        pred_chapters.append(pred_chapter)
        confidences.append(confidence)
        correct_e2e.append(pred_code == rec["true_code"])
        correct_ch.append(pred_chapter == rec["true_chapter"])

    confidences = np.array(confidences)
    correct_e2e = np.array(correct_e2e)
    correct_ch  = np.array(correct_ch)

    # ── Metrics ────────────────────────────────────────────────────────────────
    ch_accuracy   = float(correct_ch.mean())
    e2e_accuracy  = float(correct_e2e.mean())

    # Within-chapter accuracy (only where Stage-1 was correct)
    ch_correct_mask = np.array(correct_ch)
    within_ch = float(correct_e2e[ch_correct_mask].mean()) if ch_correct_mask.any() else 0.0

    # ECE (15 bins)
    n_bins = 15
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confidences >= bin_edges[i]) & (confidences < bin_edges[i + 1])
        if mask.sum() > 0:
            bin_acc  = correct_e2e[mask].mean()
            bin_conf = confidences[mask].mean()
            ece += (mask.sum() / len(confidences)) * abs(bin_acc - bin_conf)

    # Coverage@τ
    tau = cfg["threshold"]
    covered = confidences >= tau
    coverage = float(covered.mean())
    acc_on_covered = float(correct_e2e[covered].mean()) if covered.any() else 0.0

    # Per-chapter accuracy
    chapter_groups: dict[str, list] = defaultdict(list)
    for rec, correct in zip(records, correct_e2e):
        chapter_groups[rec["true_chapter"]].append(bool(correct))
    chapter_accuracy = {
        ch: {
            "accuracy": round(float(np.mean(vals)), 4),
            "n": len(vals),
        }
        for ch, vals in sorted(chapter_groups.items())
    }

    # ── Print results ──────────────────────────────────────────────────────────
    print(f"\n 📈 Stage-1 (chapter) accuracy: {ch_accuracy:.3f}")
    print(f" 📈 Stage-2 (within-chapter): {within_ch:.3f}")
    print(f" 📈 End-to-end accuracy: {e2e_accuracy:.3f}")
    print(f" 📈 Macro F1: (see saved results)")
    print(f"\n 📈 ECE: {ece:.4f}")
    print(f" 📈 Coverage@τ={tau}: {coverage:.1%} (accuracy={acc_on_covered:.3f})")
    print(f"\n 📊 Per-chapter accuracy:")
    for ch, stats in chapter_accuracy.items():
        marker = " ← OVERRIDE" if ch in cfg["overrides"] else ""
        print(f"  {ch}: {stats['accuracy']:.3f} (n={stats['n']}){marker}")

    # ── Save results ───────────────────────────────────────────────────────────
    out_name = cfg.get("output_experiment") or (
        cfg["base_experiment"] + "_hybrid_" + "_".join(
            f"{ch}-{exp.split('_')[0]}" for ch, exp in cfg["overrides"].items()
        )
    )
    out_dir = config.resolve_path("outputs", "evaluations") / out_name / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "experiment":           out_name,
        "base_experiment":      cfg["base_experiment"],
        "stage1_experiment":    cfg["stage1_experiment"],
        "overrides":            cfg["overrides"],
        "n_records":            len(records),
        "e2e_accuracy":         round(e2e_accuracy, 4),
        "chapter_accuracy":     round(ch_accuracy, 4),
        "within_chapter_accuracy": round(within_ch, 4),
        "ece":                  round(ece, 4),
        f"coverage_at_{tau}":   round(coverage, 4),
        f"accuracy_on_covered": round(acc_on_covered, 4),
        "chapter_accuracy_detail": chapter_accuracy,
    }

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n 💾 Results saved: {out_dir}")
    print(f"\n{'='*70}")
    print(f" E2E Accuracy: {e2e_accuracy:.3f} | ECE: {ece:.4f} | Coverage@{tau}: {coverage:.1%}")
    print(f"{'='*70}")


# ==============================================================================
# CLI
# ==============================================================================

def parse_args() -> dict:
    parser = argparse.ArgumentParser(
        description="Hybrid hierarchical evaluation with per-chapter overrides"
    )
    parser.add_argument(
        "--base-experiment",
        default="E-010_40ep_E002Init",
    )
    parser.add_argument(
        "--stage1-experiment",
        default="E-003_Hierarchical_ICD10",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="CHAPTER=EXPERIMENT",
        help="Chapter override, e.g. Z=E-014_SupCon_Z. Repeatable.",
    )
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--output-experiment", default=None)
    args = parser.parse_args()

    # Parse overrides
    overrides = {}
    for ov in args.override:
        if "=" not in ov:
            parser.error(f"--override must be CHAPTER=EXPERIMENT, got: {ov}")
        ch, exp = ov.split("=", 1)
        overrides[ch.upper()] = exp

    return {
        "base_experiment":   args.base_experiment,
        "stage1_experiment": args.stage1_experiment,
        "overrides":         overrides,
        "threshold":         args.threshold,
        "output_experiment": args.output_experiment,
    }


if __name__ == "__main__":
    cfg = parse_args()
    main(cfg)
