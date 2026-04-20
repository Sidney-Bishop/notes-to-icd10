#!/usr/bin/env python3
"""
predict.py — ICD-10 Inference Entrypoint
=========================================
Runs a single clinical note through the calibrated two-stage hierarchical
pipeline and returns a structured prediction with a Use Case B routing
decision (auto-code vs. human review).

Usage
-----
    # Pipe a note from stdin
    echo "Patient presents with..." | uv run python scripts/predict.py

    # Pass a note directly
    uv run python scripts/predict.py --note "S: Patient reports..."

    # Read from a file
    uv run python scripts/predict.py --file note.txt

    # Return top-3 predictions, lower threshold, JSON output
    uv run python scripts/predict.py --note "..." --top-k 3 --threshold 0.6 --json

    # Use a specific experiment
    uv run python scripts/predict.py --note "..." \\
        --experiment E-004a_Hierarchical_E002Init \\
        --stage1-experiment E-003_Hierarchical_ICD10

Output (default, human-readable)
---------------------------------
    ══════════════════════════════════════════════════════════════════════
      ICD-10 Prediction
    ══════════════════════════════════════════════════════════════════════
      Chapter:    M — Musculoskeletal system
      Decision:   ✅ AUTO-CODE  (confidence 84.3% ≥ threshold 70.0%)

      Rank  Code      Confidence  Description
      ────  ────────  ──────────  ─────────────────────────────────────
         1  M25.562      84.3%    Pain in left knee
         2  M79.622       8.1%    Pain in left upper arm
         3  M54.5         4.2%    Low back pain
    ══════════════════════════════════════════════════════════════════════

Output (--json)
---------------
    {
      "top_code": "M25.562",
      "confidence": 0.843,
      "chapter": "M",
      "decision": "AUTO_CODE",
      "threshold": 0.7,
      "predictions": [
        {"rank": 1, "code": "M25.562", "confidence": 0.843},
        {"rank": 2, "code": "M79.622", "confidence": 0.081},
        {"rank": 3, "code": "M54.5",   "confidence": 0.042}
      ],
      "stage2_source": "resolver"
    }
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Project path setup — allows running from any working directory
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference import HierarchicalPredictor  # noqa: E402

# ---------------------------------------------------------------------------
# ICD-10 chapter descriptions
# ---------------------------------------------------------------------------
CHAPTER_DESCRIPTIONS: dict[str, str] = {
    "A": "Certain infectious and parasitic diseases",
    "B": "Certain infectious and parasitic diseases",
    "C": "Neoplasms",
    "D": "Diseases of blood and blood-forming organs",
    "E": "Endocrine, nutritional and metabolic diseases",
    "F": "Mental and behavioural disorders",
    "G": "Diseases of the nervous system",
    "H": "Diseases of the eye / ear",
    "I": "Diseases of the circulatory system",
    "J": "Diseases of the respiratory system",
    "K": "Diseases of the digestive system",
    "L": "Diseases of the skin and subcutaneous tissue",
    "M": "Diseases of the musculoskeletal system",
    "N": "Diseases of the genitourinary system",
    "O": "Pregnancy, childbirth and the puerperium",
    "P": "Certain conditions originating in the perinatal period",
    "Q": "Congenital malformations and chromosomal abnormalities",
    "R": "Symptoms, signs and abnormal clinical findings",
    "S": "Injury, poisoning — external causes",
    "T": "Poisoning by drugs and toxic substances",
    "U": "Codes for special purposes (COVID-19 etc.)",
    "Z": "Factors influencing health status and contact with services",
}


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _chapter_label(chapter: str) -> str:
    desc = CHAPTER_DESCRIPTIONS.get(chapter, "Unknown chapter")
    return f"{chapter} — {desc}"


def _decision(confidence: float, threshold: float) -> tuple[str, str]:
    """Return (decision_code, display_string)."""
    if confidence >= threshold:
        return "AUTO_CODE", f"✅ AUTO-CODE  (confidence {confidence:.1%} ≥ threshold {threshold:.1%})"
    else:
        return "HUMAN_REVIEW", f"👤 HUMAN REVIEW  (confidence {confidence:.1%} < threshold {threshold:.1%})"


def _print_human(
    predictions: list[dict],
    chapter: str,
    stage2_source: str,
    threshold: float,
) -> None:
    top = predictions[0]
    confidence  = top["confidence"]
    decision_code, decision_str = _decision(confidence, threshold)

    w = 70
    print()
    print("═" * w)
    print("  ICD-10 Prediction")
    print("═" * w)
    print(f"  Chapter:    {_chapter_label(chapter)}")
    print(f"  Decision:   {decision_str}")
    if stage2_source != "resolver":
        print(f"  Note:       Fallback code used (chapter {chapter} has no resolver)")
    print()
    print(f"  {'Rank':<5} {'Code':<10} {'Confidence':>10}  ")
    print(f"  {'────':<5} {'────────':<10} {'──────────':>10}  ")
    for p in predictions:
        marker = "◀" if p["rank"] == 1 else " "
        print(f"  {p['rank']:<5} {p['code']:<10} {p['confidence']:>9.1%}  {marker}")
    print("═" * w)
    print()


def _print_json(
    predictions: list[dict],
    chapter: str,
    stage2_source: str,
    threshold: float,
) -> None:
    top = predictions[0]
    confidence = top["confidence"]
    decision_code, _ = _decision(confidence, threshold)

    output = {
        "top_code":      top["code"],
        "confidence":    round(confidence, 6),
        "chapter":       chapter,
        "decision":      decision_code,
        "threshold":     threshold,
        "predictions":   predictions,
        "stage2_source": stage2_source,
    }
    print(json.dumps(output, indent=2))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="ICD-10 prediction for a single clinical note.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Input
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "--note", "-n",
        type=str,
        help="Clinical note text (SOAP format preferred).",
    )
    input_group.add_argument(
        "--file", "-f",
        type=Path,
        help="Path to a plain-text file containing the clinical note.",
    )

    # Prediction options
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=5,
        help="Number of top ICD-10 predictions to return (default: 5).",
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=0.7,
        help="Confidence threshold for auto-coding decision (default: 0.7).",
    )

    # Model selection
    parser.add_argument(
        "--experiment",
        default="E-004a_Hierarchical_E002Init",
        help="Stage-2 experiment name (default: E-004a_Hierarchical_E002Init).",
    )
    parser.add_argument(
        "--stage1-experiment",
        default="E-003_Hierarchical_ICD10",
        help="Stage-1 experiment name (default: E-003_Hierarchical_ICD10).",
    )

    # Output format
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output structured JSON instead of human-readable text.",
    )

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Read input note
    # ------------------------------------------------------------------
    if args.note:
        note = args.note
    elif args.file:
        note = args.file.read_text(encoding="utf-8")
    elif not sys.stdin.isatty():
        note = sys.stdin.read()
    else:
        parser.print_help()
        sys.exit(0)

    note = note.strip()
    if not note:
        print("Error: empty note.", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------
    # Load pipeline (suppress model loading noise when using --json)
    # ------------------------------------------------------------------
    if args.json:
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            predictor = HierarchicalPredictor(
                experiment_name=args.experiment,
                stage1_experiment=args.stage1_experiment,
            )
    else:
        predictor = HierarchicalPredictor(
            experiment_name=args.experiment,
            stage1_experiment=args.stage1_experiment,
        )

    # ------------------------------------------------------------------
    # Run inference
    # ------------------------------------------------------------------
    result = predictor.predict(note, top_k=args.top_k)

    codes         = result["codes"]
    scores        = result["scores"]
    chapter       = result.get("chapter", "?")
    stage2_source = result.get("stage2_source", "resolver")

    predictions = [
        {"rank": i + 1, "code": code, "confidence": round(score, 6)}
        for i, (code, score) in enumerate(zip(codes, scores))
    ]

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    if args.json:
        _print_json(predictions, chapter, stage2_source, args.threshold)
    else:
        _print_human(predictions, chapter, stage2_source, args.threshold)


if __name__ == "__main__":
    main()