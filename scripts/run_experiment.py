#!/usr/bin/env python3
"""
run_experiment.py — Experiment Orchestration Driver
=====================================================
Runs the full train → calibrate → evaluate pipeline for one or more
model configurations. Writes a results summary to CSV and prints a
comparison table at the end.

Skip logic: each stage is skipped if its output already exists, so
re-running after an interruption picks up where it left off.

Usage
-----
    # Single experiment
    uv run python scripts/run_experiment.py \\
        --experiment E-008_Longformer \\
        --model yikuan8/Clinical-Longformer \\
        --stage2-init outputs/evaluations/E-002_FullICD10_ClinicalBERT/model/model \\
        --gold-path data/gold/medsynth_gold_augmented.parquet

    # Multiple models in one run (sequential)
    uv run python scripts/run_experiment.py \\
        --experiments E-008_Longformer E-009_RoBERTa \\
        --models yikuan8/Clinical-Longformer allenai/biomed_roberta_base \\
        --stage2-init outputs/evaluations/E-002_FullICD10_ClinicalBERT/model/model \\
        --gold-path data/gold/medsynth_gold_augmented.parquet

    # Dry-run: show what would run without executing
    uv run python scripts/run_experiment.py \\
        --experiment E-008_Longformer \\
        --model yikuan8/Clinical-Longformer \\
        --dry-run

    # Re-evaluate existing experiments (skip training)
    uv run python scripts/run_experiment.py \\
        --experiments E-005c_Merged_ZO E-006_BiomedBERT E-007_PubMedBERT \\
        --skip-train \\
        --stage1-experiment E-003_Hierarchical_ICD10

Output
------
    outputs/experiment_results.csv   — appended after each experiment
    Console comparison table         — printed at end of run
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_CSV  = PROJECT_ROOT / "outputs" / "experiment_results.csv"

BANNER = "=" * 70

CSV_FIELDS = [
    "experiment", "model", "timestamp",
    "e2e_accuracy", "macro_f1", "ece",
    "coverage_07", "accuracy_on_covered",
    "stage1_accuracy", "stage2_accuracy",
    "train_seconds", "total_seconds", "status",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def log(msg: str, indent: int = 0) -> None:
    prefix = "  " * indent
    print(f"{prefix}{msg}", flush=True)


def run_cmd(cmd: list[str], label: str, dry_run: bool = False) -> tuple[bool, float]:
    """
    Run a subprocess command. Returns (success, elapsed_seconds).
    Streams output live so the user sees progress.
    """
    log(f"▶  {label}")
    log(f"   {' '.join(cmd)}", indent=1)

    if dry_run:
        log("   [DRY RUN — skipped]", indent=1)
        return True, 0.0

    t0 = time.time()
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    elapsed = time.time() - t0

    if result.returncode != 0:
        log(f"   ✗ FAILED (exit {result.returncode}) after {elapsed:.1f}s", indent=1)
        return False, elapsed

    log(f"   ✓ Done in {elapsed:.1f}s", indent=1)
    return True, elapsed


def should_skip_train(exp_dir: Path) -> bool:
    """True if stage2 training outputs already exist for all expected chapters."""
    stage2 = exp_dir / "stage2"
    if not stage2.exists():
        return False
    chapters = [d for d in stage2.iterdir() if d.is_dir()]
    # At minimum A and Z should be present for a complete run
    trained = {d.name for d in chapters}
    return "A" in trained and "Z" in trained


def should_skip_calibrate(exp_dir: Path) -> bool:
    """True if calibration_report.json already exists."""
    return (exp_dir / "calibration_report.json").exists()


def should_skip_evaluate(exp_dir: Path) -> bool:
    """True if eval results already exist."""
    return (exp_dir / "eval").exists()


def read_eval_results(exp_dir: Path) -> dict:
    """Read evaluation metrics from the saved eval JSON."""
    eval_dir = exp_dir / "eval"
    if not eval_dir.exists():
        return {}

    # Look for the summary JSON written by evaluate.py
    for candidate in ["summary.json", "results.json", "eval_summary.json"]:
        f = eval_dir / candidate
        if f.exists():
            return json.loads(f.read_text())

    # Fall back: scan for any JSON in eval/
    jsons = list(eval_dir.glob("*.json"))
    if jsons:
        return json.loads(jsons[0].read_text())

    return {}


def append_results_csv(row: dict) -> None:
    """Append a result row to the shared CSV, creating it with headers if needed."""
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    write_header = not RESULTS_CSV.exists()

    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def print_comparison_table(rows: list[dict]) -> None:
    """Print a formatted comparison table of all experiments in this run."""
    if not rows:
        return

    print(f"\n{BANNER}")
    print(f"  Results Summary")
    print(BANNER)

    # Header
    print(f"  {'Experiment':<35} {'E2E Acc':>8} {'F1':>7} {'ECE':>6} "
          f"{'Cov@0.7':>8} {'Status':>8}")
    print(f"  {'-'*35} {'-'*8} {'-'*7} {'-'*6} {'-'*8} {'-'*8}")

    for r in rows:
        acc  = f"{float(r.get('e2e_accuracy', 0)):.3f}" if r.get('e2e_accuracy') else "—"
        f1   = f"{float(r.get('macro_f1', 0)):.3f}"     if r.get('macro_f1')     else "—"
        ece  = f"{float(r.get('ece', 0)):.4f}"           if r.get('ece')          else "—"
        cov  = f"{float(r.get('coverage_07', 0)):.3f}"  if r.get('coverage_07')  else "—"
        st   = r.get('status', '?')

        print(f"  {r['experiment']:<35} {acc:>8} {f1:>7} {ece:>6} {cov:>8} {st:>8}")

    print(BANNER)
    print(f"\n  Full results: {RESULTS_CSV.relative_to(PROJECT_ROOT)}")


# ---------------------------------------------------------------------------
# Single experiment runner
# ---------------------------------------------------------------------------

def run_one(
    experiment: str,
    model: str | None,
    stage2_init: str | None,
    gold_path: str | None,
    stage1_experiment: str,
    code_filter: str,
    epochs: int,
    chapters: list[str] | None,
    skip_train: bool,
    skip_calibrate: bool,
    dry_run: bool,
) -> dict:
    """
    Run train → calibrate → evaluate for a single experiment.
    Returns a result dict for the summary table.
    """
    eval_base = PROJECT_ROOT / "outputs" / "evaluations"
    exp_dir   = eval_base / experiment

    log(f"\n{'─'*70}")
    log(f"  Experiment: {experiment}")
    if model:
        log(f"  Model:      {model}")
    log(f"  Started:    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"{'─'*70}")

    result = {
        "experiment": experiment,
        "model":      model or "—",
        "timestamp":  datetime.now().isoformat(),
        "status":     "pending",
    }

    t_total = time.time()

    # ------------------------------------------------------------------
    # Stage: Train
    # ------------------------------------------------------------------
    train_seconds = 0.0
    if skip_train or should_skip_train(exp_dir):
        if should_skip_train(exp_dir):
            log("⏭  Train: skipping — outputs already exist")
        else:
            log("⏭  Train: skipping (--skip-train)")
    else:
        cmd = [
            "uv", "run", "python", "scripts/train.py",
            "--experiment",  experiment,
            "--mode",        "hierarchical",
            "--stage",       "2",
            "--code-filter", code_filter,
            "--epochs",      str(epochs),
        ]
        if model:
            cmd += ["--model", model]
        if stage2_init:
            cmd += ["--stage2-init", stage2_init]
        if gold_path:
            cmd += ["--gold-path", gold_path]
        if chapters:
            cmd += ["--chapters"] + chapters

        ok, train_seconds = run_cmd(cmd, "Training", dry_run)
        result["train_seconds"] = round(train_seconds)

        if not ok:
            result["status"] = "train_failed"
            result["total_seconds"] = round(time.time() - t_total)
            append_results_csv(result)
            return result

    # ------------------------------------------------------------------
    # Stage: Calibrate
    # ------------------------------------------------------------------
    if skip_calibrate or should_skip_calibrate(exp_dir):
        if should_skip_calibrate(exp_dir):
            log("⏭  Calibrate: skipping — calibration_report.json exists")
        else:
            log("⏭  Calibrate: skipping (--skip-calibrate)")
    else:
        cmd = [
            "uv", "run", "python", "scripts/calibrate.py",
            "--experiment",       experiment,
            "--stage1-experiment", stage1_experiment,
        ]
        ok, _ = run_cmd(cmd, "Calibration", dry_run)
        if not ok:
            result["status"] = "calibrate_failed"
            result["total_seconds"] = round(time.time() - t_total)
            append_results_csv(result)
            return result

    # ------------------------------------------------------------------
    # Stage: Evaluate
    # ------------------------------------------------------------------
    if should_skip_evaluate(exp_dir) and not dry_run:
        log("⏭  Evaluate: skipping — eval/ already exists")
    else:
        cmd = [
            "uv", "run", "python", "scripts/evaluate.py",
            "--experiment",        experiment,
            "--mode",              "hierarchical",
            "--stage1-experiment", stage1_experiment,
        ]
        ok, _ = run_cmd(cmd, "Evaluation", dry_run)
        if not ok:
            result["status"] = "evaluate_failed"
            result["total_seconds"] = round(time.time() - t_total)
            append_results_csv(result)
            return result

    # ------------------------------------------------------------------
    # Read results
    # ------------------------------------------------------------------
    if not dry_run:
        metrics = read_eval_results(exp_dir)
        result.update({
            "e2e_accuracy":        metrics.get("e2e_accuracy"),
            "macro_f1":            metrics.get("macro_f1"),
            "ece":                 metrics.get("ece"),
            "coverage_07":         metrics.get("coverage_at_threshold"),
            "accuracy_on_covered": metrics.get("accuracy_on_covered") or metrics.get("accuracy_at_threshold"),
            "stage1_accuracy":     metrics.get("stage1_accuracy"),
            "stage2_accuracy":     metrics.get("within_chapter_accuracy"),
        })

    result["status"]        = "ok" if not dry_run else "dry_run"
    result["total_seconds"] = round(time.time() - t_total)
    append_results_csv(result)

    log(f"\n  ✅ {experiment} complete in {result['total_seconds']}s")
    if result.get("e2e_accuracy"):
        log(f"     E2E: {float(result['e2e_accuracy']):.3f}  "
            f"F1: {float(result['macro_f1']):.3f}  "
            f"ECE: {float(result['ece']):.4f}")

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Orchestrate train → calibrate → evaluate for one or more experiments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Single experiment shorthand
    p.add_argument("--experiment", "-e",
                   help="Single experiment name.")
    p.add_argument("--model", "-m",
                   help="HuggingFace model ID for single experiment.")

    # Multi-experiment mode
    p.add_argument("--experiments", nargs="+",
                   help="List of experiment names (multi-model run).")
    p.add_argument("--models", nargs="+",
                   help="List of model IDs — must match --experiments in order.")

    # Shared options
    p.add_argument("--stage2-init",
                   default="outputs/evaluations/E-002_FullICD10_ClinicalBERT/model/model",
                   help="Path to E-002 flat model for stage-2 init.")
    p.add_argument("--gold-path",
                   default="data/gold/medsynth_gold_augmented.parquet",
                   help="Gold layer parquet path.")
    p.add_argument("--stage1-experiment",
                   default="E-003_Hierarchical_ICD10",
                   help="Stage-1 router experiment name.")
    p.add_argument("--code-filter",
                   default="billable",
                   choices=["all", "billable"])
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--chapters", nargs="+",
                   help="Limit training to specific chapters.")

    # Control flags
    p.add_argument("--skip-train",      action="store_true",
                   help="Skip training — calibrate + evaluate only.")
    p.add_argument("--skip-calibrate",  action="store_true",
                   help="Skip calibration.")
    p.add_argument("--dry-run",         action="store_true",
                   help="Show what would run without executing.")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Build experiment list
    if args.experiments and args.models:
        if len(args.experiments) != len(args.models):
            print("Error: --experiments and --models must have the same number of items.",
                  file=sys.stderr)
            sys.exit(1)
        pairs = list(zip(args.experiments, args.models))
    elif args.experiments:
        # Re-evaluate existing experiments — no model needed
        pairs = [(e, None) for e in args.experiments]
    elif args.experiment:
        pairs = [(args.experiment, args.model)]
    else:
        print("Error: provide --experiment or --experiments.", file=sys.stderr)
        sys.exit(1)

    print(f"\n{BANNER}")
    print(f"  run_experiment.py — Experiment Orchestration")
    print(f"  Experiments: {len(pairs)}")
    print(f"  Dry-run:     {args.dry_run}")
    print(f"  Started:     {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(BANNER)

    all_results = []
    t0 = time.time()

    for experiment, model in pairs:
        result = run_one(
            experiment       = experiment,
            model            = model,
            stage2_init      = args.stage2_init,
            gold_path        = args.gold_path,
            stage1_experiment= args.stage1_experiment,
            code_filter      = args.code_filter,
            epochs           = args.epochs,
            chapters         = args.chapters,
            skip_train       = args.skip_train,
            skip_calibrate   = args.skip_calibrate,
            dry_run          = args.dry_run,
        )
        all_results.append(result)

        if result["status"] not in ("ok", "dry_run"):
            log(f"\n⚠️  {experiment} failed with status: {result['status']}")
            log("   Continuing with remaining experiments...")

    total = time.time() - t0
    print(f"\n{BANNER}")
    print(f"  All experiments complete in {total/60:.1f} min")
    print(BANNER)

    print_comparison_table(all_results)


if __name__ == "__main__":
    main()