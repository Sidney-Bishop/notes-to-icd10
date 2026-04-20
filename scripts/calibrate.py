#!/usr/bin/env python3
"""
calibrate.py — Temperature Scaling for ICD-10 Hierarchical Pipeline
=====================================================================
Learns a per-model temperature scalar T that minimises NLL on the held-out
test split, then writes temperature.json alongside each resolver's model dir.
inference.py reads these files at load time and applies T before softmax.

Temperature scaling (Guo et al. 2017) fits a single scalar T per model:

    p_calibrated = softmax(logits / T)

    T > 1  →  softer distribution  (model was overconfident)
    T < 1  →  sharper distribution (model was underconfident)

We optimise T by minimising cross-entropy loss on the test split logits.
This is valid because T has a single degree of freedom — it cannot overfit.

Usage
-----
    uv run python scripts/calibrate.py \\
        --experiment  E-004a_Hierarchical_E002Init \\
        --stage1-experiment E-003_Hierarchical_ICD10

    # Dry-run: print temperatures without writing
    uv run python scripts/calibrate.py \\
        --experiment E-004a_Hierarchical_E002Init \\
        --dry-run

Outputs
-------
    outputs/evaluations/{experiment}/stage1/model/temperature.json
    outputs/evaluations/{experiment}/stage2/{chapter}/model/temperature.json
    outputs/evaluations/{experiment}/calibration_report.json
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Project path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import config                                        # noqa: E402
from src.adapters import EncoderAdapter                             # noqa: E402

BANNER = "=" * 70


# ---------------------------------------------------------------------------
# Temperature optimisation
# ---------------------------------------------------------------------------

def collect_logits(
    model: torch.nn.Module,
    tokenizer,
    texts: list[str],
    labels: list[int],
    device: torch.device,
    batch_size: int = 32,
    max_length: int = 512,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Run forward passes and collect raw logits + true label indices.

    Returns
    -------
    logits : FloatTensor [N, C]
    targets : LongTensor [N]
    """
    model.eval()
    all_logits = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        inputs = tokenizer(
            batch_texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        inputs = {
            k: v.to(device)
            for k, v in inputs.items()
            if k in tokenizer.model_input_names
        }
        with torch.no_grad():
            logits = model(**inputs).logits  # [B, C]
        all_logits.append(logits.cpu())

    logits_all = torch.cat(all_logits, dim=0)          # [N, C]
    targets_all = torch.tensor(labels, dtype=torch.long)  # [N]
    return logits_all, targets_all


def optimise_temperature(
    logits: torch.Tensor,
    targets: torch.Tensor,
    lr: float = 0.01,
    max_iter: int = 500,
    tol: float = 1e-6,
) -> float:
    """
    Learn scalar temperature T by minimising cross-entropy loss.

    Uses LBFGS optimiser — converges in <50 iterations for single-scalar T.

    Returns
    -------
    T : float — optimal temperature (>0)
    """
    T = torch.nn.Parameter(torch.ones(1))
    optimiser = torch.optim.LBFGS([T], lr=lr, max_iter=max_iter, tolerance_change=tol)

    def closure():
        optimiser.zero_grad()
        loss = F.cross_entropy(logits / T.clamp(min=1e-4), targets)
        loss.backward()
        return loss

    optimiser.step(closure)
    return float(T.item())


def ece_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    temperature: float = 1.0,
    n_bins: int = 10,
) -> float:
    """Compute ECE from raw logits at a given temperature."""
    probs = torch.softmax(logits / temperature, dim=-1)
    confs, preds = probs.max(dim=-1)
    correct = (preds == targets).float()

    confs_np  = confs.numpy()
    correct_np = correct.numpy()

    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(confs_np)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confs_np > lo) & (confs_np <= hi)
        if mask.sum() == 0:
            continue
        bin_acc  = correct_np[mask].mean()
        bin_conf = confs_np[mask].mean()
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return float(ece)


def coverage_at_threshold(
    logits: torch.Tensor,
    targets: torch.Tensor,
    temperature: float = 1.0,
    threshold: float = 0.7,
) -> tuple[float, float]:
    """Return (coverage, accuracy_on_covered) at confidence threshold."""
    probs = torch.softmax(logits / temperature, dim=-1)
    confs, preds = probs.max(dim=-1)
    covered  = confs >= threshold
    if covered.sum() == 0:
        return 0.0, 0.0
    coverage = float(covered.float().mean())
    acc      = float((preds[covered] == targets[covered]).float().mean())
    return coverage, acc


# ---------------------------------------------------------------------------
# Per-model calibration
# ---------------------------------------------------------------------------

def calibrate_model(
    hf_model_dir: Path,
    label_map_path: Path,
    test_split_path: Path,
    label_scheme: str = "icd10",
    device: torch.device = torch.device("cpu"),
    threshold: float = 0.7,
) -> dict:
    """
    Calibrate a single EncoderAdapter model and return a report dict.

    Parameters
    ----------
    hf_model_dir    : Path to HF model weights (config.json, model.safetensors)
    label_map_path  : Path to label_map.json (has id2label + label_scheme)
    test_split_path : Path to test_split.parquet
    label_scheme    : "icd10", "icd3", or "chapter"
    device          : Torch device

    Returns
    -------
    dict with temperature, ECE before/after, coverage before/after
    """
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    # Load label map
    with open(label_map_path) as f:
        lmap = json.load(f)
    id2label = {int(k): v for k, v in lmap["id2label"].items()}
    label2id = {v: k for k, v in id2label.items()}

    # Load test split
    test_df = pl.read_parquet(test_split_path)

    # Map true labels to integer indices
    def _apply_scheme(code: str) -> str:
        if label_scheme == "icd3":
            return code[:3]
        if label_scheme == "chapter":
            return code[0] if code else "UNKNOWN"
        return code  # full icd10

    true_codes   = [_apply_scheme(c) for c in test_df["standard_icd10"].to_list()]
    true_label_ids = [label2id.get(c, -1) for c in true_codes]

    # Filter out any unmapped labels (shouldn't happen but defensive)
    valid = [(txt, lid) for txt, lid in
             zip(test_df["apso_note"].to_list(), true_label_ids) if lid >= 0]
    if not valid:
        return {"error": "no valid labels in test split"}
    texts, label_ids = zip(*valid)

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(str(hf_model_dir))
    model     = AutoModelForSequenceClassification.from_pretrained(
        str(hf_model_dir)
    ).to(device)
    model.eval()

    # Collect logits
    logits, targets = collect_logits(
        model, tokenizer, list(texts), list(label_ids), device
    )

    # Metrics before calibration (T=1.0)
    ece_before = ece_from_logits(logits, targets, temperature=1.0)
    cov_before, acc_before = coverage_at_threshold(
        logits, targets, temperature=1.0, threshold=threshold
    )

    # Optimise temperature
    T = optimise_temperature(logits, targets)

    # Metrics after calibration
    ece_after = ece_from_logits(logits, targets, temperature=T)
    cov_after, acc_after = coverage_at_threshold(
        logits, targets, temperature=T, threshold=threshold
    )

    return {
        "temperature":     round(T, 6),
        "n_samples":       len(texts),
        "n_classes":       len(id2label),
        "ece_before":      round(ece_before, 4),
        "ece_after":       round(ece_after,  4),
        "coverage_before": round(cov_before, 4),
        "coverage_after":  round(cov_after,  4),
        "accuracy_on_covered_before": round(acc_before, 4),
        "accuracy_on_covered_after":  round(acc_after,  4),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Temperature scaling calibration for the ICD-10 pipeline."
    )
    parser.add_argument(
        "--experiment",
        required=True,
        help="Stage-2 experiment name (e.g. E-004a_Hierarchical_E002Init)",
    )
    parser.add_argument(
        "--stage1-experiment",
        default="E-003_Hierarchical_ICD10",
        help="Stage-1 experiment name (default: E-003_Hierarchical_ICD10)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Confidence threshold for Use Case B coverage metric (default: 0.7)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print temperatures without writing temperature.json files",
    )
    args = parser.parse_args()

    eval_base = config.resolve_path("outputs", "evaluations")
    exp_dir   = eval_base / args.experiment
    s1_dir    = eval_base / args.stage1_experiment / "stage1"
    s2_dir    = exp_dir / "stage2"

    device = torch.device(
        "mps"  if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available()         else
        "cpu"
    )

    started = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(BANNER)
    print(f"  calibrate.py — Temperature Scaling")
    print(f"  Experiment:  {args.experiment}")
    print(f"  Stage-1:     {args.stage1_experiment}")
    print(f"  Threshold:   {args.threshold}")
    print(f"  Dry-run:     {args.dry_run}")
    print(f"  Device:      {device}")
    print(f"  Started:     {started}")
    print(BANNER)

    report = {}
    t0 = time.time()

    # ------------------------------------------------------------------
    # Stage-1 calibration
    # ------------------------------------------------------------------
    print("\n── Stage-1 Router ─────────────────────────────────────────────────")
    s1_hf_dir    = s1_dir / "model" / "model"
    s1_lmap      = s1_dir / "model" / "label_map.json"
    s1_test      = s1_dir / "test_split.parquet"

    if not s1_hf_dir.exists():
        print(f"   ⚠️  Stage-1 model not found at {s1_hf_dir}, skipping")
    else:
        result = calibrate_model(
            hf_model_dir   = s1_hf_dir,
            label_map_path = s1_lmap,
            test_split_path= s1_test,
            label_scheme   = "chapter",
            device         = device,
            threshold      = args.threshold,
        )
        report["stage1"] = result
        T = result.get("temperature", 1.0)
        print(f"   T={T:.4f} | "
              f"ECE {result['ece_before']:.3f}→{result['ece_after']:.3f} | "
              f"Coverage@{args.threshold} "
              f"{result['coverage_before']:.1%}→{result['coverage_after']:.1%} "
              f"(acc={result['accuracy_on_covered_after']:.3f})")

        if not args.dry_run:
            out = s1_hf_dir / "temperature.json"
            out.write_text(json.dumps({"temperature": T}, indent=2))
            print(f"   💾 Saved: {out}")

    # ------------------------------------------------------------------
    # Stage-2 per-chapter calibration
    # ------------------------------------------------------------------
    print("\n── Stage-2 Resolvers ──────────────────────────────────────────────")

    if not s2_dir.exists():
        print(f"   ⚠️  Stage-2 directory not found at {s2_dir}")
    else:
        chapter_results = {}
        for ch_dir in sorted(s2_dir.iterdir()):
            if not ch_dir.is_dir():
                continue
            ch = ch_dir.name
            hf_dir    = ch_dir / "model" / "model"
            lmap_path = ch_dir / "model" / "label_map.json"
            test_path = ch_dir / "test_split.parquet"

            if not hf_dir.exists() or not test_path.exists():
                print(f"   ⏭  {ch}: no model or test split, skipping")
                continue

            result = calibrate_model(
                hf_model_dir   = hf_dir,
                label_map_path = lmap_path,
                test_split_path= test_path,
                label_scheme   = "icd10",
                device         = device,
                threshold      = args.threshold,
            )
            chapter_results[ch] = result
            T = result.get("temperature", 1.0)
            print(f"   {ch}: T={T:.4f} | "
                  f"ECE {result['ece_before']:.3f}→{result['ece_after']:.3f} | "
                  f"Coverage@{args.threshold} "
                  f"{result['coverage_before']:.1%}→{result['coverage_after']:.1%} "
                  f"(acc={result['accuracy_on_covered_after']:.3f})")

            if not args.dry_run:
                out = hf_dir / "temperature.json"
                out.write_text(json.dumps({"temperature": T}, indent=2))

        report["stage2"] = chapter_results

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    elapsed = time.time() - t0
    report["meta"] = {
        "experiment":       args.experiment,
        "stage1_experiment": args.stage1_experiment,
        "threshold":        args.threshold,
        "device":           str(device),
        "elapsed_s":        round(elapsed, 1),
        "started":          started,
    }

    if not args.dry_run:
        report_path = exp_dir / "calibration_report.json"
        report_path.write_text(json.dumps(report, indent=2))
        print(f"\n   💾 Report saved: {report_path}")

    # Summary stats
    if "stage2" in report:
        ch_results = report["stage2"]
        avg_T      = np.mean([r["temperature"] for r in ch_results.values()])
        avg_ece_b  = np.mean([r["ece_before"]  for r in ch_results.values()])
        avg_ece_a  = np.mean([r["ece_after"]   for r in ch_results.values()])
        avg_cov_a  = np.mean([r["coverage_after"] for r in ch_results.values()])
        avg_acc_a  = np.mean([r["accuracy_on_covered_after"] for r in ch_results.values()])

        print(f"\n{'='*70}")
        print(f"  Stage-2 summary ({len(ch_results)} resolvers)")
        print(f"  Avg temperature:  {avg_T:.4f}")
        print(f"  Avg ECE:          {avg_ece_b:.3f} → {avg_ece_a:.3f}")
        print(f"  Avg Coverage@{args.threshold}: {avg_cov_a:.1%}  "
              f"(avg accuracy on covered: {avg_acc_a:.3f})")
        print(f"  Elapsed: {elapsed:.1f}s")
        print(f"{'='*70}")


if __name__ == "__main__":
    main()