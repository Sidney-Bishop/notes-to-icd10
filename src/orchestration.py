"""
src/orchestration.py — testable orchestration primitives for the full
de-leaked rebuild.

Everything here is unit-testable: pure formatting/logic, tmp-dir-testable
filesystem checks, command construction as DATA, and a subprocess executor
that streams to console+file and captures the REAL exit code.

The thin CLI (scripts/run_full_deleaked_rebuild.py) wires these together.

Design notes:
  - No hardcoded absolute paths — uses src.config.config.project_root.
  - Phase specs are data (build_phase_specs) so command construction is
    testable WITHOUT executing anything: e.g. assert every ModernBERT phase
    sets --stage1-model to its backbone (never the roberta-base default),
    every gold path is the de-leaked one, dependency order holds.
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Constants — the de-leaked rebuild configuration
# ---------------------------------------------------------------------------

GOLD_DELEAKED = "data/gold/medsynth_gold_apso_deleaked.parquet"

MODEL_CLINICALBERT = "emilyalsentzer/Bio_ClinicalBERT"
MODEL_CLINICAL_MODERNBERT = "Simonlee711/Clinical_ModernBERT"
MODEL_BIOCLINICAL_MODERNBERT = "thomas-sounack/BioClinical-ModernBERT-base"

# The roberta-base default in run_experiment.py is a trap: a hierarchical
# rebuild that forgets --stage1-model trains its router on the wrong backbone.
ROBERTA_DEFAULT = "roberta-base"


# ---------------------------------------------------------------------------
# Formatting (pure)
# ---------------------------------------------------------------------------

def now_iso() -> str:
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


def event_line(msg: str, when: Optional[str] = None) -> str:
    """Format one events.log line."""
    return f"[{when or now_iso()}] {msg}"


def classify_exit(returncode: int) -> str:
    """OK for 0, FAILED otherwise."""
    return "OK" if returncode == 0 else "FAILED"


# ---------------------------------------------------------------------------
# Filesystem checks (tmp-dir testable)
# ---------------------------------------------------------------------------

def statline(path: Path) -> str:
    """'path (size_bytes, mtime_iso)' if present, else 'path MISSING'."""
    p = Path(path)
    if p.is_file():
        size = p.stat().st_size
        mtime = datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%dT%H:%M:%S")
        return f"{p} ({size} bytes, {mtime})"
    return f"{p} MISSING"


def preflight(paths: list[Path], dry_run: bool = False) -> tuple[bool, list[str]]:
    """
    Verify inputs exist. Returns (ok, log_lines).
    In dry_run, a missing input is logged but does NOT fail (the input would
    be produced by an earlier phase at real run time).
    """
    lines = ["  INPUTS (read):"]
    missing = False
    for p in paths:
        line = statline(p)
        lines.append(f"    READ  {line}")
        if line.endswith("MISSING"):
            missing = True
    if missing:
        if dry_run:
            lines.append("  (DRY: input not present yet — produced by an earlier phase at run time)")
            return True, lines
        lines.append("  ✗ PRE-FLIGHT FAILED — required input missing.")
        return False, lines
    return True, lines


def postflight(paths: list[Path]) -> tuple[bool, list[str]]:
    """Verify expected outputs exist. Returns (ok, log_lines) with a VERDICT."""
    lines = ["  OUTPUTS (written, verified):"]
    missing = False
    for p in paths:
        line = statline(p)
        lines.append(f"    WROTE {line}")
        if line.endswith("MISSING"):
            missing = True
    if missing:
        lines.append("  VERDICT: ✗ expected artifact MISSING (step may have failed silently)")
        return False, lines
    lines.append("  VERDICT: ✓ expected artifacts present")
    return True, lines


def flat_done(eval_base: Path, experiment: str) -> bool:
    """True if a flat experiment's model weights already exist (resume guard)."""
    return (Path(eval_base) / experiment / "model" / "model.safetensors").is_file()


# ---------------------------------------------------------------------------
# Phase specs (command construction as DATA — testable without executing)
# ---------------------------------------------------------------------------

@dataclass
class PhaseSpec:
    name: str
    cmd: list[str]                      # the exact argv (no shell)
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    is_flat_pretrain: bool = False
    flat_experiment: Optional[str] = None   # for resume skip check
    supports_dry_run: bool = True           # False → CLI skips it in dry-run
                                            # (underlying script has no --dry-run)


def _flat_cmd(experiment: str, model: str, gold: str) -> list[str]:
    return [
        "env", "PYTHONPATH=.", "uv", "run", "python", "scripts/train.py",
        "--experiment", experiment, "--mode", "flat", "--label-scheme", "icd10",
        "--model", model, "--code-filter", "billable", "--batch-size", "16",
        "--epochs", "40", "--max-length", "512", "--gold-path", gold,
    ]


def _hier_cmd(experiment: str, model: str, stage2_init: str, gold: str,
              dry_run: bool) -> list[str]:
    cmd = [
        "env", "PYTHONPATH=.", "uv", "run", "python", "scripts/run_experiment.py",
        "--experiment", experiment, "--model", model,
        "--stage2-init", stage2_init,
        "--train-stage1", "--stage1-model", model,   # MUST be model, never roberta default
        "--gold-path", gold, "--epochs", "20", "--code-filter", "billable",
    ]
    if dry_run:
        cmd.append("--dry-run")
    return cmd


# --- SupCon chain + MIMIC builders -----------------------------------------
# EHIER = the de-leaked hierarchical router/base. calibrate + hybrid default to
# the LEAKY E-003/E-010; we MUST pass EHIER explicitly or we recreate the
# train/serve mismatch that historically cratered E-015.

def _presplits_cmd(experiment: str, gold: str) -> list[str]:
    return [
        "env", "PYTHONPATH=.", "uv", "run", "python", "scripts/prepare_splits.py",
        "--experiment", experiment, "--gold-path", gold, "--code-filter", "billable",
    ]


def _supcon_zbase_cmd(experiment: str, stage2_init: str, gold: str) -> list[str]:
    return [
        "env", "PYTHONPATH=.", "uv", "run", "python", "scripts/train.py",
        "--experiment", experiment, "--mode", "hierarchical", "--stage", "2",
        "--code-filter", "billable", "--epochs", "20",
        "--stage2-init", stage2_init, "--gold-path", gold,
        "--use-presplit", "--chapters", "Z",
    ]


def _supcon_train_cmd(source_experiment: str, experiment: str) -> list[str]:
    return [
        "env", "PYTHONPATH=.", "uv", "run", "python", "scripts/train_supcon_z.py",
        "--source-experiment", source_experiment, "--experiment", experiment,
    ]


def _supcon_calibrate_cmd(experiment: str, stage1_experiment: str) -> list[str]:
    return [
        "uv", "run", "python", "scripts/calibrate.py",
        "--experiment", experiment, "--stage1-experiment", stage1_experiment,
    ]


def _supcon_hybrid_cmd(base_experiment: str, stage1_experiment: str,
                       override_z: str) -> list[str]:
    return [
        "env", "PYTHONPATH=.", "uv", "run", "python", "scripts/evaluate_hybrid.py",
        "--base-experiment", base_experiment,
        "--stage1-experiment", stage1_experiment,
        "--override", f"Z={override_z}", "--threshold", "0.7",
    ]


def _mimic_cmd(base_experiment: str, stage1_experiment: str) -> list[str]:
    return [
        "env", "PYTHONPATH=.", "uv", "run", "python",
        "scripts/validation/validate_mimic_evaluate.py",
        "--base-experiment", base_experiment,
        "--stage1-experiment", stage1_experiment,
        "--deleaked-reference",
    ]


def build_phase_specs(dry_run: bool = False) -> list[PhaseSpec]:
    """
    The full de-leaked rebuild as data. Pure — constructs no files, runs
    nothing. Tests assert properties over this structure.
    """
    eb = "outputs/evaluations"
    gold = GOLD_DELEAKED
    E002, EHIER = "E-020_Flat_ClinicalBERT_Deleaked", "E-021_Hier_ClinicalBERT_Deleaked"
    E023, E024 = "E-023_Flat_ClinicalModernBERT_Deleaked", "E-024_Hier_ClinicalModernBERT_Deleaked"
    E025, E026 = "E-025_Flat_BioClinicalModernBERT_Deleaked", "E-026_Hier_BioClinicalModernBERT_Deleaked"

    specs: list[PhaseSpec] = []

    specs.append(PhaseSpec(
        name="flat_clinicalbert",
        cmd=_flat_cmd(E002, MODEL_CLINICALBERT, gold),
        inputs=[gold], outputs=[f"{eb}/{E002}/model/model.safetensors"],
        is_flat_pretrain=True, flat_experiment=E002,
    ))
    specs.append(PhaseSpec(
        name="hier_clinicalbert",
        cmd=_hier_cmd(EHIER, MODEL_CLINICALBERT, f"{eb}/{E002}", gold, dry_run),
        inputs=[gold, f"{eb}/{E002}/model/model.safetensors"],
        outputs=[f"{eb}/{EHIER}/stage1/model/model.safetensors",
                 f"{eb}/{EHIER}/eval/summary.json"],
    ))
    specs.append(PhaseSpec(
        name="flat_clinical_modernbert",
        cmd=_flat_cmd(E023, MODEL_CLINICAL_MODERNBERT, gold),
        inputs=[gold], outputs=[f"{eb}/{E023}/model/model.safetensors"],
        is_flat_pretrain=True, flat_experiment=E023,
    ))
    specs.append(PhaseSpec(
        name="hier_clinical_modernbert",
        cmd=_hier_cmd(E024, MODEL_CLINICAL_MODERNBERT, f"{eb}/{E023}", gold, dry_run),
        inputs=[gold, f"{eb}/{E023}/model/model.safetensors"],
        outputs=[f"{eb}/{E024}/stage1/model/model.safetensors",
                 f"{eb}/{E024}/eval/summary.json"],
    ))
    specs.append(PhaseSpec(
        name="flat_bioclinical_modernbert",
        cmd=_flat_cmd(E025, MODEL_BIOCLINICAL_MODERNBERT, gold),
        inputs=[gold], outputs=[f"{eb}/{E025}/model/model.safetensors"],
        is_flat_pretrain=True, flat_experiment=E025,
    ))
    specs.append(PhaseSpec(
        name="hier_bioclinical_modernbert",
        cmd=_hier_cmd(E026, MODEL_BIOCLINICAL_MODERNBERT, f"{eb}/{E025}", gold, dry_run),
        inputs=[gold, f"{eb}/{E025}/model/model.safetensors"],
        outputs=[f"{eb}/{E026}/stage1/model/model.safetensors",
                 f"{eb}/{E026}/eval/summary.json"],
    ))

    # --- SupCon Z chain (depends on EHIER from hier_clinicalbert) -----------
    ESUP_BASE = "E-022_Deleaked_SupConBase"
    ESUP = "E-022_SupCon_Z_Deleaked"
    specs.append(PhaseSpec(
        name="supcon_presplits",
        cmd=_presplits_cmd(ESUP_BASE, gold),
        inputs=[gold],
        outputs=[f"{eb}/{ESUP_BASE}/stage2/Z/train_split.parquet",
                 f"{eb}/{ESUP_BASE}/stage2/Z/val_split.parquet"],
        supports_dry_run=False,   # prepare_splits.py has no --dry-run
    ))
    specs.append(PhaseSpec(
        name="supcon_zbase",
        cmd=_supcon_zbase_cmd(ESUP_BASE, f"{eb}/{E002}", gold),
        inputs=[gold, f"{eb}/{ESUP_BASE}/stage2/Z/train_split.parquet"],
        outputs=[f"{eb}/{ESUP_BASE}/stage2/Z/model/model.safetensors"],
    ))
    specs.append(PhaseSpec(
        name="supcon_train",
        cmd=_supcon_train_cmd(ESUP_BASE, ESUP),
        inputs=[f"{eb}/{ESUP_BASE}/stage2/Z/model/model.safetensors"],
        # train_supcon_z.py saves FLAT (stage2/Z/model.safetensors), unlike
        # train.py which nests under model/. The nested path here caused a
        # false postflight MISSING during run 20260603_200854.
        outputs=[f"{eb}/{ESUP}/stage2/Z/model.safetensors"],
    ))
    specs.append(PhaseSpec(
        name="supcon_calibrate",
        cmd=_supcon_calibrate_cmd(ESUP, EHIER),   # EHIER, NOT the E-003 default
        inputs=[f"{eb}/{ESUP}/stage2/Z/model/model.safetensors"],
        outputs=[f"{eb}/{ESUP}/calibration_report.json"],
    ))
    specs.append(PhaseSpec(
        name="supcon_hybrid",
        cmd=_supcon_hybrid_cmd(EHIER, EHIER, ESUP),  # base+stage1 = EHIER, not leaky defaults
        inputs=[f"{eb}/{ESUP}/calibration_report.json"],
        # evaluate_hybrid.py writes <base>_hybrid_<ch>-<id>/eval/summary.json
        # where <id> = override_exp.split('_')[0]. Replicate that derivation.
        outputs=[f"{eb}/{EHIER}_hybrid_Z-{ESUP.split('_')[0]}/eval/summary.json"],
        supports_dry_run=False,   # evaluate_hybrid.py has no --dry-run
    ))

    # --- MIMIC de-leaked eval (depends on EHIER) ----------------------------
    specs.append(PhaseSpec(
        name="mimic_deleaked",
        cmd=_mimic_cmd(EHIER, EHIER),
        inputs=["data/mimic/gold/mimic_gold.parquet",
                f"{eb}/{EHIER}/stage1/model/model.safetensors"],
        outputs=[f"{eb}/mimic_iv_validation/summary.json"],
        supports_dry_run=False,   # validate_mimic_evaluate.py has no --dry-run
    ))
    return specs


# ---------------------------------------------------------------------------
# Executor (mirrors run_experiment.run_cmd; tees to console+file, real exit code)
# ---------------------------------------------------------------------------

@dataclass
class StepResult:
    name: str
    returncode: int
    elapsed_s: float
    status: str


def run_step(name: str, cmd: list[str], logfile: Path, cwd: Path,
             dry_run: bool = False) -> StepResult:
    """
    Execute cmd, streaming combined stdout/stderr to BOTH console and logfile,
    returning the real exit code. Mirrors run_experiment.run_cmd's contract.
    """
    if dry_run:
        return StepResult(name, 0, 0.0, "dry_run")

    t0 = time.time()
    Path(logfile).parent.mkdir(parents=True, exist_ok=True)
    with open(logfile, "w") as lf:
        proc = subprocess.Popen(
            cmd, cwd=str(cwd),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            lf.write(line)
        proc.wait()
    elapsed = time.time() - t0
    return StepResult(name, proc.returncode, elapsed, classify_exit(proc.returncode))
