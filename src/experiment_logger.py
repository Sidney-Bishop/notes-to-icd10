"""
src/experiment_logger.py — Structured Experiment Logging
=========================================================
Single source of truth for recording what was run, when, with what
parameters, and where every artifact was saved.

Writes to two files that persist across sessions:

    outputs/run.log              — human-readable append-only run log
    outputs/experiments.json     — machine-readable experiment registry

Every script (train.py, calibrate.py, evaluate.py, run_experiment.py)
calls this module to log its actions. No more grepping for files.

Usage
-----
    from src.experiment_logger import ExperimentLogger

    logger = ExperimentLogger("E-008_Balanced")

    # Log a stage starting
    logger.log_start("train_stage2", params={
        "model": "emilyalsentzer/Bio_ClinicalBERT",
        "epochs": 10,
        "gold_path": "data/gold/medsynth_gold_augmented.parquet",
        "chapters": "all",
    })

    # Log a stage completing with artifact paths
    logger.log_complete("train_stage2", artifacts={
        "stage2_dir": "outputs/evaluations/E-008_Balanced/stage2/",
        "chapters_trained": ["A", "B", "C", ...],
    })

    # Log a stage failing
    logger.log_failed("train_stage2", reason="OOM on chapter M")

    # Log evaluation results
    logger.log_results({
        "e2e_accuracy": 0.342,
        "macro_f1": 0.249,
        "ece": 0.064,
        "coverage_07": 0.187,
        "stage1_accuracy": 0.974,
    })

    # Print experiment status
    logger.print_status()

Public functions
----------------
    status()        — print a summary table of all experiments
    registry()      — return the full experiments.json as a dict
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from src.config import config

# ---------------------------------------------------------------------------
# Paths — delegate to config singleton rather than re-walking the filesystem
# ---------------------------------------------------------------------------

def _run_log_path() -> Path:
    return config.project_root / "outputs" / "run.log"

def _registry_path() -> Path:
    return config.project_root / "outputs" / "experiments.json"


# ---------------------------------------------------------------------------
# Registry helpers
# ---------------------------------------------------------------------------

def _load_registry() -> dict:
    p = _registry_path()
    if p.exists():
        try:
            return json.loads(p.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def _save_registry(registry: dict) -> None:
    p = _registry_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(registry, indent=2, default=str))


def _append_run_log(line: str) -> None:
    p = _run_log_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as f:
        f.write(line + "\n")


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _now_iso() -> str:
    return datetime.now().isoformat()


# ---------------------------------------------------------------------------
# ExperimentLogger
# ---------------------------------------------------------------------------

class ExperimentLogger:
    """
    Per-experiment structured logger.

    Parameters
    ----------
    experiment_name : str
        The experiment directory name, e.g. "E-008_Balanced".
    script : str
        Name of the calling script, e.g. "train.py". Used in log lines.
    """

    def __init__(self, experiment_name: str, script: str = "") -> None:
        self.experiment_name = experiment_name
        self.script = script or "unknown"
        self._registry = _load_registry()
        # Ensure experiment entry exists
        if experiment_name not in self._registry:
            self._registry[experiment_name] = {
                "experiment": experiment_name,
                "created": _now_iso(),
                "stages": {},
            }
            _save_registry(self._registry)

    # ------------------------------------------------------------------
    # Core logging methods
    # ------------------------------------------------------------------

    def log_start(self, stage: str, params: dict | None = None) -> None:
        """
        Record that a stage has started.

        Parameters
        ----------
        stage : str
            Stage identifier, e.g. "train_stage1", "train_stage2",
            "calibrate", "evaluate".
        params : dict
            Key parameters for this run — model, epochs, gold_path, etc.
            Anything you'd need to reproduce the run from scratch.
        """
        ts = _now()
        params = params or {}

        # Human-readable log line
        param_str = " | ".join(f"{k}={v}" for k, v in params.items())
        line = f"{ts} | START    | {self.experiment_name:<30} | {stage:<20} | {self.script} | {param_str}"
        _append_run_log(line)
        print(f"📝 [{self.experiment_name}] {stage} started", flush=True)

        # Registry update
        self._registry[self.experiment_name]["stages"][stage] = {
            "status": "running",
            "started": _now_iso(),
            "params": params,
        }
        _save_registry(self._registry)

    def log_complete(self, stage: str, artifacts: dict | None = None) -> None:
        """
        Record that a stage completed successfully.

        Parameters
        ----------
        stage : str
            Stage identifier matching the log_start call.
        artifacts : dict
            Paths and metadata for every artifact written by this stage.
            Be specific — include the full path for every file saved.
        """
        ts = _now()
        artifacts = artifacts or {}

        artifact_str = " | ".join(f"{k}={v}" for k, v in artifacts.items())
        line = f"{ts} | COMPLETE | {self.experiment_name:<30} | {stage:<20} | {self.script} | {artifact_str}"
        _append_run_log(line)
        print(f"✅ [{self.experiment_name}] {stage} complete", flush=True)

        # Registry update
        stage_entry = self._registry[self.experiment_name]["stages"].get(stage, {})
        stage_entry.update({
            "status": "complete",
            "completed": _now_iso(),
            "artifacts": artifacts,
        })
        self._registry[self.experiment_name]["stages"][stage] = stage_entry
        _save_registry(self._registry)

    def log_failed(self, stage: str, reason: str = "") -> None:
        """Record that a stage failed."""
        ts = _now()
        line = f"{ts} | FAILED   | {self.experiment_name:<30} | {stage:<20} | {self.script} | reason={reason}"
        _append_run_log(line)
        print(f"❌ [{self.experiment_name}] {stage} FAILED: {reason}", flush=True)

        stage_entry = self._registry[self.experiment_name]["stages"].get(stage, {})
        stage_entry.update({
            "status": "failed",
            "failed": _now_iso(),
            "reason": reason,
        })
        self._registry[self.experiment_name]["stages"][stage] = stage_entry
        _save_registry(self._registry)

    def log_skipped(self, stage: str, reason: str = "") -> None:
        """Record that a stage was skipped."""
        ts = _now()
        line = f"{ts} | SKIPPED  | {self.experiment_name:<30} | {stage:<20} | {self.script} | reason={reason}"
        _append_run_log(line)

        stage_entry = self._registry[self.experiment_name]["stages"].get(stage, {})
        stage_entry.update({
            "status": "skipped",
            "skipped": _now_iso(),
            "reason": reason,
        })
        self._registry[self.experiment_name]["stages"][stage] = stage_entry
        _save_registry(self._registry)

    def log_results(self, metrics: dict) -> None:
        """
        Record evaluation results for this experiment.

        Parameters
        ----------
        metrics : dict
            Expected keys: e2e_accuracy, macro_f1, ece, coverage_07,
            stage1_accuracy, stage2_accuracy. Any additional keys are
            also stored.
        """
        ts = _now()
        key_metrics = {
            k: metrics[k] for k in
            ["e2e_accuracy", "macro_f1", "ece", "coverage_07",
             "stage1_accuracy", "stage2_accuracy"]
            if k in metrics
        }
        metric_str = " | ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                for k, v in key_metrics.items())
        line = f"{ts} | RESULTS  | {self.experiment_name:<30} | {'evaluate':<20} | {self.script} | {metric_str}"
        _append_run_log(line)

        self._registry[self.experiment_name]["results"] = {
            **metrics,
            "recorded": _now_iso(),
        }
        _save_registry(self._registry)

    def log_note(self, note: str) -> None:
        """Append a free-text note to the run log."""
        ts = _now()
        line = f"{ts} | NOTE     | {self.experiment_name:<30} | {'—':<20} | {self.script} | {note}"
        _append_run_log(line)

    # ------------------------------------------------------------------
    # Status helpers
    # ------------------------------------------------------------------

    def print_status(self) -> None:
        """Print this experiment's current status."""
        entry = self._registry.get(self.experiment_name, {})
        stages = entry.get("stages", {})
        results = entry.get("results", {})

        print(f"\n{'─'*60}")
        print(f" Experiment: {self.experiment_name}")
        print(f" Created:    {entry.get('created', '—')[:19]}")
        print(f"{'─'*60}")

        for stage, info in stages.items():
            status = info.get("status", "?")
            icon = {"complete": "✅", "running": "🔄", "failed": "❌",
                    "skipped": "⏭"}.get(status, "❓")
            ts = (info.get("completed") or info.get("failed") or
                  info.get("started") or "—")[:19]
            print(f"  {icon} {stage:<25} {status:<10} {ts}")

        if results:
            print(f"\n  Results:")
            for k in ["e2e_accuracy", "macro_f1", "ece", "coverage_07",
                      "stage1_accuracy"]:
                if k in results:
                    v = results[k]
                    print(f"    {k:<25} {v:.4f}" if isinstance(v, float)
                          else f"    {k:<25} {v}")
        print()


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def status() -> None:
    """
    Print a summary table of all experiments in the registry.
    Call from command line: uv run python -c "from src.experiment_logger import status; status()"
    """
    registry = _load_registry()
    if not registry:
        print("No experiments registered yet.")
        return

    print(f"\n{'='*90}")
    print(f" Experiment Registry — {_now()}")
    print(f"{'='*90}")
    print(f" {'Experiment':<35} {'Stage1':<10} {'Stage2':<10} {'Calib':<10} {'Eval':<10} {'E2E':>7} {'F1':>7}")
    print(f" {'-'*35} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*7} {'-'*7}")

    STATUS_ICONS = {
        "complete": "✅",
        "running":  "🔄",
        "failed":   "❌",
        "skipped":  "⏭",
        None:       "—",
    }

    for exp_name, entry in sorted(registry.items()):
        stages = entry.get("stages", {})
        results = entry.get("results", {})

        def st(stage_key: str) -> str:
            s = stages.get(stage_key, {}).get("status")
            return STATUS_ICONS.get(s, "—")

        e2e = f"{results['e2e_accuracy']:.3f}" if "e2e_accuracy" in results else "—"
        f1  = f"{results['macro_f1']:.3f}"     if "macro_f1"      in results else "—"

        print(f" {exp_name:<35} {st('train_stage1'):<10} {st('train_stage2'):<10} "
              f"{st('calibrate'):<10} {st('evaluate'):<10} {e2e:>7} {f1:>7}")

    print(f"{'='*90}")
    print(f" Registry: {_registry_path()}")
    print(f" Run log:  {_run_log_path()}\n")


def registry() -> dict:
    """Return the full experiments registry as a dict."""
    return _load_registry()


def get_experiment(experiment_name: str) -> dict:
    """Return the registry entry for a single experiment."""
    return _load_registry().get(experiment_name, {})


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "status":
        status()
    elif len(sys.argv) > 2 and sys.argv[1] == "show":
        entry = get_experiment(sys.argv[2])
        print(json.dumps(entry, indent=2, default=str))
    else:
        print("Usage:")
        print("  uv run python src/experiment_logger.py status")
        print("  uv run python src/experiment_logger.py show E-008_Balanced")