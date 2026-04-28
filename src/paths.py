"""
src/paths.py — Canonical Artifact Path Resolution
==================================================
Single source of truth for where every artifact lives on disk.
Import this module in every script — never hardcode paths elsewhere.

All path functions auto-detect the layout convention used when the
experiment was created, so old experiments (notebook-built) and new
experiments (script-built) work without any manual intervention.

Layout conventions (auto-detected in priority order):
  FLAT   : stage2/A/model.safetensors          ← E-008+ (script-built)
  SINGLE : stage2/A/model/model.safetensors    ← E-006 (script-built)
  NESTED : stage2/A/model/model/model.safetensors ← E-002 (notebook-built)

Usage
-----
    from src.paths import ExperimentPaths

    p = ExperimentPaths("E-008_Balanced")
    p.stage1_model_dir()          # Path to stage-1 weights dir
    p.stage2_model_dir("Z")       # Path to chapter Z weights dir
    p.stage2_label_map("Z")       # Path to chapter Z label_map.json
    p.stage2_test_split("Z")      # Path to chapter Z test_split.parquet
    p.calibration_report()        # Path to calibration_report.json
    p.eval_summary()              # Path to eval/summary.json
    p.temperature(stage=1)        # Path to stage-1 temperature.json
    p.temperature(stage=2, ch="Z") # Path to chapter Z temperature.json
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from src.config import config


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _eval_base() -> Path:
    """Root evaluations directory from config."""
    return config.resolve_path("outputs", "evaluations")


def _find_model_dir(root: Path) -> Optional[Path]:
    """
    Auto-detect which layout convention a model directory uses.

    Checks in priority order:
      1. root/                    (FLAT   — model.safetensors directly here)
      2. root/model/              (SINGLE — one level of nesting)
      3. root/model/model/        (NESTED — two levels, notebook legacy)

    Returns the first directory that contains config.json or model.safetensors,
    or None if no weights are found.
    """
    candidates = [
        root,
        root / "model",
        root / "model" / "model",
    ]
    for candidate in candidates:
            # Must have model weights — config.json alone is not enough
            if candidate.is_dir() and (candidate / "model.safetensors").exists():
                return candidate
    return None


def _find_label_map(root: Path) -> Optional[Path]:
    """
    Find label_map.json — sits alongside or just above the model weights.
    """
    candidates = [
        root / "label_map.json",
        root / "model" / "label_map.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _find_temperature(root: Path) -> Optional[Path]:
    """
    Find temperature.json — written by calibrate.py next to the weights.
    """
    candidates = [
        root / "temperature.json",
        root / "model" / "temperature.json",
        root / "model" / "model" / "temperature.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


# ---------------------------------------------------------------------------
# ExperimentPaths — main API
# ---------------------------------------------------------------------------

class ExperimentPaths:
    """
    Resolves all artifact paths for a given experiment.

    Parameters
    ----------
    experiment_name : str
        The experiment directory name, e.g. "E-008_Balanced".
    stage1_experiment : str | None
        If Stage-1 router lives in a different experiment (e.g. shared
        E-003 router), specify it here. Defaults to experiment_name.
    """

    def __init__(
        self,
        experiment_name: str,
        stage1_experiment: Optional[str] = None,
    ) -> None:
        self.experiment_name  = experiment_name
        self.stage1_experiment = stage1_experiment or experiment_name
        self._base            = _eval_base() / experiment_name
        self._s1_base         = _eval_base() / self.stage1_experiment / "stage1"
        self._s2_base         = self._base / "stage2"

    # ------------------------------------------------------------------
    # Top-level experiment directory
    # ------------------------------------------------------------------

    @property
    def experiment_dir(self) -> Path:
        return self._base

    @property
    def stage2_base(self) -> Path:
        return self._s2_base

    # ------------------------------------------------------------------
    # Stage-1
    # ------------------------------------------------------------------

    def stage1_model_dir(self) -> Optional[Path]:
        """Directory containing Stage-1 model weights (auto-detected)."""
        return _find_model_dir(self._s1_base)

    def stage1_label_map(self) -> Optional[Path]:
        """Path to Stage-1 label_map.json."""
        return _find_label_map(self._s1_base)

    def stage1_test_split(self) -> Path:
        """Path to Stage-1 test_split.parquet."""
        return self._s1_base / "test_split.parquet"

    def stage1_temperature(self) -> Path:
        """
        Canonical write location for Stage-1 temperature.json.
        Always written to stage1/ root (flat), regardless of where
        the weights are.
        """
        return self._s1_base / "temperature.json"

    def stage1_temperature_existing(self) -> Optional[Path]:
        """Find existing Stage-1 temperature.json (any convention)."""
        return _find_temperature(self._s1_base)

    # ------------------------------------------------------------------
    # Stage-2
    # ------------------------------------------------------------------

    def chapter_dir(self, chapter: str) -> Path:
        """Root directory for a specific chapter."""
        return self._s2_base / chapter.upper()

    def stage2_model_dir(self, chapter: str) -> Optional[Path]:
        """Directory containing chapter model weights (auto-detected)."""
        return _find_model_dir(self.chapter_dir(chapter))

    def stage2_label_map(self, chapter: str) -> Optional[Path]:
        """Path to chapter label_map.json."""
        return _find_label_map(self.chapter_dir(chapter))

    def stage2_test_split(self, chapter: str) -> Path:
        """Path to chapter test_split.parquet."""
        return self.chapter_dir(chapter) / "test_split.parquet"

    def stage2_train_split(self, chapter: str) -> Path:
        return self.chapter_dir(chapter) / "train_split.parquet"

    def stage2_val_split(self, chapter: str) -> Path:
        return self.chapter_dir(chapter) / "val_split.parquet"

    def stage2_temperature(self, chapter: str) -> Path:
        """
        Canonical write location for chapter temperature.json.
        Always written to stage2/{ch}/ root (flat).
        """
        return self.chapter_dir(chapter) / "temperature.json"

    def stage2_temperature_existing(self, chapter: str) -> Optional[Path]:
        """Find existing chapter temperature.json (any convention)."""
        return _find_temperature(self.chapter_dir(chapter))

    def stage2_results(self) -> Path:
        """Path to stage2_results.json (skip chapters + trained chapters)."""
        return self._s2_base / "stage2_results.json"

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def calibration_report(self) -> Path:
        """Path to calibration_report.json."""
        return self._base / "calibration_report.json"

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def eval_dir(self) -> Path:
        return self._base / "eval"

    def eval_summary(self) -> Path:
        return self.eval_dir() / "summary.json"

    # ------------------------------------------------------------------
    # Experiment log
    # ------------------------------------------------------------------

    def train_result(self) -> Path:
        return self._base / "train_result.json"

    # ------------------------------------------------------------------
    # Existence checks
    # ------------------------------------------------------------------

    def stage1_trained(self) -> bool:
        """True if Stage-1 model weights exist."""
        return self.stage1_model_dir() is not None

    def stage2_trained(self, chapter: str) -> bool:
        """True if chapter model weights exist."""
        return self.stage2_model_dir(chapter) is not None

    def stage2_all_trained(self, chapters: list[str]) -> bool:
        """True if all specified chapters have model weights."""
        return all(self.stage2_trained(ch) for ch in chapters)

    def calibrated(self) -> bool:
        return self.calibration_report().exists()

    def evaluated(self) -> bool:
        return self.eval_summary().exists()

    # ------------------------------------------------------------------
    # Debug
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        s1 = self.stage1_model_dir()
        return (
            f"ExperimentPaths('{self.experiment_name}')\n"
            f"  stage1_model : {s1 or 'NOT FOUND'}\n"
            f"  stage2_base  : {self._s2_base}\n"
            f"  calibrated   : {self.calibrated()}\n"
            f"  evaluated    : {self.evaluated()}"
        )


# ---------------------------------------------------------------------------
# Convenience function for scripts that need a quick path
# ---------------------------------------------------------------------------

def experiment_paths(
    experiment_name: str,
    stage1_experiment: Optional[str] = None,
) -> ExperimentPaths:
    """Shorthand constructor."""
    return ExperimentPaths(experiment_name, stage1_experiment)