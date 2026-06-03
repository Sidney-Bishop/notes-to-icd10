"""
tests/test_paths_stage2_splits.py
=================================
Extends ExperimentPaths coverage for the Stage-2 per-chapter SPLIT paths,
which had no tests — the gap that let E-016 ship with model weights but only
test_split.parquet (no train/val), silently breaking downstream SupCon.

Two concerns:
  1. Path construction — stage2_{train,val,test}_split return the right paths.
  2. Split COMPLETENESS — detect the E-016 state: model present, but train/val
     splits missing. This is the guard whose absence cost real debugging time.

Conventions match tests/test_paths.py (patched_paths fixture, _make_* helpers).
The completeness tests assert a stage2_splits_complete() method that SHOULD
exist; until paths.py provides it they go red, documenting the missing guard.

Run with:
    uv run pytest tests/test_paths_stage2_splits.py -v
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ---------------------------------------------------------------------------
# Fixtures (mirrors test_paths.py)
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_eval_base(tmp_path: Path) -> Path:
    eval_base = tmp_path / "outputs" / "evaluations"
    eval_base.mkdir(parents=True)
    return eval_base


@pytest.fixture
def patched_paths(fake_eval_base: Path, monkeypatch):
    """Patch ExperimentPaths' eval-base resolution to the tmp dir."""
    import src.paths as paths_mod
    # test_paths.py patches the same internal; mirror whatever it targets.
    monkeypatch.setattr(paths_mod, "_eval_base", lambda: fake_eval_base, raising=False)
    return fake_eval_base


def _make_model_file(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "model.safetensors").touch()
    (directory / "config.json").write_text("{}")


def _make_split(directory: Path, name: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{name}_split.parquet").touch()


# ---------------------------------------------------------------------------
# 1. Path construction
# ---------------------------------------------------------------------------

class TestStage2SplitPaths:
    def test_train_split_path(self, patched_paths):
        from src.paths import ExperimentPaths
        p = ExperimentPaths("E-TEST")
        assert p.stage2_train_split("Z") == p.chapter_dir("Z") / "train_split.parquet"

    def test_val_split_path(self, patched_paths):
        from src.paths import ExperimentPaths
        p = ExperimentPaths("E-TEST")
        assert p.stage2_val_split("Z") == p.chapter_dir("Z") / "val_split.parquet"

    def test_test_split_path(self, patched_paths):
        from src.paths import ExperimentPaths
        p = ExperimentPaths("E-TEST")
        assert p.stage2_test_split("Z") == p.chapter_dir("Z") / "test_split.parquet"

    def test_split_paths_differ_per_chapter(self, patched_paths):
        from src.paths import ExperimentPaths
        p = ExperimentPaths("E-TEST")
        assert p.stage2_train_split("Z") != p.stage2_train_split("M")


# ---------------------------------------------------------------------------
# 2. Split completeness — the E-016 guard (RED until paths.py adds the method)
# ---------------------------------------------------------------------------

class TestStage2SplitCompleteness:
    """
    E-016 had model weights + test_split.parquet but NO train/val splits.
    stage2_trained() returned True (weights present), so nothing flagged the
    incomplete state. We want a method that detects it.
    """
    def test_complete_when_all_three_splits_present(self, patched_paths):
        from src.paths import ExperimentPaths
        p = ExperimentPaths("E-TEST")
        ch = p.chapter_dir("Z")
        _make_model_file(ch / "model")
        for name in ("train", "val", "test"):
            _make_split(ch, name)
        assert p.stage2_splits_complete("Z") is True

    def test_incomplete_when_train_val_missing(self, patched_paths):
        """The exact E-016 state: model + test_split only."""
        from src.paths import ExperimentPaths
        p = ExperimentPaths("E-TEST")
        ch = p.chapter_dir("Z")
        _make_model_file(ch / "model")
        _make_split(ch, "test")          # only test split, like E-016
        assert p.stage2_splits_complete("Z") is False

    def test_incomplete_when_val_missing(self, patched_paths):
        from src.paths import ExperimentPaths
        p = ExperimentPaths("E-TEST")
        ch = p.chapter_dir("Z")
        _make_split(ch, "train")
        _make_split(ch, "test")
        assert p.stage2_splits_complete("Z") is False

    def test_trained_but_splits_incomplete_are_independent(self, patched_paths):
        """
        Documents the E-016 trap: stage2_trained() True does NOT imply splits
        complete. Both signals are needed; they measure different things.
        """
        from src.paths import ExperimentPaths
        p = ExperimentPaths("E-TEST")
        ch = p.chapter_dir("Z")
        _make_model_file(ch / "model")
        _make_split(ch, "test")
        assert p.stage2_trained("Z") is True            # weights present
        assert p.stage2_splits_complete("Z") is False   # but splits are not
