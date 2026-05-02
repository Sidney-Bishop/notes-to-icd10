"""
tests/test_paths.py
===================
Unit tests for src/paths.py — ExperimentPaths path resolution.

Tests the three layout conventions (FLAT, SINGLE, NESTED) and verifies
that auto-detection works correctly for each. Uses tmp_path fixtures so
no real experiment artifacts are required.

Run with:
    uv run pytest tests/test_paths.py -v
"""

import json
import pytest
from pathlib import Path
from unittest.mock import patch


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_eval_base(tmp_path: Path) -> Path:
    """Creates a fake outputs/evaluations/ directory tree."""
    eval_base = tmp_path / "outputs" / "evaluations"
    eval_base.mkdir(parents=True)
    return eval_base


def _make_model_file(directory: Path) -> None:
    """Write a minimal model.safetensors sentinel file."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "model.safetensors").touch()
    (directory / "config.json").write_text("{}")


def _make_label_map(directory: Path, num_labels: int = 22) -> None:
    """Write a minimal label_map.json."""
    directory.mkdir(parents=True, exist_ok=True)
    label_map = {
        "label2id": {str(i): i for i in range(num_labels)},
        "id2label": {str(i): str(i) for i in range(num_labels)},
        "num_labels": num_labels,
    }
    (directory / "label_map.json").write_text(json.dumps(label_map))


def _make_temperature(directory: Path, t: float = 1.25) -> None:
    """Write a minimal temperature.json."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "temperature.json").write_text(json.dumps({"temperature": t}))


@pytest.fixture
def patched_paths(fake_eval_base: Path, monkeypatch):
    """
    Patch _eval_base() to return our tmp directory.
    Returns the fake eval_base path for test use.
    """
    import src.paths as paths_module
    monkeypatch.setattr(paths_module, "_eval_base", lambda: fake_eval_base)
    return fake_eval_base


# ---------------------------------------------------------------------------
# _find_model_dir — layout convention detection
# ---------------------------------------------------------------------------

class TestFindModelDir:

    def test_flat_layout(self, tmp_path):
        """FLAT: model.safetensors directly in root."""
        from src.paths import _find_model_dir
        _make_model_file(tmp_path)
        result = _find_model_dir(tmp_path)
        assert result == tmp_path

    def test_single_nested_layout(self, tmp_path):
        """SINGLE: model.safetensors in root/model/."""
        from src.paths import _find_model_dir
        model_dir = tmp_path / "model"
        _make_model_file(model_dir)
        result = _find_model_dir(tmp_path)
        assert result == model_dir

    def test_double_nested_layout(self, tmp_path):
        """NESTED: model.safetensors in root/model/model/ (notebook legacy)."""
        from src.paths import _find_model_dir
        model_dir = tmp_path / "model" / "model"
        _make_model_file(model_dir)
        result = _find_model_dir(tmp_path)
        assert result == model_dir

    def test_flat_takes_priority_over_nested(self, tmp_path):
        """FLAT layout is returned first when both flat and nested exist."""
        from src.paths import _find_model_dir
        _make_model_file(tmp_path)
        _make_model_file(tmp_path / "model")
        result = _find_model_dir(tmp_path)
        assert result == tmp_path

    def test_returns_none_when_no_weights(self, tmp_path):
        """Returns None when no model.safetensors exists anywhere."""
        from src.paths import _find_model_dir
        result = _find_model_dir(tmp_path)
        assert result is None

    def test_config_json_alone_is_not_enough(self, tmp_path):
        """config.json without model.safetensors returns None."""
        from src.paths import _find_model_dir
        tmp_path.mkdir(parents=True, exist_ok=True)
        (tmp_path / "config.json").write_text("{}")
        result = _find_model_dir(tmp_path)
        assert result is None


# ---------------------------------------------------------------------------
# _find_label_map
# ---------------------------------------------------------------------------

class TestFindLabelMap:

    def test_label_map_in_root(self, tmp_path):
        from src.paths import _find_label_map
        _make_label_map(tmp_path)
        result = _find_label_map(tmp_path)
        assert result == tmp_path / "label_map.json"

    def test_label_map_in_model_subdir(self, tmp_path):
        from src.paths import _find_label_map
        _make_label_map(tmp_path / "model")
        result = _find_label_map(tmp_path)
        assert result == tmp_path / "model" / "label_map.json"

    def test_returns_none_when_missing(self, tmp_path):
        from src.paths import _find_label_map
        result = _find_label_map(tmp_path)
        assert result is None


# ---------------------------------------------------------------------------
# _find_temperature
# ---------------------------------------------------------------------------

class TestFindTemperature:

    def test_temperature_in_root(self, tmp_path):
        from src.paths import _find_temperature
        _make_temperature(tmp_path, t=1.3)
        result = _find_temperature(tmp_path)
        assert result == tmp_path / "temperature.json"

    def test_temperature_in_model_subdir(self, tmp_path):
        from src.paths import _find_temperature
        _make_temperature(tmp_path / "model", t=0.9)
        result = _find_temperature(tmp_path)
        assert result == tmp_path / "model" / "temperature.json"

    def test_temperature_in_nested_model(self, tmp_path):
        from src.paths import _find_temperature
        _make_temperature(tmp_path / "model" / "model", t=1.1)
        result = _find_temperature(tmp_path)
        assert result == tmp_path / "model" / "model" / "temperature.json"

    def test_returns_none_when_missing(self, tmp_path):
        from src.paths import _find_temperature
        result = _find_temperature(tmp_path)
        assert result is None


# ---------------------------------------------------------------------------
# ExperimentPaths — construction and basic properties
# ---------------------------------------------------------------------------

class TestExperimentPathsConstruction:

    def test_experiment_dir(self, patched_paths):
        from src.paths import ExperimentPaths
        p = ExperimentPaths("E-009_Balanced_E002Init")
        assert p.experiment_dir == patched_paths / "E-009_Balanced_E002Init"

    def test_stage2_base(self, patched_paths):
        from src.paths import ExperimentPaths
        p = ExperimentPaths("E-009_Balanced_E002Init")
        assert p.stage2_base == patched_paths / "E-009_Balanced_E002Init" / "stage2"

    def test_separate_stage1_experiment(self, patched_paths):
        """Stage-1 from a different experiment (shared E-003 router)."""
        from src.paths import ExperimentPaths
        p = ExperimentPaths(
            "E-009_Balanced_E002Init",
            stage1_experiment="E-003_Hierarchical_ICD10"
        )
        assert "E-003_Hierarchical_ICD10" in str(p.stage1_test_split())
        assert "E-009_Balanced_E002Init" in str(p.stage2_base)

    def test_defaults_stage1_to_experiment_name(self, patched_paths):
        from src.paths import ExperimentPaths
        p = ExperimentPaths("E-009_Balanced_E002Init")
        assert "E-009_Balanced_E002Init" in str(p.stage1_test_split())


# ---------------------------------------------------------------------------
# ExperimentPaths — Stage-1 path resolution
# ---------------------------------------------------------------------------

class TestExperimentPathsStage1:

    def test_stage1_model_dir_flat(self, patched_paths):
        from src.paths import ExperimentPaths
        exp = "E-009_Balanced_E002Init"
        stage1_dir = patched_paths / exp / "stage1"
        _make_model_file(stage1_dir)

        p = ExperimentPaths(exp)
        assert p.stage1_model_dir() == stage1_dir

    def test_stage1_model_dir_returns_none_when_missing(self, patched_paths):
        from src.paths import ExperimentPaths
        p = ExperimentPaths("E-MISSING")
        assert p.stage1_model_dir() is None

    def test_stage1_trained_true(self, patched_paths):
        from src.paths import ExperimentPaths
        exp = "E-009_Balanced_E002Init"
        _make_model_file(patched_paths / exp / "stage1")
        p = ExperimentPaths(exp)
        assert p.stage1_trained() is True

    def test_stage1_trained_false(self, patched_paths):
        from src.paths import ExperimentPaths
        p = ExperimentPaths("E-MISSING")
        assert p.stage1_trained() is False

    def test_stage1_label_map(self, patched_paths):
        from src.paths import ExperimentPaths
        exp = "E-009_Balanced_E002Init"
        stage1_dir = patched_paths / exp / "stage1"
        _make_label_map(stage1_dir)
        p = ExperimentPaths(exp)
        assert p.stage1_label_map() == stage1_dir / "label_map.json"

    def test_stage1_temperature_write_path(self, patched_paths):
        """temperature() returns the canonical write path (always flat)."""
        from src.paths import ExperimentPaths
        exp = "E-009_Balanced_E002Init"
        p = ExperimentPaths(exp)
        expected = patched_paths / exp / "stage1" / "temperature.json"
        assert p.stage1_temperature() == expected


# ---------------------------------------------------------------------------
# ExperimentPaths — Stage-2 path resolution
# ---------------------------------------------------------------------------

class TestExperimentPathsStage2:

    def test_chapter_dir(self, patched_paths):
        from src.paths import ExperimentPaths
        exp = "E-009_Balanced_E002Init"
        p = ExperimentPaths(exp)
        assert p.chapter_dir("Z") == patched_paths / exp / "stage2" / "Z"

    def test_chapter_dir_uppercase(self, patched_paths):
        """Chapter letter is uppercased automatically."""
        from src.paths import ExperimentPaths
        p = ExperimentPaths("E-009_Balanced_E002Init")
        assert p.chapter_dir("z") == p.chapter_dir("Z")

    def test_stage2_model_dir_flat(self, patched_paths):
        from src.paths import ExperimentPaths
        exp = "E-009_Balanced_E002Init"
        ch_dir = patched_paths / exp / "stage2" / "M"
        _make_model_file(ch_dir)
        p = ExperimentPaths(exp)
        assert p.stage2_model_dir("M") == ch_dir

    def test_stage2_model_dir_single_nested(self, patched_paths):
        from src.paths import ExperimentPaths
        exp = "E-009_Balanced_E002Init"
        ch_model_dir = patched_paths / exp / "stage2" / "M" / "model"
        _make_model_file(ch_model_dir)
        p = ExperimentPaths(exp)
        assert p.stage2_model_dir("M") == ch_model_dir

    def test_stage2_trained_true(self, patched_paths):
        from src.paths import ExperimentPaths
        exp = "E-009_Balanced_E002Init"
        _make_model_file(patched_paths / exp / "stage2" / "Z")
        p = ExperimentPaths(exp)
        assert p.stage2_trained("Z") is True

    def test_stage2_trained_false(self, patched_paths):
        from src.paths import ExperimentPaths
        p = ExperimentPaths("E-MISSING")
        assert p.stage2_trained("Z") is False

    def test_stage2_all_trained(self, patched_paths):
        from src.paths import ExperimentPaths
        exp = "E-009_Balanced_E002Init"
        for ch in ["A", "B", "C"]:
            _make_model_file(patched_paths / exp / "stage2" / ch)
        p = ExperimentPaths(exp)
        assert p.stage2_all_trained(["A", "B", "C"]) is True
        assert p.stage2_all_trained(["A", "B", "C", "Z"]) is False

    def test_stage2_label_map(self, patched_paths):
        from src.paths import ExperimentPaths
        exp = "E-009_Balanced_E002Init"
        ch_dir = patched_paths / exp / "stage2" / "Z"
        _make_label_map(ch_dir)
        p = ExperimentPaths(exp)
        assert p.stage2_label_map("Z") == ch_dir / "label_map.json"

    def test_stage2_temperature_write_path(self, patched_paths):
        from src.paths import ExperimentPaths
        exp = "E-009_Balanced_E002Init"
        p = ExperimentPaths(exp)
        expected = patched_paths / exp / "stage2" / "Z" / "temperature.json"
        assert p.stage2_temperature("Z") == expected


# ---------------------------------------------------------------------------
# ExperimentPaths — existence checks
# ---------------------------------------------------------------------------

class TestExperimentPathsExistenceChecks:

    def test_calibrated_true(self, patched_paths):
        from src.paths import ExperimentPaths
        exp = "E-009_Balanced_E002Init"
        report_path = patched_paths / exp / "calibration_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("{}")
        p = ExperimentPaths(exp)
        assert p.calibrated() is True

    def test_calibrated_false(self, patched_paths):
        from src.paths import ExperimentPaths
        p = ExperimentPaths("E-MISSING")
        assert p.calibrated() is False

    def test_evaluated_true(self, patched_paths):
        from src.paths import ExperimentPaths
        exp = "E-009_Balanced_E002Init"
        eval_path = patched_paths / exp / "eval" / "summary.json"
        eval_path.parent.mkdir(parents=True, exist_ok=True)
        eval_path.write_text("{}")
        p = ExperimentPaths(exp)
        assert p.evaluated() is True

    def test_evaluated_false(self, patched_paths):
        from src.paths import ExperimentPaths
        p = ExperimentPaths("E-MISSING")
        assert p.evaluated() is False