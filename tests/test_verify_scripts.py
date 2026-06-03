"""
tests/test_verify_scripts.py
============================
Unit tests for verify_scripts.py after the testability refactor.

The headline test is test_runtime_import_*: the OLD check [9] was dead code
(`... if False else None` + a hardcoded check(..., True)) that ALWAYS passed
without importing anything. These tests assert the check now reflects ACTUAL
import success/failure — against a real module and a deliberately-broken one.

Run with:
    uv run pytest tests/test_verify_scripts.py -v
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import verify_scripts as vs


# ---------------------------------------------------------------------------
# Fixture: a minimal fake project root with the files the checks read
# ---------------------------------------------------------------------------

@pytest.fixture
def good_root(tmp_path: Path) -> Path:
    """A project tree where every check should PASS."""
    (tmp_path / "scripts").mkdir()
    (tmp_path / "src").mkdir()
    (tmp_path / "data" / "medsynth").mkdir(parents=True)
    (tmp_path / "data" / "gold").mkdir(parents=True)

    src, scripts, gold = tmp_path / "src", tmp_path / "scripts", tmp_path / "data" / "gold"

    # required src files
    (src / "experiment_logger.py").write_text(
        "class ExperimentLogger:\n"
        "    pass\n"
        "def status():\n"
        "    return 'run.log experiments.json'\n"
    )
    (src / "paths.py").write_text("# paths\n")
    (src / "inference.py").write_text(
        "from src.paths import ExperimentPaths\n"
        "def f(): ExperimentPaths('x').stage2_model_dir('Z')\n"
    )
    (src / "gatekeeper.py").write_text("# gk\n")
    (src / "preprocessing.py").write_text("# pp\n")

    # required scripts
    (scripts / "train.py").write_text(
        "from src.experiment_logger import ExperimentLogger\n"
        "exp_logger.log_start; exp_logger.log_complete\n"
        "hasattr(obj, key)\n"
        "gold_path.is_absolute()\n"
    )
    (scripts / "calibrate.py").write_text("ExperimentLogger\nlog_start\n")
    (scripts / "evaluate.py").write_text("ExperimentLogger\nlog_results\n")
    (scripts / "prepare_splits.py").write_text("is_absolute()\n")
    (scripts / "prepare_data.py").write_text(
        "hf_hub_download\nEXPECTED_SHA256\n_verify_sha256\n--offline\n"
    )
    (scripts / "generate_manifest.py").write_text("git_commit\nsha256\nschema\n")

    # DVC pointers + manifest
    (tmp_path / "data" / "medsynth" / "icd10_notes.parquet.dvc").write_text("x")
    (gold / "cdc_fy2026_icd10.parquet.dvc").write_text("x")
    (gold / "medsynth_gold_apso_deleaked.parquet.dvc").write_text("x")
    (gold / "MANIFEST_20260603.json").write_text("{}")
    return tmp_path


# ---------------------------------------------------------------------------
# Aggregate: a good tree passes everything
# ---------------------------------------------------------------------------

def test_good_root_passes_all_checks(good_root):
    results = vs.run_all_checks(good_root)
    failed = [(n, d) for n, passed, d in results if not passed]
    assert failed == [], f"unexpected failures: {failed}"


# ---------------------------------------------------------------------------
# Individual checks detect the problems they exist to catch
# ---------------------------------------------------------------------------

def test_missing_required_file_is_flagged(good_root):
    (good_root / "scripts" / "train.py").unlink()
    results = dict((n, p) for n, p, _ in vs.check_required_files(good_root))
    assert results["train.py"] is False

def test_z_override_detected(good_root):
    (good_root / "src" / "inference.py").write_text('pred_chapter = "Z"\nExperimentPaths stage2_model_dir\n')
    results = dict((n, p) for n, p, _ in vs.check_inference(good_root))
    assert results["Z override removed"] is False

def test_missing_sha256_flagged(good_root):
    (good_root / "scripts" / "prepare_data.py").write_text("hf_hub_download\n--offline\n")
    results = dict((n, p) for n, p, _ in vs.check_prepare_data(good_root))
    assert results["SHA256 constants defined"] is False


# ---------------------------------------------------------------------------
# THE HEADLINE: runtime import check actually imports (old check [9] did not)
# ---------------------------------------------------------------------------

class TestRuntimeImport:
    def test_passes_for_importable_module(self, good_root):
        # good_root's experiment_logger.py is valid Python
        results = vs.check_runtime_import(good_root)
        name, passed, detail = results[0]
        assert passed is True, f"expected importable, got: {detail}"

    def test_fails_for_broken_module(self, good_root):
        """
        The test the OLD code could never pass: a module with a syntax error
        must make the check return False. Old check [9] hardcoded True and
        never imported, so it would have reported this broken module as OK.
        """
        (good_root / "src" / "experiment_logger.py").write_text(
            "class ExperimentLogger:\n    this is not valid python <<<\n"
        )
        results = vs.check_runtime_import(good_root)
        name, passed, detail = results[0]
        assert passed is False
        assert detail  # carries the actual exception message

    def test_fails_for_import_time_error(self, good_root):
        """A module that raises at import (not just syntax) is also caught."""
        (good_root / "src" / "experiment_logger.py").write_text(
            "raise RuntimeError('boom at import')\n"
        )
        results = vs.check_runtime_import(good_root)
        _, passed, detail = results[0]
        assert passed is False
        assert "boom at import" in detail
