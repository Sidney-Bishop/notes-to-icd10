"""
tests/test_experiment_logger.py
================================
Unit tests for src/experiment_logger.py.

Verifies that log_start, log_complete, log_results, and log_failed
write correctly to both outputs/experiments.json and outputs/run.log.
Uses tmp_path so no real outputs are touched.

Note on registry structure: log_results() stores metrics under
registry[experiment_name]["results"][metric_key], not at the top level.

Run with:
    uv run pytest tests/test_experiment_logger.py -v
"""

import json
import pytest
from pathlib import Path


# ---------------------------------------------------------------------------
# Fixture — redirect logger outputs to tmp_path
# ---------------------------------------------------------------------------

@pytest.fixture
def logger_tmp(tmp_path: Path, monkeypatch):
    """
    Patches _project_root() to return tmp_path so all logger
    writes go to a temporary directory. Returns the tmp_path.
    """
    import src.experiment_logger as el_module
    monkeypatch.setattr(el_module, "_project_root", lambda: tmp_path)
    (tmp_path / "outputs").mkdir(parents=True, exist_ok=True)
    return tmp_path


def _read_registry(logger_tmp):
    return json.loads(
        (logger_tmp / "outputs" / "experiments.json").read_text()
    )


def _read_run_log(logger_tmp):
    p = logger_tmp / "outputs" / "run.log"
    return p.read_text() if p.exists() else ""


# ---------------------------------------------------------------------------
# ExperimentLogger construction
# ---------------------------------------------------------------------------

class TestExperimentLoggerConstruction:

    def test_creates_registry_file(self, logger_tmp):
        from src.experiment_logger import ExperimentLogger
        ExperimentLogger("E-TEST", script="test.py")
        assert (logger_tmp / "outputs" / "experiments.json").exists()

    def test_run_log_written_after_action(self, logger_tmp):
        """run.log is written lazily — only after the first action."""
        from src.experiment_logger import ExperimentLogger
        el = ExperimentLogger("E-TEST", script="test.py")
        el.log_start("train_stage2")
        assert (logger_tmp / "outputs" / "run.log").exists()

    def test_experiment_registered_on_init(self, logger_tmp):
        from src.experiment_logger import ExperimentLogger
        ExperimentLogger("E-TEST", script="test.py")
        registry = _read_registry(logger_tmp)
        assert "E-TEST" in registry


# ---------------------------------------------------------------------------
# log_start
# ---------------------------------------------------------------------------

class TestLogStart:

    def test_log_start_writes_to_registry(self, logger_tmp):
        from src.experiment_logger import ExperimentLogger
        el = ExperimentLogger("E-TEST", script="test.py")
        el.log_start("train_stage2", params={"epochs": 20, "lr": 2e-5})
        assert "E-TEST" in _read_registry(logger_tmp)

    def test_log_start_writes_to_run_log(self, logger_tmp):
        from src.experiment_logger import ExperimentLogger
        el = ExperimentLogger("E-TEST", script="test.py")
        el.log_start("train_stage2", params={"epochs": 20})
        run_log = _read_run_log(logger_tmp)
        assert "train_stage2" in run_log
        assert "E-TEST" in run_log

    def test_log_start_with_no_params(self, logger_tmp):
        from src.experiment_logger import ExperimentLogger
        el = ExperimentLogger("E-TEST", script="test.py")
        el.log_start("train_stage1")  # should not raise


# ---------------------------------------------------------------------------
# log_complete
# ---------------------------------------------------------------------------

class TestLogComplete:

    def test_log_complete_writes_to_run_log(self, logger_tmp):
        from src.experiment_logger import ExperimentLogger
        el = ExperimentLogger("E-TEST", script="test.py")
        el.log_start("train_stage2")
        el.log_complete("train_stage2", artifacts={"stage2_dir": "/tmp/stage2"})
        run_log = _read_run_log(logger_tmp)
        assert "train_stage2" in run_log

    def test_log_complete_with_no_artifacts(self, logger_tmp):
        from src.experiment_logger import ExperimentLogger
        el = ExperimentLogger("E-TEST", script="test.py")
        el.log_complete("train_stage2")  # should not raise


# ---------------------------------------------------------------------------
# log_results
# ---------------------------------------------------------------------------

class TestLogResults:

    def test_log_results_writes_metrics_to_registry(self, logger_tmp):
        """Metrics stored under registry[exp]['results'][key]."""
        from src.experiment_logger import ExperimentLogger
        el = ExperimentLogger("E-TEST", script="test.py")
        el.log_results({
            "e2e_accuracy": 0.839,
            "macro_f1":     0.762,
            "stage1_accuracy": 0.987,
        })
        registry = _read_registry(logger_tmp)
        results = registry["E-TEST"]["results"]
        assert results.get("e2e_accuracy") == pytest.approx(0.839)
        assert results.get("macro_f1")     == pytest.approx(0.762)

    def test_log_results_multiple_calls_overwrite(self, logger_tmp):
        """Second log_results call replaces the results block."""
        from src.experiment_logger import ExperimentLogger
        el = ExperimentLogger("E-TEST", script="test.py")
        el.log_results({"e2e_accuracy": 0.5, "macro_f1": 0.4})
        el.log_results({"e2e_accuracy": 0.839, "macro_f1": 0.762})
        results = _read_registry(logger_tmp)["E-TEST"]["results"]
        assert results["e2e_accuracy"] == pytest.approx(0.839)

    def test_log_results_writes_to_run_log(self, logger_tmp):
        from src.experiment_logger import ExperimentLogger
        el = ExperimentLogger("E-TEST", script="test.py")
        el.log_results({"e2e_accuracy": 0.839, "macro_f1": 0.762})
        assert "E-TEST" in _read_run_log(logger_tmp)


# ---------------------------------------------------------------------------
# log_failed
# ---------------------------------------------------------------------------

class TestLogFailed:

    def test_log_failed_writes_to_run_log(self, logger_tmp):
        from src.experiment_logger import ExperimentLogger
        el = ExperimentLogger("E-TEST", script="test.py")
        el.log_failed("train_stage2", reason="OOM on chapter M")
        assert "E-TEST" in _read_run_log(logger_tmp)

    def test_log_failed_with_no_reason(self, logger_tmp):
        from src.experiment_logger import ExperimentLogger
        el = ExperimentLogger("E-TEST", script="test.py")
        el.log_failed("train_stage2")  # should not raise


# ---------------------------------------------------------------------------
# Multiple experiments in registry
# ---------------------------------------------------------------------------

class TestMultipleExperiments:

    def test_two_experiments_coexist(self, logger_tmp):
        from src.experiment_logger import ExperimentLogger
        el1 = ExperimentLogger("E-001", script="test.py")
        el1.log_results({"e2e_accuracy": 0.872, "macro_f1": 0.841})

        el2 = ExperimentLogger("E-002", script="test.py")
        el2.log_results({"e2e_accuracy": 0.733, "macro_f1": 0.634})

        registry = _read_registry(logger_tmp)
        assert "E-001" in registry
        assert "E-002" in registry
        assert registry["E-001"]["results"]["e2e_accuracy"] == pytest.approx(0.872)
        assert registry["E-002"]["results"]["e2e_accuracy"] == pytest.approx(0.733)

    def test_writing_e002_does_not_corrupt_e001(self, logger_tmp):
        from src.experiment_logger import ExperimentLogger
        el1 = ExperimentLogger("E-001", script="test.py")
        el1.log_results({"e2e_accuracy": 0.872, "macro_f1": 0.841})

        el2 = ExperimentLogger("E-002", script="test.py")
        el2.log_results({"e2e_accuracy": 0.733, "macro_f1": 0.634})

        registry = _read_registry(logger_tmp)
        assert registry["E-001"]["results"]["macro_f1"] == pytest.approx(0.841)


# ---------------------------------------------------------------------------
# status() and registry() public functions
# ---------------------------------------------------------------------------

class TestPublicFunctions:

    def test_registry_returns_dict(self, logger_tmp):
        from src.experiment_logger import ExperimentLogger, registry
        el = ExperimentLogger("E-TEST", script="test.py")
        el.log_results({"e2e_accuracy": 0.839})
        result = registry()
        assert isinstance(result, dict)
        assert "E-TEST" in result

    def test_status_runs_without_error(self, logger_tmp, capsys):
        from src.experiment_logger import ExperimentLogger, status
        el = ExperimentLogger("E-TEST", script="test.py")
        el.log_results({"e2e_accuracy": 0.839, "macro_f1": 0.762})
        status()  # should not raise
        captured = capsys.readouterr()
        assert "E-TEST" in captured.out