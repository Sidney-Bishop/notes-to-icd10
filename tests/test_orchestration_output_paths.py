"""
test_orchestration_output_paths.py
==================================
Regression tests for the phase-spec OUTPUT paths — the gap that let the
SupCon postflight falsely report MISSING during the de-leaked rebuild.

The orchestrator declares an `outputs` list per phase and postflight checks
those paths exist. If a declared path does not match where the script actually
writes, postflight reports a false MISSING and halts the chain. These tests
pin each de-leaked tail-step's declared output to the script's real save
convention (verified against the artifacts produced by run 20260603_200854).

Save conventions (verified on disk):
  - train.py (supcon_zbase)        → stage2/Z/model/model.safetensors  (NESTED)
  - train_supcon_z.py (supcon_train) → stage2/Z/model.safetensors        (FLAT)
  - calibrate.py                   → <exp>/calibration_report.json
  - evaluate_hybrid.py             → <base>_hybrid_Z-<id>/eval/summary.json
                                     where <id> = override_exp.split('_')[0]
  - validate_mimic_evaluate.py     → mimic_iv_validation/summary.json

Run: uv run pytest tests/test_orchestration_output_paths.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import orchestration as orch


def _spec(name):
    for s in orch.build_phase_specs():
        if s.name == name:
            return s
    raise AssertionError(f"no phase spec named {name}")


def _only_output(name):
    s = _spec(name)
    assert len(s.outputs) >= 1, f"{name} has no declared outputs"
    return s.outputs


class TestTailStepOutputPaths:
    def test_supcon_zbase_output_is_nested(self):
        """train.py nests weights under model/ — this one was already correct."""
        outs = _only_output("supcon_zbase")
        assert any(o.endswith("stage2/Z/model/model.safetensors") for o in outs), outs

    def test_supcon_train_output_is_flat(self):
        """
        train_supcon_z.py saves FLAT (stage2/Z/model.safetensors), NOT nested.
        The spec originally declared the nested path → postflight false MISSING
        → chain halted. This pins the flat convention.
        """
        outs = _only_output("supcon_train")
        assert any(o.endswith("stage2/Z/model.safetensors") for o in outs), \
            f"supcon_train output should be FLAT stage2/Z/model.safetensors, got {outs}"
        # and must NOT be the nested path that caused the bug
        assert not any(o.endswith("stage2/Z/model/model.safetensors") for o in outs), \
            f"supcon_train still declares the NESTED path (the original bug): {outs}"

    def test_supcon_calibrate_output_is_report(self):
        outs = _only_output("supcon_calibrate")
        assert any(o.endswith("calibration_report.json") for o in outs), outs

    def test_supcon_hybrid_declares_real_eval_summary(self):
        """
        evaluate_hybrid.py writes <base>_hybrid_Z-<id>/eval/summary.json where
        <id> = override_exp.split('_')[0]. The spec originally had EMPTY outputs
        (nothing checked). This pins the real derived path.
        """
        outs = _only_output("supcon_hybrid")
        # must reference a hybrid eval summary, not be empty
        assert any("_hybrid_Z-" in o and o.endswith("/eval/summary.json") for o in outs), \
            f"supcon_hybrid must declare the real hybrid eval summary path, got {outs}"

    def test_mimic_output_is_validation_summary(self):
        outs = _only_output("mimic_deleaked")
        assert any(o.endswith("mimic_iv_validation/summary.json") for o in outs), outs
