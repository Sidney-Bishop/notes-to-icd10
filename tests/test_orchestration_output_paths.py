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
  - validate_mimic_evaluate.py     → mimic_iv_validation_<PREFIX>/summary.json

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

    def test_mimic_output_is_per_experiment_validation_summary(self):
        # Output dir is now keyed on the EHIER id-prefix so the two publication
        # runs don't collide (Q12a). Default preset's hier router is E-051.
        outs = _only_output("mimic_deleaked")
        prefix = orch.PRESET_DELEAKED.hier_cbert.split("_")[0]
        expected = f"mimic_iv_validation_{prefix}/summary.json"
        assert any(o.endswith(expected) for o in outs), f"expected {expected}, got {outs}"
        # must NOT be the old fixed path that collided
        assert not any(o.endswith("mimic_iv_validation/summary.json") for o in outs), \
            f"mimic still declares the OLD colliding path: {outs}"


class TestTwoPresetsLaunchSafety:
    """
    Guards for the two publication runs (leaky + de-leaked). These are the
    launch-blocker regressions from Q12: the two runs must not collide and must
    carry the correct per-run MIMIC reference flag, and the two presets must
    differ ONLY in gold path, experiment names, and the deleaked flag.
    """

    def _by_name(self, run):
        return {s.name: s for s in orch.build_phase_specs(run=run)}

    def test_both_presets_build_same_phase_count(self):
        dl = orch.build_phase_specs(run=orch.PRESET_DELEAKED)
        lk = orch.build_phase_specs(run=orch.PRESET_LEAKY)
        assert len(dl) == len(lk), (len(dl), len(lk))

    def test_mimic_outputs_are_distinct_between_presets(self):
        """The core Q12a regression: the two runs must not write MIMIC to the
        same file, or the second overwrites the first's publication result."""
        dl = self._by_name(orch.PRESET_DELEAKED)
        lk = self._by_name(orch.PRESET_LEAKY)
        dl_mimic = next(s for n, s in dl.items() if n.startswith("mimic"))
        lk_mimic = next(s for n, s in lk.items() if n.startswith("mimic"))
        assert dl_mimic.outputs[0] != lk_mimic.outputs[0], \
            f"MIMIC outputs collide: {dl_mimic.outputs} == {lk_mimic.outputs}"

    def test_deleaked_reference_flag_is_per_preset(self):
        """Q12b: de-leaked run compares MIMIC against the de-leaked reference;
        the leaky run must NOT pass --deleaked-reference."""
        dl = self._by_name(orch.PRESET_DELEAKED)
        lk = self._by_name(orch.PRESET_LEAKY)
        dl_mimic = next(s for n, s in dl.items() if n.startswith("mimic"))
        lk_mimic = next(s for n, s in lk.items() if n.startswith("mimic"))
        assert "--deleaked-reference" in dl_mimic.cmd, dl_mimic.cmd
        assert "--deleaked-reference" not in lk_mimic.cmd, lk_mimic.cmd

    def test_no_experiment_name_shared_between_presets(self):
        """Every experiment name must differ between the two runs, so neither
        overwrites the other's artifacts on disk."""
        def exp_names(run):
            names = set()
            for s in orch.build_phase_specs(run=run):
                cmd = s.cmd
                for flag in ("--experiment", "--base-experiment"):
                    if flag in cmd:
                        names.add(cmd[cmd.index(flag) + 1])
            return names
        dl_names = exp_names(orch.PRESET_DELEAKED)
        lk_names = exp_names(orch.PRESET_LEAKY)
        assert dl_names, "no experiment names found — test wiring broken"
        assert dl_names.isdisjoint(lk_names), \
            f"experiment names shared between presets: {dl_names & lk_names}"

    def test_presets_differ_only_in_gold_names_and_flag(self):
        """
        The clean-A/B invariant: after normalising OUT the gold path, the
        experiment names, and the --deleaked-reference flag, the two presets'
        commands must be IDENTICAL. If they differ anywhere else, a
        hyperparameter has drifted between the runs and it's no longer a clean
        A/B on redaction level alone.
        """
        dl = orch.build_phase_specs(run=orch.PRESET_DELEAKED)
        lk = orch.build_phase_specs(run=orch.PRESET_LEAKY)
        assert len(dl) == len(lk)

        DL, LK = orch.PRESET_DELEAKED, orch.PRESET_LEAKY

        import re

        def normalise(cmd, run):
            out = []
            for tok in cmd:
                t = tok
                # strip the gold path
                t = t.replace(run.gold_path, "<GOLD>")
                # strip any experiment number (E-0NN) and the run suffix, so
                # only the structural role text remains
                t = re.sub(r"E-0\d+", "E-0<N>", t)
                t = t.replace(run.suffix, "<SUFFIX>")
                # drop the deleaked-reference flag entirely
                if t == "--deleaked-reference":
                    continue
                out.append(t)
            return out

        for a, b in zip(dl, lk):
            na = normalise(a.cmd, DL)
            nb = normalise(b.cmd, LK)
            assert na == nb, (
                f"phase {a.name!r}/{b.name!r} differ beyond gold+names+flag:\n"
                f"  deleaked: {na}\n  leaky:    {nb}"
            )
