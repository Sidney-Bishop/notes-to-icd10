"""
tests/test_orchestration.py
===========================
Unit tests for src/orchestration.py — the testable logic extracted from the
full de-leaked rebuild orchestrator.

The high-value tests are over build_phase_specs(): they assert the exact
properties we spent the whole session manually guarding —
  - every hierarchical phase sets --stage1-model to its backbone, NEVER the
    roberta-base default (the silent-failure trap)
  - every phase reads the DE-LEAKED gold, never the leaky/augmented gold
  - flat pretrain precedes its hierarchical consumer (dependency order)
  - exit codes propagate truthfully (a failing command yields nonzero)

Run with:
    uv run pytest tests/test_orchestration.py -v
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import orchestration as orch


# ---------------------------------------------------------------------------
# Formatting (pure)
# ---------------------------------------------------------------------------

class TestFormatting:
    def test_event_line_includes_timestamp_and_message(self):
        line = orch.event_line("hello", when="2026-01-01T00:00:00")
        assert line == "[2026-01-01T00:00:00] hello"

    def test_classify_exit_zero_is_ok(self):
        assert orch.classify_exit(0) == "OK"

    def test_classify_exit_nonzero_is_failed(self):
        assert orch.classify_exit(1) == "FAILED"
        assert orch.classify_exit(137) == "FAILED"


# ---------------------------------------------------------------------------
# statline / preflight / postflight (tmp-dir)
# ---------------------------------------------------------------------------

class TestStatline:
    def test_present_file_reports_size_and_mtime(self, tmp_path):
        f = tmp_path / "x.parquet"
        f.write_text("data")
        line = orch.statline(f)
        assert str(f) in line and "bytes" in line and "MISSING" not in line

    def test_missing_file_reports_missing(self, tmp_path):
        line = orch.statline(tmp_path / "nope.parquet")
        assert line.endswith("MISSING")


class TestPreflight:
    def test_passes_when_all_present(self, tmp_path):
        f = tmp_path / "gold.parquet"; f.write_text("g")
        ok, lines = orch.preflight([f])
        assert ok is True
        assert any("READ" in ln for ln in lines)

    def test_fails_when_missing_in_live_mode(self, tmp_path):
        ok, lines = orch.preflight([tmp_path / "absent.parquet"], dry_run=False)
        assert ok is False
        assert any("PRE-FLIGHT FAILED" in ln for ln in lines)

    def test_missing_does_not_fail_in_dry_run(self, tmp_path):
        """dry-run must keep going so all planned commands log."""
        ok, lines = orch.preflight([tmp_path / "absent.parquet"], dry_run=True)
        assert ok is True
        assert any("DRY" in ln for ln in lines)


class TestPostflight:
    def test_verdict_ok_when_all_present(self, tmp_path):
        f = tmp_path / "model.safetensors"; f.write_text("w")
        ok, lines = orch.postflight([f])
        assert ok is True
        assert any("VERDICT: ✓" in ln for ln in lines)

    def test_verdict_missing_when_absent(self, tmp_path):
        ok, lines = orch.postflight([tmp_path / "missing.safetensors"])
        assert ok is False
        assert any("MISSING" in ln for ln in lines)


class TestFlatDone:
    def test_true_when_model_present(self, tmp_path):
        exp = orch.PRESET_DELEAKED.flat_cbert
        mdir = tmp_path / exp / "model"
        mdir.mkdir(parents=True)
        (mdir / "model.safetensors").write_text("w")
        assert orch.flat_done(tmp_path, exp) is True

    def test_false_when_absent(self, tmp_path):
        assert orch.flat_done(tmp_path, "E-NONE") is False


# ---------------------------------------------------------------------------
# Phase specs — THE HIGH-VALUE TESTS
# ---------------------------------------------------------------------------

class TestPhaseSpecs:
    def test_six_core_phases_lead_in_expected_order(self):
        """The flat/hier backbone phases lead the run, in dependency order.
        (SupCon chain + MIMIC follow; covered separately.)"""
        specs = orch.build_phase_specs()
        names = [s.name for s in specs]
        assert names[:6] == [
            "flat_clinicalbert", "hier_clinicalbert",
            "flat_clinical_modernbert", "hier_clinical_modernbert",
            "flat_bioclinical_modernbert", "hier_bioclinical_modernbert",
        ]

    def test_no_phase_uses_leaky_gold(self):
        """
        No phase may reference the leaky/augmented gold. Note: some steps
        (supcon_train) legitimately take NO gold path — they inherit splits
        from --source-experiment — so we assert absence of leaky gold rather
        than presence of de-leaked gold everywhere.
        """
        for s in orch.build_phase_specs():
            joined = " ".join(s.cmd)
            assert "augmented" not in joined, f"{s.name} references augmented gold"
            assert "medsynth_gold_apso.parquet" not in joined, (
                f"{s.name} references the leaky (non-deleaked) gold"
            )

    def test_gold_bearing_phases_use_deleaked_gold(self):
        """Phases that DO take a --gold-path must use the de-leaked one."""
        for s in orch.build_phase_specs():
            if "--gold-path" in s.cmd:
                gp = s.cmd[s.cmd.index("--gold-path") + 1]
                assert gp == orch.GOLD_DELEAKED, f"{s.name} gold-path is {gp}"

    def test_no_hier_phase_uses_roberta_default_stage1(self):
        """
        The silent-failure trap: run_experiment.py defaults --stage1-model to
        roberta-base. Every hierarchical phase must pass its own backbone.
        """
        for s in orch.build_phase_specs():
            if s.name.startswith("hier_"):
                assert "--stage1-model" in s.cmd, f"{s.name} missing --stage1-model"
                idx = s.cmd.index("--stage1-model")
                stage1_model = s.cmd[idx + 1]
                assert stage1_model != orch.ROBERTA_DEFAULT, (
                    f"{s.name} would train Stage-1 on the roberta-base default"
                )

    def test_hier_stage1_model_matches_phase_backbone(self):
        """Stage-1 backbone must equal the phase's --model (not some other)."""
        for s in orch.build_phase_specs():
            if s.name.startswith("hier_"):
                model = s.cmd[s.cmd.index("--model") + 1]
                stage1 = s.cmd[s.cmd.index("--stage1-model") + 1]
                assert model == stage1, f"{s.name}: --model {model} != --stage1-model {stage1}"

    def test_flat_precedes_its_hier_consumer(self):
        """Each hier phase's stage2-init points at a flat experiment built earlier."""
        specs = orch.build_phase_specs()
        produced_flats = set()
        for s in specs:
            if s.is_flat_pretrain:
                produced_flats.add(s.flat_experiment)
            elif s.name.startswith("hier_"):
                init = s.cmd[s.cmd.index("--stage2-init") + 1]
                init_exp = init.rsplit("/", 1)[-1]
                assert init_exp in produced_flats, (
                    f"{s.name} inits from {init_exp} which no earlier flat phase produced"
                )

    def test_dry_run_appends_dry_flag_to_hier(self):
        specs = orch.build_phase_specs(dry_run=True)
        for s in specs:
            if s.name.startswith("hier_"):
                assert "--dry-run" in s.cmd

    def test_live_run_has_no_dry_flag(self):
        specs = orch.build_phase_specs(dry_run=False)
        for s in specs:
            assert "--dry-run" not in s.cmd


# ---------------------------------------------------------------------------
# SupCon chain + MIMIC specs — guard the LEAKY-default traps
# ---------------------------------------------------------------------------

class TestSupConAndMimicSpecs:
    def _by_name(self):
        return {s.name: s for s in orch.build_phase_specs()}

    def test_supcon_chain_present_in_order(self):
        names = [s.name for s in orch.build_phase_specs()]
        for step in ("supcon_presplits", "supcon_zbase", "supcon_train",
                     "supcon_calibrate", "supcon_hybrid"):
            assert step in names
        # order: presplits < zbase < train < calibrate < hybrid
        idx = {n: i for i, n in enumerate(names)}
        assert (idx["supcon_presplits"] < idx["supcon_zbase"]
                < idx["supcon_train"] < idx["supcon_calibrate"]
                < idx["supcon_hybrid"])

    def test_supcon_calibrate_uses_deleaked_router_not_e003(self):
        """
        calibrate.py defaults --stage1-experiment to E-003 (leaky router).
        The de-leaked SupCon MUST calibrate against the de-leaked router
        (E-021), or we recreate the train/serve mismatch that cratered E-015.
        """
        cal = self._by_name()["supcon_calibrate"]
        idx = cal.cmd.index("--stage1-experiment")
        assert cal.cmd[idx + 1] == orch.PRESET_DELEAKED.hier_cbert
        assert "E-003_Hierarchical_ICD10" not in cal.cmd

    def test_supcon_hybrid_base_and_stage1_are_deleaked(self):
        """hybrid defaults base=E-010, stage1=E-003 (both leaky). Must override."""
        hyb = self._by_name()["supcon_hybrid"]
        base = hyb.cmd[hyb.cmd.index("--base-experiment") + 1]
        stage1 = hyb.cmd[hyb.cmd.index("--stage1-experiment") + 1]
        assert base == orch.PRESET_DELEAKED.hier_cbert
        assert stage1 == orch.PRESET_DELEAKED.hier_cbert
        assert "E-010_40ep_E002Init" not in hyb.cmd
        assert "E-003_Hierarchical_ICD10" not in hyb.cmd

    def test_supcon_hybrid_overrides_z_with_supcon_resolver(self):
        hyb = self._by_name()["supcon_hybrid"]
        assert "--override" in hyb.cmd
        ov = hyb.cmd[hyb.cmd.index("--override") + 1]
        assert ov == f"Z={orch.PRESET_DELEAKED.supcon_z}"

    def test_mimic_carries_deleaked_reference_and_deleaked_base(self):
        m = self._by_name()["mimic_deleaked"]
        assert "--deleaked-reference" in m.cmd
        base = m.cmd[m.cmd.index("--base-experiment") + 1]
        assert base == orch.PRESET_DELEAKED.hier_cbert
        assert "E-010_40ep_E002Init" not in m.cmd

    def test_dry_run_unsupported_steps_flagged(self):
        """
        prepare_splits / evaluate_hybrid / validate_mimic_evaluate lack
        --dry-run; their specs must say so, so the CLI skips them in dry-run
        rather than passing an unsupported flag.
        """
        by = self._by_name()
        assert by["supcon_presplits"].supports_dry_run is False
        assert by["supcon_hybrid"].supports_dry_run is False
        assert by["mimic_deleaked"].supports_dry_run is False

    def test_dry_run_supported_steps_default_true(self):
        by = self._by_name()
        assert by["supcon_zbase"].supports_dry_run is True
        assert by["supcon_train"].supports_dry_run is True
        assert by["supcon_calibrate"].supports_dry_run is True

    def test_no_supcon_or_mimic_step_uses_leaky_gold(self):
        for s in orch.build_phase_specs():
            assert "augmented" not in " ".join(s.cmd)


# ---------------------------------------------------------------------------
# Executor — exit-code propagation (the riskiest line, now testable)
# ---------------------------------------------------------------------------

class TestRunStep:
    def test_dry_run_does_not_execute(self, tmp_path):
        res = orch.run_step("noop", ["false"], tmp_path / "l.log", tmp_path, dry_run=True)
        assert res.status == "dry_run"
        assert res.returncode == 0
        assert not (tmp_path / "l.log").exists()   # nothing written in dry-run

    def test_successful_command_returns_zero_and_ok(self, tmp_path):
        log = tmp_path / "ok.log"
        res = orch.run_step("echo", ["echo", "hello"], log, tmp_path)
        assert res.returncode == 0
        assert res.status == "OK"
        assert "hello" in log.read_text()

    def test_failing_command_propagates_nonzero(self, tmp_path):
        """
        THE critical test: a command that fails must yield a nonzero exit and
        FAILED status — not be silently swallowed. This is the failure mode the
        bash PIPESTATUS line risked getting wrong.
        """
        log = tmp_path / "fail.log"
        res = orch.run_step("false", ["false"], log, tmp_path)
        assert res.returncode != 0
        assert res.status == "FAILED"

    def test_output_is_written_to_logfile(self, tmp_path):
        log = tmp_path / "out.log"
        orch.run_step("echo", ["echo", "captured-line"], log, tmp_path)
        assert "captured-line" in log.read_text()
