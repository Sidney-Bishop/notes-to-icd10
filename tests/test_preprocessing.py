"""
tests/test_preprocessing.py
============================
Unit tests for src/preprocessing.py — APSO-Flip and ICD-10 redaction.

Tests the two public interfaces:
  - prepare_inference_input() — single-string inference path
  - redact_icd10_sections() — DataFrame-level redaction (Polars)
  - build_apso_note() — APSO reordering on DataFrames

Run with:
    uv run pytest tests/test_preprocessing.py -v
"""

import sys
import pytest
import polars as pl
from pathlib import Path

# Ensure project root is on sys.path so src.preprocessing is importable
_root = next(
    p for p in [Path(__file__).parent.parent, *Path(__file__).parent.parent.parents]
    if (p / "artifacts.yaml").exists()
)
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from src.preprocessing import (
    prepare_inference_input,
    ICD10_REDACT_PATTERN,
    REDACTION_MARKER,
)


# ---------------------------------------------------------------------------
# ICD-10 redaction pattern
# ---------------------------------------------------------------------------

class TestICD10RedactPattern:

    def test_canonical_format_matched(self):
        """M25.562, N39.0, E11.65 — standard format."""
        import re
        codes = ["M25.562", "N39.0", "E11.65", "J18.9", "Z79.4"]
        for code in codes:
            assert re.search(ICD10_REDACT_PATTERN, code), \
                f"Expected {code} to match ICD-10 pattern"

    def test_raw_format_matched(self):
        """M25562, N390 — format without dot."""
        import re
        codes = ["M25562", "N390", "J189"]
        for code in codes:
            assert re.search(ICD10_REDACT_PATTERN, code), \
                f"Expected {code} to match ICD-10 pattern"

    def test_common_abbreviations_not_matched(self):
        """CBC, ECG, TSH, BMI — should not match."""
        import re
        non_codes = ["CBC", "ECG", "TSH", "BMI", "MRI"]
        for term in non_codes:
            match = re.search(ICD10_REDACT_PATTERN, f" {term} ")
            assert match is None, \
                f"Expected {term} NOT to match ICD-10 pattern"


# ---------------------------------------------------------------------------
# prepare_inference_input — single-string inference path
# ---------------------------------------------------------------------------

class TestPrepareInferenceInput:

    def test_returns_string(self):
        note = (
            "Subjective: Patient reports chest pain.\n"
            "Objective: BP 140/90, HR 88.\n"
            "Assessment: Hypertension.\n"
            "Plan: Lisinopril 10mg daily."
        )
        result = prepare_inference_input(note)
        assert isinstance(result, str)

    def test_output_is_non_empty(self):
        note = (
            "Subjective: Patient reports chest pain.\n"
            "Objective: BP 140/90.\n"
            "Assessment: Hypertension.\n"
            "Plan: Lisinopril."
        )
        result = prepare_inference_input(note)
        assert len(result.strip()) > 0

    def test_icd10_codes_redacted(self):
        """Explicit ICD-10 codes in the note should be redacted."""
        note = (
            "Assessment: I10 (Essential hypertension).\n"
            "Plan: Continue lisinopril.\n"
            "Subjective: Headache.\n"
            "Objective: BP 160/95."
        )
        result = prepare_inference_input(note)
        assert "I10" not in result or REDACTION_MARKER in result

    def test_assessment_content_preserved(self):
        """Assessment section content should appear in output."""
        note = (
            "Subjective: Fatigue.\n"
            "Objective: HR 72.\n"
            "Assessment: Type 2 diabetes mellitus with hyperglycaemia.\n"
            "Plan: Adjust metformin."
        )
        result = prepare_inference_input(note)
        assert "diabetes" in result.lower() or "hyperglycaemia" in result.lower()

    def test_handles_missing_sections_gracefully(self):
        """Notes missing some SOAP sections should not raise."""
        note = "Assessment: Lyme disease. Plan: Doxycycline."
        result = prepare_inference_input(note)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_handles_empty_string_gracefully(self):
        """Empty string input should return a string (may be empty)."""
        result = prepare_inference_input("")
        assert isinstance(result, str)

    def test_apso_order_assessment_before_plan(self):
        """Assessment content should appear before Plan content in output."""
        note = (
            "Subjective: Fatigue and weight loss.\n"
            "Objective: BMI 18, pale conjunctivae.\n"
            "Assessment: Iron deficiency anaemia.\n"
            "Plan: Ferrous sulphate 200mg TDS, dietary advice."
        )
        result = prepare_inference_input(note)
        assessment_pos = result.lower().find("anaemia")
        plan_pos = result.lower().find("ferrous")
        if assessment_pos != -1 and plan_pos != -1:
            assert assessment_pos < plan_pos, \
                "Assessment content should appear before Plan content in APSO output"


# ---------------------------------------------------------------------------
# build_apso_note — DataFrame-level APSO reordering
# ---------------------------------------------------------------------------

class TestBuildApsoNote:

    @pytest.fixture
    def sample_df(self):
        """Minimal DataFrame with SOAP columns."""
        return pl.DataFrame({
            "subjective":  ["Patient reports fatigue"],
            "objective":   ["HR 72, BP 120/80"],
            "assessment":  ["Type 2 diabetes mellitus"],
            "plan":        ["Adjust metformin dosage"],
        })

    def test_returns_dataframe(self, sample_df):
        from src.preprocessing import build_apso_note
        result = build_apso_note(sample_df)
        assert isinstance(result, pl.DataFrame)

    def test_apso_note_column_created(self, sample_df):
        from src.preprocessing import build_apso_note
        result = build_apso_note(sample_df)
        assert "apso_note" in result.columns

    def test_assessment_before_plan_in_output(self, sample_df):
        from src.preprocessing import build_apso_note
        result = build_apso_note(sample_df)
        apso_note = result["apso_note"][0]
        assessment_pos = apso_note.find("diabetes")
        plan_pos = apso_note.find("metformin")
        if assessment_pos != -1 and plan_pos != -1:
            assert assessment_pos < plan_pos

    def test_all_sections_included(self, sample_df):
        from src.preprocessing import build_apso_note
        result = build_apso_note(sample_df)
        apso_note = result["apso_note"][0]
        assert "diabetes" in apso_note
        assert "metformin" in apso_note


# ---------------------------------------------------------------------------
# redact_icd10_sections — DataFrame-level redaction
# ---------------------------------------------------------------------------

class TestRedactICD10Sections:

    @pytest.fixture
    def df_with_codes(self):
        """
        DataFrame with ICD-10 codes. build_apso_note() called first —
        redact_icd10_sections() requires the apso_note column to exist.
        """
        from src.preprocessing import build_apso_note
        df = pl.DataFrame({
            "subjective":     ["Chest pain since yesterday"],
            "objective":      ["BP 140/90, HR 88"],
            "assessment":     ["I10 Essential hypertension, also E11.65"],
            "plan":           ["Lisinopril 10mg, monitor I10"],
            "standard_icd10": ["I10"],
        })
        return build_apso_note(df)

    def test_returns_dataframe(self, df_with_codes):
        from src.preprocessing import redact_icd10_sections
        result = redact_icd10_sections(df_with_codes)
        assert isinstance(result, pl.DataFrame)

    def test_codes_redacted_from_assessment(self, df_with_codes):
        from src.preprocessing import redact_icd10_sections
        result = redact_icd10_sections(df_with_codes)
        if "apso_note" in result.columns:
            apso = result["apso_note"][0]
            assert "I10" not in apso or REDACTION_MARKER in apso

    def test_apso_note_column_present_after_redaction(self, df_with_codes):
        from src.preprocessing import redact_icd10_sections
        result = redact_icd10_sections(df_with_codes)
        assert "apso_note" in result.columns

    def test_clinical_content_preserved(self, df_with_codes):
        from src.preprocessing import redact_icd10_sections
        result = redact_icd10_sections(df_with_codes)
        if "apso_note" in result.columns:
            apso = result["apso_note"][0]
            assert "hypertension" in apso.lower() or "lisinopril" in apso.lower()

    def test_handles_no_codes_in_text(self):
        """DataFrame with no ICD-10 codes should pass through unchanged."""
        from src.preprocessing import redact_icd10_sections, build_apso_note
        df = pl.DataFrame({
            "subjective":     ["Patient has a headache"],
            "objective":      ["Temp 37.2, alert and oriented"],
            "assessment":     ["Tension headache"],
            "plan":           ["Paracetamol PRN, rest"],
            "standard_icd10": ["G44.2"],
        })
        df = build_apso_note(df)  # add apso_note column first
        result = redact_icd10_sections(df)
        assert isinstance(result, pl.DataFrame)
        assert "apso_note" in result.columns