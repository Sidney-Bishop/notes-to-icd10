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
    PARENTHETICAL_ICD10_PATTERN,
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
# PARENTHETICAL_ICD10_PATTERN — full wrapper stripping
# ---------------------------------------------------------------------------

class TestParentheticalICD10Pattern:
    """
    Verify that (ICD-10: CODE) and (ICD10: CODE) wrappers are removed
    entirely by PARENTHETICAL_ICD10_PATTERN — not just the code string.

    The critical distinction from ICD10_REDACT_PATTERN:
      ICD10_REDACT_PATTERN:      "Pain in left knee (ICD-10: M25.562)"
                                 → "Pain in left knee (ICD-10: [REDACTED])"
      PARENTHETICAL_ICD10_PATTERN applied first:
                                 → "Pain in left knee"

    Leaving "(ICD-10: [REDACTED])" in the text would signal to the model
    that a label was present at that position — defeating redaction.
    """

    def test_parenthetical_with_decimal_code_removed(self):
        """(ICD-10: M25.562) — canonical format with decimal."""
        import re
        text = "Pain in the left knee (ICD-10: M25.562)."
        result = re.sub(PARENTHETICAL_ICD10_PATTERN, "", text)
        assert "(ICD-10:" not in result
        assert "M25.562" not in result
        assert "Pain in the left knee" in result

    def test_parenthetical_with_raw_code_removed(self):
        """(ICD-10: M25562) — raw format without decimal."""
        import re
        text = "Chronic knee pain (ICD-10: M25562) persists."
        result = re.sub(PARENTHETICAL_ICD10_PATTERN, "", text)
        assert "(ICD-10:" not in result
        assert "M25562" not in result
        assert "persists" in result

    def test_parenthetical_no_hyphen_removed(self):
        """(ICD10: N39.0) — variant without hyphen."""
        import re
        text = "Urinary tract infection (ICD10: N39.0)."
        result = re.sub(PARENTHETICAL_ICD10_PATTERN, "", text)
        assert "ICD10" not in result
        assert "N39.0" not in result
        assert "Urinary tract infection" in result

    def test_parenthetical_case_insensitive(self):
        """(icd-10: E11.65) — lowercase variant."""
        import re
        text = "Type 2 diabetes (icd-10: E11.65) with complications."
        result = re.sub(PARENTHETICAL_ICD10_PATTERN, "", text)
        assert "icd" not in result.lower() or "E11.65" not in result
        assert "complications" in result

    def test_no_icd_artifact_left_after_removal(self):
        """Critical: '(ICD-10: [REDACTED])' must NOT appear after both patterns applied."""
        import re
        text = "Diagnosis: Pain in left knee (ICD-10: M25.562)."
        # Apply parenthetical pattern first (as in prepare_inference_input)
        step1 = re.sub(PARENTHETICAL_ICD10_PATTERN, "", text)
        # Then apply code redaction — should find nothing left to redact
        step2 = re.sub(ICD10_REDACT_PATTERN, REDACTION_MARKER, step1)
        assert "(ICD-10: [REDACTED])" not in step2
        assert "M25.562" not in step2
        assert REDACTION_MARKER not in step2, (
            "No stray code should remain after parenthetical was stripped — "
            f"got: {step2!r}"
        )

    def test_clinical_text_outside_parenthetical_preserved(self):
        """Text outside the parenthetical should be completely unchanged."""
        import re
        text = "Assessment: Chronic left knee pain (ICD-10: M25.562). Plan: physiotherapy."
        result = re.sub(PARENTHETICAL_ICD10_PATTERN, "", text)
        assert "Chronic left knee pain" in result
        assert "physiotherapy" in result

    def test_multiple_parentheticals_in_one_note(self):
        """Multiple (ICD-10: CODE) wrappers in a single note — all removed."""
        import re
        text = (
            "Type 2 diabetes (ICD-10: E11.65) with hypertension (ICD-10: I10). "
            "Continue current management."
        )
        result = re.sub(PARENTHETICAL_ICD10_PATTERN, "", text)
        assert "(ICD-10:" not in result
        assert "E11.65" not in result
        assert "I10" not in result
        assert "Continue current management" in result

    def test_parenthetical_removed_in_redact_icd10_sections(self):
        """
        End-to-end: redact_icd10_sections() must fully remove the parenthetical
        wrapper, not leave (ICD-10: [REDACTED]) in the apso_note column.
        """
        from src.preprocessing import build_apso_note, redact_icd10_sections
        import polars as pl

        df = pl.DataFrame({
            "subjective":     ["Knee pain for 3 months"],
            "objective":      ["Swelling, reduced ROM"],
            "assessment":     ["Pain in left knee (ICD-10: M25.562)."],
            "plan":           ["Physiotherapy, NSAIDs"],
            "standard_icd10": ["M25.562"],
        })
        df = build_apso_note(df)
        result = redact_icd10_sections(df)
        apso = result["apso_note"][0]

        assert "(ICD-10: [REDACTED])" not in apso, (
            "Parenthetical wrapper was not fully removed — "
            f"'(ICD-10: [REDACTED])' still present in: {apso!r}"
        )
        assert "M25.562" not in apso
        assert "Pain in left knee" in apso

    def test_parenthetical_removed_in_prepare_inference_input(self):
        """
        End-to-end: prepare_inference_input() must fully remove the
        parenthetical wrapper on the single-string inference path.
        """
        note = (
            "Subjective: Knee pain.\n"
            "Objective: Swelling present.\n"
            "Assessment: Pain in left knee (ICD-10: M25.562).\n"
            "Plan: Physiotherapy."
        )
        result = prepare_inference_input(note)
        assert "(ICD-10: [REDACTED])" not in result, (
            "Parenthetical wrapper was not fully removed on inference path — "
            f"'(ICD-10: [REDACTED])' still present in: {result!r}"
        )
        assert "M25.562" not in result
        assert "Pain in left knee" in result


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