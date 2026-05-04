"""
preprocessing.py — APSO-Flip and ICD-10 redaction utilities.

Implements the exact preprocessing pipeline established in notebook 01-EDA_SOAP.ipynb
(Phases 3a and 3c) and consumed by all downstream model notebooks (02–05).

The canonical model input is the ``apso_note`` column produced here:
    Assessment → Plan → Subjective → Objective
    (plain concatenation, no special tokens, no dialogue)

This module provides two interfaces:

    DataFrame-level (Polars)
        Used by the data pipeline when processing the Gold layer.
        ``build_apso_note()`` — recompose sections into APSO order.
        ``redact_icd10_sections()`` — redact code strings from all SOAP columns
                                      and rebuild apso_note.

    Single-string (inference)
        Used by ``src/inference.py`` for new clinical notes at runtime.
        ``prepare_inference_input()`` — extract SOAP sections via regex,
                                        reorder to APSO, and redact.
"""

import re
import warnings
import polars as pl
from typing import Optional

# ---------------------------------------------------------------------------
# Constants — kept in one place so notebook and inference stay in sync
# ---------------------------------------------------------------------------

# Same pattern used in Phase 3b (detection) and Phase 3c (redaction).
# Anchored on the ICD-10 structural convention — letter + two digits —
# to discriminate codes from common medical abbreviations (MCL, CBC, TSH).
#
#   canonical format:  M25.562, N39.0
#   raw format:        M25562, N390, J18
ICD10_REDACT_PATTERN = (
    r'\b[A-Z][0-9]{2}\.[0-9A-Z]{1,4}\b'   # canonical: M25.562, N39.0
    r'|'
    r'\b[A-Z][0-9]{2}[0-9A-Z]{0,5}\b'     # raw: M25562, N390, J18
)

REDACTION_MARKER = "[REDACTED]"

# Section headers used for inference-time regex extraction
_SOAP_PATTERNS = {
    'subjective': r'(?i)subjective:?\s*(.*?)(?=\d\.\s*objective|objective:|$)',
    'objective':  r'(?i)objective:?\s*(.*?)(?=\d\.\s*assessment|assessment:|$)',
    'assessment': r'(?i)assessment:?\s*(.*?)(?=\d\.\s*plan|plan:|$)',
    'plan':       r'(?i)plan:?\s*(.*)',
}


# ---------------------------------------------------------------------------
# DataFrame-level helpers (Polars)
# ---------------------------------------------------------------------------

def build_apso_note(df: pl.DataFrame) -> pl.DataFrame:
    """
    Recompose pre-extracted SOAP section columns into APSO order.

    Adds an ``apso_note`` column (Assessment -> Plan -> Subjective -> Objective).
    The original ``note`` column is not modified.

    Requires columns: ``assessment``, ``plan``, ``subjective``, ``objective``.
    Null sections are treated as empty strings so every record produces a
    valid (non-null) apso_note.

    This is a direct port of Phase 3a in notebook 01-EDA_SOAP.ipynb.

    Parameters
    ----------
    df : pl.DataFrame
        Gold layer DataFrame with pre-extracted SOAP section columns.

    Returns
    -------
    pl.DataFrame
        Input DataFrame with ``apso_note`` column added.
    """
    required = {'assessment', 'plan', 'subjective', 'objective'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"DataFrame is missing required SOAP section columns: {missing}. "
            f"Run the Pydantic Gatekeeper (Phase 1e) before calling this function."
        )

    return df.with_columns([
        (
            pl.col("assessment").fill_null("") + "\n\n" +
            pl.col("plan").fill_null("")       + "\n\n" +
            pl.col("subjective").fill_null("") + "\n\n" +
            pl.col("objective").fill_null("")
        ).str.strip_chars().alias("apso_note")
    ])


def redact_icd10_sections(df: pl.DataFrame) -> pl.DataFrame:
    """
    Redact ICD-10 code strings from all SOAP section columns and rebuild apso_note.

    Replaces explicit ICD-10 strings with ``[REDACTED]`` in the four SOAP
    section columns (``assessment``, ``plan``, ``objective``, ``subjective``),
    then rebuilds ``apso_note`` from the cleaned sections so it is the single
    source of truth for downstream model input.

    The original ``note`` column is NOT modified — it is preserved for
    audit traceability and tasks where the original text is needed.

    This is a direct port of Phase 3c in notebook 01-EDA_SOAP.ipynb.

    Parameters
    ----------
    df : pl.DataFrame
        Gold layer DataFrame with ``apso_note`` and SOAP section columns present.

    Returns
    -------
    pl.DataFrame
        DataFrame with redacted section columns and rebuilt ``apso_note``.
    """
    required = {'apso_note', 'assessment', 'plan', 'subjective', 'objective'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"DataFrame is missing required columns: {missing}. "
            f"Run build_apso_note() before calling redact_icd10_sections()."
        )

    sections = ['assessment', 'plan', 'objective', 'subjective']

    # Step 1: Redact each section into a temporary _clean column
    df = df.with_columns([
        pl.col(section)
        .str.replace_all(ICD10_REDACT_PATTERN, REDACTION_MARKER)
        .alias(f"{section}_clean")
        for section in sections
    ])

    # Step 2: Overwrite originals with cleaned versions, drop temporaries
    df = df.with_columns([
        pl.col(f"{section}_clean").alias(section)
        for section in sections
    ]).drop([f"{section}_clean" for section in sections])

    # Step 3: Rebuild apso_note from the cleaned sections
    df = df.with_columns([
        (
            pl.col("assessment").fill_null("") + "\n\n" +
            pl.col("plan").fill_null("")       + "\n\n" +
            pl.col("subjective").fill_null("") + "\n\n" +
            pl.col("objective").fill_null("")
        ).str.strip_chars().alias("apso_note")
    ])

    return df


# ---------------------------------------------------------------------------
# Single-string interface (inference)
# ---------------------------------------------------------------------------

def prepare_inference_input(note: str) -> str:
    """
    Prepare a raw clinical note for model inference.

    Extracts SOAP sections via regex, reorders to APSO format
    (Assessment -> Plan -> Subjective -> Objective), and redacts any
    explicit ICD-10 code strings — reproducing the same transformation
    applied to training data in Phases 3a and 3c.

    This is the inference-time equivalent of ``build_apso_note()`` +
    ``redact_icd10_sections()`` for a single string where pre-extracted
    section columns are not available.

    Parameters
    ----------
    note : str
        Raw clinical note in SOAP format.

    Returns
    -------
    str
        APSO-reordered, ICD-10-redacted note ready for tokenisation.
        Sections missing from the input are silently omitted.

    Notes
    -----
    The regex extraction is less reliable than the Pydantic-based extraction
    used in the training pipeline (100% success on MedSynth). For production
    use with notes that may have non-standard section headers, consider routing
    through the full pipeline instead.
    """
    sections: dict = {}

    for section_name, pattern in _SOAP_PATTERNS.items():
        match = re.search(pattern, note, re.DOTALL)
        if match:
            sections[section_name] = match.group(1).strip()

    # Recompose in APSO order — same concatenation as build_apso_note()
    apso_parts = [
        sections.get('assessment', ''),
        sections.get('plan', ''),
        sections.get('subjective', ''),
        sections.get('objective', ''),
    ]
    apso_note = "\n\n".join(part for part in apso_parts if part).strip()

    # Fall back to original note if section extraction fails entirely
    if not apso_note:
        warnings.warn(
            "prepare_inference_input: no SOAP section headers detected in the "
            "provided note — falling back to the raw text without APSO reordering. "
            "Assessment-first ordering cannot be applied, which may reduce prediction "
            "quality. Ensure the note contains 'Assessment:', 'Plan:', 'Subjective:', "
            "and 'Objective:' headers.",
            UserWarning,
            stacklevel=2,
        )
        apso_note = note.strip()

    # Redact ICD-10 code strings — same pattern as Phase 3c
    apso_note = re.sub(ICD10_REDACT_PATTERN, REDACTION_MARKER, apso_note)

    return apso_note