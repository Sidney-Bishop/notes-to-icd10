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

# Broad pattern validated against full corpus audit (Phase 3b.2).
# Removes any parenthetical containing the word "ICD" — covers all
# MedSynth variants: (ICD-10: CODE), (ICD-10 Code: CODE), (ICD-10 code CODE),
# (ICD-10), (ICD-10 description), (ICD-10-CM CODE), (CODE ICD-10), etc.
# Word boundary \b prevents matching AICD (medical device abbreviation).
PARENTHETICAL_ICD10_PATTERN = r'(?i)\s*\([^)]*\bICD[^)]*\)'

REDACTION_MARKER = "[REDACTED]"

# Cleanup pattern: remove stray ([REDACTED]) artifacts.
REDACTED_ARTIFACT_PATTERN = r'\(\s*\[REDACTED\]\s*\)'

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

    df : pl.DataFrame
        Gold layer DataFrame with pre-extracted SOAP section columns.

    Returns

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

    Removes the full parenthetical "(ICD-10: CODE)" wrapper entirely, then
    replaces any remaining standalone ICD-10 strings with ``[REDACTED]`` in the
    four SOAP section columns (``assessment``, ``plan``, ``objective``,
    ``subjective``), then rebuilds ``apso_note`` from the cleaned sections.

    The original ``note`` column is NOT modified — it is preserved for
    audit traceability.

    This is a direct port of Phase 3c in notebook 01-EDA_SOAP.ipynb.

    Parameters

    df : pl.DataFrame
        Gold layer DataFrame with ``apso_note`` and SOAP section columns present.

    Returns

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
    # First strip the "(ICD-10: CODE)" parenthetical, then redact stray codes
    df = df.with_columns([
        pl.col(section)
        .str.replace_all(PARENTHETICAL_ICD10_PATTERN, '')
        .str.replace_all(ICD10_REDACT_PATTERN, REDACTION_MARKER)
        .str.replace_all(REDACTED_ARTIFACT_PATTERN, '')
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

    note : str
        Raw clinical note in SOAP format.

    Returns

    str
        APSO-reordered, ICD-10-redacted note ready for tokenisation.
        Sections missing from the input are silently omitted.

    Notes

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

    # Redact ICD-10 code strings — same logic as Phase 3c
    # Step 1: strip "(ICD-10: CODE)" parentheticals
    apso_note = re.sub(PARENTHETICAL_ICD10_PATTERN, '', apso_note)
    # Step 2: redact any remaining standalone codes
    apso_note = re.sub(ICD10_REDACT_PATTERN, REDACTION_MARKER, apso_note)
    # Step 3: clean up stray ([REDACTED]) artifacts
    apso_note = re.sub(REDACTED_ARTIFACT_PATTERN, '', apso_note)

    return apso_note

# ---------------------------------------------------------------------------
# Description redaction (Q8) — DataFrame-level
# ---------------------------------------------------------------------------
#
# Removes the human-readable diagnosis DESCRIPTION (the text the ICD-10 code
# encodes, e.g. "Pain in left knee" for M25.562) from the assessment section,
# which the code-only redaction (redact_icd10_sections) leaves behind. This is
# the "v5" redactor finalised in docs/decisions.md D012:
#
#   * dictionary-anchored: matches MedSynth's own harvested phrasings per code,
#     falling back to the CDC description when a code has no harvested phrasing;
#   * article-tolerant full-phrase match (an inserted "the"/"a"/"an" between
#     tokens does not defeat the match);
#   * removes ALL occurrences, [DIAGNOSIS] placeholder when the phrase is
#     mid-sentence, drops the whole line when the phrase stands alone;
#   * min-2-content-token guard: descriptions with fewer than two content tokens
#     (e.g. "Weakness") are skipped entirely, since matching a lone common word
#     over-redacts legitimate clinical findings (D012);
#   * assessment-only scope (D011).
#
# The LLM audit that VALIDATED this rule is advisory only and never runs here
# (D013) — this function is fully deterministic and reproducible.
#
# Measured corpus-wide on the billable gold: removes 75.6% of pre-existing
# description leakage, residual 18.6%, fires on 6,091 / 9,660 records (D012).

DESCRIPTION_PLACEHOLDER = "[DIAGNOSIS]"
_MIN_CONTENT_TOKENS = 2
_DESC_STOPWORDS = {
    "of", "the", "and", "in", "with", "to", "a", "an", "or", "unspecified",
    "other", "site", "due", "as", "not", "elsewhere", "classified", "specified",
}


def _content_tokens(text: str) -> list:
    """Lowercased alphanumeric tokens with stopwords and very short tokens removed."""
    return [
        w for w in re.findall(r"[a-z0-9]+", (text or "").lower())
        if w not in _DESC_STOPWORDS and len(w) > 2
    ]


def _desc_norm(text: str) -> str:
    """Normalise for standalone-line comparison: strip markdown, collapse whitespace."""
    return re.sub(r"\s+", " ", re.sub(r"[*_`]", "", (text or ""))).strip().lower()


def _desc_make_rx(phrase: str):
    """Compile an article-tolerant regex: phrase tokens joined by whitespace that
    may include an optional intervening article (the/a/an)."""
    tokens = [re.escape(t) for t in phrase.split()]
    if not tokens:
        return None
    return re.compile(r"\s+(?:the\s+|a\s+|an\s+)?".join(tokens), re.IGNORECASE)


def _candidate_phrases(code: str, dictionary: dict, cdc_map: dict) -> list:
    """Per-code candidate phrases: harvested dictionary phrasings (longest first)
    plus the CDC description as fallback, de-duplicated, guarded for length."""
    out = [e["phrase"] for e in dictionary.get(code, [])]
    cdc_d = cdc_map.get(code.replace(".", ""), "")
    if cdc_d:
        out.append(cdc_d)
    seen, uniq = set(), []
    for p in out:
        k = p.lower()
        if k in seen or len(p) < 3:
            continue
        if len(_content_tokens(p)) < _MIN_CONTENT_TOKENS:   # guard: skip generic single words
            continue
        seen.add(k)
        uniq.append(p)
    return sorted(uniq, key=len, reverse=True)


def _redact_one(text: str, phrase: str) -> str:
    """Remove every occurrence of `phrase` from `text`: drop the line if the phrase
    is standalone on it, else replace with the [DIAGNOSIS] placeholder. Bounded."""
    rx = _desc_make_rx(phrase)
    if rx is None:
        return text
    n = 0
    while True:
        m = rx.search(text)
        if not m:
            break
        s, e = m.span()
        line_start = text.rfind("\n", 0, s) + 1
        line_end = text.find("\n", e)
        if line_end == -1:
            line_end = len(text)
        line_core = re.sub(r"^[\s\-\*\u2022\d\.\)]+", "", text[line_start:line_end])
        line_core = re.sub(r"[\s\.\*:]+$", "", line_core)
        if _desc_norm(line_core) == _desc_norm(m.group(0)):
            text = text[:line_start] + text[line_end:]
        else:
            text = text[:s] + DESCRIPTION_PLACEHOLDER + text[e:]
        n += 1
        if n > 10:    # safety bound; no real assessment restates a label >10x
            break
    return text


def _redact_assessment_text(assessment: str, code: str, dictionary: dict, cdc_map: dict) -> str:
    """Apply the v5 description redaction to a single assessment string."""
    if not assessment:
        return assessment
    cands = _candidate_phrases(code, dictionary, cdc_map)
    if not cands:
        return assessment
    text = assessment
    for phrase in cands:
        text = _redact_one(text, phrase)
    # tidy: drop emptied bullets and collapse blank runs
    text = re.sub(r"(?im)^[ \t]*[-*\u2022]\s*$", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def redact_descriptions(
    df: pl.DataFrame,
    dictionary_path,
    cdc_path,
    code_col: str = "standard_icd10",
) -> pl.DataFrame:
    """
    Redact diagnosis DESCRIPTIONS from the assessment section and rebuild apso_note.

    The "v5" description redactor (see module docstring above and decisions
    D011/D012/D013). Operates on the ``assessment`` column, keyed per-row on the
    record's ICD-10 code, then rebuilds ``apso_note`` from the cleaned sections
    using the SAME concatenation as redact_icd10_sections() (Assessment -> Plan
    -> Subjective -> Objective), so the two redaction passes compose cleanly.

    This runs AFTER redact_icd10_sections() in the pipeline (Phase 3d after 3c):
    code strings are already removed; this removes the leftover description text.

    Deterministic and reproducible — no LLM is involved (D013).

    Parameters

    df : pl.DataFrame
        Gold DataFrame with apso_note + SOAP section columns + the code column.
    dictionary_path : str | Path
        Path to the committed q8_phrasing_dictionary.json artifact.
    cdc_path : str | Path
        Path to the CDC ICD-10 reference parquet (code_no_decimal, description).
    code_col : str
        Column holding the per-row canonical ICD-10 code (default standard_icd10).

    Returns

    pl.DataFrame
        DataFrame with description-redacted assessment and rebuilt apso_note.
    """
    import json

    required = {"apso_note", "assessment", "plan", "subjective", "objective", code_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"DataFrame is missing required columns: {missing}. "
            f"Run build_apso_note() and redact_icd10_sections() before "
            f"redact_descriptions()."
        )

    # Load the harvested phrasing dictionary (committed artifact)
    blob = json.loads(open(dictionary_path, "r", encoding="utf-8").read())
    dictionary = blob["dictionary"]

    # Load CDC description fallback map (un-dotted code -> description)
    cdc = pl.read_parquet(cdc_path)
    cdc_map = dict(zip(cdc["code_no_decimal"].to_list(), cdc["description"].to_list()))

    # Per-row redaction of the assessment (Python; one-time gold build, not hot path)
    codes = df[code_col].to_list()
    assessments = df["assessment"].to_list()
    redacted = [
        _redact_assessment_text(a, c, dictionary, cdc_map)
        for a, c in zip(assessments, codes)
    ]
    df = df.with_columns(pl.Series("assessment", redacted))

    # Rebuild apso_note from cleaned sections — IDENTICAL block to
    # redact_icd10_sections() Step 3, so the two passes stay consistent.
    df = df.with_columns([
        (
            pl.col("assessment").fill_null("") + "\n\n" +
            pl.col("plan").fill_null("")       + "\n\n" +
            pl.col("subjective").fill_null("") + "\n\n" +
            pl.col("objective").fill_null("")
        ).str.strip_chars().alias("apso_note")
    ])

    return df
