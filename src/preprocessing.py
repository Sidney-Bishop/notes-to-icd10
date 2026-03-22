# src/preprocessing.py
import re
import polars as pl
from typing import Optional

def get_surgical_expressions():
    """
    Returns Polars expressions for high-performance vectorized preprocessing.
    Utilizes all M4 Max cores.
    """
    # Pattern to find Assessment/Plan (A/P) and everything after it
    # Handles: Assessment:, Plan:, A/P:, Assessment & Plan:
    soap_pattern = r"(?i)(assessment|plan|a/p|dx|diagnosis)[\s\w&]*:.*"
    
    # Pattern to strip ICD-10 codes (M25.5, I21.9, etc) to prevent leakage
    leakage_pattern = r"(?i)\(?(icd-10|code|dx)[\s\w:]*[A-Z][0-9][0-9A-Z\.]*\)?"

    return [
        # 1. Clean whitespace
        pl.col("Note").str.replace_all(r"\s+", " ").str.strip_chars().alias("clean_note"),
        
        # 2. Extract SOAP sections
        pl.col("clean_note").str.extract(soap_pattern, 0).alias("soap_section"),
        pl.col("clean_note").str.replace(soap_pattern, "").alias("context_section"),
        
        # 3. Apply Sanitization to prevent leakage
        pl.col("soap_section").str.replace_all(leakage_pattern, "[REDACTED_CODE]"),
    ]

def finalize_surgical_text(df: pl.DataFrame) -> pl.DataFrame:
    """
    Combines SOAP, Context, and Dialogue into the final BERT-ready payload.
    Prioritizes [DIAGNOSIS] tokens at the start of the sequence.
    """
    return df.with_columns([
        pl.format(
            "[DIAGNOSIS]: {}\n\n[CONTEXT]: {}\n\n[DIALOGUE]: {}",
            pl.col("soap_section").fill_null("[NO_AP_FOUND]"),
            pl.col("context_section").fill_null(""),
            pl.col("Dialogue").fill_null("")
        ).alias("text")
    ]).with_columns([
        # Truncate to word count to stay under 512 tokens
        pl.col("text").str.split(" ").list.slice(0, 450).list.join(" ")
    ])

# Helper for single-string inference (e.g., in a demo)
def prepare_inference_input(note: str, dialogue: str = "") -> str:
    """Standardizes a new clinical encounter using the SOAP approach."""
    # Robust but slower regex for single-string use
    soap_match = re.search(r"(?i)(assessment|plan|a/p|dx)[\s\w&]*:.*", note, re.DOTALL)
    
    if soap_match:
        ap = soap_match.group(0)
        context = note[:soap_match.start()]
    else:
        ap = "[NO_AP_FOUND]"
        context = note

    full_text = f"[DIAGNOSIS]: {ap}\n\n[CONTEXT]: {context}\n\n[DIALOGUE]: {dialogue}"
    return " ".join(full_text.split()[:450])