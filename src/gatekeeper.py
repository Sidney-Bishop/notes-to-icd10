# src/gatekeeper.py
import polars as pl
import re
from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Optional, Tuple


class ClinicalRecord(BaseModel):
    """
    The Pydantic Firewall for MedSynth.
    Extracts structured SOAP signals from raw clinical notes.
    """
    id:       str       = Field(..., alias="ID")
    note:     str       = Field(..., alias="Note")
    dialogue: str       = Field(..., alias="Dialogue")
    label:    List[str] = Field(default=[], alias="ICD10")

    # --- Extracted SOAP Fields ---
    subjective: Optional[str] = None
    objective:  Optional[str] = None
    assessment: Optional[str] = None
    plan:       Optional[str] = None

    def extract_soap(self):
        """
        Surgically isolates SOAP sections using regex based on Note headers.
        """
        patterns = {
            'subjective': r'(?i)subjective:?\s*(.*?)(?=\d\.\s*objective|objective:|$)',
            'objective':  r'(?i)objective:?\s*(.*?)(?=\d\.\s*assessment|assessment:|$)',
            'assessment': r'(?i)assessment:?\s*(.*?)(?=\d\.\s*plan|plan:|$)',
            'plan':       r'(?i)plan:?\s*(.*)'
        }

        for section, pattern in patterns.items():
            match = re.search(pattern, self.note, re.DOTALL)
            if match:
                setattr(self, section, match.group(1).strip())

    @field_validator("note")
    @classmethod
    def validate_note_content(cls, v: str) -> str:
        if len(v.strip()) < 20:
            raise ValueError("Clinical note is too short to contain diagnostic signal.")
        return v

    @field_validator("label")
    @classmethod
    def validate_label_list(cls, v: List[str]) -> List[str]:
        """
        Ensure label is always a list of strings with valid ICD-10 format.

        Accepts both raw format (no decimal, e.g. M25562) and canonical format
        (with decimal, e.g. M25.562). Raw format is the convention used in the
        MedSynth source dataset and throughout Phase 1. Decimal restoration is
        applied separately in Phase 2a.

        Codes that do not match either format are logged and dropped. This
        covers the placeholder-X codes (e.g. T781XXA) identified in Phase 1b —
        these are retained as-is since they are structurally valid ICD-10-CM
        codes absent from the CM descriptions file.
        """
        if v is None:
            return []
        if isinstance(v, str):
            v = [v]
        if not isinstance(v, list):
            raise ValueError("Label must be a string or list of strings")

        # Accepts:
        #   Raw format:       letter + 2-7 alphanumeric chars, no decimal (e.g. M25562)
        #   Canonical format: letter + 2 chars + decimal + 1-4 chars (e.g. M25.562)
        icd10_pattern = r'^[A-Z][0-9A-Z]{2,7}$|^[A-Z][0-9A-Z]{2}\.[0-9A-Z]{1,4}$'

        cleaned_labels = []
        for item in v:
            label_str = str(item).strip().upper()
            if re.match(icd10_pattern, label_str):
                cleaned_labels.append(label_str)

            # Codes not matching either format are silently dropped.
            # The record still passes validation with a reduced label list.
            # If all codes are dropped the label will be an empty list —
            # this will not raise an error but will be visible as [] in df_valid.

        return cleaned_labels


def validate_dataframe(
    df: pl.DataFrame,
    chunk_size: Optional[int] = None
) -> Tuple[pl.DataFrame, pl.DataFrame, List[ClinicalRecord]]:
    """
    Zero-Trust Validator with optional chunking for large datasets.

    Parameters
    ----------
    df : pl.DataFrame
        The prepared DataFrame from Phase 1e (with string-cast ID column).
    chunk_size : int, optional
        Number of rows per chunk. If None, processes the full DataFrame at once.

    Returns
    -------
    valid_df : pl.DataFrame
        Records that passed all validation checks, promoted to Silver.
    error_df : pl.DataFrame
        Records that failed validation, with error type and message.
    valid_objects : list[ClinicalRecord]
        Validated Pydantic objects for downstream SOAP extraction access.
    """
    all_valid_objects = []
    all_error_logs    = []

    # Determine if chunking is needed
    if chunk_size is None or len(df) < chunk_size:
        chunks = [df]
    else:
        chunks = [df[i:i + chunk_size] for i in range(0, len(df), chunk_size)]

    for chunk_idx, chunk in enumerate(chunks):
        records = chunk.to_dicts()

        for rec in records:
            try:
                validated = ClinicalRecord(**rec)
                validated.extract_soap()
                all_valid_objects.append(validated)
            except Exception as e:
                error_msg  = str(e).lower()
                error_type = "unknown"
                if "too short" in error_msg:
                    error_type = "note_length"
                elif "label" in error_msg:
                    error_type = "label_format"
                elif "soap" in error_msg:
                    error_type = "soap_extraction"

                all_error_logs.append({
                    "ID":            rec.get("ID", "UNKNOWN"),
                    "error_type":    error_type,
                    "error_message": str(e)
                })

    # Reconstruct DataFrames from validated objects
    valid_records_data = [obj.model_dump() for obj in all_valid_objects]
    valid_df  = pl.DataFrame(valid_records_data) if valid_records_data else pl.DataFrame()
    error_df  = pl.DataFrame(all_error_logs)     if all_error_logs     else pl.DataFrame()

    return valid_df, error_df, all_valid_objects