"""
prepare_data.py — Headless Gold layer preparation pipeline.

Migrates the deterministic, automatable phases from notebook 01-EDA_SOAP.ipynb
to a reproducible command-line script. Interactive phases (visual auditors) are
excluded by design — this script is the headless, CI-safe complement to the
notebook, not a replacement for it.

Phases executed
---------------
    Phase 0   Infrastructure & path validation (via ArtifactConfig)
    Phase 1a  Raw MedSynth ingestion from HuggingFace
    Phase 1b  CDC FY2026 ICD-10 validation and status classification
    Phase 1e  Pydantic Firewall — schema validation and SOAP section extraction
    Phase 1g  DuckDB Silver Vault persistence
    Phase 2a  Decimal restoration (raw → canonical ICD-10 format)
    Phase 2c  Code status annotation (billable / noisy_111 / placeholder_x)
    Phase 3a  APSO-Flip (Assessment-first reordering)
    Phase 3b  ICD-10 leakage detection (quantification only)
    Phase 3c  ICD-10 code redaction ([REDACTED] substitution)
    Phase 4   Gold layer Parquet export with post-write verification

Phases excluded (require human visual review — run in notebook 01)
------------------------------------------------------------------
    Phase 1c  Token volume visualisation (matplotlib/seaborn charts)
    Phase 1d  Raw MedSynth Discovery Auditor (Panel browser widget)
    Phase 1f  Surgical Signal Auditor (Panel browser widget)
    Phase 2b  Gold Layer Discovery Auditor (Panel browser widget)
    Phase 3b.1 Leakage Forensic Auditor (Panel browser widget)
    Phase 3c.1 Post-Redaction Integrity Auditor (Panel browser widget)

Usage
-----
    uv run python scripts/prepare_data.py
    uv run python scripts/prepare_data.py --no-duckdb        # skip Silver Vault
    uv run python scripts/prepare_data.py --dry-run          # validate only, no export
    uv run python scripts/prepare_data.py --offline          # skip CDC download, use cache

Output
------
    data/gold/medsynth_gold_apso_{timestamp}.parquet

    The output filename is intentionally timestamped — downstream loaders
    (load_gold_parquet) select the most recent file by sort order, so
    re-running this script safely produces a new artifact without overwriting
    the previous one.
"""

import re
import io
import sys
import json
import zipfile
import argparse
import requests
import polars as pl
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------------
# Path bootstrap — same minimal pattern used in notebook 01 Phase 0.
# Adds the project root to sys.path so 'import src' works regardless of cwd.
# ---------------------------------------------------------------------------

def _bootstrap_project_root() -> Path:
    """Walk upward from CWD to find the project root (contains artifacts.yaml)."""
    current = Path.cwd()
    while current != current.parent:
        if (current / "artifacts.yaml").exists():
            return current.resolve()
        current = current.parent
    raise FileNotFoundError(
        "Could not find 'artifacts.yaml' in current or parent directories.\n"
        "Run this script from within the project tree."
    )

PROJECT_ROOT = _bootstrap_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Now safe to import project modules
from src.config import config                           # noqa: E402
from src.gatekeeper import validate_dataframe           # noqa: E402
from src.preprocessing import build_apso_note, redact_icd10_sections, ICD10_REDACT_PATTERN  # noqa: E402


# ==============================================================================
# PHASE 1a: Raw MedSynth ingestion
# ==============================================================================

# def phase_1a_ingest() -> pl.DataFrame:
#     """
#     Load raw MedSynth from HuggingFace and apply structural validation.

#     Mirrors the ingestion logic in notebook 01 Phase 1a:
#       - Fetches the train split of Ahmad0067/MedSynth
#       - Strips whitespace from column names (present in the HuggingFace release)
#       - Validates: no null IDs, no null Notes, ICD10 column present
#       - Sentinel-imputes the 2 null Dialogue records identified in Phase 1a

#     Returns the raw DataFrame with columns: ID, Note, Dialogue, ICD10.
#     The ICD10 column is kept as-is (list of raw strings without decimals).
#     """
#     print("\n── Phase 1a: Ingestion ──────────────────────────────────────────────")

#     try:
#         from datasets import load_dataset
#     except ImportError as e:
#         raise ImportError(
#             "The 'datasets' package is required. Install with: pip install datasets"
#         ) from e

#     print("📥 Loading MedSynth from HuggingFace (Ahmad0067/MedSynth)...")
#     ds = load_dataset("Ahmad0067/MedSynth", split="train")
#     df = pl.from_arrow(ds.data.table)
#     df = df.rename({c: c.strip() for c in df.columns})

#     print(f"   ✅ Loaded {len(df):,} records, columns: {df.columns}")

#     # Structural integrity assertions (mirrors Phase 1a checks)
#     assert "ID"       in df.columns, "❌ ID column missing"
#     assert "Note"     in df.columns, "❌ Note column missing"
#     assert "Dialogue" in df.columns, "❌ Dialogue column missing"
#     assert "ICD10"    in df.columns, "❌ ICD10 column missing"

#     null_ids   = df.filter(pl.col("ID").is_null()).height
#     null_notes = df.filter(pl.col("Note").is_null()).height
#     if null_ids > 0:
#         raise ValueError(f"❌ {null_ids} null IDs found — cannot proceed without unique record identifiers.")
#     if null_notes > 0:
#         raise ValueError(f"❌ {null_notes} null Notes found — cannot proceed with empty clinical notes.")

#     null_dialogues = df.filter(pl.col("Dialogue").is_null()).height
#     if null_dialogues > 0:
#         print(f"   ⚠️  {null_dialogues} null Dialogue(s) — applying sentinel imputation")
#         df = df.with_columns([
#             pl.col("Dialogue").fill_null("[NO_TRANSCRIPT_AVAILABLE]")
#         ])

#     # Cast ID to String for downstream consistency
#     df = df.with_columns([pl.col("ID").cast(pl.String)])

#     print(f"   ✅ Structural integrity confirmed: {len(df):,} records")
#     return df


def phase_1a_ingest() -> pl.DataFrame:
    print("\n── Phase 1a: Ingestion ──────────────────────────────────────────────")
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError("The 'datasets' package is required. Install with: pip install datasets") from e

    print("📥 Loading MedSynth from HuggingFace (Ahmad0067/MedSynth)...")
    ds = load_dataset("Ahmad0067/MedSynth", split="train")
    df = pl.from_arrow(ds.data.table)
    df = df.rename({c: c.strip() for c in df.columns})

    print(f"   ✅ Loaded {len(df):,} records, columns: {df.columns}")

    # --- Robustness fix 2026-05-04 ---
    # Old HF release: ID, Note, Dialogue, ICD10
    # New HF release: Note, Dialogue, ICD10, ICD10_desc (no ID)
    if "ID" not in df.columns:
        print("   ⚠️  ID column missing — generating sequential IDs")
        df = df.with_columns(pl.int_range(0, len(df)).cast(pl.String).alias("ID"))
    
    # Drop extra column if present (doesn't hurt pipeline)
    if "ICD10_desc" in df.columns:
        df = df.drop("ICD10_desc")
    
    # Now validate required columns
    assert "ID" in df.columns, "❌ ID column missing"
    assert "Note" in df.columns, "❌ Note column missing"
    assert "Dialogue" in df.columns, "❌ Dialogue column missing"
    assert "ICD10" in df.columns, "❌ ICD10 column missing"

    null_ids = df.filter(pl.col("ID").is_null()).height
    null_notes = df.filter(pl.col("Note").is_null()).height
    if null_ids > 0:
        raise ValueError(f"❌ {null_ids} null IDs found")
    if null_notes > 0:
        raise ValueError(f"❌ {null_notes} null Notes found")

    null_dialogues = df.filter(pl.col("Dialogue").is_null()).height
    if null_dialogues > 0:
        print(f"   ⚠️  {null_dialogues} null Dialogue(s) — applying sentinel imputation")
        df = df.with_columns([pl.col("Dialogue").fill_null("[NO_TRANSCRIPT_AVAILABLE]")])

    df = df.with_columns([pl.col("ID").cast(pl.String)])
    print(f"   ✅ Structural integrity confirmed: {len(df):,} records")
    return df


# ==============================================================================
# PHASE 1b: CDC FY2026 ICD-10 validation
# ==============================================================================

_CDC_URL = (
    "https://ftp.cdc.gov/pub/health_statistics/nchs/publications/"
    "ICD10CM/2026/icd10cm-Code%20Descriptions-2026.zip"
)
_CDC_CACHE_FILE = "cdc_fy2026_icd10.parquet"


def _load_cdc_reference(gold_dir: Path, offline: bool = False) -> pl.DataFrame:
    """
    Download (or restore from cache) the CDC FY2026 ICD-10 code descriptions.

    Returns a DataFrame with columns: code_no_decimal, description.
    Caches the result in data/gold/ to avoid repeated downloads.
    """
    cache_path = gold_dir / _CDC_CACHE_FILE

    if cache_path.exists():
        print(f"   📂 CDC reference cache hit: {cache_path.name}")
        return pl.read_parquet(cache_path)

    if offline:
        raise FileNotFoundError(
            f"CDC reference not cached at {cache_path} and --offline was specified.\n"
            "Run once without --offline to populate the cache."
        )

    print(f"   📥 Downloading CDC FY2026 ICD-10 reference from CDC FTP...")
    response = requests.get(_CDC_URL, timeout=60)
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        target = next(
            (f for f in z.namelist() if "codes" in f.lower() and f.endswith(".txt")),
            None,
        )
        if target is None:
            raise FileNotFoundError(
                f"Could not find a 'codes*.txt' file in CDC zip. "
                f"Available: {z.namelist()}"
            )
        with z.open(target) as f:
            lines = [line.decode("utf-8", errors="ignore").strip() for line in f]

    df = pl.DataFrame({"raw": lines}).filter(pl.col("raw").str.len_chars() > 0)
    df = df.with_columns([
        pl.col("raw").str.extract(r"^\s*(\S+)", 1)
          .str.replace(".", "", literal=True)
          .str.to_uppercase()
          .alias("code_no_decimal"),
        pl.col("raw").str.extract(r"^\s*\S+\s+(.+)$", 1).alias("description"),
    ])
    df = df.select(["code_no_decimal", "description"]).unique()

    df.write_parquet(cache_path, compression="zstd")
    print(f"   ✅ CDC reference loaded: {len(df):,} codes (cached to {cache_path.name})")
    return df


def phase_1b_validate_icd10(
    df_raw: pl.DataFrame,
    gold_dir: Path,
    offline: bool = False,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Validate ICD-10 codes against the CDC FY2026 reference.

    Mirrors notebook 01 Phase 1b. Returns (df_raw_with_status, cdc_df).

    The returned df_checked has a 'status' column per code:
        'billable'                  — confirmed leaf code in CDC FY2026
        'non_billable_parent'       — valid ICD-10 parent, too broad for billing
        'invalid_or_malformed'      — does not appear in CDC reference
    """
    print("\n── Phase 1b: CDC ICD-10 Validation ─────────────────────────────────")
    cdc_df = _load_cdc_reference(gold_dir, offline=offline)

    cdc_codes = set(cdc_df["code_no_decimal"].to_list())

    # Explode ICD10 list column → one row per (ID, code)
    df_exploded = (
        df_raw
        .with_columns([pl.col("ICD10").list.explode().alias("raw_code")])
        .select(["ID", "raw_code"])
        .filter(pl.col("raw_code").is_not_null())
    )

    def classify_code(code: str) -> str:
        if not code:
            return "invalid_or_malformed"
        code_upper = code.strip().upper().replace(".", "")
        if code_upper in cdc_codes:
            return "billable"
        # Parent codes: shorter prefixes (3-char) that exist in CDC as descriptors
        # The CDC file includes both billable leaf nodes and non-billable headers.
        # "non_billable_parent" = in CDC as a non-leaf; determined by absence as a
        # standalone billable code but presence as a 3-char prefix of billable codes.
        # Notebook 01 Phase 1b uses a simpler heuristic: if it's in the CDC dict
        # at all it's either billable or non_billable_parent; otherwise invalid.
        # We replicate that: check the 3-char prefix.
        if len(code_upper) == 3 and any(c.startswith(code_upper) for c in cdc_codes):
            return "non_billable_parent (Noisy 111)"
        # Placeholder-X codes: structurally valid ICD-10-CM injury codes with X placeholders
        # (e.g. T781XXA) — absent from CDC descriptions but clinically legitimate.
        if re.match(r"^[A-Z][0-9]{2}[0-9A-Z]*X[0-9A-Z]*$", code_upper):
            return "placeholder_x"
        return "invalid_or_malformed"

    statuses = [classify_code(c) for c in df_exploded["raw_code"].to_list()]
    df_checked = df_exploded.with_columns([pl.Series("status", statuses)])

    # Summary stats (mirrors Phase 1b.1 composition audit)
    status_counts = (
        df_checked
        .group_by("status")
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
    )
    print("   📊 ICD-10 Status Distribution:")
    for row in status_counts.iter_rows(named=True):
        print(f"      {row['status']:<40} {row['count']:>6,}")

    billable_n = df_checked.filter(pl.col("status") == "billable").height
    total_n    = df_checked.height
    print(f"\n   ✅ Billable codes: {billable_n:,} / {total_n:,} "
          f"({billable_n/total_n*100:.1f}%)")

    return df_checked, cdc_df


# ==============================================================================
# PHASE 1e: Pydantic Firewall → Silver layer
# ==============================================================================

def phase_1e_pydantic_firewall(df_raw: pl.DataFrame) -> pl.DataFrame:
    """
    Run the Pydantic Gatekeeper to validate schema and extract SOAP sections.

    Mirrors notebook 01 Phase 1e. Returns df_valid (Silver layer) with
    columns: id, note, dialogue, label, subjective, objective, assessment, plan.

    The validate_dataframe function in src/gatekeeper.py is the canonical
    implementation — this phase just calls it and reports results.
    """
    print("\n── Phase 1e: Pydantic Firewall ──────────────────────────────────────")

    df_valid, df_errors, _ = validate_dataframe(df_raw)

    n_valid  = len(df_valid)
    n_errors = len(df_errors)
    n_total  = n_valid + n_errors

    if n_errors > 0:
        print(f"   ⚠️  {n_errors:,} records failed validation:")
        if "error_type" in df_errors.columns:
            type_counts = (
                df_errors.group_by("error_type")
                .agg(pl.len().alias("count"))
                .sort("count", descending=True)
            )
            for row in type_counts.iter_rows(named=True):
                print(f"      {row['error_type']:<30} {row['count']:>5,}")
    else:
        print(f"   ✅ All {n_total:,} records passed validation")

    print(f"   ✅ Silver layer: {n_valid:,} records, columns: {df_valid.columns}")

    # Verify 100% SOAP extraction (expected on MedSynth)
    missing_soap = df_valid.filter(
        pl.col("assessment").is_null() | pl.col("plan").is_null()
    ).height
    if missing_soap > 0:
        print(f"   ⚠️  {missing_soap:,} records missing assessment or plan section")

    return df_valid


# ==============================================================================
# PHASE 1g: DuckDB Silver Vault
# ==============================================================================

def phase_1g_silver_vault(df_valid: pl.DataFrame) -> None:
    """
    Persist the Silver layer to DuckDB and verify label cardinality.

    Mirrors notebook 01 Phase 1g. Creates or replaces the silver_vault table
    in the project's audit_trail.ddb database.
    """
    print("\n── Phase 1g: DuckDB Silver Vault ────────────────────────────────────")

    with config.duckdb_connection() as con:
        con.register("df_valid_frame", df_valid)
        con.execute("CREATE OR REPLACE TABLE silver_vault AS SELECT * FROM df_valid_frame")

        # Label cardinality audit — mirrors Phase 1g SQL exactly
        # UNNEST is required because DuckDB's COUNT(DISTINCT) on list columns
        # does not reliably count distinct array values.
        try:
            label_stats = con.execute("""
                SELECT
                    COUNT(DISTINCT code) as unique_codes,
                    COUNT(DISTINCT id)   as total_records
                FROM (
                    SELECT id, UNNEST(label) as code
                    FROM silver_vault
                )
            """).pl()
            unique_codes  = label_stats["unique_codes"].item()
            total_records = label_stats["total_records"].item()
            print(f"   📊 Silver Vault — {total_records:,} records, {unique_codes:,} unique ICD codes")
        except Exception as e:
            # label column might not be a list type in some schema variants
            print(f"   ⚠️  Label cardinality audit skipped: {e}")
            raw_count = con.execute("SELECT COUNT(*) as n FROM silver_vault").pl()["n"].item()
            print(f"   📊 Silver Vault — {raw_count:,} records")

    print(f"   ✅ silver_vault written to {config.db_path}")


# ==============================================================================
# PHASE 2a: Decimal restoration (raw → canonical ICD-10)
# ==============================================================================

def phase_2a_decimal_restoration(df_valid: pl.DataFrame) -> pl.DataFrame:
    """
    Restore decimal points in raw ICD-10 codes → canonical format (e.g. M25562 → M25.562).

    Mirrors notebook 01 Phase 2a exactly. Produces a 'standard_icd10' column
    containing the canonical single-code string for each record.

    MedSynth records contain one ICD-10 code per record despite the label
    column being a list — the list structure is an artefact of the Pydantic
    schema. This phase takes label[0] and restores the decimal.
    """
    print("\n── Phase 2a: Decimal Restoration ────────────────────────────────────")

    def restore_decimal(code: str) -> str:
        """
        Insert decimal point at position 3 for codes >= 4 chars, otherwise return as-is.
        E.g. 'M25562' → 'M25.562', 'J18' → 'J18', 'M25.562' → 'M25.562' (already correct).
        """
        if not code or not isinstance(code, str):
            return code
        code = code.strip().upper()
        if "." in code:
            return code          # already canonical
        if len(code) >= 4:
            return code[:3] + "." + code[3:]
        return code              # 3-char parent code, no decimal needed

    # Extract first label from the list column and restore decimal
    df_gold = df_valid.with_columns([
        pl.col("label")
          .list.first()
          .map_elements(restore_decimal, return_dtype=pl.String)
          .alias("standard_icd10")
    ])

    # Verify canonical format: all codes match letter + 2 digits + optional decimal
    invalid_format = df_gold.filter(
        ~pl.col("standard_icd10").str.contains(r"^[A-Z][0-9]{2}(\.[0-9A-Z]{1,4})?$")
    ).height
    if invalid_format > 0:
        print(f"   ⚠️  {invalid_format:,} codes do not match canonical format — check Phase 1e")
    else:
        print(f"   ✅ Decimal restoration complete: {len(df_gold):,} codes in canonical format")

    # Sample for human review (mirrors Phase 2a output)
    n_sample = min(5, len(df_gold))
    sample = df_gold.select(["id", "standard_icd10"]).sample(n=n_sample, seed=42)
    print(f"   📋 Sample (n={n_sample}): {sample['standard_icd10'].to_list()}")

    return df_gold


# ==============================================================================
# PHASE 2c: Code status annotation
# ==============================================================================

def phase_2c_status_annotation(
    df_gold: pl.DataFrame,
    df_checked: pl.DataFrame,
    cdc_df: pl.DataFrame,
) -> pl.DataFrame:
    """
    Embed CDC validation status as 'code_status' column in the Gold layer.

    Mirrors notebook 01 Phase 2c. Status values:
        'billable'       — confirmed leaf code in CDC FY2026
        'noisy_111'      — valid ICD-10 parent code, too broad for billing
        'placeholder_x'  — structurally valid ICD-10-CM injury code,
                           absent from CDC descriptions (placeholder X convention)
    """
    print("\n── Phase 2c: Status Annotation ──────────────────────────────────────")

    # Build a code → status lookup from df_checked (one row per code)
    # Use the first status assigned to each code (they should be consistent)
    code_status_map = (
        df_checked
        .select(["raw_code", "status"])
        .unique(subset=["raw_code"])
    )

    # Normalise status labels to match Phase 2c conventions in the notebook
    def normalise_status(s: str) -> str:
        if "billable" in s.lower() and "non" not in s.lower():
            return "billable"
        if "noisy" in s.lower() or "non_billable" in s.lower() or "parent" in s.lower():
            return "noisy_111"
        if "placeholder" in s.lower():
            return "placeholder_x"
        return "invalid"

    normalised_statuses = [
        normalise_status(s) for s in code_status_map["status"].to_list()
    ]
    code_status_map = code_status_map.with_columns([
        pl.Series("code_status_norm", normalised_statuses)
    ])

    # Join on the standard_icd10 column (decimal-restored codes)
    # df_checked uses raw (no-decimal) codes — strip decimals for the join
    code_status_lookup = dict(
        zip(
            code_status_map["raw_code"].to_list(),
            code_status_map["code_status_norm"].to_list(),
        )
    )

    def lookup_status(canonical_code: str) -> str:
        if not canonical_code:
            return "invalid"
        raw = canonical_code.replace(".", "").upper()
        return code_status_lookup.get(raw, "billable")  # default billable if not found

    statuses = [lookup_status(c) for c in df_gold["standard_icd10"].to_list()]
    df_gold = df_gold.with_columns([pl.Series("code_status", statuses)])

    # Distribution summary
    dist = (
        df_gold.group_by("code_status")
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
    )
    print("   📊 Code status distribution:")
    for row in dist.iter_rows(named=True):
        pct = row["count"] / len(df_gold) * 100
        print(f"      {row['code_status']:<20} {row['count']:>6,}  ({pct:.1f}%)")

    print(f"   ✅ Status annotation complete: {len(df_gold):,} records")
    return df_gold


# ==============================================================================
# PHASE 3a: APSO-Flip
# ==============================================================================

def phase_3a_apso_flip(df_gold: pl.DataFrame) -> pl.DataFrame:
    """
    Reorder notes from SOAP to APSO (Assessment-first).

    Delegates to src.preprocessing.build_apso_note() — the canonical
    implementation ported from notebook 01 Phase 3a.

    Adds an 'apso_token_estimate' column (words × 1.3) mirroring the
    token pressure metric computed in Phase 1c/3a of the notebook.
    """
    print("\n── Phase 3a: APSO-Flip ──────────────────────────────────────────────")

    df_gold = build_apso_note(df_gold)

    # Token pressure estimate (mirrors Phase 3a stats in notebook)
    df_gold = df_gold.with_columns([
        (pl.col("apso_note")
         .str.count_matches(r"\S+")
         .cast(pl.Float64) * 1.3)
        .cast(pl.Int64)
        .alias("apso_token_estimate")
    ])

    at_risk = df_gold.filter(pl.col("apso_token_estimate") > 512).height
    pct_at_risk = at_risk / len(df_gold) * 100
    avg_tokens  = df_gold["apso_token_estimate"].mean()

    print(f"   📊 Token pressure: {at_risk:,} records > 512 tokens ({pct_at_risk:.1f}%)")
    print(f"   📊 Mean APSO token estimate: {avg_tokens:.0f}")
    print(f"   ✅ APSO-Flip complete: {len(df_gold):,} records")
    return df_gold


# ==============================================================================
# PHASE 3b: Leakage detection
# ==============================================================================

def phase_3b_leakage_detection(df_gold: pl.DataFrame) -> pl.DataFrame:
    """
    Quantify ICD-10 code leakage in apso_note before redaction.

    Mirrors notebook 01 Phase 3b: identification only, no modification.
    Adds a boolean 'has_leakage' column and reports summary statistics.
    """
    print("\n── Phase 3b: Leakage Detection ──────────────────────────────────────")

    df_gold = df_gold.with_columns([
        pl.col("apso_note")
          .str.contains(ICD10_REDACT_PATTERN)
          .alias("has_leakage")
    ])

    leakage_n   = df_gold.filter(pl.col("has_leakage")).height
    leakage_pct = leakage_n / len(df_gold) * 100

    print(f"   📊 Leakage detected: {leakage_n:,} records ({leakage_pct:.1f}%) contain ICD-10 strings")
    print(f"   ✅ Leakage quantification complete — redaction in Phase 3c")

    return df_gold


# ==============================================================================
# PHASE 3c: ICD-10 code redaction
# ==============================================================================

def phase_3c_redaction(df_gold: pl.DataFrame) -> pl.DataFrame:
    """
    Replace ICD-10 code strings with [REDACTED] in all SOAP section columns.

    Delegates to src.preprocessing.redact_icd10_sections() — the canonical
    implementation ported from notebook 01 Phase 3c.

    Post-redaction: asserts zero remaining code strings in apso_note.
    """
    print("\n── Phase 3c: ICD-10 Redaction ───────────────────────────────────────")

    df_gold = redact_icd10_sections(df_gold)

    # Post-redaction integrity check (mirrors Phase 4 prerequisite guard)
    remaining = df_gold.filter(
        pl.col("apso_note").str.contains(ICD10_REDACT_PATTERN)
    ).height

    if remaining > 0:
        raise ValueError(
            f"❌ Redaction incomplete: {remaining:,} records still contain ICD-10 strings.\n"
            "This indicates a pattern coverage gap — review ICD10_REDACT_PATTERN."
        )

    # Drop the intermediate leakage flag (not needed in final export)
    if "has_leakage" in df_gold.columns:
        df_gold = df_gold.drop("has_leakage")

    print(f"   ✅ Redaction complete: 0 ICD-10 strings remaining in apso_note")
    return df_gold


# ==============================================================================
# PHASE 4: Gold layer Parquet export
# ==============================================================================

def phase_4_export(df_gold: pl.DataFrame, dry_run: bool = False) -> Path:
    """
    Write the final Gold layer to a timestamped Parquet file.

    Mirrors notebook 01 Phase 4: pre-export validation, write, post-write
    verification, and audit log entry.

    Parameters
    ----------
    df_gold : pl.DataFrame
        Fully processed Gold layer DataFrame.
    dry_run : bool
        If True, validate and report without writing to disk.

    Returns
    -------
    Path
        Absolute path to the written Parquet file.
        Returns None on dry_run.
    """
    print("\n── Phase 4: Gold Layer Export ───────────────────────────────────────")

    # Pre-export redaction guard (mirrors Phase 4 cell exactly)
    icd_remaining = df_gold.filter(
        pl.col("apso_note").str.contains(ICD10_REDACT_PATTERN)
    ).height
    if icd_remaining > 0:
        raise ValueError(
            f"❌ Pre-export guard failed: {icd_remaining:,} records still contain ICD-10 strings.\n"
            "Run Phase 3c before exporting."
        )

    print(f"   ✅ Pre-export guard: 0 ICD-10 strings in apso_note")
    print(f"   📋 Final schema: {len(df_gold):,} rows × {len(df_gold.columns)} cols")
    print(f"   📋 Columns: {df_gold.columns}")

    if dry_run:
        print("   ℹ️  --dry-run: skipping file write")
        return None

    gold_dir    = config.resolve_path("data", "gold")
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_path = gold_dir / f"medsynth_gold_apso_{timestamp}.parquet"

    print(f"\n   💾 Writing to: {export_path}")
    df_gold.write_parquet(export_path, compression="snappy")

    file_size_mb = export_path.stat().st_size / (1024 * 1024)
    print(f"   ✅ Written: {file_size_mb:.1f} MB")

    # Post-write verification (mirrors Phase 4 step 4)
    df_verify = pl.read_parquet(export_path)
    if df_verify.height != len(df_gold) or df_verify.columns != df_gold.columns:
        raise ValueError(
            f"❌ Post-write verification failed.\n"
            f"   Source:  {len(df_gold):,} rows, {df_gold.columns}\n"
            f"   Written: {df_verify.height:,} rows, {df_verify.columns}"
        )
    print(f"   ✅ Post-write verification passed: {df_verify.height:,} rows")

    # Audit log
    config.log_event(
        phase="prepare_data",
        action="gold_export",
        details={
            "path":        str(export_path),
            "rows":        len(df_gold),
            "columns":     df_gold.columns,
            "size_mb":     round(file_size_mb, 2),
            "script":      "scripts/prepare_data.py",
        },
        notebook="prepare_data.py",
    )

    print(f"\n{'='*70}")
    print(f"  ✅ Gold layer ready: {export_path.name}")
    print(f"{'='*70}")
    return export_path


# ==============================================================================
# Entry point
# ==============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare the MedSynth Gold layer (headless notebook 01 equivalent)."
    )
    parser.add_argument(
        "--no-duckdb",
        action="store_true",
        help="Skip Phase 1g (DuckDB Silver Vault persistence).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run all phases but skip the final Parquet write.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Skip CDC download — use cached reference only. "
             "Raises if cache is absent.",
    )
    args = parser.parse_args()

    start = datetime.now()
    print(f"\n{'='*70}")
    print(f"  prepare_data.py — MedSynth Gold Layer Pipeline")
    print(f"  Started: {start.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Project root: {PROJECT_ROOT}")
    print(f"{'='*70}")

    gold_dir = config.resolve_path("data", "gold")

    # ── Phase 0: Infrastructure (ArtifactConfig.__init__ handles this) ─────
    print("\n── Phase 0: Infrastructure ──────────────────────────────────────────")
    validation = config.validate_config()
    if validation["errors"]:
        for err in validation["errors"]:
            print(f"   ❌ {err}")
        raise RuntimeError("Configuration errors — cannot proceed.")
    print("   ✅ Configuration valid")

    # ── Phase 1a ────────────────────────────────────────────────────────────
    df_raw = phase_1a_ingest()

    # ── Phase 1b ────────────────────────────────────────────────────────────
    df_checked, cdc_df = phase_1b_validate_icd10(df_raw, gold_dir, offline=args.offline)

    # ── Phase 1e ────────────────────────────────────────────────────────────
    df_valid = phase_1e_pydantic_firewall(df_raw)

    # ── Phase 1g ────────────────────────────────────────────────────────────
    if not args.no_duckdb:
        phase_1g_silver_vault(df_valid)
    else:
        print("\n── Phase 1g: DuckDB Silver Vault (skipped via --no-duckdb) ─────────")

    # ── Phase 2a ────────────────────────────────────────────────────────────
    df_gold = phase_2a_decimal_restoration(df_valid)

    # ── Phase 2c ────────────────────────────────────────────────────────────
    df_gold = phase_2c_status_annotation(df_gold, df_checked, cdc_df)

    # ── Phase 3a ────────────────────────────────────────────────────────────
    df_gold = phase_3a_apso_flip(df_gold)

    # ── Phase 3b ────────────────────────────────────────────────────────────
    df_gold = phase_3b_leakage_detection(df_gold)

    # ── Phase 3c ────────────────────────────────────────────────────────────
    df_gold = phase_3c_redaction(df_gold)

    # ── Phase 4 ─────────────────────────────────────────────────────────────
    export_path = phase_4_export(df_gold, dry_run=args.dry_run)

    elapsed = (datetime.now() - start).total_seconds()
    print(f"\n  Total runtime: {elapsed:.1f}s")


if __name__ == "__main__":
    main()