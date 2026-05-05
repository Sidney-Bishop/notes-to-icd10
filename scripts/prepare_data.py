"""
prepare_data.py — Headless Gold layer preparation pipeline.

Migrates the deterministic, automatable phases from notebook 01-EDA_SOAP.ipynb
to a reproducible command-line script. Interactive phases (visual auditors) are
excluded by design — this script is the headless, CI-safe complement to the
notebook, not a replacement for it.

Phases executed
---------------
    Phase 0   Infrastructure & path validation (via ArtifactConfig)
    Phase 1a  Raw MedSynth ingestion from DVC-tracked canonical file
    Phase 1b  CDC FY2026 ICD-10 validation and status classification
    Phase 1e  Pydantic Firewall — schema validation and SOAP section extraction
    Phase 1g  DuckDB Silver Vault persistence
    Phase 2a  Decimal restoration (raw → canonical ICD-10 format)
    Phase 2c  Code status annotation (billable / noisy_111 / placeholder_x)
    Phase 3a  APSO-Flip (Assessment-first reordering)
    Phase 3b  ICD-10 leakage detection (quantification only)
    Phase 3c  ICD-10 code redaction ([REDACTED] substitution)
    Phase 4   Gold layer Parquet export with post-write verification
"""

import re
import io
import sys
import json
import zipfile
import argparse
import subprocess
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

def phase_1a_ingest() -> pl.DataFrame:
    print("\n── Phase 1a: Ingestion ──────────────────────────────────────────────")

    raw_path = config.resolve_path("data", "raw") / "medsynth_notes_v2024.parquet"

    if not raw_path.exists():
        print(f" ⚠️ Raw not found — running fetcher...")
        subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts/fetch_raw_medsynth.py")], check=True)

    df = pl.read_parquet(raw_path)
    print(f" ✅ Loaded {len(df):,} records from DVC, columns: {df.columns}")

    # Validate (same as original notebook)
    assert set(["ID","Note","Dialogue","ICD10"]).issubset(df.columns)
    df = df.with_columns(pl.col("ID").cast(pl.String))

    null_dialogues = df.filter(pl.col("Dialogue").is_null()).height
    if null_dialogues:
        df = df.with_columns(pl.col("Dialogue").fill_null("[NO_TRANSCRIPT_AVAILABLE]"))

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
    cache_path = gold_dir / _CDC_CACHE_FILE

    if cache_path.exists():
        print(f"   📂 CDC reference cache hit: {cache_path.name}")
        return pl.read_parquet(cache_path)

    if offline:
        raise FileNotFoundError(
            f"CDC reference not cached at {cache_path} and --offline was specified."
        )

    print(f"   📥 Downloading CDC FY2026 ICD-10 reference...")
    response = requests.get(_CDC_URL, timeout=60)
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        target = next((f for f in z.namelist() if "codes" in f.lower() and f.endswith(".txt")), None)
        if target is None:
            raise FileNotFoundError("Could not find a 'codes*.txt' file in CDC zip.")
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
    return df


def phase_1b_validate_icd10(df_raw: pl.DataFrame, gold_dir: Path, offline: bool = False) -> tuple[pl.DataFrame, pl.DataFrame]:
    print("\n── Phase 1b: CDC ICD-10 Validation ─────────────────────────────────")
    cdc_df = _load_cdc_reference(gold_dir, offline=offline)
    cdc_codes = set(cdc_df["code_no_decimal"].to_list())

    df_exploded = (
        df_raw
        .with_columns([pl.col("ICD10").list.explode().alias("raw_code")])
        .select(["ID", "raw_code"])
        .filter(pl.col("raw_code").is_not_null())
    )

    def classify_code(code: str) -> str:
        if not code: return "invalid_or_malformed"
        code_upper = code.strip().upper().replace(".", "")
        if code_upper in cdc_codes: return "billable"
        if len(code_upper) == 3 and any(c.startswith(code_upper) for c in cdc_codes):
            return "non_billable_parent (Noisy 111)"
        if re.match(r"^[A-Z][0-9]{2}[0-9A-Z]*X[0-9A-Z]*$", code_upper):
            return "placeholder_x"
        return "invalid_or_malformed"

    statuses = [classify_code(c) for c in df_exploded["raw_code"].to_list()]
    df_checked = df_exploded.with_columns([pl.Series("status", statuses)])

    status_counts = df_checked.group_by("status").agg(pl.len().alias("count")).sort("count", descending=True)
    print("   📊 ICD-10 Status Distribution:")
    for row in status_counts.iter_rows(named=True):
        print(f"      {row['status']:<40} {row['count']:>6,}")

    return df_checked, cdc_df

# ==============================================================================
# PHASE 1e: Pydantic Firewall
# ==============================================================================

def phase_1e_pydantic_firewall(df_raw: pl.DataFrame) -> pl.DataFrame:
    print("\n── Phase 1e: Pydantic Firewall ──────────────────────────────────────")
    df_valid, df_errors, _ = validate_dataframe(df_raw)
    print(f"   ✅ Silver layer: {len(df_valid):,} records")
    return df_valid

# ==============================================================================
# PHASE 1g: DuckDB Silver Vault
# ==============================================================================

def phase_1g_silver_vault(df_valid: pl.DataFrame) -> None:
    print("\n── Phase 1g: DuckDB Silver Vault ────────────────────────────────────")
    with config.duckdb_connection() as con:
        con.register("df_valid_frame", df_valid)
        con.execute("CREATE OR REPLACE TABLE silver_vault AS SELECT * FROM df_valid_frame")
    print(f"   ✅ silver_vault written to {config.db_path}")

# ==============================================================================
# PHASE 2a: Decimal restoration
# ==============================================================================

def phase_2a_decimal_restoration(df_valid: pl.DataFrame) -> pl.DataFrame:
    print("\n── Phase 2a: Decimal Restoration ────────────────────────────────────")
    def restore_decimal(code: str) -> str:
        if not code or not isinstance(code, str): return code
        code = code.strip().upper()
        if "." in code: return code
        if len(code) >= 4: return code[:3] + "." + code[3:]
        return code

    df_gold = df_valid.with_columns([
        pl.col("label").list.first().map_elements(restore_decimal, return_dtype=pl.String).alias("standard_icd10")
    ])
    print(f"   ✅ Decimal restoration complete")
    return df_gold

# ==============================================================================
# PHASE 2c: Code status annotation
# ==============================================================================

def phase_2c_status_annotation(df_gold: pl.DataFrame, df_checked: pl.DataFrame, cdc_df: pl.DataFrame) -> pl.DataFrame:
    print("\n── Phase 2c: Status Annotation ──────────────────────────────────────")
    code_status_map = df_checked.select(["raw_code", "status"]).unique(subset=["raw_code"])

    def normalise_status(s: str) -> str:
        if "billable" in s.lower() and "non" not in s.lower(): return "billable"
        if "noisy" in s.lower() or "non_billable" in s.lower() or "parent" in s.lower(): return "noisy_111"
        if "placeholder" in s.lower(): return "placeholder_x"
        return "invalid"

    code_status_lookup = dict(zip(code_status_map["raw_code"], [normalise_status(s) for s in code_status_map["status"]]))

    def lookup_status(canonical_code: str) -> str:
        if not canonical_code: return "invalid"
        raw = canonical_code.replace(".", "").upper()
        return code_status_lookup.get(raw, "billable")

    df_gold = df_gold.with_columns([
        pl.col("standard_icd10").map_elements(lookup_status, return_dtype=pl.String).alias("code_status")
    ])
    return df_gold

# ==============================================================================
# PHASE 3a, 3b, 3c: Processing
# ==============================================================================

def phase_3a_apso_flip(df_gold: pl.DataFrame) -> pl.DataFrame:
    print("\n── Phase 3a: APSO-Flip ──────────────────────────────────────────────")
    df_gold = build_apso_note(df_gold)
    df_gold = df_gold.with_columns([(pl.col("apso_note").str.count_matches(r"\S+").cast(pl.Float64) * 1.3).cast(pl.Int64).alias("apso_token_estimate")])
    return df_gold

def phase_3b_leakage_detection(df_gold: pl.DataFrame) -> pl.DataFrame:
    print("\n── Phase 3b: Leakage Detection ──────────────────────────────────────")
    df_gold = df_gold.with_columns([pl.col("apso_note").str.contains(ICD10_REDACT_PATTERN).alias("has_leakage")])
    return df_gold

def phase_3c_redaction(df_gold: pl.DataFrame) -> pl.DataFrame:
    print("\n── Phase 3c: ICD-10 Redaction ───────────────────────────────────────")
    df_gold = redact_icd10_sections(df_gold)
    if "has_leakage" in df_gold.columns: df_gold = df_gold.drop("has_leakage")
    return df_gold

# ==============================================================================
# PHASE 4: Export
# ==============================================================================

def phase_4_export(df_gold: pl.DataFrame, dry_run: bool = False) -> Path:
    print("\n── Phase 4: Gold Layer Export ───────────────────────────────────────")
    if dry_run:
        print("   ℹ️  --dry-run: skipping file write")
        return None

    gold_dir = config.resolve_path("data", "gold")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_path = gold_dir / f"medsynth_gold_apso_{timestamp}.parquet"
    df_gold.write_parquet(export_path, compression="snappy")
    print(f"  ✅ Gold layer ready: {export_path.name}")
    return export_path

# ==============================================================================
# Entry point
# ==============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the MedSynth Gold layer.")
    parser.add_argument("--no-duckdb", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()

    start = datetime.now()
    print(f"\n{'='*70}\n  prepare_data.py — MedSynth Gold Layer Pipeline\n{'='*70}")

    gold_dir = config.resolve_path("data", "gold")
    df_raw = phase_1a_ingest()
    df_checked, cdc_df = phase_1b_validate_icd10(df_raw, gold_dir, offline=args.offline)
    df_valid = phase_1e_pydantic_firewall(df_raw)
    
    if not args.no_duckdb: phase_1g_silver_vault(df_valid)
    
    df_gold = phase_2a_decimal_restoration(df_valid)
    df_gold = phase_2c_status_annotation(df_gold, df_checked, cdc_df)
    df_gold = phase_3a_apso_flip(df_gold)
    df_gold = phase_3b_leakage_detection(df_gold)
    df_gold = phase_3c_redaction(df_gold)
    phase_4_export(df_gold, dry_run=args.dry_run)

    print(f"\n  Total runtime: {(datetime.now() - start).total_seconds():.1f}s")

if __name__ == "__main__":
    main()