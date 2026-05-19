"""
scripts/validation/validate_mimic_prepare.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MIMIC-IV-Note Validation Pipeline — Medallion Preparation (Bronze → Gold)

PURPOSE
-------
Prepares the MIMIC-IV discharge note dataset for validation against the
notes-to-icd10 model (E-010). This script implements the data preparation
half of the validation pipeline:

    Bronze  →  raw MIMIC-IV files ingested via DuckDB
    Silver  →  joined, filtered (ICD-10 only, billable codes only)
    Gold    →  APSO-preprocessed, stratified sample, ready for inference

The companion script `validate_mimic_evaluate.py` loads the Gold layer and
runs the E-010 model, computing E2E accuracy, F1, and ECE.

RESTRICTED DATA — DO NOT COMMIT DERIVED DATA
---------------------------------------------
MIMIC-IV is a credentialed dataset (PhysioNet, DUA v1.5.0). Derived data,
intermediate files, and results MUST NOT be committed to git. All MIMIC
data is written to `data/mimic/` which is gitignored.

To access MIMIC-IV-Note (v2.2):
  1. Complete CITI "Data or Specimens Only Research" training
  2. Apply at: https://physionet.org/content/mimic-iv-note/2.2/
  3. Sign the PhysioNet Credentialed Health Data Use Agreement 1.5.0

To access MIMIC-IV clinical data (diagnoses_icd.csv.gz):
  4. Apply at: https://physionet.org/content/mimiciv/2.2/
  5. Sign the separate DUA for the clinical database

REQUIRED FILES (set paths in .env or pass as CLI args)
-------------------------------------------------------
  discharge.csv.gz      — from MIMIC-IV-Note v2.2 /note/
  diagnoses_icd.csv.gz  — from MIMIC-IV clinical v2.2 /hosp/

MEDALLION OUTPUT (written to data/mimic/, gitignored)
------------------------------------------------------
  data/mimic/bronze/    — registered DuckDB views of raw files
  data/mimic/silver/    — joined + filtered Parquet
  data/mimic/gold/      — APSO-preprocessed + stratified sample Parquet

USAGE
-----
  # Set environment variables (or pass as --args)
  export MIMIC_DISCHARGE_PATH=~/Downloads/mimic-iv-note-2.2/note/discharge.csv.gz
  export MIMIC_DIAGNOSES_PATH=~/Downloads/mimic-iv-note-2.2/note/diagnoses_icd.csv.gz

  uv run python scripts/validation/validate_mimic_prepare.py
  uv run python scripts/validation/validate_mimic_prepare.py --sample-size 5000
  uv run python scripts/validation/validate_mimic_prepare.py --full-run

CITATION
--------
Johnson, A.E.W., et al. MIMIC-IV-Note: Deidentified free-text clinical notes.
PhysioNet (2023). https://doi.org/10.13026/1n74-ne17

Johnson, A., et al. MIMIC-IV (version 2.2). PhysioNet (2023).
https://doi.org/10.13026/6mm1-ek67
"""

import sys
import os
import argparse
import time
from pathlib import Path

# ── Project root resolution ────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import duckdb
import polars as pl
from src.config import config
from src.preprocessing import prepare_inference_input

# ── MIMIC section header mapping to APSO ──────────────────────────────────
# MIMIC discharge summaries use different section headers than MedSynth SOAP.
# This mapping defines the priority order for APSO-Flip on MIMIC notes.
# Sections listed first are placed first in the APSO-flipped note.
MIMIC_SECTION_PRIORITY = [
    # Assessment-equivalent (placed first — highest diagnostic signal)
    "Discharge Diagnosis:",
    "Discharge Diagnoses:",
    "PRIMARY DIAGNOSIS:",
    "PRIMARY DIAGNOSES:",
    "Assessment:",
    "ASSESSMENT:",
    "IMPRESSION:",
    # Plan-equivalent
    "Brief Hospital Course:",
    "BRIEF HOSPITAL COURSE:",
    "Plan:",
    "PLAN:",
    "TRANSITIONAL ISSUES:",
    # Subjective-equivalent
    "Chief Complaint:",
    "CHIEF COMPLAINT:",
    "History of Present Illness:",
    "HISTORY OF PRESENT ILLNESS:",
    # Objective-equivalent (lowest priority)
    "Physical Exam:",
    "PHYSICAL EXAM:",
    "ADMISSION PHYSICAL EXAM:",
    "Pertinent Results:",
    "PERTINENT RESULTS:",
    "ADMISSION LABS:",
]


def _apso_flip_mimic(text: str) -> str:
    """
    Apply APSO-Flip to a MIMIC-IV discharge summary.

    MIMIC notes use different section headers than MedSynth SOAP notes.
    This function extracts sections by header, reorders them in APSO
    priority, and concatenates — reproducing the training-time APSO-Flip
    for MIMIC's note format.

    Falls back to the full note text if no recognised headers are found,
    consistent with prepare_inference_input() behaviour.

    Parameters
    ----------
    text : str
        Raw MIMIC-IV discharge summary text.

    Returns
    -------
    str
        APSO-reordered note text, ready for tokenisation.
    """
    import re

    # Build regex to split on any recognised section header
    # Pattern: header at start of line (possibly after whitespace)
    all_headers = [re.escape(h) for h in MIMIC_SECTION_PRIORITY]
    split_pattern = r'(?m)^(?:' + '|'.join(all_headers) + r')'

    # Find all header positions
    found_sections = {}
    lines = text.split('\n')
    current_header = '__preamble__'
    current_content = []

    for line in lines:
        stripped = line.strip()
        matched_header = None
        for header in MIMIC_SECTION_PRIORITY:
            if stripped == header or stripped.startswith(header + ' '):
                matched_header = header
                break

        if matched_header:
            # Save previous section
            if current_header not in found_sections:
                found_sections[current_header] = []
            found_sections[current_header].append('\n'.join(current_content).strip())
            current_header = matched_header
            current_content = []
        else:
            current_content.append(line)

    # Save last section
    if current_header not in found_sections:
        found_sections[current_header] = []
    found_sections[current_header].append('\n'.join(current_content).strip())

    # Reorder in APSO priority — drop preamble and empty sections
    ordered_parts = []
    for header in MIMIC_SECTION_PRIORITY:
        if header in found_sections:
            content = ' '.join(found_sections[header]).strip()
            if content:
                ordered_parts.append(f"{header}\n{content}")

    if not ordered_parts:
        # No recognised headers — fall back to full text
        return text.strip()

    return '\n\n'.join(ordered_parts)


def _resolve_mimic_paths(args) -> tuple[Path, Path]:
    """Resolve MIMIC file paths from CLI args or environment variables."""
    discharge_path = (
        Path(args.discharge_path) if args.discharge_path
        else Path(os.environ.get(
            'MIMIC_DISCHARGE_PATH',
            str(Path.home() / 'Downloads/mimic-iv-note-2.2/note/discharge.csv.gz')
        ))
    )
    diagnoses_path = (
        Path(args.diagnoses_path) if args.diagnoses_path
        else Path(os.environ.get(
            'MIMIC_DIAGNOSES_PATH',
            str(Path.home() / 'Downloads/mimic-iv-note-2.2/note/diagnoses_icd.csv.gz')
        ))
    )

    for p in [discharge_path, diagnoses_path]:
        if not p.exists():
            raise FileNotFoundError(
                f"MIMIC file not found: {p}\n"
                f"Set MIMIC_DISCHARGE_PATH / MIMIC_DIAGNOSES_PATH in .env, "
                f"or pass --discharge-path / --diagnoses-path."
            )
    return discharge_path, diagnoses_path


def _ensure_mimic_dirs() -> dict[str, Path]:
    """Create and return the MIMIC medallion directories."""
    mimic_base = config.project_root / "data" / "mimic"
    dirs = {
        layer: mimic_base / layer
        for layer in ["bronze", "silver", "gold"]
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def _load_our_label_map() -> dict[str, str]:
    """
    Load all chapter label maps and return a nodot→dotted code dict.

    Returns
    -------
    dict
        Mapping from code-without-dots (e.g. 'A419') to
        code-with-dots (e.g. 'A41.9') for all 1,916 codes in our label map.
    """
    stage2_dir = config.project_root / "outputs" / "evaluations" / \
                 "E-010_40ep_E002Init" / "stage2"
    import json
    nodot_to_dotted = {}
    for lm_path in stage2_dir.glob("*/label_map.json"):
        with open(lm_path) as f:
            lmap = json.load(f)
        for code in lmap["label2id"]:
            nodot_to_dotted[code.replace(".", "")] = code
    return nodot_to_dotted


def run_bronze(
    conn: duckdb.DuckDBPyConnection,
    discharge_path: Path,
    diagnoses_path: Path,
    dirs: dict[str, Path],
) -> None:
    """
    Bronze layer: register raw MIMIC files as DuckDB views and log provenance.

    No data transformation at this layer — raw files only.
    """
    print("\n── Bronze: Registering raw MIMIC files ──────────────────────────────")

    # Register raw files as DuckDB views (no copy — read directly from gz)
    conn.execute(f"""
        CREATE OR REPLACE VIEW bronze_discharge AS
        SELECT * FROM read_csv_auto('{discharge_path}', compression='gzip')
    """)
    conn.execute(f"""
        CREATE OR REPLACE VIEW bronze_diagnoses AS
        SELECT * FROM read_csv_auto('{diagnoses_path}', compression='gzip')
    """)

    # Log provenance to DuckDB
    conn.execute("""
        CREATE TABLE IF NOT EXISTS mimic_provenance (
            layer       VARCHAR,
            source_file VARCHAR,
            n_records   INTEGER,
            created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    n_discharge = conn.execute("SELECT COUNT(*) FROM bronze_discharge").fetchone()[0]
    n_diagnoses = conn.execute("SELECT COUNT(*) FROM bronze_diagnoses").fetchone()[0]

    conn.execute("""
        INSERT INTO mimic_provenance (layer, source_file, n_records)
        VALUES
            ('bronze', 'discharge.csv.gz', ?),
            ('bronze', 'diagnoses_icd.csv.gz', ?)
    """, [n_discharge, n_diagnoses])

    print(f" ✅ discharge:  {n_discharge:,} records")
    print(f" ✅ diagnoses:  {n_diagnoses:,} records")


def run_silver(
    conn: duckdb.DuckDBPyConnection,
    dirs: dict[str, Path],
    nodot_to_dotted: dict[str, str],
) -> int:
    """
    Silver layer: join discharge notes to ICD-10 primary diagnoses,
    filter to codes in our label map, normalise code format.

    Returns
    -------
    int
        Number of records in the silver layer.
    """
    print("\n── Silver: Join + filter + normalise ────────────────────────────────")

    # Build a VALUES table of our label map codes for SQL join
    # DuckDB supports this natively
    code_values = ", ".join(
        f"('{nodot}', '{dotted}')"
        for nodot, dotted in nodot_to_dotted.items()
    )

    silver_path = dirs["silver"] / "mimic_silver.parquet"

    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS our_codes (
            code_nodot  VARCHAR PRIMARY KEY,
            code_dotted VARCHAR
        )
    """)
    conn.execute("DELETE FROM our_codes")
    conn.execute(f"INSERT INTO our_codes VALUES {code_values}")

    # Join: discharge notes → ICD-10 primary diagnosis → our label map
    conn.execute(f"""
        COPY (
            SELECT
                d.note_id,
                d.subject_id,
                d.hadm_id,
                d.text                              AS raw_text,
                diag.icd_code                       AS icd_code_nodot,
                oc.code_dotted                      AS icd_code,
                LEFT(oc.code_dotted, 1)             AS icd_chapter
            FROM bronze_discharge d
            INNER JOIN (
                -- Primary diagnosis only (seq_num = 1), ICD-10 only
                SELECT subject_id, hadm_id, icd_code
                FROM bronze_diagnoses
                WHERE icd_version = '10'
                  AND seq_num = 1
            ) diag
                ON d.hadm_id = diag.hadm_id
            INNER JOIN our_codes oc
                ON diag.icd_code = oc.code_nodot
            -- One discharge note per admission (take the one with highest seq)
            QUALIFY ROW_NUMBER() OVER (
                PARTITION BY d.hadm_id
                ORDER BY d.note_seq DESC
            ) = 1
        )
        TO '{silver_path}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)

    n_silver = conn.execute(
        f"SELECT COUNT(*) FROM read_parquet('{silver_path}')"
    ).fetchone()[0]

    conn.execute("""
        INSERT INTO mimic_provenance (layer, source_file, n_records)
        VALUES ('silver', 'mimic_silver.parquet', ?)
    """, [n_silver])

    # Chapter distribution
    chapter_dist = conn.execute(f"""
        SELECT icd_chapter, COUNT(*) AS n
        FROM read_parquet('{silver_path}')
        GROUP BY icd_chapter
        ORDER BY n DESC
    """).fetchall()

    print(f" ✅ Silver records: {n_silver:,}")
    print(f" Chapter distribution (top 10):")
    for ch, n in chapter_dist[:10]:
        print(f"   {ch}: {n:,}")

    return n_silver


def run_gold(
    conn: duckdb.DuckDBPyConnection,
    dirs: dict[str, Path],
    sample_size: int,
    full_run: bool,
    seed: int = 42,
) -> int:
    """
    Gold layer: APSO-preprocess notes, apply stratified sample.

    APSO preprocessing is done in Python (Polars UDF) since it requires
    regex section extraction. The result is written to Parquet.

    Parameters
    ----------
    sample_size : int
        Number of records to sample (stratified by chapter).
        Ignored if full_run=True.
    full_run : bool
        If True, use all silver records (no sampling).

    Returns
    -------
    int
        Number of records in the gold layer.
    """
    print(f"\n── Gold: APSO preprocessing + {'full run' if full_run else f'stratified sample (n={sample_size:,})'} ──")

    silver_path = dirs["silver"] / "mimic_silver.parquet"
    gold_path   = dirs["gold"]   / "mimic_gold.parquet"

    # Load silver into Polars for APSO preprocessing
    silver_df = pl.read_parquet(silver_path)
    print(f" Loaded {len(silver_df):,} silver records")

    # Stratified sample by chapter
    if not full_run:
        n_chapters = silver_df["icd_chapter"].n_unique()
        per_chapter = max(1, sample_size // n_chapters)
        silver_df = (
            silver_df
            .with_columns(pl.lit(1).alias("_row"))
            .sort("_row")
            .group_by("icd_chapter")
            .agg(pl.all().sample(n=per_chapter, seed=seed, with_replacement=False))
            .explode(pl.all().exclude("icd_chapter"))
            .drop("_row")
            .sample(fraction=1.0, shuffle=True, seed=seed)  # shuffle
        )
        print(f" Stratified sample: {len(silver_df):,} records ({per_chapter} per chapter)")

    # Apply APSO-Flip to raw MIMIC text
    print(f" Applying APSO-Flip to {len(silver_df):,} notes...", flush=True)
    t0 = time.perf_counter()

    apso_notes = [
        _apso_flip_mimic(text)
        for text in silver_df["raw_text"].to_list()
    ]

    elapsed = time.perf_counter() - t0
    print(f" Done in {elapsed:.1f}s ({len(silver_df)/elapsed:.0f} notes/sec)")

    # Add apso_note column, drop raw_text (save space)
    gold_df = silver_df.with_columns(
        pl.Series("apso_note", apso_notes)
    ).drop("raw_text")

    # Write gold layer
    gold_df.write_parquet(gold_path, compression="zstd")

    n_gold = len(gold_df)
    conn.execute("""
        INSERT INTO mimic_provenance (layer, source_file, n_records)
        VALUES ('gold', 'mimic_gold.parquet', ?)
    """, [n_gold])

    # Summary statistics
    avg_len = gold_df["apso_note"].str.len_chars().mean()
    print(f" ✅ Gold records: {n_gold:,}")
    print(f" Avg APSO note length: {avg_len:.0f} chars")
    print(f" Columns: {gold_df.columns}")
    print(f" Saved to: {gold_path}")

    return n_gold


def main(args) -> None:
    t_start = time.perf_counter()

    print("=" * 70)
    print("  validate_mimic_prepare.py — MIMIC-IV Medallion Preparation")
    print(f"  Sample size: {'FULL RUN' if args.full_run else args.sample_size:,}" if not args.full_run else "  Sample size: FULL RUN")
    print(f"  Experiment:  E-010_40ep_E002Init")
    print("=" * 70)

    # Resolve paths
    discharge_path, diagnoses_path = _resolve_mimic_paths(args)
    print(f"\n📥 discharge:  {discharge_path}")
    print(f"📥 diagnoses:  {diagnoses_path}")

    # Create medallion directories
    dirs = _ensure_mimic_dirs()

    # Load our label map
    nodot_to_dotted = _load_our_label_map()
    print(f"\n📋 Label map: {len(nodot_to_dotted):,} billable codes")

    # DuckDB connection — in-memory for pipeline, provenance written to project db
    with config.duckdb_connection() as conn:
        run_bronze(conn, discharge_path, diagnoses_path, dirs)
        n_silver = run_silver(conn, dirs, nodot_to_dotted)
        n_gold = run_gold(
            conn, dirs,
            sample_size=args.sample_size,
            full_run=args.full_run,
            seed=42,
        )

        # Final provenance summary
        print("\n── Provenance log ───────────────────────────────────────────────────")
        prov = conn.execute("""
            SELECT layer, source_file, n_records, created_at
            FROM mimic_provenance
            ORDER BY created_at
        """).fetchall()
        for row in prov:
            print(f"  {row[0]:8s} | {row[1]:35s} | {row[2]:>8,} records | {row[3]}")

    elapsed = time.perf_counter() - t_start
    print(f"\n{'='*70}")
    print(f" ✅ Preparation complete — {elapsed:.0f}s")
    print(f" Gold layer: {dirs['gold'] / 'mimic_gold.parquet'}")
    print(f"\n Next step:")
    print(f"   uv run python scripts/validation/validate_mimic_evaluate.py")
    print(f"{'='*70}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="MIMIC-IV validation pipeline — medallion preparation"
    )
    parser.add_argument(
        "--discharge-path",
        default=None,
        help="Path to discharge.csv.gz (overrides MIMIC_DISCHARGE_PATH env var)"
    )
    parser.add_argument(
        "--diagnoses-path",
        default=None,
        help="Path to diagnoses_icd.csv.gz (overrides MIMIC_DIAGNOSES_PATH env var)"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=5000,
        help="Stratified sample size (default: 5000). Ignored if --full-run."
    )
    parser.add_argument(
        "--full-run",
        action="store_true",
        help="Use all matched records (~78k). Takes ~2-3 hours on MPS."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
