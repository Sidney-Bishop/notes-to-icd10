"""
prepare_data.py — Headless Gold layer preparation pipeline.
Locked to canonical HF datasets: SidneyBishop/notes-to-icd10
"""
import hashlib, re, sys, argparse
from pathlib import Path
from datetime import datetime

import polars as pl
from huggingface_hub import hf_hub_download

# Bootstrap: locate project root before src/ is on sys.path.
# find_project_root() is defined in src/config.py — single source of truth.
def _find_root() -> "Path":
    current = Path.cwd()
    while current != current.parent:
        if (current / "artifacts.yaml").exists():
            return current.resolve()
        current = current.parent
    raise FileNotFoundError("artifacts.yaml not found — run from within the project tree.")

PROJECT_ROOT = _find_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Now src/ is importable — use the canonical implementation going forward.
from src.config import find_project_root  # noqa: E402 (import after sys.path setup)

from src.config import config
from src.gatekeeper import validate_dataframe
from src.preprocessing import build_apso_note, redact_icd10_sections, ICD10_REDACT_PATTERN

# --- Canonical HF sources ---
HF_REPO_ID      = "SidneyBishop/notes-to-icd10"
HF_MEDSYNTH_PATH = "data/medsynth/icd10_notes.parquet"
HF_CDC_PATH      = "data/reference/cdc_fy2026_icd10.parquet"

# SHA256 hashes of the canonical source files on HF Hub.
# These are the single source of truth — if either hash changes it means
# the upstream dataset has been modified, which would silently invalidate
# the gold layer and all trained models. The pipeline raises immediately
# rather than continuing on unexpected data.
EXPECTED_SHA256 = {
    "icd10_notes.parquet":      "7fa03f67b113b57a5f17349c712946553b4b186e1a11f39d74e0821d02fc5ac8",
    "cdc_fy2026_icd10.parquet": "2433adf954c3f49296a40761b83afb98c2d61cd78ca43f335fbdd4167e5fb93d",
}


def _sha256(path: Path) -> str:
    """Compute SHA256 of a file in streaming chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify_sha256(path: Path, key: str) -> None:
    """
    Assert that a file matches its canonical SHA256 hash.

    Parameters
    ----------
    path : Path
        File to verify.
    key : str
        Filename key in EXPECTED_SHA256 (e.g. "icd10_notes.parquet").

    Raises
    ------
    ValueError
        If the hash does not match. The error message instructs the user
        to delete the cached file and retry so the pipeline can re-download
        a clean copy.
    KeyError
        If key is not in EXPECTED_SHA256 (programming error, not user error).
    """
    expected = EXPECTED_SHA256[key]
    actual   = _sha256(path)
    if actual != expected:
        raise ValueError(
            f"\n❌ SHA256 mismatch for {path.name}"
            f"\n   expected : {expected}"
            f"\n   actual   : {actual}"
            f"\n   → Delete {path} and re-run to fetch a clean copy."
        )
    print(f"   🔒 SHA256 verified: {path.name}")

def phase_1a_ingest() -> pl.DataFrame:
    print("\n── Phase 1a: Ingestion ──────────────────────────────────────────────")
    raw_path = config.resolve_path("data", "medsynth") / "icd10_notes.parquet"
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    if not raw_path.exists():
        print(f" 📥 Downloading canonical MedSynth from HF...")
        hf_file = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=HF_MEDSYNTH_PATH,
            repo_type="dataset"
        )
        # Verify the downloaded file matches the canonical hash before
        # reading or caching — rejects corrupt or tampered downloads.
        _verify_sha256(Path(hf_file), "icd10_notes.parquet")
        df = pl.read_parquet(hf_file)
        # Ensure canonical schema
        if "ID" not in df.columns:
            df = df.with_columns(pl.int_range(0, len(df)).cast(pl.Utf8).alias("ID"))
        if df.schema["ICD10"] == pl.Utf8:
            df = df.with_columns(
                pl.col("ICD10").map_elements(
                    lambda x: [x.strip().upper()] if x else [],
                    return_dtype=pl.List(pl.Utf8)
                ).alias("ICD10")
            )
        df = df.select(["ID", "Note", "Dialogue", "ICD10"])
        df.write_parquet(raw_path, compression="zstd")
        print(f" ✅ Cached {len(df):,} records")
    else:
        _verify_sha256(raw_path, "icd10_notes.parquet")
        df = pl.read_parquet(raw_path)
        print(f" ✅ Loaded {len(df):,} records from cache")

    return df.with_columns([
        pl.col("ID").cast(pl.Utf8),
        pl.col("Dialogue").fill_null("[NO_TRANSCRIPT_AVAILABLE]")
    ])

def _load_cdc(gold_dir: Path, offline=False):
    cache = gold_dir / "cdc_fy2026_icd10.parquet"
    cache.parent.mkdir(parents=True, exist_ok=True)

    if cache.exists():
        _verify_sha256(cache, "cdc_fy2026_icd10.parquet")
        return pl.read_parquet(cache)

    if offline:
        raise FileNotFoundError("CDC cache missing and offline=True")

    print(" 📥 Downloading CDC FY2026 from HF...")
    hf_file = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=HF_CDC_PATH,
        repo_type="dataset"
    )
    # Verify before caching — rejects corrupt or tampered downloads.
    _verify_sha256(Path(hf_file), "cdc_fy2026_icd10.parquet")
    df = pl.read_parquet(hf_file)
    df.write_parquet(cache, compression="zstd")
    print(f" ✅ CDC FY2026: {len(df):,} billable codes")
    return df

def phase_1b(df_raw, gold_dir, offline=False):
    print("\n── Phase 1b: CDC Validation ─────────────────────────────────")
    cdc = _load_cdc(gold_dir, offline)
    codes = set(cdc["code_no_decimal"].to_list())

    exploded = df_raw.select([
        "ID",
        pl.col("ICD10").list.explode().alias("raw_code")
    ]).filter(pl.col("raw_code").is_not_null())

    def cls(c):
        u = c.strip().upper().replace(".", "")
        if u in codes:
            return "billable"
        if len(u) == 3 and any(x.startswith(u) for x in codes):
            return "non_billable_parent (Noisy 111)"
        if re.match(r"^[A-Z][0-9]{2}[0-9A-Z]*X[0-9A-Z]*$", u):
            return "placeholder_x"
        return "invalid_or_malformed"

    checked = exploded.with_columns(
        pl.Series("status", [cls(c) for c in exploded["raw_code"]])
    )

    for r in checked.group_by("status").agg(pl.len().alias("c")).sort("c", descending=True).iter_rows():
        print(f" {r[0]:<40} {r[1]:>6,}")

    return checked, cdc

def phase_1e(df_raw):
    print("\n── Phase 1e: Pydantic Firewall ──────────────────────────────────────")
    valid, _, _ = validate_dataframe(df_raw)
    print(f" ✅ Silver: {len(valid):,}")
    return valid

def phase_1g(df_valid):
    print("\n── Phase 1g: DuckDB ────────────────────────────────────")
    with config.duckdb_connection() as con:
        con.register("df", df_valid)
        con.execute("CREATE OR REPLACE TABLE silver_vault AS SELECT * FROM df")

def phase_2a(df_valid):
    print("\n── Phase 2a: Decimal Restoration ────────────────────────────────────")
    def rest(c):
        if not c:
            return c
        c = c.strip().upper()
        return c if "." in c else (c[:3] + "." + c[3:] if len(c) >= 4 else c)

    return df_valid.with_columns(
        pl.col("label").list.first().map_elements(rest, return_dtype=pl.Utf8).alias("standard_icd10")
    )

def phase_2c(df_gold, df_checked, cdc):
    print("\n── Phase 2c: Status Annotation ──────────────────────────────────────")
    m = df_checked.select(["raw_code", "status"]).unique("raw_code")

    def norm(s):
        s = s.lower()
        if "billable" in s and "non" not in s:
            return "billable"
        elif "noisy" in s or "parent" in s:
            return "noisy_111"
        elif "placeholder" in s:
            return "placeholder_x"
        else:
            return "invalid"

    lookup = dict(zip(m["raw_code"], [norm(s) for s in m["status"]]))

    return df_gold.with_columns(
        pl.col("standard_icd10").map_elements(
            lambda c: lookup.get(c.replace(".", "").upper(), "billable") if c else "invalid",
            return_dtype=pl.Utf8
        ).alias("code_status")
    )

def phase_3a(df):
    print("\n── Phase 3a: APSO-Flip ──────────────────────────────────────────────")
    df = build_apso_note(df)
    return df.with_columns(
        ((pl.col("apso_note").str.count_matches(r"\S+") * 1.3).cast(pl.Int64)).alias("apso_token_estimate")
    )

def phase_3b(df):
    return df.with_columns(
        pl.col("apso_note").str.contains(ICD10_REDACT_PATTERN).alias("has_leakage")
    )

def phase_3c(df):
    df = redact_icd10_sections(df)
    return df.drop("has_leakage") if "has_leakage" in df.columns else df

def phase_4(df, dry=False):
    print("\n── Phase 4: Export ───────────────────────────────────────")
    if dry:
        print(" (dry-run, skipping write)")
        return
    p = config.resolve_path("data", "gold") / f"medsynth_gold_apso_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
    p.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(p, compression="snappy")
    print(f" ✅ {p.name}")

def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--no-duckdb", action="store_true")
    pa.add_argument("--dry-run", action="store_true")
    pa.add_argument("--offline", action="store_true")
    a = pa.parse_args()

    print(f"\n{'='*70}\n prepare_data.py — HF-locked canonical\n{'='*70}")

    gold = config.resolve_path("data", "gold")
    raw = phase_1a_ingest()
    chk, cdc = phase_1b(raw, gold, a.offline)
    val = phase_1e(raw)

    if not a.no_duckdb:
        phase_1g(val)

    g = phase_2a(val)
    g = phase_2c(g, chk, cdc)
    g = phase_3a(g)
    g = phase_3b(g)
    g = phase_3c(g)
    phase_4(g, a.dry_run)

if __name__ == "__main__":
    main()