"""
prepare_data.py — Headless Gold layer preparation pipeline.
"""
import re, io, sys, zipfile, argparse, requests, polars as pl
from pathlib import Path
from datetime import datetime

def _bootstrap_project_root() -> Path:
    current = Path.cwd()
    while current!= current.parent:
        if (current / "artifacts.yaml").exists():
            return current.resolve()
        current = current.parent
    raise FileNotFoundError("artifacts.yaml not found")
PROJECT_ROOT = _bootstrap_project_root()
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))

from src.config import config
from src.gatekeeper import validate_dataframe
from src.preprocessing import build_apso_note, redact_icd10_sections, ICD10_REDACT_PATTERN

def phase_1a_ingest() -> pl.DataFrame:
    print("\n── Phase 1a: Ingestion ──────────────────────────────────────────────")
    raw_path = config.resolve_path("data", "medsynth") / "icd10_notes.parquet"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    if not raw_path.exists():
        print(f" ⚠️ Building canonical file...")
        from src.data_loader import load_medsynth_raw
        df = load_medsynth_raw()
        if "ID" not in df.columns:
            df = df.with_columns(pl.int_range(0, len(df)).cast(pl.Utf8).alias("ID"))
        if df.schema["ICD10"] == pl.Utf8:
            df = df.with_columns(
                pl.col("ICD10").map_elements(
                    lambda x: [x.strip().upper()] if x else [],
                    return_dtype=pl.List(pl.Utf8)
                ).alias("ICD10")
            )
        df = df.select(["ID","Note","Dialogue","ICD10"])
        df.write_parquet(raw_path, compression="zstd")
        print(f" ✅ Created {len(df):,} records")
    else:
        df = pl.read_parquet(raw_path)
        print(f" ✅ Loaded {len(df):,} records")
    return df.with_columns([
        pl.col("ID").cast(pl.Utf8),
        pl.col("Dialogue").fill_null("[NO_TRANSCRIPT_AVAILABLE]")
    ])

_CDC_URL = "https://ftp.cdc.gov/pub/health_statistics/nchs/publications/ICD10CM/2026/icd10cm-Code%20Descriptions-2026.zip"
def _load_cdc(gold_dir: Path, offline=False):
    cache = gold_dir / "cdc_fy2026_icd10.parquet"
    if cache.exists(): return pl.read_parquet(cache)
    if offline: raise FileNotFoundError("CDC cache missing")
    print(" 📥 Downloading CDC FY2026...")
    r = requests.get(_CDC_URL, timeout=60); r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        target = next(f for f in z.namelist() if "codes" in f.lower() and f.endswith(".txt"))
        lines = [l.decode("utf-8", errors="ignore").strip() for l in z.open(target)]
    df = pl.DataFrame({"raw": lines}).filter(pl.col("raw").str.len_chars()>0)
    df = df.with_columns([
        pl.col("raw").str.extract(r"^\s*(\S+)",1).str.replace(".","",literal=True).str.to_uppercase().alias("code_no_decimal"),
        pl.col("raw").str.extract(r"^\s*\S+\s+(.+)$",1).alias("description")
    ]).select(["code_no_decimal","description"]).unique()
    df.write_parquet(cache, compression="zstd")
    return df

def phase_1b(df_raw, gold_dir, offline=False):
    print("\n── Phase 1b: CDC Validation ─────────────────────────────────")
    cdc = _load_cdc(gold_dir, offline)
    codes = set(cdc["code_no_decimal"].to_list())
    exploded = df_raw.select(["ID", pl.col("ICD10").list.explode().alias("raw_code")]).filter(pl.col("raw_code").is_not_null())
    def cls(c):
        u = c.strip().upper().replace(".","")
        if u in codes: return "billable"
        if len(u)==3 and any(x.startswith(u) for x in codes): return "non_billable_parent (Noisy 111)"
        if re.match(r"^[A-Z][0-9]{2}[0-9A-Z]*X[0-9A-Z]*$", u): return "placeholder_x"
        return "invalid_or_malformed"
    checked = exploded.with_columns(pl.Series("status", [cls(c) for c in exploded["raw_code"]]))
    for r in checked.group_by("status").agg(pl.len().alias("c")).sort("c",descending=True).iter_rows():
        print(f" {r[0]:<40} {r[1]:>6,}")
    return checked, cdc

def phase_1e(df_raw):
    print("\n── Phase 1e: Pydantic Firewall ──────────────────────────────────────")
    valid,_,_ = validate_dataframe(df_raw)
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
        if not c: return c
        c=c.strip().upper()
        return c if "." in c else (c[:3]+"."+c[3:] if len(c)>=4 else c)
    return df_valid.with_columns(pl.col("label").list.first().map_elements(rest, return_dtype=pl.Utf8).alias("standard_icd10"))

def phase_2c(df_gold, df_checked, cdc):
    print("\n── Phase 2c: Status Annotation ──────────────────────────────────────")
    m = df_checked.select(["raw_code","status"]).unique("raw_code")
    def norm(s):
        s=s.lower()
        return "billable" if "billable" in s and "non" not in s else ("noisy_111" if "noisy" in s or "parent" in s else ("placeholder_x" if "placeholder" in s else "invalid"))
    lookup = dict(zip(m["raw_code"], [norm(s) for s in m["status"]]))
    return df_gold.with_columns(pl.col("standard_icd10").map_elements(lambda c: lookup.get(c.replace(".","").upper(),"billable") if c else "invalid", return_dtype=pl.Utf8).alias("code_status"))

def phase_3a(df):
    print("\n── Phase 3a: APSO-Flip ──────────────────────────────────────────────")
    df = build_apso_note(df)
    return df.with_columns(((pl.col("apso_note").str.count_matches(r"\S+")*1.3).cast(pl.Int64)).alias("apso_token_estimate"))

def phase_3b(df): return df.with_columns(pl.col("apso_note").str.contains(ICD10_REDACT_PATTERN).alias("has_leakage"))
def phase_3c(df): df = redact_icd10_sections(df); return df.drop("has_leakage") if "has_leakage" in df.columns else df

def phase_4(df, dry=False):
    print("\n── Phase 4: Export ───────────────────────────────────────")
    if dry: return
    p = config.resolve_path("data","gold") / f"medsynth_gold_apso_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
    df.write_parquet(p, compression="snappy")
    print(f" ✅ {p.name}")

def main():
    import argparse
    pa=argparse.ArgumentParser(); pa.add_argument("--no-duckdb",action="store_true"); pa.add_argument("--dry-run",action="store_true"); pa.add_argument("--offline",action="store_true"); a=pa.parse_args()
    print(f"\n{'='*70}\n prepare_data.py\n{'='*70}")
    gold=config.resolve_path("data","gold")
    raw=phase_1a_ingest()
    chk,cdc=phase_1b(raw,gold,a.offline)
    val=phase_1e(raw)
    if not a.no_duckdb: phase_1g(val)
    g=phase_2a(val); g=phase_2c(g,chk,cdc); g=phase_3a(g); g=phase_3b(g); g=phase_3c(g); phase_4(g,a.dry_run)

if __name__=="__main__": main()