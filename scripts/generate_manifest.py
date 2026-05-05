"""
generate_manifest.py — Phase 4 export manifest with SHA256 hashes
Locked to canonical HF datasets: SidneyBishop/notes-to-icd10
"""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone

import polars as pl

def _bootstrap_project_root() -> Path:
    current = Path.cwd()
    while current!= current.parent:
        if (current / "artifacts.yaml").exists():
            return current.resolve()
        current = current.parent
    raise FileNotFoundError("artifacts.yaml not found")

PROJECT_ROOT = _bootstrap_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import config

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    print("Running full prepare_data.py...")
    subprocess.run([sys.executable, "scripts/prepare_data.py"], check=True)

    gold_dir = config.resolve_path("data", "gold")
    latest = sorted(gold_dir.glob("medsynth_gold_apso_*.parquet"))[-1]
    print(f"\nLatest gold: {latest.name}")

    df = pl.read_parquet(latest)

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip(),
        "hf_repo": "SidneyBishop/notes-to-icd10",
        "files": {
            "gold_parquet": {
                "path": str(latest.relative_to(PROJECT_ROOT)),
                "sha256": sha256(latest),
                "rows": len(df),
                "size_bytes": latest.stat().st_size,
            }
        },
        "validation_split": {
            row[0]: int(row[1]) for row in df.group_by("code_status").agg(pl.len()).sort("code_status").iter_rows()
        },
        "schema": {k: str(v) for k, v in df.schema.items()}
    }

    for name, path in [
        ("medsynth", config.resolve_path("data", "medsynth") / "icd10_notes.parquet"),
        ("cdc_fy2026", config.resolve_path("data", "gold") / "cdc_fy2026_icd10.parquet")
    ]:
        if path.exists():
            manifest["files"][name] = {
                "path": str(path.relative_to(PROJECT_ROOT)),
                "sha256": sha256(path),
                "rows": len(pl.read_parquet(path)),
                "size_bytes": path.stat().st_size,
            }

    out = gold_dir / f"MANIFEST_{latest.stem}.json"
    out.write_text(json.dumps(manifest, indent=2))
    print(f"\n✅ Manifest written: {out}")
    print(json.dumps(manifest, indent=2))

if __name__ == "__main__":
    main()
