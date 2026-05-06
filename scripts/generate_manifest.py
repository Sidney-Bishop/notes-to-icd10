"""
generate_manifest.py — Phase 4 export manifest with SHA256 hashes
Locked to canonical HF datasets: SidneyBishop/notes-to-icd10

Usage
-----
# Generate manifest for an existing gold file (no pipeline re-run):
    python scripts/generate_manifest.py --gold-path data/gold/medsynth_gold_apso_20260505_194721.parquet

# Run the full pipeline first, then generate manifest:
    python scripts/generate_manifest.py

# Dry run — print manifest without writing to disk:
    python scripts/generate_manifest.py --gold-path <path> --dry-run
"""
import argparse
from pathlib import Path
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone

import polars as pl

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

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    pa = argparse.ArgumentParser(
        description="Generate a SHA256-anchored provenance manifest for a gold parquet."
    )
    pa.add_argument(
        "--gold-path",
        type=Path,
        default=None,
        help=(
            "Path to an existing gold parquet. If omitted, runs the full "
            "prepare_data.py pipeline first and picks the latest output."
        ),
    )
    pa.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the manifest to stdout without writing it to disk.",
    )
    args = pa.parse_args()

    gold_dir = config.resolve_path("data", "gold")

    if args.gold_path is not None:
        # Use the provided path — no pipeline re-run needed.
        latest = args.gold_path.resolve()
        if not latest.exists():
            print(f"❌ Gold file not found: {latest}")
            sys.exit(1)
        print(f"\n── Using existing gold file ────────────────────────────────────")
        print(f"   {latest.name}")
    else:
        # No path given — run the full pipeline and pick the latest output.
        print("\n── Running full prepare_data.py pipeline ───────────────────────")
        subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "prepare_data.py")],
            check=True
        )
        candidates = sorted(gold_dir.glob("medsynth_gold_apso_*.parquet"))
        if not candidates:
            print("❌ No gold parquet found after pipeline run.")
            sys.exit(1)
        latest = candidates[-1]

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

    manifest_json = json.dumps(manifest, indent=2)
    print(f"\n{manifest_json}")

    if args.dry_run:
        print("\n⏭  Dry run — manifest not written to disk.")
    else:
        out = gold_dir / f"MANIFEST_{latest.stem}.json"
        out.write_text(manifest_json)
        print(f"\n✅ Manifest written: {out}")

if __name__ == "__main__":
    main()
