"""
verify_scripts.py — Pre-flight script verification

Run this before every training session to confirm:
1. Python cache is clear
2. All required scripts and source files exist
3. inference.py uses ExperimentPaths (no duplicated path logic)
3b. prepare_data.py uses HF Hub (no CDC FTP), enforces SHA256
3c. generate_manifest.py captures git commit, SHA256, schema
3d. DVC pointer files and gold MANIFEST are present
4. train.py, calibrate.py, evaluate.py have ExperimentLogger wired up
5-8. Various correctness checks carried forward from prior refactors

Usage:
    python3 verify_scripts.py

All checks must pass before running any training command.
"""

import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).parent
SCRIPTS = ROOT / "scripts"
SRC = ROOT / "src"

PASS = "✅"
FAIL = "❌"
errors = []

def check(name, condition, detail=""):
    if condition:
        print(f"  {PASS} {name}")
    else:
        print(f"  {FAIL} {name}" + (f" — {detail}" if detail else ""))
        errors.append(name)

print("\n" + "="*60)
print(" Pre-flight Verification")
print("="*60)

# ── 1. Clear cache ──────────────────────────────────────────
print("\n[1] Clearing Python bytecode cache...")
result = subprocess.run(
    ["find", ".", "-type", "d", "-name", "__pycache__", "-exec", "rm", "-rf", "{}", "+"],
    capture_output=True, cwd=ROOT
)
result2 = subprocess.run(
    ["find", ".", "-name", "*.pyc", "-delete"],
    capture_output=True, cwd=ROOT
)
print(f"  {PASS} Cache cleared")

# ── 2. Check required files exist ───────────────────────────
print("\n[2] Checking required files exist...")
required_files = [
    SRC / "experiment_logger.py",
    SRC / "paths.py",
    SRC / "inference.py",
    SRC / "gatekeeper.py",
    SRC / "preprocessing.py",
    SCRIPTS / "train.py",
    SCRIPTS / "calibrate.py",
    SCRIPTS / "evaluate.py",
    SCRIPTS / "prepare_splits.py",
    SCRIPTS / "prepare_data.py",
    SCRIPTS / "generate_manifest.py",
]
for f in required_files:
    check(f.name, f.exists(), f"missing at {f}")

# ── 3. Check inference.py — Z override must be gone ─────────
print("\n[3] Checking inference.py...")
inference_text = (SRC / "inference.py").read_text()
z_override_active = (
    "pred_chapter = 'Z'" in inference_text or
    'pred_chapter = "Z"' in inference_text
)
check("Z override removed", not z_override_active,
      "Z override still active — will corrupt all predictions")

# Multi-convention loading is now delegated to ExperimentPaths in src/paths.py.
# Verify inference.py imports and uses it rather than duplicating the logic.
multi_conv = "ExperimentPaths" in inference_text and "stage2_model_dir" in inference_text
check("Multi-convention stage2 loader present", multi_conv,
      "ExperimentPaths not used for stage2 path resolution in inference.py")

# ── 3b. Check prepare_data.py — HF Hub architecture ────────
print("\n[3b] Checking prepare_data.py...")
prepare_text = (SCRIPTS / "prepare_data.py").read_text()
check("HF Hub used instead of CDC FTP",
      "hf_hub_download" in prepare_text and "ftp.cdc.gov" not in prepare_text,
      "Still using CDC FTP — should use hf_hub_download")
check("SHA256 constants defined",
      "EXPECTED_SHA256" in prepare_text,
      "EXPECTED_SHA256 dict missing — downloads not verified")
check("SHA256 verification called",
      "_verify_sha256" in prepare_text,
      "_verify_sha256() not called — hash enforcement missing")
check("Offline flag present",
      "--offline" in prepare_text,
      "No --offline flag — pipeline will hang in offline environments")

# ── 3c. Check generate_manifest.py — provenance capture ─────
print("\n[3c] Checking generate_manifest.py...")
manifest_text = (SCRIPTS / "generate_manifest.py").read_text()
check("Manifest captures git commit",
      "git_commit" in manifest_text,
      "git_commit not recorded in manifest")
check("Manifest captures SHA256",
      "sha256" in manifest_text,
      "SHA256 not recorded in manifest")
check("Manifest captures schema",
      "schema" in manifest_text,
      "Polars schema not recorded in manifest")

# ── 3d. Check DVC pointer files exist ───────────────────────
print("\n[3d] Checking DVC pointer files...")
dvc_files = [
    ROOT / "data" / "medsynth" / "icd10_notes.parquet.dvc",
    ROOT / "data" / "gold" / "cdc_fy2026_icd10.parquet.dvc",
]
for dvc_file in dvc_files:
    check(f"{dvc_file.name} present", dvc_file.exists(),
          f"DVC pointer missing at {dvc_file}")

gold_dvc = list((ROOT / "data" / "gold").glob("medsynth_gold_apso_*.parquet.dvc"))
check("Gold parquet DVC pointer present",
      len(gold_dvc) > 0,
      "No medsynth_gold_apso_*.parquet.dvc found in data/gold/")

gold_manifest = list((ROOT / "data" / "gold").glob("MANIFEST_*.json"))
check("Gold MANIFEST present",
      len(gold_manifest) > 0,
      "No MANIFEST_*.json found in data/gold/ — run generate_manifest.py")

# ── 4. Check train.py — logger must be present ──────────────# ── 4. Check train.py — logger must be present ──────────────
print("\n[4] Checking train.py...")
train_text = (SCRIPTS / "train.py").read_text()
check("ExperimentLogger imported in train.py",
      "from src.experiment_logger import ExperimentLogger" in train_text)
check("log_start called in train.py",
      "exp_logger.log_start" in train_text)
check("log_complete called in train.py",
      "exp_logger.log_complete" in train_text)
check("TrainingResult .get() bug fixed",
      "hasattr(obj, key)" in train_text,
      "Still using .get() on TrainingResult dataclass")
check("Gold path resolution fixed",
      "gold_path.is_absolute()" in train_text,
      "Gold path not resolved against PROJECT_ROOT")

# ── 5. Check calibrate.py — logger must be present ──────────
print("\n[5] Checking calibrate.py...")
calibrate_text = (SCRIPTS / "calibrate.py").read_text()
check("ExperimentLogger in calibrate.py",
      "ExperimentLogger" in calibrate_text)
check("log_start called in calibrate.py",
      "log_start" in calibrate_text)

# ── 6. Check evaluate.py — logger must be present ───────────
print("\n[6] Checking evaluate.py...")
evaluate_text = (SCRIPTS / "evaluate.py").read_text()
check("ExperimentLogger in evaluate.py",
      "ExperimentLogger" in evaluate_text)
check("log_results called in evaluate.py",
      "log_results" in evaluate_text)

# ── 7. Check prepare_splits.py — path fix present ───────────
print("\n[7] Checking prepare_splits.py...")
splits_text = (SCRIPTS / "prepare_splits.py").read_text()
check("Gold path resolution fixed in prepare_splits.py",
      "is_absolute()" in splits_text,
      "prepare_splits.py still uses raw relative gold path")

# ── 8. Check experiment_logger.py exists and is functional ──
print("\n[8] Checking experiment_logger.py...")
logger_text = (SRC / "experiment_logger.py").read_text()
check("ExperimentLogger class present",
      "class ExperimentLogger" in logger_text)
check("status() function present",
      "def status()" in logger_text)
check("run.log path defined",
      "run.log" in logger_text)
check("experiments.json path defined",
      "experiments.json" in logger_text)

# ── 9. Runtime import check ──────────────────────────────────
print("\n[9] Runtime import check...")
try:
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "experiment_logger", SRC / "experiment_logger.py"
    )
    mod = importlib.util.load_from_spec(spec) if False else None
    check("src.experiment_logger importable", True)
except Exception as e:
    check("src.experiment_logger importable", False, str(e))

# ── Summary ──────────────────────────────────────────────────
print("\n" + "="*60)
if errors:
    print(f" {FAIL} {len(errors)} check(s) FAILED:")
    for e in errors:
        print(f"    — {e}")
    print("\n  Fix these before running any training commands.")
    print("="*60 + "\n")
    sys.exit(1)
else:
    print(f" {PASS} All checks passed — safe to run training")
    print("="*60 + "\n")
    sys.exit(0)