"""
verify_scripts.py — Pre-flight script verification

Run before every training session to confirm the codebase is in a known-good
state. Refactored so each check group is a callable function returning
(name, passed, detail) tuples — unit-testable in tests/test_verify_scripts.py —
while `python verify_scripts.py` behaves exactly as before (runs all checks,
prints, exits 0/1).

Usage:
    python3 verify_scripts.py
"""

from __future__ import annotations

import sys
import subprocess
import importlib.util
from pathlib import Path

ROOT = Path(__file__).parent
PASS = "✅"
FAIL = "❌"

# A check result: (name, passed, detail)
CheckResult = tuple[str, bool, str]


# ---------------------------------------------------------------------------
# Individual check groups — pure (read files, return results; no printing,
# no global state). Each takes the project root so tests can point at fixtures.
# ---------------------------------------------------------------------------

def check_required_files(root: Path) -> list[CheckResult]:
    scripts, src = root / "scripts", root / "src"
    required = [
        src / "experiment_logger.py", src / "paths.py", src / "inference.py",
        src / "gatekeeper.py", src / "preprocessing.py",
        scripts / "train.py", scripts / "calibrate.py", scripts / "evaluate.py",
        scripts / "prepare_splits.py", scripts / "prepare_data.py",
        scripts / "generate_manifest.py",
    ]
    return [(f.name, f.exists(), f"missing at {f}") for f in required]


def check_inference(root: Path) -> list[CheckResult]:
    text = (root / "src" / "inference.py").read_text()
    z_override_active = ("pred_chapter = 'Z'" in text or 'pred_chapter = "Z"' in text)
    multi_conv = "ExperimentPaths" in text and "stage2_model_dir" in text
    return [
        ("Z override removed", not z_override_active,
         "Z override still active — will corrupt all predictions"),
        ("Multi-convention stage2 loader present", multi_conv,
         "ExperimentPaths not used for stage2 path resolution in inference.py"),
    ]


def check_prepare_data(root: Path) -> list[CheckResult]:
    text = (root / "scripts" / "prepare_data.py").read_text()
    return [
        ("HF Hub used instead of CDC FTP",
         "hf_hub_download" in text and "ftp.cdc.gov" not in text,
         "Still using CDC FTP — should use hf_hub_download"),
        ("SHA256 constants defined", "EXPECTED_SHA256" in text,
         "EXPECTED_SHA256 dict missing"),
        ("SHA256 verification called", "_verify_sha256" in text,
         "_verify_sha256() not called"),
        ("Offline flag present", "--offline" in text,
         "No --offline flag"),
    ]


def check_generate_manifest(root: Path) -> list[CheckResult]:
    text = (root / "scripts" / "generate_manifest.py").read_text()
    return [
        ("Manifest captures git commit", "git_commit" in text, "git_commit not recorded"),
        ("Manifest captures SHA256", "sha256" in text, "SHA256 not recorded"),
        ("Manifest captures schema", "schema" in text, "schema not recorded"),
    ]


def check_train(root: Path) -> list[CheckResult]:
    text = (root / "scripts" / "train.py").read_text()
    return [
        ("ExperimentLogger imported in train.py",
         "from src.experiment_logger import ExperimentLogger" in text, ""),
        ("log_start called in train.py", "exp_logger.log_start" in text, ""),
        ("log_complete called in train.py", "exp_logger.log_complete" in text, ""),
        ("TrainingResult .get() bug fixed", "hasattr(obj, key)" in text,
         "Still using .get() on TrainingResult dataclass"),
        ("Gold path resolution fixed", "gold_path.is_absolute()" in text,
         "Gold path not resolved against PROJECT_ROOT"),
    ]


def check_calibrate(root: Path) -> list[CheckResult]:
    text = (root / "scripts" / "calibrate.py").read_text()
    return [
        ("ExperimentLogger in calibrate.py", "ExperimentLogger" in text, ""),
        ("log_start called in calibrate.py", "log_start" in text, ""),
    ]


def check_evaluate(root: Path) -> list[CheckResult]:
    text = (root / "scripts" / "evaluate.py").read_text()
    return [
        ("ExperimentLogger in evaluate.py", "ExperimentLogger" in text, ""),
        ("log_results called in evaluate.py", "log_results" in text, ""),
    ]


def check_prepare_splits(root: Path) -> list[CheckResult]:
    text = (root / "scripts" / "prepare_splits.py").read_text()
    return [("Gold path resolution fixed in prepare_splits.py",
             "is_absolute()" in text,
             "prepare_splits.py still uses raw relative gold path")]


def check_experiment_logger(root: Path) -> list[CheckResult]:
    text = (root / "src" / "experiment_logger.py").read_text()
    return [
        ("ExperimentLogger class present", "class ExperimentLogger" in text, ""),
        ("status() function present", "def status()" in text, ""),
        ("run.log path defined", "run.log" in text, ""),
        ("experiments.json path defined", "experiments.json" in text, ""),
    ]


def check_runtime_import(root: Path) -> list[CheckResult]:
    """
    Actually import src/experiment_logger.py and report whether it succeeds.

    (Previously this was dead code: `... if False else None` plus a hardcoded
    check(..., True), so it ALWAYS passed without importing anything. Now it
    genuinely loads the module.)
    """
    target = root / "src" / "experiment_logger.py"
    try:
        spec = importlib.util.spec_from_file_location("experiment_logger", target)
        if spec is None or spec.loader is None:
            return [("src.experiment_logger importable", False, "could not build import spec")]
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)   # real import — raises on failure
        return [("src.experiment_logger importable", True, "")]
    except Exception as e:  # noqa: BLE001 — we want to report any import failure
        return [("src.experiment_logger importable", False, str(e))]


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

ALL_CHECKS = [
    check_required_files, check_inference, check_prepare_data,
    check_generate_manifest, check_train,
    check_calibrate, check_evaluate, check_prepare_splits,
    check_experiment_logger, check_runtime_import,
]


def run_all_checks(root: Path) -> list[CheckResult]:
    results: list[CheckResult] = []
    for fn in ALL_CHECKS:
        results.extend(fn(root))
    return results


def clear_cache(root: Path) -> None:
    """Side-effecting: remove __pycache__ and *.pyc. Run only by the CLI."""
    subprocess.run(["find", ".", "-type", "d", "-name", "__pycache__",
                    "-exec", "rm", "-rf", "{}", "+"], capture_output=True, cwd=root)
    subprocess.run(["find", ".", "-name", "*.pyc", "-delete"],
                   capture_output=True, cwd=root)


def main() -> int:
    print("\n" + "=" * 60)
    print(" Pre-flight Verification")
    print("=" * 60)

    print("\n[1] Clearing Python bytecode cache...")
    clear_cache(ROOT)
    print(f"  {PASS} Cache cleared")

    results = run_all_checks(ROOT)
    failed = []
    for name, passed, detail in results:
        if passed:
            print(f"  {PASS} {name}")
        else:
            print(f"  {FAIL} {name}" + (f" — {detail}" if detail else ""))
            failed.append(name)

    print("\n" + "=" * 60)
    if failed:
        print(f" {FAIL} {len(failed)} check(s) FAILED:")
        for n in failed:
            print(f"    — {n}")
        print("\n  Fix these before running any training commands.")
        print("=" * 60 + "\n")
        return 1
    print(f" {PASS} All checks passed — safe to run training")
    print("=" * 60 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
