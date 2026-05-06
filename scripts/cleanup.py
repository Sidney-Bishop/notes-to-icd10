"""
scripts/cleanup.py — Experiment storage cleanup utility

Removes training checkpoints from completed experiments, keeping only
the final model weights needed for inference and evaluation.

The HuggingFace Trainer saves intermediate checkpoints during training
for resume capability. Once training completes successfully, these are
redundant — each checkpoint is ~3.6GB per resolver, and a full hierarchical
run produces 20 Stage-2 resolvers + 1 Stage-1 = ~75GB of checkpoints per
experiment.

What is KEPT:
  - Final model weights (model.safetensors, config.json, tokenizer files)
  - Label maps (label_map.json)
  - Temperature calibration (temperature.json)
  - Evaluation results (e2e_results.json, stage2_results.json, eval/)
  - Registry metadata (registry/)

What is DELETED:
  - checkpoint-N/ directories inside any model directory
  - Top-level checkpoints/ directories inside experiment directories

Usage
-----
    # Preview what would be deleted (safe — no changes made)
    uv run python scripts/cleanup.py --dry-run

    # Clean all experiments
    uv run python scripts/cleanup.py

    # Clean a specific experiment only
    uv run python scripts/cleanup.py --experiment E-003_Hierarchical_ICD10

    # Clean all experiments except the current best
    uv run python scripts/cleanup.py --keep E-010_40ep_E002Init
"""

import argparse
import shutil
import sys
from pathlib import Path

# Bootstrap: locate project root before src/ is on sys.path
def _find_root() -> Path:
    current = Path.cwd()
    while current != current.parent:
        if (current / "artifacts.yaml").exists():
            return current.resolve()
        current = current.parent
    raise FileNotFoundError("artifacts.yaml not found — run from within the project tree.")

PROJECT_ROOT = _find_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import config


def _find_checkpoint_dirs(experiment_dir: Path) -> list[Path]:
    """
    Find all checkpoint directories within an experiment directory.

    Matches both patterns:
      - checkpoint-N/   (inside model dirs, created by _finalize_model_dir)
      - checkpoints/    (top-level, created by older training runs)
    """
    checkpoints = []

    # Pattern 1: checkpoint-N/ anywhere in the tree
    for p in experiment_dir.rglob("checkpoint-*"):
        if p.is_dir():
            checkpoints.append(p)

    # Pattern 2: top-level checkpoints/ directories
    for p in experiment_dir.rglob("checkpoints"):
        if p.is_dir() and p not in checkpoints:
            checkpoints.append(p)

    # Sort shallowest first — parents deleted first, children skipped via exists() check
    return sorted(checkpoints, key=lambda p: len(p.parts))


def _format_size(size_bytes: int) -> str:
    """Format bytes as human-readable string."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def _dir_size(path: Path) -> int:
    """Compute total size of a directory in bytes."""
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def cleanup(
    experiments: list[Path],
    dry_run: bool = True,
    verbose: bool = True,
) -> tuple[int, int]:
    """
    Remove checkpoint directories from a list of experiment paths.

    Parameters
    ----------
    experiments : list[Path]
        Experiment directories to clean.
    dry_run : bool
        If True, print what would be deleted without deleting anything.
    verbose : bool
        If True, print per-checkpoint details.

    Returns
    -------
    (total_dirs, total_bytes) — number of directories and bytes freed/to free.
    """
    total_dirs  = 0
    total_bytes = 0

    for exp_dir in experiments:
        checkpoints = _find_checkpoint_dirs(exp_dir)
        if not checkpoints:
            if verbose:
                print(f"  ✅ {exp_dir.name} — already clean")
            continue

        exp_bytes = sum(_dir_size(c) for c in checkpoints)
        total_bytes += exp_bytes
        total_dirs  += len(checkpoints)

        print(f"\n  📁 {exp_dir.name}  ({len(checkpoints)} checkpoint dirs, "
              f"{_format_size(exp_bytes)})")

        for ckpt in checkpoints:
            size = _dir_size(ckpt)
            rel  = ckpt.relative_to(exp_dir)
            if verbose:
                print(f"     {'[DRY RUN] would delete' if dry_run else 'deleting'}: "
                      f"{rel}  ({_format_size(size)})")
            if not dry_run:
                if ckpt.exists():
                    shutil.rmtree(ckpt)
                else:
                    if verbose:
                        print(f"     (already removed as part of parent)")

    return total_dirs, total_bytes


def main():
    pa = argparse.ArgumentParser(
        description="Remove training checkpoints from completed experiments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    pa.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Preview what would be deleted without making any changes (safe).",
    )
    pa.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="Clean a specific experiment only (e.g. E-003_Hierarchical_ICD10).",
    )
    pa.add_argument(
        "--keep",
        type=str,
        default=None,
        help="Skip this experiment (e.g. --keep E-010_40ep_E002Init).",
    )
    pa.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-checkpoint details.",
    )
    args = pa.parse_args()

    eval_dir = config.resolve_path("outputs", "evaluations")

    print("=" * 60)
    print(f" Experiment Checkpoint Cleanup")
    print(f" {'DRY RUN — no files will be deleted' if args.dry_run else '⚠️  LIVE RUN — files will be permanently deleted'}")
    print("=" * 60)

    # Collect experiment directories to process
    if args.experiment:
        target = eval_dir / args.experiment
        if not target.exists():
            print(f"❌ Experiment not found: {target}")
            sys.exit(1)
        experiments = [target]
    else:
        experiments = sorted(
            p for p in eval_dir.iterdir()
            if p.is_dir() and p.name != "registry"
        )

    # Apply --keep filter
    if args.keep:
        experiments = [e for e in experiments if e.name != args.keep]
        print(f"\n  Skipping: {args.keep}")

    print(f"\n  Scanning {len(experiments)} experiment(s)...\n")

    total_dirs, total_bytes = cleanup(
        experiments,
        dry_run=args.dry_run,
        verbose=not args.quiet,
    )

    print("\n" + "=" * 60)
    if total_dirs == 0:
        print("  ✅ Nothing to clean — all experiments are already tidy.")
    elif args.dry_run:
        print(f"  DRY RUN: would delete {total_dirs} checkpoint directories")
        print(f"           freeing {_format_size(total_bytes)}")
        print(f"\n  Run without --dry-run to actually delete.")
    else:
        print(f"  ✅ Deleted {total_dirs} checkpoint directories")
        print(f"     Freed {_format_size(total_bytes)}")
    print("=" * 60)


if __name__ == "__main__":
    main()