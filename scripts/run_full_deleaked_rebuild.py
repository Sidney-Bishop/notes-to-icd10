#!/usr/bin/env python3
"""
scripts/run_full_deleaked_rebuild.py — full de-leaked rebuild orchestrator.

THIN wiring layer. All logic lives in src/orchestration.py (unit-tested in
tests/test_orchestration.py). This file only:
  - parses --dry-run
  - runs the verify_scripts.py codebase-integrity gate (abort on failure)
  - iterates the tested phase specs: resume-skip → preflight → run_step →
    postflight, logging a structured events.log
  - logs to ~/full_rebuild_runs/<timestamp>/ (outside the repo)

Usage:
    uv run python scripts/run_full_deleaked_rebuild.py --dry-run
    uv run python scripts/run_full_deleaked_rebuild.py
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import config
from src import orchestration as orch


def main() -> int:
    ap = argparse.ArgumentParser(description="Full de-leaked rebuild orchestrator")
    ap.add_argument("--dry-run", action="store_true",
                    help="Plan and log every step without executing training.")
    args = ap.parse_args()
    dry = args.dry_run

    root = config.project_root
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    rundir = Path.home() / "full_rebuild_runs" / ts
    rundir.mkdir(parents=True, exist_ok=True)
    events = rundir / "events.log"

    def ev(msg: str) -> None:
        line = orch.event_line(msg)
        print(line)
        with open(events, "a") as f:
            f.write(line + "\n")

    ev("=" * 60)
    ev(f"FULL DE-LEAKED REBUILD — run {ts}")
    ev(f"RUNDIR: {rundir}")
    ev(f"DRY: {dry}")

    # ── GUARD: verify_scripts.py ────────────────────────────────────────────
    branch = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                            cwd=root, capture_output=True, text=True).stdout.strip()
    ev(f"GUARD: branch={branch}")
    if dry:
        ev("GUARD: [DRY] would run verify_scripts.py and abort on failure")
    else:
        ev("GUARD: running verify_scripts.py (codebase integrity gate)")
        vlog = rundir / "00_verify_scripts.log"
        res = orch.run_step("verify_scripts", ["python3", "verify_scripts.py"],
                            vlog, root, dry_run=False)
        if res.returncode != 0:
            ev("GUARD: ✗ verify_scripts.py FAILED — aborting (fix codebase first)")
            return 1
        ev("GUARD: verify_scripts.py PASSED")

    # ── Phases ──────────────────────────────────────────────────────────────
    specs = orch.build_phase_specs(dry_run=dry)
    eval_base = config.resolve_path("outputs", "evaluations")
    any_fail = False

    for i, spec in enumerate(specs, 1):
        ev("-" * 60)
        ev(f"PHASE {i}/{len(specs)} — {spec.name}")

        # resume skip for flat pretrains
        if spec.is_flat_pretrain and orch.flat_done(eval_base, spec.flat_experiment):
            ev(f"  ⏭ {spec.flat_experiment} already complete — skipping")
            continue

        # dry-run skip for steps whose script lacks --dry-run
        if dry and not spec.supports_dry_run:
            ev(f"  [DRY] would run: {' '.join(spec.cmd)}")
            ev(f"  [DRY] (skipped — {spec.name} has no --dry-run flag)")
            continue

        # preflight
        ok, lines = orch.preflight([root / p for p in spec.inputs], dry_run=dry)
        for ln in lines:
            ev(ln)
        if not ok:
            ev(f"  ✗ aborting {spec.name} — inputs missing")
            any_fail = True
            break

        # execute
        ev(f"  COMMAND: {' '.join(spec.cmd)}")
        logfile = rundir / f"{i:02d}_{spec.name}.log"
        result = orch.run_step(spec.name, spec.cmd, logfile, root, dry_run=dry)
        ev(f"  STEP END {spec.name}  exit={result.returncode}  "
           f"elapsed={result.elapsed_s:.0f}s  status={result.status}  log={logfile}")

        if result.status == "FAILED":
            ev(f"  ✗ {spec.name} FAILED — stopping chain")
            any_fail = True
            break

        # postflight (live only)
        if not dry and spec.outputs:
            ok, lines = orch.postflight([root / p for p in spec.outputs])
            for ln in lines:
                ev(ln)
            if not ok:
                ev(f"  ✗ {spec.name} produced no artifacts — stopping chain")
                any_fail = True
                break

    ev("=" * 60)
    ev("RUN COMPLETE" if not any_fail else "RUN STOPPED (failure above)")
    ev(f"Events log:     {events}")
    ev(f"Per-phase logs: {rundir}/")
    ev("Quick checks:")
    ev(f"  failures →  grep -E 'FAILED|MISSING|exit=[^0]' {events}")
    ev("  results  →  cat outputs/experiment_results.csv")
    return 1 if any_fail else 0


if __name__ == "__main__":
    sys.exit(main())
