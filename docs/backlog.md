# Backlog

Agreed-to-do-eventually, not in flight. `- [ ]` items; link to Q-IDs / D-IDs where
the rationale lives elsewhere. Promote to `status.md` "Next" when work starts.

## Reproducibility & data

- [ ] Stand up a portable DVC remote and push all tracked artifacts (Q2).
- [ ] Replace the `.dvc/CONFIGURE_LOCAL` placeholder with documented remote setup (Q2).
- [ ] Resolve orphaned `icd10cm_2026.parquet`: push or `dvc remove` (Q3).
- [ ] Reconcile dataset record/code counts in `03_dataset.qmd` (Q1).
- [ ] Freeze the canonical 971 split into version control / DVC (D002).
- [ ] Decide `uv.lock` pinning vs the paper's library basis (Q4).

## Results & reporting

- [ ] Run a single gated reproduction seed; confirm 84–87% before scaling up (Q4).
- [ ] Run the multi-seed reproduction (≥3 seeds) on the frozen split; report mean ± std.
- [ ] Decide headline: E-010 85.8% vs E-014 hybrid 86.7%; give both error bars or demote (Q5).
- [ ] Build a single generated source-of-truth results file; have README +
      publication read from it instead of hand-typed tables (kills original drift).
- [ ] Reconcile / rewrite `Our_paper.md` draft (still reports the old ~66.9%
      E-004a/E-005a generation) against current numbers.

## Code understanding (refactor read-through)

- [ ] Deep-read notebooks 01–05 and reconcile with the script pipeline.
- [ ] Document `serve.py` (FastAPI serving) — confirm whether current/used.
- [ ] Document `train_supcon_z.py` + `evaluate_hybrid.py` internals.
- [ ] Document the MIMIC-IV validation scripts (Q6).
- [ ] Confirm CUDA path (architecture.md notes it as unconfirmed/untested).

## Documentation hygiene

- [ ] Add root `README.md` aligned to the new docs convention (short, outward-
      facing, links into `docs/`).
- [ ] Decide whether earlier foundational decisions (hierarchical-over-flat,
      ClinicalBERT choice, 40-epoch warm start, DVC-over-git-LFS) get promoted to
      formal D-entries when re-affirmed during the refactor (D003). NOTE: promotion
      is earned only by a genuine re-examination — the contemporaneous act of
      revisiting (phrased "revisiting choice X, with new evidence Y") is what makes
      the D-entry honest. Do NOT back-date a D-entry to the original choice; that
      would reconstruct history and poison the file. If a foundational decision is
      never revisited, it stays here as background, not as a D-entry.

## Repo hygiene — outputs/ tracking (surfaced 2026-06-01)

- [ ] **Decide the `outputs/` git policy.** Currently messy/middle: `.safetensors`
      and split parquets are gitignored (correct), but small JSON artifacts under
      `outputs/evaluations/` ARE tracked (label_map.json, temperature.json,
      stage2_results.json, eval/*.json, calibration_report.json) — so every run
      shows dozens of "modified" files. Pick a clean state:
      (a) gitignore all of `outputs/`, keep results in MLflow/DVC; or
      (b) track only a curated `results/` snapshot (e.g. eval summary.json,
      chapter_accuracy.json), ignore the rest. Tie this to the "single
      results.json source of truth" item.
- [ ] **Clean up old-path label_maps (D007 residue).** git tracks
      `stage2/{chapter}/label_map.json` (pre-fix top-level location) while the
      corrected run writes `stage2/{chapter}/model/label_map.json`. The tracked
      copies are at the old split-layout path. Once the outputs/ policy is set,
      reconcile so tracked paths match the post-D007 layout (or are ignored).
- [ ] Note: today's run outputs were deliberately NOT committed (docs commit
      76f6dc0 stands alone). The 0.838 result is durably recorded in journal.md +
      decisions.md (D008), so the uncommitted output churn loses nothing.

## Pipeline README — scope boundary (surfaced 2026-06-01)

- [ ] **Create root `README.md`** from `docs/pipeline_readme_section.md` (Mermaid
      diagram + the 7 verified commands + gotchas). Ready to drop in; just needs
      placing + committing.
- [ ] **The pipeline diagram covers the CORE reproduction path ONLY** — the
      train→eval chain we ran and verified end to end this session:
      prepare_data → prepare_splits → train(flat/stage1/stage2) → calibrate →
      evaluate. Every argument in it is verified.
- [ ] **NOT yet in the diagram (need verification before drawing — don't guess):**
      - graph reranker fit/build step (it ran during evaluate but its own
        fit/train invocation wasn't traced this session; see Q9 sklearn drift)
      - hybrid / SupCon variants — E-014_SupCon_Z, E-010..._hybrid_Z-E-014
        (sibling experiment dirs; produce the 86.7 number behind Q5)
      - ModernBERT trials — E-012, E-013
      - MIMIC-IV validation scripts (Q6)
      - serve.py (FastAPI serving — confirm whether current/used)
      These are real entry points that exist in the repo but were NOT run or
      read deeply this session. Verify each (read argparse + trace I/O) BEFORE
      adding to the diagram. Until then the README should label its diagram
      "core reproduction pipeline," with these noted as "advanced / see docs."

## README reconciliation (surfaced 2026-06-01)

- [ ] **Decide whether the headline number gets a provisional caveat.** The top of
      README.md (Results table, Overview, "83.9% accuracy") still presents the
      number as clean/publishable. The verified 2026-06-01 run established it as
      provisional (D005 regime: code-only redaction, semantic labels retained →
      biased upward by leakage; not publishable until Q8). Editorial call: either
      mark the headline provisional, or leave it and rely on the in-section caveat.
- [ ] **Reconcile the two pipeline recipes in README.md into one canonical version.**
      The README now carries both the original script-pipeline block (stage-1 under
      E-003, 40-epoch E-002, build_graph.py + verify_scripts.py) and the verified
      2026-06-01 subsection (stage-1 under E-010, 30-epoch E-002, no separate
      graph-build step), with a reconciliation note between them. Before collapsing
      to one: verify build_graph.py and verify_scripts.py (read argparse + I/O —
      they were NOT traced this session), and decide which stage-1 experiment layout
      is canonical. Ties into the "which clone/regime is source of truth" question.
