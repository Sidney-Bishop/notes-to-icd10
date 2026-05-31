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
      formal D-entries when re-affirmed during the refactor (D003).
