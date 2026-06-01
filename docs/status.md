# Status

Live "where are we" view. **State file — overwrite freely** as reality changes.

*Last updated: 2026-05-31*

## Done

- Documentation system established (`docs/` + this convention). See D003.
- Test-set discrepancy diagnosed and fixed: billable filter added to
  `prepare_splits.py` (filter-then-split), self-check corrected. See D001/D002.
- Canonical regime fixed: billable-only, seed 42, **test N = 971**, deterministic.
- Old eval regime (972/85.4%) preserved at branch `snapshot/2026-05-10-eval`
  (`9a73669`). Fix committed on `fix/splits-billable-filter` (`a990306`,`df3a4d4`).
- Fresh-clone reproducibility tested; gaps found and logged (Q2, Q3).
- **Gold REGENERATED from verified-canonical HF raw (D006):** raw SHA256s match
  pinned values; regenerated gold is clean (0 code leaks, 9,660 billable) and
  reproduces the May-10 regime deterministically (delta vs May-5 = exactly the
  redaction fix). 971 split regenerated identically. Foundation for the retrain
  is verified end-to-end.

## In progress

- Ground-up code re-read to rebuild an accurate mental model (drove
  `architecture.md`). Core path read: prepare_data, prepare_splits, train
  (structure), calibrate, evaluate, inference, preprocessing, config, paths,
  graph_reranker, adapters, gatekeeper. **Not yet deeply read:** the notebooks
  (01–05), `serve.py`, `augment.py`, `train_supcon_z.py` internals, MIMIC-IV
  validation scripts.

## Retrain progress (Q7 resolved — pipeline unblocked)

- **D007 fixed + Q7 RESOLVED.** train.py save-path bug fixed; stage-1 retrained,
  verified loadable (val 0.935, base-init). E-002 retrained clean at 30 epochs
  (val 0.841, converged/plateaued ep27; layout verified). Both write correct
  `model/` layout.
- **Stage-2 RUNNING** (--epochs 20, verified from notebook 05; warm-start from
  E-002; skips P/Q/U). Hyperparams: lr 2e-5, batch 16, warmup 0.1.
- **Next after stage-2:** gate-check chapter `model/` layouts → calibrate →
  evaluate on 971 split → PROVISIONAL number (D005). Expect somewhat below
  historical 83.9/85.8 due to base-init stage-1 (~3% router gap) + D005 leakage.

## Still blocked / deferred

- **Stage-1 init gap (open decision):** base-init (0.935) vs historical E-001-init
  (0.964). Accept and proceed, or retrain stage-1 from E-001 (needs a loadable
  E-001, likely has its own D007 split). Currently proceeding with base-init.
- Full fresh-clone reproduction — blocked on Q2 (no portable DVC remote) and Q3
  (orphaned/missing ontology artifact). The data can be assembled manually from
  the original working directory in the meantime.
- MIMIC-IV validation completeness — dependent on PhysioNet access (Q6).

## Next (suggested order)

0. **Retrain stage-1 cleanly and back it up (Q7, D004).** This is the new
   prerequisite for everything downstream — hierarchical eval is blocked until a
   loadable, backed-up stage-1 router exists. Retrain → verify single-directory
   layout loads → DVC-track the weights. Only then can the 971-regime evaluate run.

   **Sequence agreed 2026-05-31 (D005/D006):**
   a. ~~Re-track May-10 gold~~ → SUPERSEDED by D006: regenerate gold from HF raw. ✅ DONE.
   b. Regenerate the 971 splits from canonical gold. ✅ DONE (971 confirmed).
   c. **← NEXT: Retrain.** Stage-1 first as a gate (verify it writes `stage1/model/`
      in a loadable single-dir layout, resolving Q7), THEN stage-2 + calibrate.
   d. Evaluate on the 971 split → PROVISIONAL number (D005: code-only redaction,
      not publishable; reproducibility check vs historical 83.9/85.8/77.2%).
1. Reconcile dataset counts (Q1) — cheap, removes a visible inconsistency.
2. Freeze the canonical 971 split into version control / DVC so it can't drift.
3. Decide environment pinning (Q4) and run a single gated reproduction seed;
   confirm it lands in the documented 84–87% range before committing to more.
4. Decide the headline figure question (Q5).
5. Stand up a portable DVC remote (Q2); resolve the orphaned artifact (Q3).
6. Once a clean multi-seed run exists, propagate numbers from one generated
   source of truth into README / publication (kills the original drift problem).

## Branches in play

- `main` — pre-investigation baseline (original repo).
- `fix/splits-billable-filter` — the split fix + earlier decisions log (original
  repo). Not yet merged to main; not pushed to origin.
- `snapshot/2026-05-10-eval` (`9a73669`) — preserved old eval regime.
- `notes-to-icd10-fresh` — diagnostic fresh clone; partially pulled.
