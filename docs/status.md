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

## Blocked

- **#0 (was top Next): re-run evaluate on the 971 split — BLOCKED on Q7.** The
  E-003 stage-1 model on disk is unloadable (weights split from config/tokenizer)
  and unbacked (gitignored, not in DVC). Hierarchical eval cannot run until
  stage-1 is retrained and backed up. Decided not to force a load (D004). So:
  971 is the current split regime, 966/83.9% is the last evaluated regime, and no
  971 number can be produced until Q7 is resolved.
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
