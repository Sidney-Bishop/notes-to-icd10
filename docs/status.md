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
- Fresh-clone reproducibility tested; two infra gaps found and logged (Q2, Q3).

## In progress

- Ground-up code re-read to rebuild an accurate mental model (drove
  `architecture.md`). Core path read: prepare_data, prepare_splits, train
  (structure), calibrate, evaluate, inference, preprocessing, config, paths,
  graph_reranker, adapters, gatekeeper. **Not yet deeply read:** the notebooks
  (01–05), `serve.py`, `augment.py`, `train_supcon_z.py` internals, MIMIC-IV
  validation scripts.

## Blocked

- Full fresh-clone reproduction — blocked on Q2 (no portable DVC remote) and Q3
  (orphaned/missing ontology artifact). The data can be assembled manually from
  the original working directory in the meantime.
- MIMIC-IV validation completeness — dependent on PhysioNet access (Q6).

## Next (suggested order)

0. **Reconcile split vs eval.** The 971 split exists but `evaluate.py` has not run
   on it; on-disk E-010 `summary.json` is still the 966/83.9% historical run. Re-run
   evaluate on the 971 split so a 971-regime accuracy number exists. Until then, do
   NOT quote 83.9% as belonging to the 971 regime (see architecture.md, journal).
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
