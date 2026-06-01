# Decisions

Append-only. Monotonic IDs (D001, D002, …). Never edit a past entry; supersede it
with a new ID that references it. Shape: Context / Decision / Rationale /
Trade-offs / When to revisit.

> These entries begin on 2026-05-31, the day the documentation system was
> established. Decisions made earlier in the project's life are referenced as
> background within these entries but are NOT logged as if recorded at the time —
> doing so would fabricate a record. See `philosophy.md`.

---

## D001 — Apply the billable filter inside `prepare_splits.py` (filter-then-split)

**Date:** 2026-05-31

**Context.** Reported test-set size disagreed across artifacts: 966 (committed
eval), 972 (a later working-tree eval), and 1,030 (a fresh `prepare_splits.py`
run). Root cause: the billable filter (`code_status == "billable"`) lived only in
`train.py`, not in `prepare_splits.py`, so the test set depended on which stage
filtered. The split script's own self-check validated against the *unfiltered*
population and silently passed at 1,030.

**Decision.** Add a `--code-filter {all,billable}` argument to
`prepare_splits.py`, default `billable`, applied *before* the per-chapter split
(filter-then-split). Fix the self-check to validate against the filtered count.

**Rationale.** Filter-then-split yields an honest stratified 10% of the 9,660
billable records — the regime every reported experiment uses (per `train.py` and
the dataset docs). It keeps the split and training stages in lockstep so the
discrepancy cannot recur.

**Trade-offs.** Not byte-identical to the historical committed run. Verified test
count is **971** (filter-then-split), vs the historical **966**
(split-then-filter). The +5 delta is the cost of a defensible, reproducible
split; the prior regime is preserved at branch `snapshot/2026-05-10-eval`
(commit `9a73669`).

**When to revisit.** If a reviewer requires exact continuity with the historical
966 number, or if the split strategy changes (e.g. different stratification).

---

## D002 — Canonical regime is billable-only, test N = 971

**Date:** 2026-05-31

**Context.** Following D001, the project needs one canonical test set so multi-seed
results are comparable.

**Decision.** The canonical evaluation regime is billable-only (9,660 records),
filter-then-split at seed 42, producing a deterministic test set of **971**
records. All future reported numbers use this regime.

**Rationale.** Verified by running the fixed `prepare_splits.py` on the locked
gold: filter reported `10,240 → 9,660`, totals train/val/test = 7,728/961/971.
Deterministic at seed 42. The 966-vs-971 difference is per-chapter rounding, not
randomness (`int(9660×0.2×0.5)=966` is the un-rounded ideal; 22 per-chapter
roundings sum to 971).

**Trade-offs.** Supersedes the historical 966; documentation that quoted 966 must
be updated to 971. Numbers from a fresh reproduction will not exactly equal the
old single-run figures.

**When to revisit.** If the gold dataset's record count is reconciled (see
`open_questions.md`) in a way that changes the split.

---

## D003 — Establish the `docs/` convention; no retrospective fabrication

**Date:** 2026-05-31

**Context.** The project's prior lack of documented intent forced re-derivation of
decisions from code and artifacts. (This is the proximate reason for the current
refactor.)

**Decision.** Adopt the eight-file documentation convention described in
`philosophy.md` (seven `docs/` files + root README). Append-only logs start now;
no back-dated entries for earlier choices.

**Rationale.** A trustworthy record requires that recorded fact be distinguishable
from reconstruction. Fabricating history to look complete would defeat the
purpose.

**Trade-offs.** Earlier decisions (hierarchical-over-flat, ClinicalBERT-over-
alternatives, the 40-epoch warm start, DVC-over-git-LFS) are real and important
but are referenced as background rather than logged as D-entries, because they
predate this record. They can be promoted to formal D-entries later *if* re-made
or re-affirmed during the refactor.

**When to revisit.** Never rescinds; amend the convention via a new D-ID.

---

## D004 — Do not manufacture a number from the broken stage-1 model; stop and document

**Date:** 2026-05-31

**Context.** While attempting the "re-run evaluate.py on the 971 split" step
(status #0), `evaluate.py` failed loading the Stage-1 tokenizer. Investigation
showed the E-003 stage-1 model on disk is in a broken, unbacked state:
- weights (`stage1/model.safetensors`, 433 MB) sit at the `stage1/` top level;
- config + tokenizer (`config.json`, `tokenizer.json`, `tokenizer_config.json`)
  sit one level down in `stage1/model/`;
- so neither directory is a complete loadable model, and `_find_model_dir`
  (which returns the first dir containing `model.safetensors`) returns the
  top-level dir that has weights but no tokenizer → tokenizer load fails;
- the weights are gitignored (`.gitignore:78`, `*.safetensors`) AND not in DVC
  (no `.dvc` pointer) → the file exists only on this disk, tracked by nothing.

(En route, `sentencepiece` was installed as a candidate fix; it was a red
herring — the error message named it but the real cause was the split layout.)

**Decision.** Do not shuffle files to force `evaluate.py` to load this model.
Stop, and document the finding. A 971-regime number obtained from a model whose
provenance is this uncertain would be a liability, not progress — it is exactly
the kind of untrustworthy artifact this refactor exists to eliminate.

**Rationale.** The goal of the reproducibility work is numbers whose origin is
known and defensible. Manufacturing a figure from a broken, unbacked,
murky-provenance model contradicts that goal even if the file-shuffle made eval
run. Better to record the truth (eval is blocked on a broken artifact) and fix
the artifact properly.

**Trade-offs.** No 971-regime accuracy number exists yet, and won't until
stage-1 is rebuilt. Accepted — a delayed honest number beats an immediate
untrustworthy one.

**When to revisit.** Resolved when stage-1 is cleanly retrained and backed up
(see open_questions.md Q7); at that point a real evaluation on the 971 split can
run and the number can be recorded.

---

## D005 — Retrain on code-only-redacted data (descriptions retained); result is provisional

**Date:** 2026-05-31

**Context.** The EDA notebook (cell 52, "Forensic Alert: Diagnostic Leakage
Detected") *advocates* redacting both (a) literal ICD-10 code strings and (b) the
semantic diagnosis labels (e.g. "Obesity, unspecified") from the note text,
calling redaction "a prerequisite for a valid training run." **Verified: the
notebook advocates this but never implements it** — the redaction code cells
(53, 56, 57) are empty, and the gold data retains the semantic labels. What is
actually implemented is code-string redaction only (broadened by the revised
PARENTHETICAL_ICD10_PATTERN, commits e46d089/502198f). So the model input still
contains plain-text diagnosis descriptions like "pain in left knee" — a known,
residual form of label leakage.

**Decision.** Proceed now with a retrain on the current code-only-redacted
canonical data (May-10 gold, revised regex). Do NOT implement full semantic-label
redaction yet (deferred — see Q8).

**Rationale.** To test whether retraining on the ratified canonical dataset
reproduces any of the existing historical results (83.9% / 85.8% / 77.2%), whose
provenance is currently uncertain. A close match corroborates the historical
numbers; a divergence is itself a finding. The retained semantic labels are a
known upward bias on the *absolute* accuracy, but they do not invalidate the
*reproducibility comparison*, because the historical runs carried the same
retained labels.

**Trade-offs.** Any number from this retrain is **provisional and biased upward
by retained-label leakage — NOT a publishable result.** It is a reproducibility
check against historical numbers, nothing more. The publishable number requires
the Q8 full-redaction rerun.

**When to revisit.** After Q8 (semantic redaction implemented): re-run and report
the delta between code-only and code+label redaction; that delta quantifies how
much the retained labels were inflating accuracy.

---

## D006 — Canonical gold is REGENERATED from HF raw via current code, not a stored file

**Date:** 2026-05-31 (revised same day — supersedes the initial "bless the May-10
file" draft, which was wrong in spirit: see below)

**Context.** Two stored gold files existed (May-10 untracked / revised regex;
May-5 DVC-tracked / narrow regex), differing in ~1,539 rows. The initial instinct
was to pick the May-10 file as canonical and re-track it. But the project's design
principle (confirmed by the project owner) is: **HF holds only the raw/near-raw
inputs; everything downstream — silver, gold, splits, models — is regenerated by
running the code.** Gold is deliberately NOT published, to force regeneration and
prevent stale-artifact drift.

**Decision.** Canonical gold is not a stored file at all — it is **whatever the
current pipeline (`prepare_data.py`, revised redaction regex) produces from the
HF-published raw inputs.** Regenerate it rather than trusting either stored file.
The canonical thing to lock is therefore **raw data (on HF, SHA256-pinned) + the
code version that processes it**, not a gold artifact.

**Enacted & verified 2026-05-31:**
- Both raw inputs hash-verified against the pinned SHA256s — local raw == HF
  canonical raw, byte-for-byte.
- `prepare_data.py` regenerated gold: 10,240 records, 9,660 billable, 0 bare
  ICD-10 codes in `apso_note`, 355 `[REDACTED]` markers.
- Regenerated gold vs the May-5 baseline differs ONLY in the redaction-touched
  columns (`assessment` 1,530 / `apso_note` 1,539 rows + minor S/O/P) — the exact
  fingerprint of the revised-regex fix, nothing else. The regeneration reproduced
  the May-10 regime from verified raw.
- The 971 billable split regenerated identically from the fresh gold.
  → The raw→gold→split chain is demonstrably deterministic and reproducible.

**Trade-offs.** Slightly more to run (regenerate vs load a file), but it's the
honest expression of the reproducibility principle and proves the pipeline works
from raw. The stored gold files are now incidental regenerations, not sources of
truth.

**When to revisit.** When Q8 (semantic redaction) lands, the pipeline changes and
canonical gold = the output of the *then*-current code; this decision's mechanism
(regenerate from raw) stays, only the code version advances.

---

## D007 — Fix `adapter.save()` call in all three training paths (root cause of Q7)

**Date:** 2026-05-31

**Context.** Q7 (stage-1 model unloadable — weights split from config/tokenizer)
was assumed to be after-the-fact damage. Reading `train.py` against
`src/adapters.py` showed it is **produced by the trainer itself**, in all three
training paths:
- flat (line ~378): `adapter.save(output_base)`
- stage-1 (line ~472): `adapter.save(s1_dir)`
- stage-2 (line ~636): `adapter.save(ch_dir)`

Each computes `model_dir = <parent> / "model"`, trains into `model_dir`, but then
calls `adapter.save(<parent>)`. Inline comments ("save to parent, adapter adds
/model") reveal the cause: a **wrong assumption that `adapter.save()` appends
`/model`**. It does not — `EncoderAdapter.save()`'s own docstring states it writes
a FLAT layout directly into the dir passed. So `save(<parent>)` writes the
complete model to the parent dir, while `_finalize_model_dir(model_dir, …)` writes
config/tokenizer into the `model/` subdir → weights stranded one level above
config+tokenizer → unloadable (the Q7 split). This also explains why
`_find_model_dir` carries FLAT/SINGLE/NESTED auto-detection: the bug has been
producing inconsistent layouts.

**Decision.** Change all three call sites from `adapter.save(<parent>)` to
`adapter.save(model_dir)`, so weights, tokenizer, and label_map are written into
the same `…/model/` dir that `_finalize_model_dir`, `_find_model_dir`, and
`inference.py` expect.

**Rationale.** `model/` is the single-directory layout the loader and the path
resolver already target. Fixing at the save site (rather than teaching the loader
to read split layouts) produces a complete, conventional HF model dir and makes
the FLAT/SINGLE/NESTED workaround unnecessary for newly-trained models.

**Trade-offs.** Existing on-disk models trained by the buggy code remain split
until retrained; the loader's auto-detection still covers them. New models will be
clean. This is the prerequisite that unblocks the Q7 retrain.

**When to revisit.** If `EncoderAdapter.save()`'s layout contract changes, re-check
these call sites; ideally add a post-train assertion that `model_dir` contains
config.json + tokenizer + weights together.

---

## D008 — Provisional E2E number accepted as reproducibility check (2026-06-01)

**Decision.** Accept E2E 0.838 (macro F1 0.766, ECE 0.049) on the 971 split as the
provisional number for the rebuilt pipeline, and treat it as a successful
reproduction of the historical 83.9% — NOT as a publishable result.

**Context.** First clean end-to-end run after the D007 fix and full retrain
(stage-1 base-init 0.935; E-002 30-epoch converged; stage-2 19 resolvers @20 epochs
warm-started from E-002; temperature-calibrated). Stage-1 chapter accuracy 0.952 vs
historical 0.964.

**Rationale.** The number reproduces history within ~0.1pt, from regenerated-from-HF
gold through a verified-loadable model chain, on the canonical 971 split. That
validates the rebuild: the contradictory-numbers problem that started this effort is
resolved — we can now produce the project's headline number reproducibly.

**Why provisional, not publishable.** D005 regime retains semantic diagnosis labels
(code-only redaction), biasing the number upward via residual leakage. The first
publishable number is gated on implementing semantic-label redaction (Q8) and will
be lower. The base-init stage-1 (vs historical E-001-init) is a second, smaller
caveat (~1pt router gap).

**When to revisit.** Supersede once Q8 (semantic redaction) produces a leakage-free
number; revisit the stage-1 init gap if strict E-001-init reproduction is needed.

---

## D009 — Canonicalize experiment naming on notebook source; E-009 over E-010 drift (2026-06-01)

**Decision.** The canonical hierarchical experiment is **`E-009_Balanced_E002Init`**
(notebook 05, `cfg["experiment_name"]`). `E-010_40ep_E002Init` is naming drift —
no notebook defines it — and is to be deprecated and deleted. The full canonical
pipeline is documented in `docs/canonical_pipeline.md`, derived directly from the
notebook `cfg` dicts (02–05) and the scripts.

**Context.** All five notebooks were read in full (cfg/path/training cells, not
skimmed). The naming and init chain are now pinned from source:
- E-001_Baseline_ICD3 (nb02): Bio_ClinicalBERT, 30ep, lr 2e-5, batch 16.
- E-002_FullICD10_ClinicalBERT (nb03): Bio_ClinicalBERT, 40ep, lr 2e-5, batch 16.
- E-003_Hierarchical_ICD10 (nb04): Stage-1 router init from E-001 (5ep); Stage-2
  cold-start (the 12.7% failure). Stage-1 trained ONCE here.
- E-009_Balanced_E002Init (nb05): loads Stage-1 from E-003 registry (no retrain),
  Stage-2 init from E-002 (20ep). cfg stage1_source = E-003_Hierarchical_ICD10.

**Rationale.** Names must match the authoritative source (the notebooks). E-010
exists only in the README headline and run dirs — it was never defined by a
notebook cfg, so it cannot be canonical. The designed architecture trains Stage-1
once (E-003) and reuses it by reference; therefore `--stage1-experiment` is
`E-003_Hierarchical_ICD10` in calibrate/evaluate, regardless of the stage-2
experiment name. This also corrects this session's earlier error of training
Stage-1 under E-010.

**Consequences.**
- The README headline (83.9%, "E-010") is unverified drift. Until a fresh
  canonical run produces a number, the reference E-009 e2e is 0.798 (notebook's
  logged value; overview cites 77.2% from an earlier measurement).
- A clean end-to-end re-run under canonical names (canonical_pipeline.md §3) is
  required to produce the authoritative number. This supersedes D008 (the 0.838
  run, which used the E-010-throughout deviation and 30-epoch E-002).
- The E-010 run directories and README recipe are to be reconciled to E-009.

**When to revisit.** After the canonical re-run lands a verified number, update
the reference figure and supersede D008.
