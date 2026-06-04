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

---

## D010 — Canonical E-009 number, produced by a verified-clean run (2026-06-01)

**Decision.** The authoritative E-009 end-to-end figure is **0.849** (macro F1
0.774, ECE 0.0242) on the canonical 971-item billable test split. This supersedes
the D008 number (0.838) and the D009 placeholder reference (0.798 logged). It is
recorded as the faithful, reproducible output of the documented pipeline — **and,
per D005, it remains provisional and NOT publishable** (see leakage note below).

**Provenance — why this number is citable as what it claims to be.** Produced by
the full canonical re-run under source-verified naming (canonical_pipeline.md §3),
with a gate passed after every stage:
- E-001 (675 ICD-3, 30ep, code-filter all) → val_acc 0.869; model/ nested + loads (675).
- E-002 (1,926 billable, 40ep) → val_acc 0.740, reproduces historical ~0.733;
  model/ nested + loads (1926).
- E-003 Stage-1 (22 chapters, 5ep, init from E-001) → val_acc 0.964, reproduces
  historical exactly; stage1/model/ nested + loads (22). Replaces the D007-broken
  on-disk E-003 stage-1.
- E-009 Stage-2 (19 resolvers, 20ep, warm-started from E-002) → every chapter's
  LOAD REPORT shows the 1926→N head reinit that proves E-002 transfer (NOT the
  silent cold-start that caused the old 12.7% failure); [presplit] sizes matched
  prepare_splits; spot-checked Z (263) and T (15) load.
- Calibrate: Stage-1 temperature (1.1701) written to the **E-003** path, not E-009
  — confirms the D009 "train Stage-1 once, reuse by reference" design works in
  practice. 19 resolvers calibrated, P/Q/U skipped; avg ECE 0.688→0.077.
- Evaluate: Stage-1 loaded from E-003; 19 resolvers from E-009; graph reranker
  read from data/graph/. Test N = 971 (no split mismatch). Internally coherent:
  chapter routing 0.984 × within-chapter 0.863 ≈ 0.849 e2e — no hidden inflation
  path. Q9 sklearn warning fired on the linker pickles as predicted (certified
  benign).

**Why provisional, not publishable (D005 stands, now stated concretely).** The
redaction removes the ICD-10 *code* strings but retains the human-readable
diagnosis description they encode — e.g. "pain in left knee" for M25.562 — which
in the APSO notes sits adjacent to where the code was. The model can therefore
read the answer off the description text rather than infer the diagnosis from the
Subjective/Objective clinical findings. 0.849 measures "the pipeline as currently
built," not "what the model can actually diagnose." The first publishable number
is gated on Q8 (description-level redaction + regenerate gold + rerun the chain)
and is expected to come in **below** 0.849 — that drop is the result, the
quantified leakage, not a regression.

**When to revisit.** Supersede once Q8 produces a leakage-free number on
regenerated gold.

---

## D011 — Description-redaction: assessment-only scope, dictionary-anchored, LLM-audited (2026-06-02)

**Decision.** The Q8 description-redaction (removing the diagnosis label text the
ICD-10 code encodes) operates on the **assessment section only**, using a
**deterministic phrasing-dictionary redactor** (CDC description as fallback),
validated by an **advisory local-LLM audit**. This supersedes the vague
"semantic-label redaction" sketch and sets the method to be migrated into
`src/preprocessing.py` + `prepare_data.py`.

**Scope = assessment-only (evidence-based).** Per-section overlap audit of all
9,660 billable records (full CDC description present, overlap=1.0):
assessment 76.2%, plan 14.9%, subjective 15.7%, objective 6.3%. **Note-level
(any section) = 77.9%** — only **1.7pp (169 records)** more than assessment-only.
Eyeballing the P/S/O overlaps (dashboard tab 3) showed they are overwhelmingly
*scattered legitimate clinical vocabulary* ("physical therapy for the left knee",
"NSAIDs for pain") — i.e. token co-occurrence, NOT contiguous label restatement.
Redacting P/S/O would gut clinical signal the model should use. Therefore:
redact the assessment (where the contiguous label lives); **consciously accept
the 169 blind-spot records as residual leak.** 77.9% is the honest note-level
exposure figure; 76.2% is the assessment redaction target.

**Method = dictionary-anchored deterministic rule.** MedSynth's actual diagnosis
phrasings were harvested from the ~26% of raw notes carrying a `(ICD-10: CODE)`
tag → `code → {phrasings, freq}` (1,056 codes, 1,342 phrasings after quarantine).
Phrasings with CDC-overlap <0.3 (197 — mostly comorbidity intrusions, e.g.
F12.20→"Hypertension") were quarantined, NOT used. Redactor matches the full
phrase, removes ALL occurrences, uses a `[DIAGNOSIS]` placeholder mid-sentence
and drops the line when the phrase stands alone. A **min-2-content-token guard**
skips dangerously generic single-word descriptions ("Weakness") that would
over-redact legitimate findings — those become accepted residual.

**LLM = advisory auditor only (reproducibility-preserving).** The deterministic
rule redacts; a local LLM (Path 1 direct API, temperature 0, served via oMLX)
only *judges* original→processed and flags leak/over-redaction. It never writes
gold. This keeps the published number's redaction reproducible; the LLM audit is
re-runnable given a pinned stack but is not load-bearing for the result.

**Validation.** On a 50-record sample: the dictionary redactor scored **45/50
clean by LLM verdict; the 2 "over_redacted" were false alarms** (assessments that
were label-only, so correct full removal looked like section deletion) → effective
**~94% clean**, vs a **50% baseline** for an earlier CDC-fuzzy matcher (which left
label fragments and broke sentences). Over-redaction of legitimate content is
eliminated. Residual = 3/50 leak_remains, all one class: the label **restated in
a surface form the dictionary lacks** (inserted article "the", dropped prefix
"Old", rewording). Accepted as residual for now; an optional-article match is a
noted future refinement, deliberately NOT added yet (it raises over-redaction
risk to fix the less-costly failure mode).

**Honest coverage caveat.** "~94% clean" is over the records the rule **fires on**
(5,610 of 9,660). The 3,270 no-match + 780 guard-skip records are untouched and
may retain leaks. So: of records we redact, ~94% are clean; this is NOT a claim
that 94% of all leakage is removed. Total residual leakage across the corpus is
higher and must be stated as such in the paper.

**When to revisit.** After migrating to scripts + regenerating gold + rerunning
the E-009 chain, the resulting accuracy (expected below 0.849) is the first
publishable number; compare against 0.849 to quantify the leakage's contribution.

---

## D012 — Q8 redactor finalized at v5 (article-tolerant); v4 recall fix rejected for over-redaction (2026-06-02)

**Supersedes the redactor-detail in D011** (scope and architecture in D011 stand;
this records the final redactor version and the corpus-wide residual, which D011
predated). The redactor to migrate is **v5**.

**Why this decision exists.** D011 validated the *approach* on a 50-record sample
(~94% clean). Afterward we measured the redactor corpus-wide and ran a 500-record
stratified audit. Those revealed the recall gap that the 50-sample could not, and
drove two more redactor iterations. This records what we found and why v5 is final.

**Corpus-wide deterministic residual (the honest figure).** Re-running the
token-overlap leak test on the redactor *output* across all 9,660 billable
records (no LLM, no sample):
- pre-redaction leak: 7,372 (76.3%)
- **v3** (D011's redactor): residual 2,270 (**23.5%**), removed 69.2%, fired 5,610.
- **v5** (final): residual 1,799 (**18.6%**), removed 75.6%, fired 6,091.

"~94% clean" in D011 was over the *fired* set; **18.6% of all billable records
still leak after v5**. Both numbers are true; they answer different questions.

**Stratified 500-record LLM audit (fired 300 + nomatch 200).**
- fired stratum: **96% clean** — confirms redaction quality at scale (not just n=50).
- nomatch stratum: **61% genuinely still leak** — the recall gap is REAL, not a
  metric artifact. Of records the rule does not fire on, most truly retain the label.

**Recall-gap decomposition (122 confirmed nomatch leaks categorized).**
- article_insertion 20% — dictionary phrase defeated by an inserted "the"/"a"
  (e.g. "Disease of **the** stomach…"). **Fixable → fixed in v5.**
- guard_skipped 26% — single-content-token descriptions ("Weakness", "Other Shock")
  skipped by the min-2-token guard. **Accepted residual** (skipping is correct;
  matching them over-redacts — see v4 below).
- on_CDC_fallback 34% — code absent from dictionary; CDC string didn't match. NB:
  inspection showed SOME of these are LLM false-positives / code-assessment
  mismatches (assessment is about a different condition than the code), i.e. NOT
  true misses. The 34% overstates real residual.
- reworded 20% — label genuinely paraphrased ("Old MI" → "history of myocardial
  infarction", "prosthetic heart valve" → "mechanical mitral valve replacement").
  **Genuine deterministic ceiling** — cannot be caught without semantic matching,
  which would over-redact. Accepted residual.

**v4 attempted, then REJECTED (the key breadcrumb).** v4 = v3 + two recall fixes:
(a) optional-article matching, (b) standalone-short-phrase recovery (drop the guard
when a short phrase is a whole standalone line). v4 reached **13.3% residual** —
better recall. BUT a risk-weighted precision audit (200 records, oversampling the
new behaviors) showed fix (b) **reintroduced over-redaction of clinical findings**:
it cut short labels out of *finding sentences* — "moderate **wheezing** and…" →
"moderate  and…" (R06.2), "Moderate **aphasia**, difficulty…" (R47.01),
"consistent with acute **pyonephrosis**" (N13.6). Embedded-short over-redaction is
exactly what the guard existed to prevent. Fix (a), article-tolerance, was clean
(its apparent over-redactions were label-only-assessment false alarms).

**v5 = v3 + article-tolerance ONLY (fix (a) kept, fix (b) dropped).** Deliberately
chose **18.6% residual over v4's 13.3%**: the extra ~5pp recall came at the cost of
corrupting clinical findings the model must learn from. **Over-redaction (damaging
legitimate content) is a worse failure than residual leak** — we accept a higher
leak rate to guarantee we are not mangling findings. This trade is the core of the
decision.

**LLM-judge fallibility (method note).** The local-LLM auditor mis-flagged twice,
both caught by *reading cases, not trusting tallies*: (1) it read the `[DIAGNOSIS]`
placeholder as a surviving leak (fixed by declaring the marker in the prompt);
(2) it called label-only assessments "over_redacted" when correct full removal left
an empty section. Lesson recorded: the LLM is an advisory flagger; verdicts are
confirmed by reading, especially before acting on a tally.

**Final accepted residual = 18.6%**, composed of: reworded (genuine ceiling) +
guard-skipped single-word labels (skipping is safer than over-redacting) +
CDC-fallback (partly real misses, partly LLM false-positives). Consistent with the
project stance: residual leak is documented and accepted; over-redaction is not.

**Migrate v5** (article-tolerant full-phrase match, [DIAGNOSIS] placeholder
mid-sentence / drop-line standalone, remove-all-occurrences, min-2-token guard,
dictionary + CDC fallback) into `src/preprocessing.py` + `prepare_data.py`, behind
a flag, then rerun. Port-verification gate: confirm the migrated function
reproduces 18.6% residual / 6,091 fired before trusting the rerun.

---

## D013 — LLM-assisted audit: methodology & reproducibility (publication record) (2026-06-02)

Publication-grade record of the local-LLM auditor used throughout Q8. Captures
what model, served how, doing what job, with what guardrails, and how it shaped
the redaction decisions WITHOUT contaminating the reproducible gold artifact.
Cross-refs: D011 (architecture), D012 (v5 outcome & the convergence path),
`prompts/q8_audit_prompt.md` (the prompt, committed). NB a fuller serving writeup
(`serving_local_models.md`) is not yet in the repo; the operational specifics
needed to reproduce the audit are recorded inline in §3 below.

**1. Role — advisory auditor, never redactor.** The deterministic dictionary rule
(`redact_descriptions`, v5) produces the gold. The LLM only *judges* the rule's
output per record — verdict ∈ {clean, leak_remains, over_redacted, both} — to
measure redaction quality and surface failure modes. The LLM NEVER writes, edits,
or selects gold content. This is the central reproducibility guarantee: **the
published de-leaked gold does not depend on any stochastic LLM output.** A reader
can regenerate the gold from the committed dictionary + deterministic code without
ever invoking the LLM.

**2. Model — agnostic design, named instance.** The method is model-agnostic: it
accepts any sufficiently capable instruction-following model served locally; the
model is selected from the live oMLX roster at run time (NOT hardcoded), and its
identity/capability is recorded from the runtime roster rather than assumed. The
results reported in this work were produced with the specific instance
**`Qwen3-Coder-30B-A3B-Instruct-MLX-6bit`**. Both facts belong in the paper: the
design does not depend on one model, and the reported numbers are reproducible with
this exact instance.

**3. Serving — local, Path 1 direct API.** Served via **oMLX** at
`http://127.0.0.1:8000`, OpenAI-compatible **Path 1 direct API**
(`/v1/chat/completions`). (Path 2, the `claude -p` agentic route, was NOT used for
the audit.) The Path 1 vs Path 2 distinction and the "capability recorded from
roster, not hardcoded" pattern are described in an external serving writeup
(`serving_local_models.md`) which should be added to the repo before publication;
until then the reproducibility-critical specifics (model §2, endpoint, API, temp
§4) are captured here in D013. Auth token is a local placeholder.

**4. Determinism — temperature 0, top_p 1.** All audit calls use temperature 0,
top_p 1, so verdicts are re-runnable on the pinned stack. Honest caveat for the
paper: this yields determinism *given the same model weights and serving stack* —
it is an engineering reproducibility property, not a mathematical guarantee across
model or runtime updates. The audit is therefore reproducible as an artifact
(pinned stack), and the deterministic gold is reproducible independently of it.

**5. Prompt — versioned artifact, evolved.** The audit prompt is a committed file,
`prompts/q8_audit_prompt.md`, read directly by the audit cell (NOT pasted inline —
avoids drift between documented and executed prompt). It evolved during the work;
the material change was declaring the `[DIAGNOSIS]` placeholder as a SUCCESS marker
after the judge initially mis-read it as a surviving leak (see §7). The prompt
defines the four verdicts, requires reasoning-before-verdict, and includes worked
examples.

**6. How the LLM shaped convergence (v1→v5).** The auditor *informed* but did not
*decide* the redactor evolution. It flagged the v4 short-standalone fix as
reintroducing over-redaction of clinical findings (→ rejected, see D012), and its
500-record stratified audit confirmed the recall gap was real (nomatch stratum 61%
genuine leaks). Every redaction DECISION was made by a human reading the flagged
cases; the LLM was a flagger whose tallies were verified by eyeball. The paper's
honest characterization: **LLM-assisted auditing, human-adjudicated decisions,
deterministic redaction.** The convergence to v5 (and the rejection of v4) is
attributable to human judgment over LLM-surfaced evidence, not to the LLM itself.

**7. Limitations (for the paper).** (a) The judge is fallible: it mis-flagged at
least twice — reading the `[DIAGNOSIS]` placeholder as a leak, and calling
label-only assessments "over_redacted" when correct full removal emptied the
section. Both were caught by reading cases, not trusting tallies; this is why
human adjudication is load-bearing. (b) LLM-audited samples (50; 500 stratified;
200 precision) are samples, not the full corpus — the deterministic residual
(D012, 18.6%) is the exhaustive measure; the LLM clean-rates (96% fired) are
sampled. (c) "Clean rate" is over the records the rule *fires on*, not all records.
(d) Audit reproducibility depends on the pinned local stack (§4).

---

## D014 — De-leaked rerun (E-015): experiment spec, gold provenance, Stage-1 reuse (2026-06-03)

Records the controlled rerun that produces the first leakage-corrected number, the
provenance facts needed to make it a valid A/B against the canonical 0.849 (D010),
and a reproducibility gap in D010 that this entry closes.

**Gold provenance — what 0.849 actually trained on (closes a D010 gap).** D010
recorded the 0.849 number but NOT its exact data input. The experiments log
(`outputs/experiments.json`, E-009_Balanced_E002Init → train_stage2 → params)
shows the authoritative answer: **`gold_path: data/gold/medsynth_gold_apso.parquet`**.
This is exactly the file the Q8 description-redactor was applied to, so the
de-leaked gold (`medsynth_gold_apso_deleaked.parquet`) is the correct counterpart
and the A/B is valid. (The `medsynth_gold_augmented.parquet` referenced as the
`--gold-path` default and in other experiments' logs was NOT E-009's input; it is
unrelated to this comparison and is not even present on disk.)

**E-009 recipe recovered (from experiments.json), mirrored exactly for E-015.**
model `emilyalsentzer/Bio_ClinicalBERT`; mode hierarchical; stage 2; epochs 20;
batch 16; lr 2e-05; weight_decay 0.01; warmup 0.1; max_length 512; code_filter
billable; stage2_init `E-002_FullICD10_ClinicalBERT`; chapters all; seed 42;
calibrate stage1_experiment `E-003_Hierarchical_ICD10`, threshold 0.7. The rerun
changes exactly ONE variable: `--gold-path` → `medsynth_gold_apso_deleaked.parquet`.
Epochs held at 20 (NOT the 40 used by E-010/012/013) precisely to keep the A/B
clean against E-009.

**Stage-1 reuse — a deliberate control, with a caveat to report.** The rerun
retrains Stage-2 only and REUSES the E-003 Stage-1 chapter router (which was
trained on leaky data). Rationale: Stage-1 is the coarse 22-way chapter routing;
the description leakage primarily aids fine-grained within-chapter Stage-2
resolution. Holding Stage-1 fixed (identical to 0.849) means the measured delta is
cleanly attributable to Stage-2 retraining on de-leaked data — the better
controlled experiment. CAVEAT for the paper: the number is therefore NOT
"fully de-leaked end-to-end"; Stage-1 still saw leaky data in training. This must
be stated. A fully-clean variant (retrain Stage-1 on de-leaked data too) is a
possible follow-up.

**Experiment name: E-015_E009_Deleaked.** Chosen as the next number in the REAL
sequence on disk (E-009, E-010, E-010_hybrid_Z, E-012, E-013, E-014 already exist
— see Q11 on docs drift), signalling it is E-009's recipe on de-leaked gold.

**Invocation (validated via --dry-run before launch).** Run from project root with
`PYTHONPATH=.` (run_experiment.py lacks the self-bootstrap that prepare_data.py
has — Q11):

    PYTHONPATH=. uv run python scripts/run_experiment.py \
      --experiment E-015_E009_Deleaked \
      --model emilyalsentzer/Bio_ClinicalBERT \
      --stage2-init outputs/evaluations/E-002_FullICD10_ClinicalBERT/model \
      --stage1-experiment E-003_Hierarchical_ICD10 \
      --gold-path data/gold/medsynth_gold_apso_deleaked.parquet \
      --epochs 20 --code-filter billable

**De-leaked gold provenance.** Produced by `scripts/prepare_data.py
--redact-descriptions` (flag gates phase_3d AND switches the output filename so
both golds coexist). Verified on disk: 9,660 billable rows (matches original),
[REDACTED] present (code redaction, 3c), [DIAGNOSIS] present (desc redaction, 3d),
0 [DIAGNOSIS] in the original gold, residual leak 18.6% — identical to the
validated v5 (D012). The local-LLM auditor is NOT involved in any of this (D013);
the gold is fully deterministic.

**RESULT (2026-06-03, run completed; ~83 min).**

    Metric        E-009 (leaky, D010)   E-015 (de-leaked)   Δ
    E2E accuracy            0.849              0.482        −0.367  (−43% rel)
    Macro F1                0.774              0.368        −0.406
    ECE                     0.0242             0.1313       +0.107  (worse)
    Coverage@0.7            (—)                0.475
    N (test)                971                966
    Stage-1 chapter acc     ~0.97              0.758 (eval) / 0.972 (calib)

The headline: removing the diagnosis-description leakage collapses E2E accuracy
from 0.849 to **0.482** — a 36.7-point absolute drop. The description leakage was
responsible for the large majority of the apparent performance. The 0.849 baseline
was substantially inflated, confirming the D005/D010 suspicion. This is the
first leakage-corrected number and it is a legitimate, reportable result.

**Critical interpretation caveat — 0.482 is a LOWER BOUND, conflating two effects.**
The Stage-1 router (reused from E-003, trained on LEAKY text) measured acc=0.972 on
its own calibration data but only **0.758** on the de-leaked eval set. Same router,
different text: it had partly learned to route chapters using description tokens we
then redacted, so its routing degrades on de-leaked notes. Thus the 0.482 reflects
BOTH (a) the intended effect — Stage-2 can no longer read the answer off the text —
AND (b) an INTRODUCED train/serve mismatch — Stage-1 trained on leaky text, evaluated
on de-leaked text. Effect (b) is an artifact of reusing the leaky-trained router
(the deliberate control in this experiment), not a property of the de-leaked data.
A fully-clean number requires retraining Stage-1 on de-leaked data too
(`--train-stage1` rerun). The TRUE de-leaked performance is therefore expected
somewhat ABOVE 0.482 (less Stage-1 damage) — but still far below 0.849.
This exactly realises the D014 caveat ("not fully-de-leaked end-to-end") and shows
it is material, not academic. Additionally, residual leakage (18.6%) means even a
fully-Stage-1-retrained number would slightly over-estimate true clean performance.

**Follow-up COMPLETED — E-016, the true clean end-to-end de-leaked run (2026-06-03).**
E-015's Stage-1-reuse lower bound was resolved by a full rebuild: `--train-stage1`
with `--gold-path ...deleaked.parquet`, so BOTH stages train on the de-leaked data
(Stage-1 router + Stage-2 resolvers), then calibrate + evaluate against the new
router. Experiment `E-016_Deleaked_FullRebuild`. Same recipe otherwise (Bio_ClinicalBERT
for both stages — E-003's Stage-1 was also Bio_ClinicalBERT, confirmed from its config;
epochs 20, seed 42, billable, E-002 Stage-2 init). ~139 min.

    Metric        E-009 (leaky)  E-015 (S1 reused, LB)  E-016 (full de-leaked)
    Stage-1 acc       0.9835          0.758  ← mismatch       0.948  ← fixed
    Stage-2 within    0.8628          0.637                   0.598
    E2E accuracy      0.849           0.482                   0.567
    Macro F1          0.774           0.368                   0.446
    ECE               0.0242          0.1313                  0.0703
    Cov@0.7           0.8115          0.475                   0.482
    N (test)          971             966                     966

**HEADLINE CLEAN NUMBER: E-016 E2E = 0.567** (Macro F1 0.446, ECE 0.0703). This is the
true leakage-corrected, fully-end-to-end-de-leaked result: every stage trained on
de-leaked data and evaluated on de-leaked data, no leaky component anywhere. The
Stage-1 accuracy recovered to 0.948 (from E-015's mismatched 0.758), confirming
E-015's 0.482 was an artifact-contaminated lower bound — retraining Stage-1 on the
de-leaked data recovered +0.085 E2E (0.482→0.567) by removing the train/serve
mismatch, exactly as predicted.

**The finding for the paper:** removing the diagnosis-description leakage drops true
end-to-end accuracy from **0.849 → 0.567**, a 28.2-point absolute fall (≈33% relative).
The leakage was inflating the headline by roughly a third. Report 0.567 as the clean
number and 0.849 as the leaky baseline. Remaining caveat: the de-leaked gold still
carries 18.6% residual leakage (D012), so 0.567 still slightly OVER-estimates a
perfectly-clean ceiling — state this, but it is a small documented effect, not an
artifact. E-015 (0.482) is retained in the record as the intermediate lower-bound
step that demonstrated the Stage-1 mismatch was material.

Calibration note: E-016 Stage-1 calibrated cleanly (ECE 0.120→0.035, T=1.72). Avg
Stage-2 ECE 0.413→0.196; a few small resolvers (A, B, C, T) hit the T=0.05 floor —
over-confident on tiny chapters, worth a glance later but not blocking the headline.

## D015 — Full de-leaked rebuild (all backbones): the backbone ranking INVERTED (2026-06-04)

The decisive follow-up to D014. Where D014/E-016 produced the first clean
end-to-end ClinicalBERT number (0.567), this entry records the full rebuild of
EVERY leaky-regime experiment on de-leaked data — both ModernBERT backbones, the
SupCon chain, and MIMIC — run end-to-end by the tested orchestrator
(`scripts/run_full_deleaked_rebuild.py`, run `20260603_200854`). It produced the
single most important finding of the project: **removing description leakage does
not merely deflate scores, it reverses which model backbone is best.**

**Provenance / controls.** De-leaked gold `medsynth_gold_apso_deleaked.parquet`
(D014 provenance). All models trained AND evaluated fully de-leaked (Stage-1 +
Stage-2), no leaky component. Split is now CONTENT-ADDRESSED (sorted by id before
partitioning — see the split-determinism fix, commit 6ebee5e), so the test
partition is reproducible and IDENTICAL across every backbone — the A/B is exact,
not approximate. N(test) = 966.

**RESULT — de-leaked backbone comparison (all hierarchical, same test set):**

    Backbone                 Leaky E2E   De-leaked E2E   Exp      within-ch  Stage-1
    Clinical ModernBERT        0.488        0.769  ←BEST  E-024     0.807      0.953
    ClinicalBERT (Bio_)        0.849        0.592         E-021     0.620      0.956
    BioClinical ModernBERT     0.070        0.398         E-026     0.423      0.940

    LEAKY ranking:     ClinicalBERT (0.849) >> ClinicalModernBERT (0.488) >> BioClinical (0.070)
    DE-LEAKED ranking: ClinicalModernBERT (0.769) >> ClinicalBERT (0.592) >> BioClinical (0.398)

**HEADLINE: the ranking inverts.** On the leaky benchmark ClinicalBERT appeared to
beat Clinical ModernBERT decisively (0.849 vs 0.488), and the paper had concluded
"domain pretraining outweighs architectural modernity." De-leaked, Clinical
ModernBERT (0.769) beats ClinicalBERT (0.592) by ~18 points. The leaked
description signal was strong enough that the older ClinicalBERT exploited it best,
manufacturing a backbone conclusion that is the OPPOSITE of the truth on clean data.

**Why this is real, not an artifact (scrutiny applied before accepting it).**
(1) Same test set: E-021 and E-024 share identical test ids (content-addressed
split — verified by set comparison on the M-chapter test parquet, equal N=111,
same ids). (2) The gain is UNIFORM: ModernBERT beats ClinicalBERT in 17 of 19
chapters, deltas +0.09 to +0.36, spread across the code space rather than
concentrated — the signature of a stronger backbone, not a localised residual leak.
The two exceptions (A n=6, B n=16) are tiny-N noise. (3) Routers are near-identical
(0.953 vs 0.956), localising the entire difference to Stage-2 within-chapter
resolution (0.807 vs 0.620), exactly where description leakage operated.

**Headline number update vs D014.** D014 cited E-016 = 0.567 as the clean
ClinicalBERT number (old position-addressed split). The rebuild's E-021 = 0.592 on
the content-addressed split is the number to cite going forward — it is reproducible
(anyone can regenerate the exact partition from the gold), where E-016's 0.567 was
on a partition that could not be reliably reproduced (the very fragility the split
fix addressed). Cite 0.592 as primary; 0.567 corroborates. Leakage drop is
0.849 → 0.592 ≈ 25.7pp (~30% relative).

**Caveats (unchanged, restated):** 18.6% residual leak (D012) means de-leaked
numbers slightly OVER-estimate a perfectly clean ceiling. ModernBERT used
max_length=512 (matched to ClinicalBERT for a fair A/B, NOT ModernBERT-optimal —
the one paper-derived, non-registry-verified hyperparameter choice). E-026's leaky
0.070 → de-leaked 0.398 went UP (leakage depressed it, not inflated) — an asymmetry
not fully explained; likely the leaky E-013 run was a training failure (resolvers
hit best-epoch-1) rather than a leakage effect, so the de-leaked rerun simply
trained properly. Stated honestly in the paper, not papered over.

## D016 — SupCon "solves the Z chapter" was ~97% leakage artifact (2026-06-04)

Re-evaluation of the E-014 SupCon result on de-leaked data (experiment E-022,
same rebuild run). The leaky benchmark had reported Supervised Contrastive
fine-tuning lifting the administrative Z-chapter by +21.2pp (62.1%→83.3%) and the
system to 86.7%, described as "the Z-chapter problem is solved."

**RESULT (de-leaked, E-022 hybrid = E-021 router + SupCon-Z override):**

    Quantity                 Leaky claim            De-leaked (E-022)
    Z resolver (val acc)     62.1% → 83.3% (+21.2)  0.614 → 0.674 (+6.1pp)
    Hybrid system E2E        85.8% → 86.7% (+0.9)   0.592 → 0.597 (+0.5pp)

**Finding:** the dramatic leaky Z gain was overwhelmingly leakage. De-leaked,
SupCon yields a modest genuine improvement on the Z resolver's validation accuracy
(~6pp) that nearly vanishes at the system level (+0.5pp). The effect is small enough
to be partition-sensitive (an earlier de-leaked attempt, E-018 on the old split,
showed ~0). The claim that SupCon "solves" the Z chapter is WITHDRAWN; the honest
statement is that contrastive learning gives a small, real, system-negligible
improvement that the leaky benchmark overstated by roughly an order of magnitude at
the system level. A second independent instance of the same lesson as D015: a leaky
benchmark can manufacture an apparent algorithmic win that does not exist.

Calibrate/hybrid used the DE-LEAKED router (E-021), explicitly overriding the
scripts' leaky E-003/E-010 defaults (the override the orchestrator phase-specs
enforce and the tests guard). Note: the rebuild's orchestrator initially reported
this step's output as MISSING due to a phase-spec output-path bug (train_supcon_z
saves flat, spec expected nested) — fixed in commit d870b6f with a regression test;
the SupCon result itself was unaffected (recovered by running the final 3 steps
manually).

## D017 — MIMIC-IV de-leaked: 8.1%, domain shift dominates independent of leakage (2026-06-04)

Real-world validation of the de-leaked model (E-021) against 4,877 real MIMIC-IV
discharge summaries, replacing D014-era leaky-model MIMIC figures.

    Metric          Leaky model    De-leaked model (E-021)
    E2E accuracy       12.0%             8.1%
    Macro F1           0.069            0.042
    ECE                0.298            0.318
    Coverage@0.7       22.7%            20.2%

**Finding:** both the leaky (12.0%) and de-leaked (8.1%) models collapse to
near-floor on real notes. The synthetic-to-real gap is driven by DOMAIN SHIFT and is
largely INDEPENDENT of leakage — real notes use abbreviations, shorthand, copy-paste
artefacts, and institution-specific terminology absent from the synthetic training
data. Whatever the synthetic headline, neither model is usable on real notes without
domain adaptation.

**Caveat (train/serve asymmetry):** the de-leaked model was trained on
description-redacted notes, but real MIMIC notes contain descriptions, so the
de-leaked 8.1% carries a train/serve mismatch and is not a perfectly clean
real-world estimate. A fully fair measurement needs inference-time parity between
training and serving text (open follow-up, cf. D014 inference-parity thread). Either
way both regimes land near the floor — domain shift dominates regardless.

Provenance note: `validate_mimic_evaluate.py`'s hardcoded `MEDSYNTH_REFERENCE_DELEAKED`
still prints 0.567/56.7% (the E-016 reference); the de-leaked synthetic reference to
cite is now E-021 = 0.592 (D015). The 8.1% MIMIC number itself is unaffected.
