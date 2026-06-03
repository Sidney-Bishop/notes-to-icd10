# Open Questions

Unresolved tensions carried forward. Monotonic IDs (Q1, Q2, …). Resolve by adding
a `Status: RESOLVED — see Dxxx` line, never by deleting.

---

## Q1 — Dataset record/code count: 9,660/1,926 vs 9,578/1,914?

`03_dataset.qmd` reports both: 9,660 records / 1,926 codes as the headline, and
9,578 records / 1,914 codes as "post-Pydantic-validation." The relationship (the
gatekeeper drops ~82 records / ~12 codes) is implied but not stated cleanly, and
the reported figures should be made internally consistent.

**Decide:** which count is canonical for the paper, and state the gatekeeper delta
explicitly rather than leaving two numbers.

**Status:** OPEN.

---

## Q2 — DVC remote plumbing is broken (but the raw data IS reproducible from HF)

**Refined 2026-05-31.** Earlier framing ("fresh clone cannot reproduce the data")
was too broad. The accurate picture:

- The **raw canonical inputs ARE published and reproducible**: the HF dataset
  `SidneyBishop/notes-to-icd10` holds `data/medsynth/icd10_notes.parquet` (19.8 MB)
  and `data/reference/cdc_fy2026_icd10.parquet` (1.4 MB). Both verified locally
  against `prepare_data.py`'s pinned SHA256s — byte-for-byte canonical.
- Gold is NOT published (by design — D006); it's regenerated from the raw.
- The actual gap is the **DVC remote configuration**: the committed default remote
  is the placeholder `.dvc/CONFIGURE_LOCAL`; the real remote lives in gitignored
  `.dvc/config.local`, which a fresh clone doesn't get. So `dvc pull` fails on a
  clean clone until manually reconfigured — but `prepare_data.py` falls back to
  `hf_hub_download`, so the data is still obtainable without DVC.

**Decide:** stand up a portable DVC remote (or document the HF-direct path as the
primary route, since `prepare_data.py` already supports it) and replace the
`CONFIGURE_LOCAL` placeholder with real setup steps, so a clean clone reproduces
without manual DVC surgery.

**Status:** OPEN — but narrower than first thought. Data reproducibility is intact
via HF + `prepare_data.py`; only the DVC convenience layer needs fixing.

---

## Q3 — `data/ontology/icd10cm_2026.parquet` missing from DVC remote; no confirmed consumer

`dvc status` reports this file "not in cache" — DVC-tracked (pointer committed)
but bytes never pushed to the remote. It still exists in the original working dir.

**Verified 2026-05-31 (do not re-confuse with the file below):**
- The reference file the pipeline actually READS for ICD-10/CDC validation is
  **`cdc_fy2026_icd10.parquet`** (loaded in EDA notebook Phase 1b:
  `config.resolve_path("data","gold") / "cdc_fy2026_icd10.parquet"`). That file is
  DVC-tracked, present, and pulls fine — no problem.
- **`icd10cm_2026.parquet`** (the missing one) is NOT referenced by name in the EDA
  notebook or in any script under `scripts/` or `src/` (grep verified). It has no
  confirmed code consumer in what has been searched. (Notebooks 02–05 not yet
  searched — see caveat.)
- The two files have near-identical names and were conflated earlier in this
  session; they are different artifacts. `cdc_fy2026_icd10` = the live reference;
  `icd10cm_2026` = the missing, unreferenced one.

**Decide:** confirm whether `icd10cm_2026.parquet` is read by notebooks 02–05 or
any path not yet searched. If a real consumer exists → push it to the DVC remote.
If genuinely unreferenced → `dvc remove` the pointer so it stops breaking
`dvc pull`. Do not assume it is dead until 02–05 are checked.

**Status:** OPEN. (Mislabelled "orphan", then over-corrected to "is read"; this is
the verified middle: no consumer found in searched files, unconfirmed in 02–05.)

---

## Q4 — Environment drift vs the paper's basis

A fresh `uv sync` resolves to newer libraries (transformers 5.9, torch 2.12;
sklearn historically drifted 1.1.2→1.8.0, scispacy churn noted) than the stack
that produced the documented results. A clean reproduction might land outside the
documented 84–87% range due to environment, not method.

**Decide:** whether to pin `uv.lock` to the paper's versions before the
reproduction run, and gate seed 1 against the documented range (halt + pin if it
misses) rather than silently adopting a regressed number.

**Status:** OPEN.

---

## Q5 — Headline figure for the publication: 85.8% (E-010) vs 86.7% (E-014 hybrid)?

E-010 mean-of-4 (~85.8%) and the E-014 SupCon-Z hybrid (~86.7%) are both quoted as
"best." If 86.7% appears near the headline it cannot be a single run while 85.8%
has error bars — that asymmetry reads as cherry-picking.

**Decide:** either run E-014 across the same seeds so both carry ± std, or demote
86.7% to a clearly-labelled preliminary result and headline 85.8% ± std.

**Status:** OPEN.

---

## Q6 — Synthetic→real generalisation gap

E-010 reports ~85.8% on synthetic MedSynth but ~12% E2E on real MIMIC-IV
discharge summaries (per publication drafts). The gap is attributed to domain
shift, not architecture, but MIMIC-IV validation completeness depends on
PhysioNet access.

**Decide:** how prominently to report the gap and whether any
domain-adaptation/fine-tuning on real data is in scope (per charter, training on
real data is currently out of scope).

**Status:** OPEN.

---

## Q7 — RESOLVED (2026-05-31, via D007): stage-1 model now loadable

**Status: RESOLVED — see D007.**

Root cause was `train.py` calling `adapter.save(<parent>)` in all three training
paths, splitting weights from config/tokenizer. Fixed (D007) to
`adapter.save(model_dir)`. Verified by gate run (stage-1 only, ~31 min):

- On disk: NO `model.safetensors` stranded at `stage1/` top level; `stage1/model/`
  contains config.json + tokenizer.json + tokenizer_config.json + model.safetensors
  + label_map.json — one complete model directory.
- Load test: `AutoTokenizer.from_pretrained` + `AutoModelForSequenceClassification
  .from_pretrained` on `stage1/model/` both succeed, 22 labels. This is the exact
  operation that previously threw "Couldn't instantiate the backend tokenizer."

Stage-1 router trained healthily from base Bio_ClinicalBERT: best epoch 3,
val_acc 0.920, macro_f1 0.933 (final epoch 6: val_acc 0.935). Note: trained from
base, not E-001-initialised, so routing is slightly below the historical ~96%
(expected; E-001 init can be added for the full run if desired).

---

## Q8 — Redact the diagnosis DESCRIPTION (not just the code), then rerun

> **STATUS (2026-06-02): redactor FINALIZED (v5), migration in progress.** Scope &
> architecture in **D011**; final redactor version, corpus residual, and the
> rejected-v4 breadcrumb in **D012**. The redactor is **v5** (assessment-only,
> dictionary-anchored, article-tolerant, min-2-token guard, LLM-advisory audit).
> Measured corpus-wide: **18.6% residual** leak after redaction (removes 75.6% of
> pre-existing leakage; fires on 6,091/9,660). Stratified 500-record audit: fired
> stratum 96% clean, nomatch stratum 61% genuinely leaks (real recall gap). A more
> aggressive v4 reached 13.3% residual but was **rejected** for reintroducing
> over-redaction of clinical findings (see D012). Residual composition: reworded
> (genuine ceiling) + guard-skipped single-word labels + CDC-fallback (partly LLM
> false-positives). **Remaining work:** migrating v5 into `src/preprocessing.py` +
> `prepare_data.py` behind a flag on branch `q8/description-redaction`; then
> port-verify (must reproduce 18.6%/6,091), regenerate gold, rerun E-009 → first
> publishable number. Inference parity DEFERRED (code unknown at inference — see
> D012/journal).

The current canonical gold redacts the ICD-10 *code* strings but retains the
human-readable diagnosis description they encode — e.g. "pain in left knee" for
M25.562 — in the model input (see D005, D010). In the APSO notes that description
sits adjacent to where the code was, so the model can read the answer off it
rather than infer the diagnosis from the Subjective/Objective clinical findings.
This is the residual leakage that makes the canonical 0.849 (D010) provisional and
is a likely contributor to the synthetic→real gap (Q6).

**QUANTIFIED (2026-06-01, EDA notebook audit).** The inventory we never had now
exists. Of the **9,660 billable records**, measuring CDC-description content-token
overlap with the (re-redacted) assessment field:

| overlap ≥ | records | % |
|---|---|---|
| 1.0 (full description present) | 7,360 | **76.2%** |
| 0.8 | 7,562 | 78.3% |
| 0.6 | 8,092 | 83.8% |
| 0.5 | 8,333 | 86.3% |
| 0.3 | 8,596 | 89.0% |

Clean join (0 records missing a CDC description). The leak is near-verbatim:
M25.562 "Pain in left knee" → assessment "Pain in the left knee" / "Left knee
pain"; N39.0 "Urinary tract infection, site not specified" → "Urinary Tract
Infection, site not specified". This is the dominant case, not a tail: ~3 of 4
billable training records carry the full label in the input. Per-record scores
persisted to `outputs/audits/q8_leakage_scores.parquet` (+ summary json).

**Two honesty bounds on this number.** (1) It measures description *presence*, NOT
causal accuracy inflation — "76.2% contain the label" is not "accuracy drops 76pt".
Some assessments remain diagnosable from the rest of the note after removal. The
*magnitude* of inflation is only knowable from the Q8 rerun; this count is the
*scope of exposure*. (2) Token-overlap can hit 1.0 by chance on very short
descriptions, but the overlap=1.0 cohort is dominated by multi-token exact
restatements (see examples), so that is not what drives 76.2%.

**The approach (anchored, not blind fuzzy matching).** We have the CDC reference
table (`data/ontology/icd10cm_2026.parquet`, columns `code_no_decimal` /
`description` — this is the file the audit used; a second copy exists at
`data/gold/cdc_fy2026_icd10.parquet`, not yet compared), which carries the
official textual description
per code. So for each affected record the redaction is *anchored*: take that
record's own reference description(s) and match them near the (former) code
position, tolerating word-order and minor variants ("pain in left knee" ↔ "left
knee pain"), NOT searching for arbitrary paraphrases anywhere in the note. Every
removed span then traces back to a specific reference string for that specific
record — auditable.

**Plan / sequence.**
1. *Notebook (lab, not fix).* In the EDA notebook: load gold, identify the
   affected rows (description still present after the existing code-redaction),
   **count them** (this is the inventory we never built — answers "how many
   records have the issue"), prototype the anchored redaction, export a ~20-row
   before/after sample for human confirmation.
2. *Confirm efficacy — both directions.* The sample must show that the leak was
   removed AND that legitimate clinical content the model should reason from was
   NOT gutted. Over-redaction (deleting findings) is as much a failure as
   under-redaction (leaving the description). A rule that scores 100% on "removed"
   by deleting half the note is worse than the disease.
3. *Move to the pipeline (the actual fix).* Only after sign-off: move the proven
   rule into `src/preprocessing.py` and wire it as a phase in `prepare_data.py`,
   so regenerating gold applies it and every downstream stage inherits it. The
   notebook proves it; the scripts make it real.
4. *Regenerate gold + rerun the full E-009 chain* for the first publishable
   number. Report the **delta** vs the code-only regime (0.849). Expect a
   meaningful drop — that delta IS the quantified leakage.

**Pending verification (do NOT treat as established).** Two premises this plan
rests on must be confirmed from source before building on them:
- *What the existing redaction actually removes.* Have not yet read
  `src/preprocessing.py` (`ICD10_REDACT_PATTERN`, `redact_icd10_sections`,
  `build_apso_note`). Need it to define "affected rows" correctly and to extend
  rather than duplicate/fight the existing logic. (Note: `prepare_data.py` phase
  3b already computes a per-record `has_leakage` flag, then phase 3c redacts and
  *drops* it — half the inventory is built and discarded; keeping it is most of
  Q8's counting step.)
- *That descriptions are inserted verbatim and adjacent to the code.* If
  MedSynth templates/paraphrases the description into the note prose rather than
  inserting it literally, the anchored match becomes guided-fuzzy rather than
  near-exact. A 5-minute inspection of a few gold records (locate code position,
  read the preceding span) settles it. The redaction rule's own precision/recall
  must be verified before any rerun number is trusted.

**Status:** OPEN, HIGH priority. Gates the first *publishable* number; the D010
0.849 is only a provisional reproducibility result.

## Q9 — graph reranker uses sklearn artifacts pickled under old version (2026-06-01)

During the E-010 hierarchical evaluate, the graph reranker loaded TfidfVectorizer
and TfidfTransformer pickled under sklearn 1.1.2, now running under 1.8.0
("InconsistentVersionWarning ... use at your own risk"). The run completed and
produced E2E 0.838, but the reranker's contribution may be subtly affected by the
version mismatch. This is a concrete instance of the env drift flagged in Q4.

**To resolve:** either re-fit the TF-IDF artifacts under the pinned sklearn, or pin
sklearn to a version compatible with the stored pickles, then re-run evaluate and
confirm the number is stable. Cheap to check; matters for a publishable number.

**Status:** VERIFIED BENIGN (2026-06-01). scispacy 0.5.5 pins no sklearn version; a --dry-run behavioural check showed the UMLS linker produces correct concepts under sklearn 1.8.0 (M25.562 → C0030193 "Pain" @0.97). The InconsistentVersionWarning is precautionary/cosmetic, not breakage. Optional future tidy: pin sklearn to silence the warning, but no correctness issue. Downgraded from low-medium to cosmetic.

---

## Q10 — DSPy/GEPA for hardening the Q8 audit judge (parked, audit-only)

**Idea.** Use DSPy (and an optimizer such as GEPA) with the local LLM stack to
*compile* the Q8 audit prompt rather than hand-write it — optimizing the local
LLM's leak/over-redaction verdicts to better match ground truth, making the audit
*judge* more reliable.

**Hard guardrail (non-negotiable).** DSPy may touch the **audit only**, NEVER the
redaction path. The published number's redaction must stay a deterministic,
re-runnable rule (D011). An optimizer-produced, stochastically-generated prompt
must not become load-bearing for the gold the model trains on — that would be a
reproducibility regression dressed as an improvement. LLM audits; deterministic
rule redacts. This boundary survives DSPy.

**Prerequisite.** Optimizing the judge requires a **human-labeled** gold set
(~150–200 records hand-marked clean/leak_remains/over_redacted). The LLM cannot be
optimized toward its own verdicts (circular). Building that labeled set is the
real cost and the gating task.

**Priority / timing.** LOW, and explicitly **post-first-publishable-number**.
Inserting a prompt-optimization framework now is scope creep that delays the
migration→regenerate→rerun path. The current deterministic dictionary redactor
(~94% clean on fired records, D011) is a stronger basis for a publishable result
than a higher-scoring but non-reproducible LLM redactor.

**Status:** PARKED (2026-06-02). Revisit only after the first publishable number
exists, and only for audit-judge hardening.

---

## Q11 — Repo hygiene gaps surfaced during the Q8 migration (low priority, post-number)

Three real but non-blocking gaps found while migrating the redactor and tracing the
rerun. Logged so they are not forgotten; none blocks the E-015 number.

**(a) No unit test suite.** A project targeting reproducible/publishable numbers has
no automated tests. The redaction logic especially (the v5 redactor: article
tolerance, min-2-token guard, remove-all, placeholder-vs-drop-line, apso rebuild)
is intricate and currently verified only by throwaway in-session scripts. The
config-path bug (resolve_path('data','ontology')) this session would have been
caught instantly by a single test exercising the flag path. The v5 behaviour checks
already written in-session (standalone-drop, placeholder, article-tolerance, guard,
remove-all) are the obvious seed for tests/test_preprocessing.py. Priority: real but
post-first-number.

**(b) run_experiment.py lacks sys.path bootstrap.** Unlike prepare_data.py (which
walks up to artifacts.yaml and inserts PROJECT_ROOT on sys.path), run_experiment.py
does not, so it fails with ModuleNotFoundError: src unless launched with
PYTHONPATH=. Fix: add the same _find_root() bootstrap block for consistency. Minor
papercut.

**(c) Docs vs experiments drift.** The experiments directory contains runs the docs
have no context for: E-010_hybrid_Z, E-012_40ep_ClinicalModernBERT,
E-013_40ep_BioClinicalModernBERT, E-014_SupCon_Z. The decisions/journal record
effectively stops around E-010. These later experiments are undocumented here.
Worth a reconciliation pass (what they were, whether any supersedes the E-009
baseline) — but AFTER the leakage-corrected number, since that is the current
through-line.

**Status:** OPEN, low priority. Revisit after the E-015 number and the README/paper
reconciliation.
