# Notes_to_ICD10 — Project Brief

**Kind:** Persistent orientation document for fresh Claude instances.
**Update style:** Append-only Revision History at the bottom. State sections (1-10) get overwritten when materially changed; the Revision History records what changed and why.
**Read this if:** You are a Claude instance picking up this project cold, with no conversation history. This brief is meant to survive context compaction by living on disk.

---

## 0. If you are a fresh Claude reading this

This brief is the orientation layer, not a replacement for the foundation docs.
Read in this order:

1. **This file, Section 1 (Project-at-a-glance)** — 30 seconds for the shape of
   the project.
2. **`docs/charter.md`** (41 lines) — what the project is for, who it's for, the
   non-negotiables (reproducibility, no leakage, honest reporting).
3. **`docs/philosophy.md`** (69 lines) — how the `docs/` system works: the
   state-vs-history split, the routing rule, the D-ID/Q-ID conventions. Read this
   before writing to any doc.
4. **`docs/decisions.md`** — the append-only decision log, **D001-D017**. The
   most recent, D014-D017 (2026-06-03/04), carry the de-leaked findings that are
   the current headline of the project. **These are more current than the state
   docs** (see the staleness note below).
5. **The last 2-3 dated entries of `docs/journal.md`** (747 lines) — the most
   recent session-by-session context.
6. **This file, Sections 2-10** — for the rest of the orientation, working
   conventions, load-bearing constraints, and honest gaps.
7. **`docs/canonical_pipeline.md`** (237 lines) — when you need to actually run
   the training pipeline. Every experiment name, init source, and hyperparameter
   is source-verified from the notebooks; its "Gotchas" section (§4) is
   load-bearing.

**Important staleness note.** The state docs (`charter.md`, `status.md`,
`architecture.md`, `canonical_pipeline.md`) were written **2026-05-31 to
2026-06-01**, capturing the project at a *provisional, pre-de-leaking* state.
`decisions.md` runs three days further, to **D017 (2026-06-04)**, and captures the
de-leaked rebuild and its headline finding (the backbone-ranking inversion). Where
the state docs and D014-D017 disagree about the current result, **D014-D017 are
authoritative** and this brief's Section 6 reflects them. `status.md` in
particular describes stage-2 as still "RUNNING" — that is stale; the full
de-leaked rebuild has since completed.

That's ~5-8 minutes to functional context. The brief is the map; the foundation
docs are the territory.

---

## 1. Project at a glance

| Aspect | Value |
|---|---|
| **What it is** | Research system predicting billable ICD-10 codes from clinical notes; two-stage hierarchical classifier (chapter router → per-chapter resolver) on Bio_ClinicalBERT, trained on synthetic MedSynth. Intended to support a publication. |
| **Hardware** | MacBook Pro M5 Max, 128 GB unified memory, macOS. MPS is the active backend (CUDA path exists in code, second in priority, not exercised here). |
| **Environment** | Python via `uv` (`uv.lock`); invoke everything as `uv run …`. |
| **Core architecture** | Two-stage hierarchical: Stage-1 22-way chapter router; Stage-2 per-chapter resolvers **warm-started from the flat E-002 model** (the warm start is the project's central architectural finding). 19 trainable chapters; U/P/Q skipped. |
| **Data** | HF dataset `SidneyBishop/notes-to-icd10`, SHA256-locked download, Medallion (silver→gold) build. Gold = `data/gold/medsynth_gold_apso.parquet` (leaky) and `..._deleaked.parquet` (de-leaked). |
| **Stage** | Publication-prep. The de-leaked rebuild is complete and committed; the paper has been rewritten around the corrected findings. **Two final publication runs (leaky + de-leaked, from the seeded pipeline) are planned but NOT yet run.** |
| **Headline finding** | Description-level leakage inflated the headline ~30% AND **inverted the backbone ranking**: de-leaked, Clinical ModernBERT (76.9%) beats Bio_ClinicalBERT (59.2%), the reverse of the leaky benchmark. SupCon "solving Z" was a ~97% leakage artifact. See D015/D016/D017. |
| **Docs convention** | State-vs-history split (`philosophy.md`): state docs (charter, architecture, status, canonical_pipeline) overwritten; history docs (journal, decisions, open_questions) append-only; backlog fluid. D-IDs and Q-IDs monotonic and permanent. |
| **Current work state** | De-leaked rebuild done (E-020-026), paper rewritten (10 sections), orchestrator output-path bug fixed + tested, training now seeded (commit 15e42a0), reproducibility caveat in paper. **In flight:** parameterizing the orchestrator to run BOTH publication configs (leaky + de-leaked) fresh from the seeded pipeline. **Unresolved:** a MIMIC output-path collision (both runs would write the same summary.json) must be fixed before launch. |

---

## 2. Detailed overview

### What this project is

Notes_to_ICD10 predicts billable ICD-10 diagnostic codes from APSO-structured
clinical notes (Assessment, Plan, Subjective, Objective) using a two-stage
hierarchical classifier built on `emilyalsentzer/Bio_ClinicalBERT`. It is trained
and evaluated on the **synthetic** MedSynth dataset; MIMIC-IV real notes are used
only to quantify the synthetic→real gap, never for training. The output is a
reproducible pipeline and a defensible, honestly-reported headline result on the
full ~1,926-code billable set in a low-resource regime (~4 training examples per
code) — intended to support a publication.

### What this project explicitly is not

- **Not a clinical tool.** No live deployment, no coding decisions on real
  patients. A research prototype, not a certified coder (`charter.md`).
- **Not trained on real data.** MIMIC-IV is validation-only (quantifying domain
  shift), subject to PhysioNet access.
- **Not a leaderboard chase.** The goal is a clear, reproducible, honestly-reported
  result — not a state-of-the-art number at any cost.

### The shape of the work

The project's spine is a **leakage story**. The original benchmark redacted
ICD-10 *codes* from note text but left their human-readable *descriptions* (e.g.
"pain in left knee" for M25.562) in place — so the model could read the answer
rather than infer it. The whole recent arc (Q8 → D011/D012 → D014-D017) is:
discover the description leakage, build a deterministic redactor to remove it, and
re-run everything on de-leaked data. The finding is that leakage did not merely
inflate scores — it **inverted an architectural conclusion** and **manufactured an
apparent algorithmic win (SupCon)** that does not exist. The paper's contribution
is therefore a cautionary methodological result about evaluating clinical NLP on
synthetic data.

---

## 3. Core architecture: the two-stage hierarchical classifier + warm start

The single most important architectural finding is that the hierarchical design
**requires a warm-started Stage-2**. The structure:

- **Stage-1 — 22-way chapter router.** Classifies a note to its ICD-10 chapter
  (chapter = first letter of the code). Trained once (in the E-003 notebook /
  experiment) and reused; near-perfect (~95% de-leaked).
- **Stage-2 — per-chapter resolvers.** One classifier per chapter, resolving the
  specific code within that chapter. **Warm-started from the flat E-002 ICD-10
  model.** Cold-started resolvers collapse (~11% E2E, the E-003 failure); warm
  starting is what makes the architecture viable. This lesson survives de-leaking.
- **19 trainable chapters; U/P/Q skipped** (too few records; Stage-1 prediction
  used directly via a fallback).
- **Calibration:** temperature scaling per model (Guo et al. 2017) →
  `temperature.json`, read at inference load.
- **Inference gate:** T-scaled softmax; confidence gate at τ=0.7; low-confidence
  or Z-chapter notes go through a GraphReranker (UMLS graph + Z-phrase
  dictionary), found to have minimal impact on a well-calibrated model.

### The de-leaked headline (what changed the story)

On the **leaky** benchmark, Bio_ClinicalBERT appeared to dominate (0.849 E2E) and
Clinical ModernBERT looked far worse. On **de-leaked** data the ranking
**inverts**: Clinical ModernBERT reaches **76.9%** (E-024), Bio_ClinicalBERT
**59.2%** (E-021), winning 17 of 19 chapters — the entire difference in
within-chapter resolution (routers near-identical). The leaked description signal
was best exploited by the older model, manufacturing a backbone conclusion that is
the opposite of the truth on clean data (D015).

---

## 4. Documentation map

The project follows a **state-vs-history split** (`philosophy.md`). State docs
describe what is true now and are overwritten; history docs are append-only with
supersede-don't-edit discipline.

### State docs (overwritten)

| File | Lines | Purpose | Read when |
|---|---|---|---|
| `docs/charter.md` | 41 | What the project is for, scope, non-negotiables | First, for intent |
| `docs/philosophy.md` | 69 | How the docs system works; routing rule; D/Q conventions | Before writing to any doc |
| `docs/architecture.md` | 151 | How the system is built (data → models → inference) | When wiring components or tracing the pipeline |
| `docs/status.md` | 82 | Work state — **STALE (2026-05-31), predates de-leaking** | For historical mid-refactor state; use Section 6 here for current |
| `docs/canonical_pipeline.md` | 237 | Source-verified training pipeline: names, inits, hyperparams, gotchas | When actually running training/eval |

### History docs (append-only)

| File | Lines | Purpose | Read when |
|---|---|---|---|
| `docs/decisions.md` | 834 | D001-D017, supersede-don't-edit. **D014-D017 carry the de-leaked findings.** | To understand *why* a choice/result is what it is |
| `docs/journal.md` | 747 | Dated session-by-session narrative log | Most recent entries for live state; older for specific findings |
| `docs/open_questions.md` | 320 | Q1-Q11, open and closed inline (resolve by pointing to the settling D-ID) | For "what don't we know" / historical reasoning |

### Fluid docs

| File | Lines | Purpose |
|---|---|---|
| `docs/backlog.md` | 103 | Agreed-but-not-in-flight work; `- [ ]` items; promote to status "Next" when started |

### Repo-root files worth knowing

| File | Purpose |
|---|---|
| `README.md` | Public-facing; still presents the provisional 83.9% headline — treat as less reliable than `decisions.md` for current results |
| `CLAUDE.md` | Project instructions for Claude Code |
| `canonical_pipeline.md` note | The authoritative run order; its §4 Gotchas each cost real debugging time |
| `Prj_Overview.md` (~98 KB) | Large overview document; background, not live state |
| `Run_notes.md` (~24 KB) | Operational run notes |
| `publications/notes_to_icd10/` | The Quarto paper (10 sections + metrics_explainer). Rewritten around the de-leaked findings; renders clean via `quarto render` |

### What to read for which task

| Task | Read first |
|---|---|
| "What's the current state?" | Section 6 here + last 2-3 `journal.md` entries (NOT status.md — it's stale) |
| "Why is X the result / default?" | `docs/decisions.md`, find the relevant D### |
| "How do I run the training pipeline?" | `docs/canonical_pipeline.md` (esp. §3 run order + §4 gotchas) |
| "What's open right now?" | `docs/open_questions.md` (Q1-Q11) + recent journal |
| "What are the de-leaked numbers?" | D014-D017 + the paper's `06_results` / `05_experiments` sections |
| "How is the data built?" | `docs/architecture.md` (Data layer) + `scripts/prepare_data.py` |

---

## 5. Architecture (essentials)

Full picture in `docs/architecture.md`. The essentials:

### Data layer (Medallion)

- **Source:** HF `SidneyBishop/notes-to-icd10`, downloaded by `prepare_data.py`
  with SHA256 verification against pinned hashes (mismatch raises immediately).
- **Phased build:** ingest → CDC billable/non-billable classification → Pydantic
  firewall (`src/gatekeeper.py`) → DuckDB silver → APSO-flip → leakage detection →
  ICD-10 redaction → export gold.
- **Two golds:** `medsynth_gold_apso.parquet` (leaky: codes redacted, descriptions
  retained) and `medsynth_gold_apso_deleaked.parquet` (descriptions also removed;
  ~18.6% residual leak remains — D012).
- **Config authority:** `src/config.py` `ArtifactConfig` singleton resolves paths
  from `artifacts.yaml`.

### Model / pipeline layer

- **Encoder abstraction:** `src/adapters.py` `EncoderAdapter` behind a
  `ModelAdapter` interface — the encoder is a config value, swappable without
  touching training code (this is why the backbone comparison was even possible).
- **Preprocessing:** APSO-Flip (Assessment-first reorder so diagnostic content
  survives 512-token truncation) applied identically at train + inference;
  regex ICD-10 redaction.
- **Training:** `scripts/train.py`, three entry points (flat, hier stage-1, hier
  stage-2). **Now seeds python/numpy/torch at the start of each** (commit
  15e42a0) — previously only the split was seeded.
- **Split:** `_split_dataframe` sorts by a stable key before splitting →
  **content-addressed, reproducible** (the split-determinism fix). Seed 42.
- **Eval:** `scripts/evaluate.py` reuses the production `HierarchicalPredictor`,
  so evaluation mirrors deployment; test set assembled from per-chapter
  `test_split.parquet` files (so the split step directly determines what's
  evaluated).

### Orchestration (the de-leaked rebuild)

- `src/orchestration.py` + `scripts/run_full_deleaked_rebuild.py` — a **tested**
  orchestrator that runs the full chain (flat + hier for 3 backbones, SupCon
  chain, MIMIC) as data (phase specs), with pre/postflight file checks. Replaced
  an untested bash script. Phase-spec output paths are guarded by tests
  (`tests/test_orchestration_output_paths.py`) after a false-MISSING bug.

---

## 6. Current state (as of 2026-06-04, more current than status.md)

### The de-leaked rebuild — complete, the headline of the project

The leakage arc (Q8 → D011/D012 → D014 → D015-D017) is resolved. Verified
de-leaked numbers, content-addressed split, N=966 test:

| Finding | Detail | Decision |
|---|---|---|
| Leakage inflated the headline ~30% | ClinicalBERT E2E 0.849 (leaky) → **0.592** (de-leaked, E-021) | D014/D015 |
| **Backbone ranking INVERTED** | Clinical ModernBERT **0.769** (E-024) > ClinicalBERT 0.592; ModernBERT wins 17/19 chapters | **D015 (headline)** |
| SupCon "solves Z" was ~97% artifact | Leaky +21.2pp → de-leaked **+0.5pp** at system level (E-022) | D016 |
| No real-world generalisation | MIMIC-IV de-leaked **8.1%** E2E; domain shift dominates, independent of leakage | D017 |

The paper (`publications/notes_to_icd10/`, 10 sections) has been fully rewritten
around these findings and renders clean. A reproducibility/run-variance limitation
has been added noting the figures are single runs on a fixed seeded split.

### In flight — the two final publication runs (NOT yet run)

The plan (decided with the user): perform **two fresh end-to-end runs from the now-
seeded pipeline** — (1) code-only redaction (leaky baseline), (2) code +
description redaction (de-leaked) — whose numbers go into the publication. Option
"two fresh runs, fresh parallel experiment series" was chosen so both runs come
from the *identical* seeded pipeline and differ only in the redaction level (the
cleanest A/B). Proposed naming: `E-04x_*_Leaky` and `E-05x_*_Deleaked`.

**Work done this session toward it:**
- Orchestrator parameterization (a `RunConfig` with `PRESET_LEAKY` /
  `PRESET_DELEAKED`) drafted so `build_phase_specs` can produce either config —
  **drafted in workspace, not yet delivered/committed.**
- Confirmed both gold files exist; ~842 GB free disk (two runs ~140 GB, fits).

**Blocking issue — must fix before launch:**
- **MIMIC output-path collision.** `validate_mimic_evaluate.py` writes to a fixed
  `outputs/evaluations/mimic_iv_validation/summary.json`. Both runs would write
  the *same* file → the second overwrites the first's MIMIC result. This must be
  resolved (parameterize the output dir per run, or move results aside between
  runs) BEFORE the two publication runs, or one MIMIC number is silently lost.
  Investigation left at: read the script's output-path logic.

### Top priorities (in order)

1. **Resolve the MIMIC output-path collision**, then finish + test the
   orchestrator parameterization (test-first, incl. a "presets differ only in gold
   + names" test).
2. **Dry-run BOTH presets** (`--dry-run`) — validates the full chain without GPU.
3. **Launch run 1 (leaky), then run 2 (de-leaked).** Only after all tests green +
   both dry-runs clean.
4. Propagate the two runs' numbers into the paper from one source of truth.

---

## 7. Important constraints (load-bearing — don't break by accident)

### The leakage story (the project's spine)

1. **Description leakage, not just code leakage.** The certified redaction removed
   ICD-10 *codes* but left *descriptions* ("pain in left knee" for M25.562). That
   is the leak the whole study corrects. Guarding against verbatim code leakage is
   NOT sufficient (D011/D012, and the paper's dataset section).
2. **Two golds, never mix them.** `medsynth_gold_apso.parquet` = leaky (code-only
   redaction); `medsynth_gold_apso_deleaked.parquet` = de-leaked. Numbers from one
   must never be compared to the other as if the same quantity.
3. **~18.6% residual leak remains** in the de-leaked gold (D012) — de-leaked
   numbers slightly *over-estimate* a perfectly clean ceiling. State this; don't
   drop it.

### Pipeline correctness (each cost real debugging time — from canonical_pipeline §4)

4. **`--stage1-experiment` is `E-003_Hierarchical_ICD10`**, not the stage-2 name —
   the router is owned by E-003 and reused.
5. **`--stage2-init` points at the experiment ROOT, not its `model/` dir.** Passing
   `.../model` resolves to `.../model/model`, fails, and silently cold-starts each
   resolver from base BERT (the 12.7% failure mode).
6. **`--batch-size` and `--epochs` must be passed explicitly.** CLI defaults (8,
   10) match no trained regime. Canonical: E-001=30, E-002=40, Stage-1=5,
   Stage-2=20; batch 16 throughout.
7. **`--code-filter` differs by experiment:** E-001 = `all`; E-002/Stage-1/Stage-2
   = `billable`.
8. **`build_graph.py` is required before evaluate, and its `--gold-path` default is
   wrong** — pass the gold path explicitly.
9. **`--use-presplit` is Stage-2 only** and needs `prepare_splits.py` to have run
   under the same experiment, or it silently self-splits.

### Reproducibility

10. **The split is content-addressed** (sorted by id before splitting) → identical
    partitions across models, which is what makes the backbone A/B exact. Don't
    reintroduce position-dependent splitting.
11. **Training is now seeded** (python/numpy/torch, commit 15e42a0). Reproducible
    up to MPS kernel non-determinism (~0.2-0.3pp). The committed paper numbers
    predate seeding → they are single runs, not multi-seed means (disclosed in the
    paper's limitations).

### Naming

12. **Canonical name for the best *original* hierarchical model is `E-009`**, not
    `E-010` (E-010 is naming drift per canonical_pipeline.md §1). The de-leaked
    rebuild used a fresh `E-02x_*_Deleaked` series; the planned publication runs
    will use `E-04x_Leaky` / `E-05x_Deleaked`.

### zsh / tooling traps (from the user's conventions)

13. **zsh `!` in double quotes triggers history expansion** ("event not found").
    Use single quotes when echoing strings with `!` (esp. `.gitignore`
    carve-outs).
14. **Avoid heredocs for long or fence-containing content** — they break on zsh
    paste. Deliver long content as a downloadable file, then `cat file >> target`.
15. **Anchored patches fail on invisible whitespace** (blank lines inside a
    multi-line anchor). Prefer single-line anchors on unique lines, and ALWAYS
    verify a patch took (`grep -c`) — a patch reporting success is not proof it
    applied. For prose edits, uploading the file for direct edit is more reliable
    than anchored patching.

---

## 8. Working conventions

How the project actually operates (from `philosophy.md` and observed practice).

### Documentation discipline

- **State-vs-history split, applied strictly** (the routing rule in
  `philosophy.md`): facts → state files; reasoning → decisions (D-ID); events →
  journal (dated); unresolved → open_questions (Q-ID); queued → backlog.
- **Append-only on decisions.md, journal.md, open_questions.md.** Never edit past
  entries. Supersede a decision with a new D-ID that references the old; mark the
  old superseded. D-IDs and Q-IDs are monotonic and permanent — never renumber.
- **Decision entry shape:** Context / Decision / Rationale / Trade-offs / When to
  revisit.
- **Date every append-only entry** (ISO `YYYY-MM-DD`).
- **No retrospective fabrication.** Do not invent D-IDs or journal entries for
  choices made before they were recorded. Mark inferred-not-verified content as
  such ("unconfirmed: …") rather than stating it confidently.
- **Update docs in the same commit as the code** that motivates them.

### Working with the user

- **One step at a time.** Deliver one command / one file per step and wait for the
  result before moving on. Do not run ahead with next steps while the user is still
  executing the current one.
- **Never dump large files into the terminal.** No asking the user to `cat` a whole
  file. Use targeted `grep -n` / `sed -n` ranges, or have them open in an editor.
  Always use `git --no-pager diff` (never bare `git diff` → pager trap).
- **Commit explicitly-named files only** — never `git add .` or `git add outputs/`
  (training churn stays unstaged). Verify what's staged before committing.
- **Prefer file downloads for content > ~20 lines** rather than inline blocks.
  For editing existing files, uploading them for direct edit beats anchored
  patches.
- **Certainty and accountability over speed.** Read source before acting; verify
  before long runs; verify a change took after making it.

### Publication integrity

- **Numbers come from one generated source of truth**, not hand-transcription;
  discrepancies are documented, not smoothed (`charter.md` non-negotiable).
- **Verify arithmetic and numbers against source files** before asserting them in
  the paper — do not reconstruct from memory. (Fabricated numbers have been a real
  failure mode across the user's projects.)

---

## 9. What this document does NOT capture

Honest limits, so a future Claude doesn't treat inferred claims as confirmed.

1. **The full journal is not summarised.** Sections here draw on the foundation
   docs (charter, philosophy, architecture, status, canonical_pipeline,
   decisions through D017) and the recent de-leaking work. Individual dated
   journal entries pre-2026-06-03 are referenced via decisions, not reproduced. If
   a specific date matters, read `journal.md` directly.
2. **`status.md` is stale (2026-05-31) and this brief supersedes it for current
   state** — but status.md's *branch list* and mid-refactor detail are not
   reproduced here. Read it for that historical detail.
3. **The two publication runs have NOT been executed.** Everything in Section 6
   about them is plan + in-progress work, not results. The E-04x/E-05x names are
   proposed, not yet created on disk.
4. **The orchestrator parameterization is drafted in a working session, not
   committed.** As of this brief, `src/orchestration.py` on disk is the
   single-config (de-leaked) version with the output-path fixes; the
   `RunConfig`/preset parameterization is not yet in the repo.
5. **The MIMIC output-path collision is identified but unresolved.** The fix
   depends on `validate_mimic_evaluate.py`'s actual output logic, not yet read.
6. **`open_questions.md` Q1-Q11 are not individually summarised.** Q8 (description
   redaction) is the spine and is resolved via D011-D017; Q1 (record/code count),
   Q2 (DVC remote), Q3 (orphaned ontology artifact) remain open reproducibility
   items. Read the file for the rest.
7. **Foundation-doc line counts** are from the 2026-06-04 inventory; they drift as
   docs are edited.
8. **Decisions.md covers D001-D017.** D018+ will need a brief refresh (Section 10).

If a foundation-doc claim in this brief seems off, **the foundation doc is
authoritative** — correct the brief, don't propagate the brief's error.

---

## 10. Maintenance instructions

- **Refresh this brief when:** a D-decision closes (D018+); the two publication
  runs complete (Section 6 becomes results, not plan); the orchestrator
  parameterization lands; the MIMIC collision is resolved; the doc structure
  changes; the paper's headline numbers change.
- **Don't refresh for:** every journal entry; routine output churn; minor backlog
  progress.
- **How to refresh:** overwrite the relevant state section(s) in place; append a
  new entry to Section 11 (Revision History). Commit with a descriptive message.
- **Append-only discipline applies to Section 11 only.** Sections 0-10 can be
  edited in place; Section 11 is the audit trail. This mirrors the project's own
  state-vs-history split (`philosophy.md`).

---

## 11. Revision history

### v1.0 — 2026-06-04

**Author:** Claude (Anthropic), drafted at the user's request during the
publication-prep session, using the persistent-on-disk-brief discipline described
in `docs/model_drift_mitigation.md`.
**Created in response to:** the user asking for a project brief (and a companion
methodology guide) modelled on the versions from another project (tst_llm), during
a long session that had itself crossed a context-compaction boundary — the exact
failure mode the brief exists to mitigate.
**Method:** Full inventory of the nine `docs/` foundation files (charter,
philosophy, architecture, status, canonical_pipeline, backlog, open_questions
headers, decisions headers D001-D017, journal line count) read directly from the
uploaded files, plus the current session's working memory of the de-leaked
rebuild, paper rewrite, orchestrator fixes, seeding fix, and the in-flight
two-publication-runs plan.
**Verification status:** Foundation-doc claims (conventions, architecture, pipeline
gotchas, decision headers) cross-checked against the uploaded docs. The de-leaked
numbers (0.592, 0.769, 8.1%, +0.5pp) are from this session's committed work and
D015-D017. The in-flight-work claims (orchestrator parameterization, MIMIC
collision) are the originating session's own working memory and are explicitly
marked as not-yet-committed in Sections 6 and 9. The staleness of status.md
relative to decisions.md was verified by date comparison (status 2026-05-31 vs
D017 2026-06-04).
**Known gaps:** recorded in Section 9.
