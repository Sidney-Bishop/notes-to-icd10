# Journal

Append-only, dated log of what actually happened — bugs hit, surprises, dead ends,
operational facts that never make it into polished docs. Newest entries at the
bottom. Do not summarise after the fact; write in the moment.

---

## 2026-05-31 — Refactor kickoff: documentation + split investigation

**Why we're here.** Returned to the project in a state of confusion: the same
model (E-010) was reporting different headline numbers across README,
REFACTORING_PLAN, and the Quarto publication. Could not form a clean mental model
without re-reading code and re-running steps. Decided to establish the `docs/`
system and revisit the code ground-up.

**Test-set size mystery — resolved.** Found three different test-set sizes in the
repo: 966 (committed `summary.json`, 30 Apr), 972 (working-tree `summary.json`,
10 May, preserved at branch `snapshot/2026-05-10-eval` commit `9a73669`), and
1,030 (fresh `prepare_splits.py` run). Root cause: billable filter lived only in
`train.py`, not `prepare_splits.py`. The split self-check validated against the
unfiltered 10,240 and printed "✅ matches" at 1,030 — a green check on the wrong
population. Fixed (see D001/D002). Verified fix produces 971.

**Also found in the same diff:** the 10 May run changed the skip-chapter fallback
codes (P22.1/Q23.1 → P22.0/Q90.9), so the two historical runs differ in model
behaviour, not just test set.

**Dataset count discrepancy — open.** `03_dataset.qmd` carries two figures:
9,660 records / 1,926 codes (reported) and 9,578 records / 1,914 codes
(post-Pydantic-validation). Not yet reconciled (see `open_questions.md`).

**Fresh-clone reproducibility test — two gaps found.** Cloned the repo clean to
test the "clone → dvc pull → byte-identical data" claim:
1. The DVC default remote is committed as a placeholder `.dvc/CONFIGURE_LOCAL`;
   the real remote URL lives in `.dvc/config.local`, which is gitignored and does
   NOT come down with a clone. So a fresh clone cannot locate the data without
   manual `dvc remote modify --local`.
2. After pointing at the real local store, 3 of 4 missing files pulled, but
   `data/ontology/icd10cm_2026.parquet` is in neither cache nor remote — its bytes
   were `dvc add`ed locally but never pushed. The file still exists in the
   original working directory. **Operational fact:** nothing in `scripts/` or
   `src/` references `icd10cm_2026` — it appears to be an orphaned tracked artifact
   with no code consumer.

**Operational facts worth remembering:**
- `dvc` is not on the bare shell; must be invoked as `uv run dvc …` (uv-managed env).
- A git branch-switch triggers a graphify hook that rebuilds a large knowledge
  graph and prints noise; harmless, ignore it.
- The split `*.parquet` files are gitignored via `.gitignore:175`
  (`outputs/evaluations/*/*/*/*.parquet`) — a plain `git add` silently no-ops on
  them; freezing them needs DVC or `git add -f`.
- `uv sync` on a fresh clone resolved to newer libs than the paper's basis
  (transformers 5.9, torch 2.12; sklearn drifted 1.1.2→1.8.0 historically). This
  is the environment-regression risk the planned reproduction gate guards against.

**Where it was left.** Split fix + decisions log committed on branch
`fix/splits-billable-filter` (commits `a990306`, `df3a4d4`) in the original repo.
Fresh clone `notes-to-icd10-fresh` left partially pulled (ontology file missing).
Documentation system being established now.

**Code-read verification pass (architecture.md).** Read live code on disk to
confirm/correct the inferred items in `architecture.md`:
- **CUDA:** `inference.py` does explicit MPS→CUDA→CPU detection — the CUDA path is
  real code, just second-priority behind MPS. Earlier "expected but untested"
  framing was wrong; corrected. Runs on M5 Max (128 GB) → MPS active.
- **Serving:** confirmed live, not a stub. `scripts/serve.py` = uvicorn launcher;
  `src/server.py` = the FastAPI app/routes. Corrected the "unconfirmed how current".
- **Warm start:** confirmed in `experiments.json` — `E-010_40ep_E002Init` has
  `stage2_init: …/E-002_FullICD10_ClinicalBERT`; E-002 trained from `none`.

**Half-applied state found — splits regenerated to 971 but eval NOT re-run.** The
E-010 `summary.json` on disk is still the historical 2026-05-06 run
(966 / 83.9% / F1 0.7628). The billable-filter split regen (D001/D002) produced
971 test records, but `evaluate.py` was never re-run on it. So the splits (971)
and the eval numbers (966-regime) are out of step on disk. This is exactly the
kind of partial state that causes the "which number is real" confusion — captured
in architecture.md and tracked in status.md. **Next session must not quote 83.9%
as a 971-regime result.** Resolve by re-running evaluate on the 971 split.

## 2026-05-31 — Attempted the 971-regime eval; hit a broken stage-1 artifact

Tried to execute status #0 (re-run evaluate.py on the 971 split). Did NOT get a
number. The attempt surfaced a chain of issues, each ruling out the last:

1. **First failure:** `evaluate.py` died loading the Stage-1 tokenizer —
   "Couldn't instantiate the backend tokenizer … need sentencepiece or tiktoken."
2. **Red herring:** installed `sentencepiece` (`uv add sentencepiece` →
   0.2.1). Re-ran — identical failure. So sentencepiece was NOT the cause; the
   error message named it but the real problem was elsewhere. (We now carry an
   extra dep we didn't need; harmless, note for cleanup.)
3. **Path resolution dead-end:** traced to `_find_model_dir` in `paths.py`. It
   returns the first candidate dir containing `model.safetensors`, checking
   `[root, root/model, root/model/model]` in order.
4. **Root cause — split artifact:** the E-003 stage-1 model is split across two
   directories. `stage1/model.safetensors` (433 MB, weights) at top level;
   `stage1/model/` has `config.json` + `tokenizer.json` + `tokenizer_config.json`
   (config + tokenizer) but NO weights. So `_find_model_dir` returns top-level
   `stage1/` (weights, no tokenizer) → tokenizer load fails. Neither directory is
   a complete loadable model.
5. **Unbacked:** the weights are gitignored (`.gitignore:78`, `*.safetensors`)
   and not in DVC (no `.dvc` pointer). The 433 MB file exists only on this disk,
   tracked by nothing. Last touched by commit `5537ec6` "capture current
   experiment state before refactor branch".

**Decision (D004):** do NOT shuffle files to force a load. A number from a model
this broken and unbacked would be untrustworthy — the opposite of the goal. Stop,
document, retrain stage-1 cleanly later (Q7, marked BLOCKING).

**Operational facts for next time:**
- `.gitignore:78` ignores `*.safetensors`; `.gitignore:175` ignores
  `outputs/evaluations/*/*/*/*.parquet`. Model + split artifacts are not in git.
- `_find_model_dir` selects on `model.safetensors` presence ALONE — it does not
  require config/tokenizer in the same dir, so a split layout mis-resolves silently.
- `evaluate.py` reads `test_split.parquet` from disk (does its own NO filtering) —
  confirmed clean; it is not a source of the split-size bug.
- Polars 1.39.2 in the lockfile is YANKED upstream (uv warned on `uv add`). Note
  under Q4 env-drift.

**Net:** the documentation system did its job — the goal was "get the 971 number"
and instead we found the stage-1 artifact can't be trusted. Better found now.

## 2026-05-31 (cont.) — Read notebooks; redaction + gold findings before retrain

Read the EDA notebook (01) and the pipeline overview properly (not second-hand).
Key findings, all verified against the files:

**Redaction advocates-but-doesn't-implement.** EDA cell 52 ("Forensic Alert")
argues forcefully that BOTH ICD-10 codes AND semantic diagnosis labels must be
redacted — "a prerequisite for a valid training run." But the redaction code cells
(53/56/57) are EMPTY, and the gold data still contains the semantic labels. So the
notebook's stated design and its actual output diverge. What ships is code-only
redaction; descriptions like "pain in left knee" remain in the model input. This
is residual label leakage, now documented as a known caveat (D005) with the fix
deferred (Q8). Decided to retrain on the code-only data anyway, as a
reproducibility check against historical numbers — NOT for a publishable figure.

**Two gold files reconciled.** `medsynth_gold_apso.parquet` (May-10) vs
`..._20260505_194721.parquet` (May-5): same 10,240 records, but ~1,539 rows differ
in `apso_note`/`assessment`. Cause: the revised redaction regex (commits
e46d089/502198f) was applied to May-10 but not May-5. The DVC-tracked file is the
OLDER leakier May-5; the corrected May-10 is untracked. Decided May-10 canonical,
to be re-tracked in DVC (D006).

**Q3 corrected twice.** The "orphaned ontology file" `icd10cm_2026.parquet`: first
called orphan, then over-corrected to "is read." Verified truth: the file the
pipeline actually reads for CDC/ICD-10 validation is `cdc_fy2026_icd10.parquet`
(EDA Phase 1b), a DIFFERENT, near-identically-named file that is tracked and
present. `icd10cm_2026.parquet` has NO confirmed consumer in the notebook or any
script searched (notebooks 02-05 not yet checked). Lesson: two similarly-named
reference files were conflated; verify the exact filename, don't pattern-match.

**Method note:** caught myself (twice, on user challenge) confirming claims I
hadn't verified. Re-grepped each before writing. The docs now state only what was
checked against files, with "unconfirmed" where 02-05 weren't available.

## 2026-05-31 (cont.) — Regenerated gold from verified raw; deterministic, clean

Acted on the "regenerate gold from HF raw rather than retrain on a stored file"
decision (D006). Results — a run of POSITIVE findings after a day of problems:

**Raw inputs verified canonical.** `shasum -a 256` on both local raw files matched
`prepare_data.py`'s pinned SHA256s exactly:
- `icd10_notes.parquet` → 7fa03f...5ac8 ✓
- `cdc_fy2026_icd10.parquet` → 2433ad...b93d ✓
So local raw == HF-published canonical raw, byte-for-byte. (HF dataset confirmed
via screenshots: `data/medsynth/icd10_notes.parquet` 19.8 MB +
`data/reference/cdc_fy2026_icd10.parquet` 1.4 MB; gold not published, by design.)

**Regeneration succeeded and is clean.** `prepare_data.py` rebuilt gold:
10,240 records, 9,660 billable (60 noisy_111 / 25 placeholder_x / 495 invalid),
0 bare ICD-10 codes in `apso_note`, 355 `[REDACTED]` markers.

**Deterministic reproduction confirmed.** Regenerated gold vs the May-5 baseline
differs ONLY in redaction-touched columns — `assessment` 1,530, `apso_note` 1,539,
+ minor subjective/objective/plan (39/34/51) — the exact fingerprint of the
revised-regex fix, no other drift, no schema change. So current code + verified
raw reproduces the May-10 (corrected) regime. The 971 split regenerated
identically from the fresh gold. → raw→gold→split is demonstrably deterministic.
This is the reproducibility claim holding, shown not asserted.

**Operational facts:**
- `prepare_data.py` runs redaction (Phase 3c) SILENTLY — no phase header printed,
  which made it look skipped. Verified it ran via the 355 [REDACTED] markers + 0
  bare codes. Don't infer "phase didn't run" from a missing header.
- `code_status` label is `invalid` in the regenerated gold, vs
  `invalid_or_malformed` in older outputs/notebook. Cosmetic relabel; billable
  count unaffected. Noted to avoid a future "which is it" confusion.
- The gold regeneration OVERWROTE the May-10 working file (same path). May-10 is
  gone (gitignored, never committed); only the May-5 timestamped file remains for
  comparison. Not a loss — May-10 is reproducible by re-running, which is the point.

**Reminder still standing:** the gold is code-only-redacted (D005). The retrain
about to run produces a PROVISIONAL number (reproducibility check vs historical
83.9/85.8/77.2%), not a publishable one. Semantic-label redaction is Q8.

## 2026-05-31 (cont.) — Root cause of Q7 found in train.py (the save-path bug)

Before launching the retrain, read train.py's save logic (gate-check) — and found
the actual cause of the broken stage-1 layout. It is NOT damage; train.py produces
it. All three training paths (flat ~378, stage-1 ~472, stage-2 ~636) do:

    model_dir = <parent> / "model"
    adapter.train(..., model_dir)      # weights → model_dir  ✓
    adapter.save(<parent>)             # writes FLAT into <parent>  ✗
    _finalize_model_dir(model_dir, …)  # config/tokenizer → model_dir

Inline comments say "save to parent, adapter adds /model" — but EncoderAdapter.save
does NOT append /model (its docstring: "FLAT — no nested model/ folder"). So save
dumps weights in the parent while config+tokenizer end up in model/ → split,
unloadable. Every model this trainer produced has the same flaw; explains why
_find_model_dir needs FLAT/SINGLE/NESTED auto-detection.

Fix (D007): change the three `adapter.save(<parent>)` → `adapter.save(model_dir)`.
One conceptual error, three sites. This is the keystone finding — the real thing
behind Q7 and the layout chaos.

**Lesson reinforced:** the gate-check (read the save path before a multi-hour run)
caught a bug that would have wasted the entire retrain — it would have produced
another unloadable model. Reading beats launching-and-hoping, again.

**Sequence now:** apply D007 edit to train.py (on branch) → retrain stage-1 →
verify single complete model/ dir → then stage-2 + evaluate.

## 2026-05-31 (cont.) — D007 verified; Q7 RESOLVED; stage-1 retrained clean

Gate run (stage-1 only, billable, from base Bio_ClinicalBERT) — passed on all counts:

- **Layout fixed (D007 works):** `stage1/model/` now holds config + tokenizer +
  model.safetensors + label_map together; NO weights stranded at `stage1/` top
  level. The exact split that caused Q7 is gone.
- **Loads clean:** AutoTokenizer + AutoModelForSequenceClassification both load
  `stage1/model/` (22 labels) — the operation that previously threw the tokenizer
  error. Q7 definitively closed.
- **Trained healthily:** ~31 min, best epoch 3 (val_acc 0.920, macro_f1 0.933),
  final epoch 6 val_acc 0.935, top-5 0.983. Loss 3.15 → ~0.06. Trained from BASE
  (not E-001-init), so routing slightly below historical ~96% — expected.

**Two benign warnings logged (not chased):**
- `cls.predictions.* UNEXPECTED / classifier.* MISSING` on load — normal when
  loading a base MLM checkpoint into a sequence-classification head; the classifier
  head is newly initialised by design.
- `LayerNorm.beta/gamma` missing/unexpected keys — legacy BERT key naming
  (pre-2020 `gamma/beta` vs modern `weight/bias`). Something in the chain carries a
  legacy-format checkpoint. Didn't break anything; worth knowing if it recurs.

**Gate-first discipline paid off again:** proved the fix on a 31-min stage-1 run
before committing to the multi-hour stage-2. Had D007 been wrong, we'd have found
out in 31 min, not 3 hours.

**Cleared for:** full stage-2 retrain (per-chapter resolvers, warm-started from
E-002) → calibrate → evaluate on the 971 split → PROVISIONAL number (D005).

## 2026-05-31 (cont.) — E-002 also has the D007 split layout; retrain order established

Checked E-002 (the 40-epoch flat model that stage-2 resolvers warm-start from)
before launching stage-2. It has the SAME D007 split bug: `model.safetensors`
stranded at the `E-002_FullICD10_ClinicalBERT/` top level, NO `model/` subdir,
not even a config/tokenizer present — just the weights blob + a test_split.
So E-002 cannot be loaded, and stage-2's `--stage2-init` would fail on it.

**Remaining run order (now known, so next session need not rediscover):**
1. Retrain E-002 — `train.py --experiment E-002_FullICD10_ClinicalBERT --mode flat
   --code-filter billable` (40 epochs, the big run — hours on M5). Writes correct
   `model/` layout now (D007 fix applies to the flat path, train.py line ~381).
2. Retrain stage-2 resolvers, `--stage2-init` from the freshly-retrained E-002.
3. Calibrate.
4. Evaluate on the 971 split → PROVISIONAL number (D005).

Stage-1 is already done and verified (Q7 resolved). The blockers are fixed; what
remains is compute on an unblocked pipeline.

## 2026-05-31 (cont.) — E-002 retrained clean (30ep, converged); stage-2 hyperparams verified; stage-2 launched

**E-002 retrains.** First retrain accidentally ran at the CLI-default 10 epochs
(--epochs not passed) → val_acc 0.316, badly undertrained, still climbing at
cutoff. ~52 min wasted. Rerun at --epochs 30 → val_acc 0.841 / macro_f1 0.759,
best epoch 27, PLATEAUED (flat 0.839 across ep28/29/30). Layout verified correct
(D007): config + tokenizer + model.safetensors (1926 labels) together in model/.

**Decision: do NOT rerun E-002 at 40.** Historical was 40 epochs (notebook 05:
E002_ACCURACY=0.7329, "40-epoch flat ICD-10, test set"), but our 30-epoch run is
converged — the extra 10 epochs would gain ~nothing on a plateaued model. Treating
30 as sufficient; if a reviewer asks, the answer is "converged by epoch 27."

**Stage-2 hyperparameters verified from notebook 05** (the experiment-defining
notebook), NOT guessed:
- stage2_num_epochs: 20   ← "20 epochs with E-002 init is near-optimal"
- stage2_learning_rate: 2e-5
- stage2_batch_size: 16
- warmup_ratio: 0.1
- skip_chapters: [U, P, Q]
- stage2_init_model: E-002_FullICD10_ClinicalBERT
(lr/batch/warmup/skip are already train.py defaults; only --epochs 20 and
--stage2-init must be passed explicitly.)

**Stage-2 epoch CONFLICT found + resolved.** notebook 05 says 20; Prj_Overview.md:986
says epochs_stage2: 10. Resolved to 20 — the experiment-defining notebook and the
publication ("20-epoch version", 05_experiments.qmd:365) both say 20; Prj_Overview
is a summary doc and is treated as stale on this point.

**Stage-2 launched:** train.py --experiment E-010_40ep_E002Init --mode hierarchical
--stage 2 --stage2-init outputs/evaluations/E-002_FullICD10_ClinicalBERT
--code-filter billable --epochs 20. Warm-starts 19 resolvers from E-002; skips P/Q/U.

**Known caveat carried into the final number:** stage-1 router was trained from
BASE Bio_ClinicalBERT (val 0.935), not E-001-init (historical stage1_accuracy=0.9640).
~3% router gap WILL propagate. The forthcoming hierarchical number is therefore
expected to land somewhat below historical 83.9/85.8 for this reason alone, on top
of the D005 leakage caveat. Decision deferred: accept the gap (proceed) vs retrain
stage-1 from E-001 (needs a loadable E-001, which likely has its own D007 split).

**Operational lesson (logged honestly).** Hyperparameters must be read from the
authoritative source (notebook 05) before proposing any command — never inferred
from memory or experiment names. The CLI default (--epochs 10) silently overrides
artifacts.yaml (3) and does NOT match any historical regime. Two misses today traced
to not reading first: the wasted 10-epoch E-002 run (~52 min real cost), and an
initial "stage-2 = 10 epochs" suggestion (caught at command-confirmation, zero
compute cost). Standing rule going forward: cite the source line for every
hyperparameter.

## 2026-06-01 — PROVISIONAL NUMBER PRODUCED: E2E 0.838 on the 971 split (D005 regime)

Full pipeline ran end to end for the first time since the rebuild:
- Stage-2: 19 resolvers trained (--epochs 20, warm-started from 30-epoch E-002),
  3 skipped (P/Q/U). All chapter model/ layouts verified (D007 holds).
- Calibrate: stage-1 + 19 resolvers, ECE avg 0.658→0.095, stage-1 T=1.2549.
  CRITICAL path fix: --stage1-experiment must be E-010_40ep_E002Init, NOT the
  default E-003_Hierarchical_ICD10 (whose stage-1/model/ is the D007-broken one —
  config+tokenizer present, NO model.safetensors/label_map). Same trap in evaluate.
- Evaluate (971 split, hierarchical): **E2E 0.838 | macro F1 0.766 | ECE 0.0487**.
  Stage-1 chapter acc 0.952 (historical 0.964 — only ~1% gap, better than the ~3%
  feared from base-init). Coverage@0.7 86.0% (acc 0.917 on covered).

**This REPRODUCES the historical 83.9%** — the rebuilt, deterministic pipeline
lands on the number the project always claimed. Reproducibility check: PASS.

**Caveats (why this is PROVISIONAL, not publishable):**
- D005 regime: code-only redaction, semantic labels RETAINED → biased UPWARD by
  residual leakage. The publishable number needs semantic-label redaction (Q8) and
  will be lower.
- It matches history precisely because history carried the same leakage.
- Env drift touched the graph reranker: sklearn TfidfVectorizer/TfidfTransformer
  unpickled from 1.1.2 under 1.8.0 ("use at your own risk" warnings). Ran, but the
  reranker's contribution may be subtly off under new sklearn. New open question.

## 2026-06-01 (cont.) — pipeline diagram drafted (core path only)

Drafted a README pipeline section (docs/pipeline_readme_section.md): Mermaid
diagram + the 7 verified commands (prepare_data → prepare_splits →
train flat/stage1/stage2 → calibrate → evaluate) + a gotchas list baking in the
--epochs and --stage1-experiment traps we hit. Every argument verified by running
or reading code this session — NOT inferred.

Deliberately scoped to the CORE reproduction path. Sibling/optional entry points
(graph reranker fit, E-014 SupCon/hybrid, E-012/E-013 ModernBERT, MIMIC-IV
validation, serve.py) are NOT in the diagram — they exist in the repo but weren't
run/read this session, so drawing them would be guesswork. Logged to backlog with
the explicit rule: verify argparse + I/O before adding any of them.

## 2026-06-01 (cont.) — all five notebooks read; canonical naming settled (E-009)

Read notebooks 01–05 in full (cfg/path/training cells verbatim, not skimmed),
plus the overview doc. Pinned the naming and transfer-learning chain from source:
E-001 (30ep) → E-002 (40ep); E-003 trains Stage-1 once (init from E-001, 5ep) +
cold-start Stage-2 (12.7% failure); E-009 (nb05) loads Stage-1 from E-003 and
inits Stage-2 from E-002 (20ep).

Key correction: **E-010 is not a real experiment** — no notebook defines it; it's
drift in the README/run-dirs. Canonical hierarchical experiment = E-009 (nb05
cfg). Stage-1 is owned by E-003 and reused, so --stage1-experiment =
E-003_Hierarchical_ICD10 in calibrate/evaluate.

Wrote docs/canonical_pipeline.md — the source-exact, sequential run order (incl.
the required build_graph.py step and the E-001→E-002→E-003→E-009 chain) with
every hyperparameter from the cfg dicts. Logged D009. This supersedes D008's
naming (the session's E-010-throughout run was a deviation). Next: re-run the
whole pipeline end-to-end under canonical E-009 naming, then reconcile README +
supersede D008's number.

## 2026-06-01 (cont.) — full command chain certified against source before re-run

Read train.py argparse + data paths, prepare_splits.py, build_graph.py,
verify_scripts.py, and re-read notebook 02 cell 16 to certify EVERY flag/path in
the canonical run BEFORE committing compute. Findings that would have silently
broken the run, now fixed in canonical_pipeline.md §3:
- **--stage2-init must be the experiment ROOT** (.../E-002_FullICD10_ClinicalBERT),
  NOT .../model. The code appends /model via a candidate path; passing /model →
  /model/model → no match → silent cold-start of all 19 resolvers from base BERT
  (the E-003 12.7% failure). Highest-risk catch.
- **--stage1-init is REQUIRED** on the E-003 stage-1 step
  (.../E-001_Baseline_ICD3/model) or stage-1 cold-starts from base BERT.
- **--batch-size 16 must be explicit** everywhere (CLI default is 8; notebooks use 16).
- **--code-filter differs by experiment, verified from each notebook's own cfg:**
  E-001 = all (nb02 cell 16: filter commented out, audit trail "all");
  E-002 = billable (nb03 cfg); stages 1/2 = billable.
- **--use-presplit is Stage-2 only;** flat + stage-1 self-split (seed 42). Stage-2
  presplit requires prepare_splits.py to have run under the SAME experiment (E-009).
- E-001 has NO weights on disk (only PNGs/label map) → must be trained; it seeds
  the stage-1 router. The old registry E-003 stage-1 (May 1) predates D006/D007
  and the gold regen, so it is NOT reused — full chain retrained for provenance.

Also moved two D007 stranded top-level safetensors aside (E-002/, E-003/stage1/)
to /tmp/d007_stranded so the re-run writes clean dirs with no load ambiguity.

Chain is now fully certified. Ready to launch the canonical E-009 re-run.

## 2026-06-01 (cont.) — data+graph gate PASSED; two hazards caught & cleared

Ran steps 0–3 of the canonical pipeline (verify_scripts → prepare_data →
prepare_splits → build_graph). All clean:
- prepare_data: both raw SHA256 verified; 10,240 records; billable 9,660.
- prepare_splits (E-009, billable): 7,728/961/971 — canonical regime reproduced
  exactly. Written to E-009_Balanced_E002Init/stage2/{chapter}/.
- build_graph (explicit --gold-path): 9,660 records, 1,926 codes, 4,889 UMLS
  concepts → 6,837 nodes / 258,954 edges. icd10_knowledge_graph.pkl + indices written.

Two hazards caught and resolved at this gate:
1. **Q9 (sklearn version warning) — VERIFIED BENIGN.** scispacy 0.5.5 declares
   scikit-learn with NO version pin; runtime warns because the UMLS linker's
   TF-IDF artifacts were pickled under sklearn 1.1.2 (now 1.8.0). A --dry-run
   behavioural check showed the linker produces correct output (e.g. M25.562 →
   C0030193 "Pain" @ 0.97 — valid CUIs, sensible names/scores). Conclusion: the
   InconsistentVersionWarning is precautionary, not actual breakage. The graph is
   sound. Q9 can be downgraded to "verified benign; warning is cosmetic."
2. **STALE GOLD LANDMINE — cleared.** data/gold/ held TWO files: the canonical
   medsynth_gold_apso.parquet (Jun 1, regenerated this session) AND a stale
   medsynth_gold_apso_20260505_194721.parquet (May 5, ~29KB larger = different
   data). build_graph.py's default resolution (sorted glob [-1]) and train.py's
   --gold-path default (auto-detect latest) BOTH grab the stale timestamped file —
   the --dry-run proved it (loaded the May-5 file). Any bare command would train
   on 3-week-old gold. FIX: moved the stale file to /tmp/stale_gold/ (recoverable)
   so only the canonical gold remains; ALSO adding explicit --gold-path to every
   train.py command (steps 4–7). Belt and suspenders. This is a real reproducibility
   hazard — bare auto-detect must not be trusted when timestamped golds linger.

Gate 1 (data+graph) PASSED. Next: E-001 (30ep) → gate model/ layout → E-002 (40ep).

## 2026-06-01 (cont.) — CANONICAL RUN COMPLETE: E-009 = 0.849 (verified, provisional); leakage now concrete

**The headline: the full pipeline ran clean, end to end, and every stage
reproduced its expected number.** This is the win the whole reproducibility effort
was driving toward — a faithful, gated, reproducible run under source-verified
naming. The contradictory-numbers problem that started this is resolved: we can
produce the project's number on demand, with provenance.

Four-stage chain retrained from scratch, gate passed after each:
- E-001 (675 ICD-3, 30ep, code-filter all) → val_acc 0.869; model/ nested, loads (675).
- E-002 (1,926 billable, 40ep) → val_acc 0.740, reproduces historical ~0.733;
  model/ nested, loads (1926); no stray top-level safetensors.
- E-003 Stage-1 (22 chapters, 5ep, init from E-001) → val_acc 0.964, reproduces
  historical EXACTLY; stage1/model/ nested, loads (22). Replaces the D007-broken
  on-disk stage-1.
- E-009 Stage-2 (19 resolvers, 20ep, warm from E-002) → all 19 trained, P/Q/U
  skipped. Every chapter's LOAD REPORT showed the 1926→N head reinit that proves
  E-002 transfer — NOT the silent cold-start that caused the old 12.7% failure.
  [presplit] sizes matched prepare_splits. Spot-checked Z (263) and T (15) load.

Calibrate (25s): Stage-1 T=1.1701 written to the **E-003** path, not E-009 — the
D009 "train Stage-1 once, reuse by reference" design works in practice. 19
resolvers calibrated, P/Q/U skipped, avg ECE 0.688→0.077. The low pre-cal
coverage (0.0%→) is just diffuse softmax being sharpened by T<1, not a defect.

Evaluate (61s): Stage-1 from E-003, 19 resolvers from E-009, graph reranker from
data/graph/. **Test N = 971** (canonical, no mismatch). **E2E 0.849, macro F1
0.774, ECE 0.0242.** Internally coherent: chapter routing 0.984 × within-chapter
0.863 ≈ 0.849 — no hidden inflation path. Q9 sklearn warning fired on the linker
pickles exactly as predicted (certified benign). Recorded as **D010**, supersedes
D008.

**The other half: that same run made the leakage concrete, and it keeps 0.849
provisional (D005 stands).** 0.849 is a genuine achievement *as an artifact*
(faithful, gated, reproducible) and NOT yet a scientific result. The redaction
removes the ICD-10 *code* but retains the diagnosis *description* it encodes —
"pain in left knee" for M25.562 — sitting right next to where the code was. The
model can read the answer off that text instead of inferring from clinical
findings. Both things are true; both are now in the record.

**Found this session (extends the picture):**
- `prepare_data.py` phase 3b computes a per-record `has_leakage` flag, then phase
  3c redacts and *drops* it. Half the leakage inventory is built and discarded —
  keeping it is most of the counting step Q8 needs.
- The real redaction logic is in `src/preprocessing.py` (`ICD10_REDACT_PATTERN`,
  `redact_icd10_sections`, `build_apso_note`) — NOT yet read. Its name suggests it
  targets codes/sections, not description text, which would explain why
  `has_leakage` never tracked the semantic leakage. Pending a read of that file.
- We have the CDC reference descriptions per code → the description-redaction can
  be *anchored* per record, not blind fuzzy matching.

**Forward plan (now in Q8, rewritten):** prototype anchored description-redaction
in the EDA notebook → confirm efficacy on a real before/after sample (check BOTH
under- and over-redaction) → move the proven rule into preprocessing.py +
prepare_data.py → regenerate gold → rerun the E-009 chain for the first
*publishable* number (expected below 0.849; the drop is the quantified leakage).
Two premises flagged as pending-verification, not fact: what the existing
redaction removes, and whether descriptions are inserted verbatim+adjacent.

Next concrete step: read `src/preprocessing.py`, then the notebook count of
affected rows.

## 2026-06-01 (cont.) — Q8 leakage QUANTIFIED: 76.2% of billable records carry the full label

Built the leakage inventory we'd been reasoning about without a number. In the EDA
notebook (Phase 0 env only — did NOT rerun the gold-rebuild phases), read the
canonical gold, re-applied the certified `redact_icd10_sections` (so we measure the
true model-input state), joined each record's CDC reference description
(`data/ontology/icd10cm_2026.parquet`, un-dotted code join, 0 missing), and scored
content-token overlap between description and assessment.

**Result (9,660 billable records):** overlap=1.0 (full description present in
assessment) **7,360 = 76.2%**; ≥0.8 78.3%; ≥0.6 83.8%; ≥0.5 86.3%; ≥0.3 89.0%.
The leak is near-verbatim — M25.562 "Pain in left knee" → "Pain in the left knee";
N39.0 "Urinary tract infection, site not specified" appears almost char-for-char.
This is the dominant case: ~3 of 4 billable training records have the label written
into the input. Per-record scores saved to outputs/audits/q8_leakage_scores.parquet.

Reframes D010's 0.849: the headline is earned on data where the answer is usually
on the page. Two bounds kept explicit: this measures description *presence* not
causal inflation (magnitude needs the rerun), and token-overlap can hit 1.0 by
chance on 1–2 token descriptions (not the driver here — the 1.0 cohort is
multi-token exact restatements). Also confirmed from source this session:
`src/preprocessing.py` redaction targets codes/parentheticals ONLY, never the
description — so the notebook's "0 leaks remaining" meant 0 *code* leaks, never the
semantic leak. Q8 updated with the table; next is prototyping the anchored
redaction and a human-eyeball before/after sample.

## 2026-06-02 — Q8 redactor BUILT & VALIDATED (~94% clean); assessment-only scope locked (D011)

Turned the Q8 leakage finding into a working, validated description-redactor —
all in the EDA notebook (the lab), nothing in gold touched yet.

**Scope decided (D011): assessment-only.** All-4-section overlap audit showed the
leak is overwhelmingly in the assessment (76.2%); note-level exposure is 77.9% —
only 1.7pp / 169 records more. Eyeballing P/S/O (dashboard tab 3) confirmed those
overlaps are scattered legitimate clinical vocabulary, not contiguous label
restatement, so redacting them would gut signal. The 169 blind-spot records are
consciously accepted as residual.

**Method: dictionary-anchored deterministic redactor.** Harvested MedSynth's own
diagnosis phrasings from the ~26% of raw notes carrying an (ICD-10:CODE) tag →
code→{phrasings}. Quarantined 197 low-CDC-overlap suspects (comorbidity
intrusions like F12.20→"Hypertension"). Final dictionary: 1,056 codes / 1,342
phrasings, CDC description as fallback. Redactor removes all occurrences,
[DIAGNOSIS] placeholder mid-sentence / drop-line when standalone, min-2-token
guard against over-redacting generic words ("Weakness"). Fires on 5,610/9,660;
3,270 no-match, 780 guard-skip.

**Validation via local-LLM audit (advisory only, never writes gold; temp 0, oMLX
direct API).** Dictionary redactor: 45/50 clean by verdict; the 2 "over_redacted"
were false alarms (label-only assessments where correct full removal looked like
deletion) → effectively ~94% clean. Earlier CDC-fuzzy matcher was the 50%
baseline (left label fragments, broke sentences). Over-redaction of real content:
eliminated. Residual: 3/50 leak_remains, all the same class — label restated in a
surface form the dictionary lacks ("the" inserted, "Old" prefix dropped). Not
chased: fixing it (optional-article match) would raise over-redaction risk to
address the less-costly failure mode.

**Process notes that mattered.** (1) The LLM judge is useful but fallible — twice
it mis-flagged (the [DIAGNOSIS] placeholder as a leak; label-only assessments as
over-redaction); reading the cases, not trusting the tally, caught both. The
prompt now declares [DIAGNOSIS] as the success marker. (2) Honest coverage:
"~94% clean" is over the FIRED set, not all records — total residual leakage is
higher and must be stated that way in the paper. (3) All redaction logic kept in
the notebook for evaluation; superseded versions preserved as markdown
breadcrumbs (CDC-fuzzy, dict v1/v2) so the trail of what we tried survives.

**Next:** migrate the validated redactor into src/preprocessing.py + prepare_data.py,
regenerate gold, rerun the E-009 chain → first publishable number (expected below
0.849; that delta quantifies the leakage). Tidy: strip empty `**\n\n**` fences
when redaction empties an assessment.

## 2026-06-02 (cont.) — Q8 redactor: from "94% on 50" to measured 18.6% residual; v4 rejected, v5 final (D012)

After committing D011 (which validated the approach on 50 records), we did the
harder measurement work. The 50-sample was encouraging but couldn't see the recall
gap; the corpus-wide numbers and a 500-record stratified audit could, and they
reshaped the picture. Full trail here because the path matters for anyone following.

**1. Corpus-wide deterministic residual.** Re-ran the token-overlap leak test on
the redactor *output* over all 9,660 billable (no LLM). v3 (D011's redactor):
residual 23.5% (2,270), removed 69.2%, fired 5,610. So the honest figure is not
"94% clean" — that was over the fired set — but "23.5% of all billable still leak
after redaction." Both true, different denominators. This became the number to
improve.

**2. Stratified 500-record LLM audit (300 fired + 200 nomatch).** Fired stratum
96% clean — confirms quality at scale, the 50-sample wasn't a fluke. Nomatch
stratum 61% genuinely leak — the records we DON'T fire on mostly DO still contain
the label. So the recall gap is real, not an artifact of the loose overlap metric.

**3. Decomposed the 122 confirmed nomatch misses.** article_insertion 20%
(inserted "the" defeats the match — fixable), guard_skipped 26% (single-word
labels we skip on purpose), on_CDC_fallback 34% (dict miss — but reading them,
some are LLM false-positives / code-assessment mismatches, not real leaks), reworded
20% (genuine paraphrase, deterministic ceiling). So ~half the gap is fixable,
~a fifth is a hard ceiling, a third needs discounting for false-positives.

**4. v4: chased recall, caught a regression.** Added (a) optional-article matching
and (b) standalone-short-phrase recovery. Residual dropped to 13.3% — tempting.
But a risk-weighted 200-record precision audit (oversampling the new behaviors)
showed fix (b) reintroduced the over-redaction the guard was built to stop: it cut
short labels out of finding sentences — "moderate wheezing and…" → "moderate  and…"
(R06.2), aphasia (R47.01), pyonephrosis (N13.6). Real damage to clinical signal.
Fix (a) was clean (apparent over-redactions were label-only-assessment false alarms).

**5. v5 = v3 + article-tolerance only.** Dropped fix (b). Residual 18.6% (1,799),
removed 75.6%, fired 6,091. We deliberately took 18.6% over v4's 13.3%: the extra
recall corrupted findings, and over-redaction is a worse failure than residual leak.
v5 is the redactor we migrate. Recorded as D012.

**Method lesson reinforced (twice).** The local-LLM judge mis-flagged twice and
both were caught by READING cases, not trusting counts: it called the [DIAGNOSIS]
placeholder a leak (fixed in the prompt), and called label-only assessments
"over_redacted" when full removal correctly emptied the section. The judge is
advisory; tallies get verified by eyeball before we act on them. This is now a
standing practice for the audit.

**Housekeeping done alongside:** consolidated the EDA notebook to a single clean
underscore-named file (01-EDA_SOAP_1.ipynb; the space-named original and the
interim _clean copy removed), with superseded redactor versions preserved as
markdown breadcrumbs. Parked DSPy/GEPA as Q10 (audit-judge hardening only,
post-first-number, needs a human-labeled set; never the redaction path).

**Next:** migrate v5 into src/preprocessing.py + prepare_data.py behind a flag
(flag gates phase 3d AND switches output filename so leaky + de-leaked golds
coexist for a clean A/B), on branch q8/description-redaction. Port-verification
gate: migrated redact_descriptions must reproduce 18.6% / 6,091 before we trust the
rerun. Then rerun E-009 → first publishable number (expect below 0.849; the delta
is the leakage contribution). Inference parity (prepare_inference_input) DEFERRED:
at inference the code is unknown, so code-keyed redaction can't apply as-is —
real train/serve asymmetry needing its own design pass; the rerun measures the
leakage effect, not a deployable model.

## 2026-06-03 — Q8 redactor MIGRATED into pipeline + de-leaked gold built; E-015 rerun launched (D014)

Morning session. Took the validated v5 redactor from notebook to production pipeline,
generated the de-leaked gold, and launched the leakage-corrected rerun. The
"read-the-source-before-running" discipline paid off repeatedly — caught three
would-be-silent traps before any of them cost a training run.

**Migration (committed 74d7cda, branch q8/description-redaction).** Added
`redact_descriptions` (v5 logic) to src/preprocessing.py — article-tolerant
full-phrase match, [DIAGNOSIS] placeholder mid-sentence / drop-line standalone,
remove-all-occurrences, min-2-token guard, dictionary + CDC fallback, rebuilds
apso_note with the same block as redact_icd10_sections. Added phase_3d +
--redact-descriptions flag to scripts/prepare_data.py; the flag gates the phase AND
switches the output filename (medsynth_gold_apso.parquet vs _deleaked.parquet) so
both golds coexist for a clean A/B; flag-off is byte-identical to the 0.849 pipeline.
Committed the phrasing dictionary as a versioned artifact at
data/ontology/q8_phrasing_dictionary.json.

**Port-verification gate (the key discipline).** Before trusting the migrated code,
re-measured residual on gold via the installed function: 9,660 scored, residual
18.6%, removed 75.6% — IDENTICAL to the notebook v5 (D012). Behaviour-tested the
five canonical cases too (standalone-drop, placeholder, article-tolerance, guard,
remove-all) — all pass. The port is faithful.

**Three traps caught by reading first (none reached a training run):**
1. config.resolve_path('data','ontology') — invalid; resolve_path is a config-key
   lookup, 'ontology' is not a registered key. Fixed to PROJECT_ROOT / 'data' /
   'ontology' (matching graph_reranker.py).
2. data_loader.load_gold_parquet globs medsynth_gold_apso_*.parquet and takes
   last-by-sort — fragile. Avoided by always passing --gold-path explicitly (the
   training scripts support it; glob is only a fallback).
3. --use-presplit reads cached per-experiment splits — but for a NEW experiment name
   the splits don't exist yet, so it splits the de-leaked gold fresh (seed 42). Safe;
   omitted the flag anyway for clarity.

**De-leaked gold built & verified.** prepare_data.py --redact-descriptions ran
clean (3a→3d→4 visible; 3b/3c silent but confirmed via markers). Exported
medsynth_gold_apso_deleaked.parquet. Verified: 9,660 rows, [REDACTED] present,
[DIAGNOSIS] present (2,951), 0 [DIAGNOSIS] in original, residual 18.6%. Original
gold untouched.

**Gold provenance pinned (closes a D010 gap).** experiments.json confirms E-009/0.849
trained on medsynth_gold_apso.parquet — exactly the file we de-leaked. A/B valid.
Recovered the full E-009 recipe; mirrored it exactly for E-015, changing only the
gold path. Stage-1 reused from E-003 (deliberate control; number is not
fully-end-to-end-de-leaked — caveat in D014).

**E-015 launched.** Dry-run validated all paths (de-leaked gold, E-003 Stage-1
reuse, E-002 init, Stage-2 only, epochs 20). Unloaded the oMLX audit model first to
free memory for training (the LLM plays no role in the run — D013). ~75 min Stage-2
train + calibrate + evaluate. Number to land in D014 + here.

**Gaps logged as Q11** (non-blocking, post-number): no unit tests; run_experiment.py
missing sys.path bootstrap; docs drift (E-012/013/014 undocumented).

**Next:** record the E-015 number; compare to 0.849 (delta = leakage contribution);
then README/paper reconciliation and the deferred items (inference parity,
serving_local_models.md, Q11).

## 2026-06-03 (cont.) — E-015 de-leaked number: 0.849 → 0.482 (D014 RESULT)

The rerun finished (~83 min, clean). The leakage-corrected E2E accuracy is **0.482**
(Macro F1 0.368, ECE 0.1313), vs the 0.849 leaky baseline (D010) — a 36.7-point drop.
Description leakage accounted for the large majority of the apparent performance;
0.849 was substantially inflated, as D005/D010 suspected. First publishable
leakage-corrected number.

Key honesty point captured in D014: **0.482 is a lower bound.** The reused E-003
Stage-1 router scored 0.972 on its own calibration data but 0.758 on the de-leaked
eval set — it was trained on leaky text and degrades when fed de-leaked notes. So
0.482 conflates the intended effect (Stage-2 can't read the answer) with an
introduced Stage-1 train/serve mismatch. True clean performance is expected somewhat
above 0.482 but far below 0.849. Follow-up: a --train-stage1 rerun on de-leaked gold
to isolate the pure effect. Reporting rule for now: "0.482, de-leaked, Stage-1 reused
(lower bound)" alongside 0.849 — not as a single clean replacement.

The run otherwise behaved exactly as designed: de-leaked gold loaded (9,660 billable),
19 Stage-2 resolvers trained + 3 chapters skipped (P/Q/U fallbacks), E-003 Stage-1
reused, epochs 20, seed 42. The LayerNorm beta/gamma and classifier-reinit warnings
were benign as expected. Number recorded in D014.

**Next:** README/Our_paper reconciliation (still show 83.9%/0.849 as clean — now
demonstrably inflated); the --train-stage1 fully-de-leaked follow-up; then the
deferred items (inference parity, serving_local_models.md, Q11).
