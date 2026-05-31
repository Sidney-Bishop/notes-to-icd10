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
