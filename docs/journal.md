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
