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
