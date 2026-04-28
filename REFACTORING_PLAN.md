# Notes-to-ICD10 — Refactoring Plan

**Document purpose:** Structured plan for the next phase of development.
Written as a handoff document — a new conversation with Claude should be
able to pick this up and proceed without needing the full project history.

**Date:** 27 April 2026
**Current best result:** E-009_Balanced_E002Init — 71.7% E2E | 0.637 F1 | 0.030 ECE

---

## Project Context (for a new conversation)

This project fine-tunes `emilyalsentzer/Bio_ClinicalBERT` with a classification
head to predict ICD-10 codes from clinical APSO notes (Assessment, Plan,
Subjective, Objective ordering). The pipeline is hierarchical: a Stage-1
22-way chapter router feeds into 19 per-chapter Stage-2 code resolvers.

Key files:
- `src/inference.py` — end-to-end prediction pipeline
- `src/paths.py` — canonical path resolution across experiments
- `src/experiment_logger.py` — structured logging to `outputs/run.log` and `outputs/experiments.json`
- `scripts/train.py` — training orchestrator
- `scripts/calibrate.py` — temperature scaling
- `scripts/evaluate.py` — evaluation pipeline
- `verify_scripts.py` — pre-flight checks (run before every training session)
- `RUN_NOTES.md` — step-by-step reproduction guide
- `Prj_Overview.md` — full project history and architectural decisions

Always run `uv run python verify_scripts.py` before any training command.

---

## Refactoring Requirements

### R-001 — Notebook Result Logging
**Priority: High**
**Area: Notebooks**
**Status: COMPLETE**

Every metric, result, and artifact produced by a notebook run should be
captured in a structured log. Currently notebooks produce output that
scrolls past and is lost between sessions.

**What was built:**
- Three lines appended to the final registry cell of each training notebook
  (02, 03, 04, 05) — `ExperimentLogger.log_results()` writes to
  `outputs/experiments.json` after each notebook completes.
- Phase 13 `OFFICIAL_PERFORMANCE_RECORD.txt` written to each experiment's
  registry directory with full metrics pulled live from scope.

**Acceptance criteria met:**
- `uv run python -c "from src.experiment_logger import status; status()"` shows
  all experiments registered with correct metrics ✅
- `outputs/run.log` contains timestamped entries ✅

---

### R-002 — Shared Notebook Utilities Module
**Priority: High**
**Area: Notebooks**

MLflow and TensorBoard setup code is copy-pasted across every notebook.
This causes drift — different notebooks initialise MLflow differently,
use different experiment names, and log different subsets of parameters.

**What to build:**
- Create `notebooks/utils/nb_setup.py` — a single shared module containing:
  - `setup_mlflow(experiment_name, run_name)` — consistent MLflow initialisation
  - `setup_tensorboard(log_dir)` — consistent TensorBoard writer
  - `log_params(cfg)` — log a config dict to MLflow
  - `log_metrics(metrics)` — log a metrics dict to MLflow
  - `end_run()` — clean shutdown
- Every notebook imports from this module instead of defining its own setup

**Acceptance criteria:**
- No MLflow or TensorBoard setup code exists in any notebook cell
- All notebooks use identical MLflow experiment naming conventions
- Removing setup from one notebook does not break any other notebook

---

### R-003 — Plot/Figure Artifact Persistence
**Priority: High**
**Area: Notebooks + Scripts**
**Status: COMPLETE**

Every matplotlib/plotly figure generated anywhere in the project is
currently displayed inline and lost. If a paper is being written, there
is no way to know which figures exist or where they came from.

**What was built:**
- `src/plot_utils.py` with `save_figure(fig, notebook, description, experiment, dpi)`
  saves to `outputs/visualizations/{notebook}/`, logs to `outputs/run.log`
  and `outputs/figure_index.json`.
- All dashboard and confusion matrix cells in notebooks 02–05 updated to
  use `save_figure()` instead of `plt.savefig()`.

**Acceptance criteria met:**
- All figures saved to `outputs/visualizations/` ✅
- `outputs/run.log` contains an entry for each figure ✅
- Timestamp in filename prevents overwrite ✅

---

### R-004 — MLflow as Active Query Source
**Priority: Medium**
**Area: Scripts + Notebooks**

MLflow is being written to but never read from. The experiment registry
(`outputs/experiments.json`) was built manually because MLflow wasn't
being queried. This creates two parallel tracking systems that can diverge.

**What to build:**
- A `scripts/mlflow_query.py` utility that answers common questions:
  - `best_run(metric='e2e_accuracy')` — return the run with the highest E2E accuracy
  - `compare_runs(experiment_names)` — return a comparison table of parameters and metrics
  - `get_artifacts(run_name)` — return all artifact paths for a run
  - `leaderboard()` — print a ranked table of all runs by E2E accuracy
- The `ExperimentLogger.status()` function should optionally pull from MLflow rather than `experiments.json`
- Decide explicitly: is `experiments.json` the source of truth, or is MLflow? Document the decision.

**Recommended decision:** MLflow is the source of truth for training metrics and parameters. `experiments.json` is a lightweight index for quick status checks and human-readable audit. `run.log` is the append-only human audit trail. All three serve different purposes and should be maintained.

**Acceptance criteria:**
- `uv run python scripts/mlflow_query.py leaderboard` prints a ranked table
- The table matches `experiments.json` — no divergence
- New training runs automatically appear in both without manual registration

---

### R-005 — Pydantic Input Validation for Inference
**Priority: Medium**
**Area: `src/inference.py`**

`HierarchicalPredictor.predict()` accepts any string with no validation.
In production this is a correctness and reliability risk.

**What to build:**
- Add a `ClinicalNoteInput` Pydantic model to `src/inference.py`:
  ```python
  from pydantic import BaseModel, field_validator

  class ClinicalNoteInput(BaseModel):
      note: str
      preprocessed: bool = False

      @field_validator('note')
      def note_must_be_non_empty(cls, v):
          if not v or not v.strip():
              raise ValueError('Clinical note cannot be empty')
          return v

      @field_validator('note')
      def warn_if_short(cls, v):
          tokens = len(v.split())
          if tokens < 20:
              warnings.warn(f'Note is very short ({tokens} words) — prediction may be unreliable')
          if tokens > 400:
              warnings.warn(f'Note is long ({tokens} words) — will be truncated to 512 tokens')
          return v
  ```
- `predict()` accepts either a raw string (backward compatible) or a `ClinicalNoteInput`
- Encoding issues (non-UTF8 characters) are caught and sanitised with a warning

**Acceptance criteria:**
- `predictor.predict("")` raises a `ValidationError` with a clear message
- `predictor.predict("hi")` issues a warning but still runs
- Existing code that passes a plain string continues to work unchanged
- Pydantic is already in the project dependencies (verify before implementing)

---

### R-006 — EDA Notebook ICD-10 Frequency Analysis
**Priority: Medium**
**Area: `notebooks/01-EDA_SOAP.ipynb`**
**Status: COMPLETE**

The EDA notebook currently gives a poor account of record distribution
per ICD-10 code. This is critical information for understanding model
performance and for any paper describing the dataset.

**What was built:**
- Cell 23 of `notebooks/01-EDA_SOAP.ipynb` — full frequency histogram with
  annotation arrow for the 11 post-publication codes.
- Cell 29 — token pressure KDE saved via `save_figure()`.

**Acceptance criteria met:**
- Frequency distribution visible in under 30 seconds ✅
- Analysis saved as figure via `save_figure()` ✅

---

### R-007 — Flat vs Hierarchical Baseline Comparison
**Priority: Medium**
**Area: Documentation + Evaluation**
**Status: COMPLETE — data now available**

71.7% E2E sounds strong but without an explicit baseline comparison there
is no frame of reference. The meaningful comparison for this project is
flat ClinicalBERT vs the hierarchical pipeline.

**Official comparison table (all results on same test set):**

| Approach | Model | Test Accuracy | Test F1 | Notes |
|---|---|---|---|---|
| Flat ICD-3 (E-001) | Bio_ClinicalBERT | 87.2% | 0.841 | 675 classes, 30 epochs |
| Flat ICD-10 (E-002) | Bio_ClinicalBERT | 73.3% | 0.634 | 1,926 classes, 40 epochs |
| Hierarchical cold start (E-003) | Bio_ClinicalBERT | 11.1% | 0.075 | Fresh Stage-2 init |
| **Hierarchical E-002 init (E-009)** | **Bio_ClinicalBERT** | **71.7%** | **0.637** | **22-way router + 19 resolvers** |

**Key finding:** E-009 hierarchical (71.7%) is within 1.6pp of E-002 flat
(73.3%). The Stage-2 within-chapter accuracy target to beat the flat
baseline is 80.4% (73.3%/91.2%). Closing this gap is the primary
remaining research objective.

**Acceptance criteria met:**
- Both flat and hierarchical results measured on same test set ✅
- TF-IDF/LR baselines not viable due to cardinality — documented ✅

---

### R-008 — Unit Tests for Core Modules
**Priority: Low (but important before any production use)**
**Area: `src/`**

No unit tests exist. The Z override bug and `TrainingResult.get()` bug
would both have been caught immediately by tests.

**What to build:**
- `tests/test_paths.py` — verify `ExperimentPaths` resolves all three conventions correctly
- `tests/test_experiment_logger.py` — verify logging writes to correct files
- `tests/test_inference_validation.py` — verify Pydantic validation (after R-005)
- `tests/test_preprocessing.py` — verify APSO flip and ICD-10 redaction

**Acceptance criteria:**
- `uv run pytest tests/` passes with no failures
- Tests are runnable without GPU (mock the model loading)
- CI can be added later — tests are the prerequisite

---

### R-009 — Dependency Version Audit
**Priority: Low**
**Area: `pyproject.toml`**
**Status: PARTIALLY COMPLETE**

sklearn version mismatch warnings appeared during evaluation (1.1.2 vs 1.8.0).
The graph reranker was pickled against an old version — this is a silent
correctness risk that could produce wrong results without any error.

**What was done:**
- `pyproject.toml` `[tool.uv] dev-dependencies` section updated to the new
  `[dependency-groups] dev` format — deprecation warning resolved ✅

**Remaining:**
- Rebuild graph reranker with current sklearn version or pin sklearn
- Audit remaining unpinned dependencies
- Ensure `uv.lock` is committed

---

## Implementation Order

| Order | Requirement | Status |
|---|---|---|
| 1 | R-003 (plot saving) | ✅ Complete |
| 2 | R-002 (shared notebook utils) | ⏳ Deferred — do after notebook re-runs |
| 3 | R-001 (notebook logging) | ✅ Complete |
| 4 | R-006 (EDA frequency analysis) | ✅ Complete |
| 5 | R-004 (MLflow querying) | ⏳ Pending |
| 6 | R-007 (baseline comparison) | ✅ Complete — data available |
| 7 | R-005 (Pydantic validation) | ⏳ Pending |
| 8 | R-008 (unit tests) | ⏳ Pending |
| 9 | R-009 (dependency audit) | 🔶 Partially complete |

---

## What Is NOT Being Refactored (and Why)

| Item | Reason |
|---|---|
| Chapter Z contrastive fine-tuning | Research task, not refactoring — separate work item |
| MIMIC-IV validation | Blocked on PhysioNet access |
| Chapter O targeted retraining | Quick win but separate from refactoring |
| Stage-1 router retraining on augmented gold | Marginal gain — E-003 Stage-1 at 96.4% is sufficient |

---

## Notebook Re-run Status

| Notebook | Experiment | Status | Key result |
|---|---|---|---|
| 01-EDA_SOAP | EDA | ✅ Complete | Gold layer exported |
| 02-Model_ClinicalBERT_Baseline_ICD3 | E-001 | ✅ Complete | F1 0.843, Acc 86.9% (30 epochs) |
| 03-Model_ClinicalBERT_Surgical_ICD10 | E-002 | ✅ Complete | F1 0.661, Acc 76.2% (40 epochs) |
| 04-Model_Hierarchical_ICD10 | E-003 | ✅ Complete | E2E 11.1% (cold start baseline) |
| 05-Model_Hierarchical_ICD10_E002Init | E-009 | ⏳ In progress | Target: 71.7% E2E |

---

## Starting Point for a New Conversation

If starting fresh, give Claude this context:

> "I am working on the Notes-to-ICD10 project. Please read REFACTORING_PLAN.md
> in the project root. We are working through the requirements in implementation
> order. The current best model is E-009_Balanced_E002Init at 71.7% E2E.
> All scripts pass `uv run python verify_scripts.py`. Before touching any code,
> read src/paths.py, src/experiment_logger.py, and src/inference.py to
> understand the existing architecture. Notebooks 01–04 have been re-run and
> documented. Notebook 05 (E-009) is the next to re-run."

---

*Last updated: 27 April 2026*
*Author: Refactoring session with Claude Sonnet 4.6*