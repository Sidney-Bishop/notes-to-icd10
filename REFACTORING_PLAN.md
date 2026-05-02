# Notes-to-ICD10 — Refactoring Plan

**Document purpose:** Structured plan for the next phase of development.
Written as a handoff document — a new conversation with Claude should be
able to pick this up and proceed without needing the full project history.

**Date:** 1 May 2026
**Current best result:** E-010_40ep_E002Init — 83.9% E2E | 0.762 F1 | 0.034 ECE | 82.1% Coverage@0.7 (95.2% acc)

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
- `Run notes.md` — step-by-step reproduction guide
- `Prj_Overview.md` — full project history and architectural decisions
- `notebooks/utils/nb_setup.py` — shared notebook utilities (R-002)

Always run `uv run python verify_scripts.py` before any training command.

---

## Critical Architectural Constraints

**1. E-002 must be trained on the same gold dataset as Stage-2 resolvers.**
When the code spaces differ, classifier heads cannot transfer and Stage-2
trains from random initialisation — E2E collapses from ~80% to ~20%.
Enforced in `scripts/train.py`.

**2. E-002 epoch count matters significantly.**
40-epoch E-002 produces +4.1pp E2E over 20-epoch E-002 on Stage-2 resolvers.
The encoder representations continue improving through epoch 40 on this dataset.
Always use 40 epochs for E-002 training.

---

## Experiment Registry Summary

| Experiment | E2E | F1 | ECE | Coverage@0.7 |
|---|---|---|---|---|
| E-001_Baseline_ICD3 | 86.9% | 0.843 | — | — |
| E-002_FullICD10_ClinicalBERT | 76.2% | 0.661 | — | — |
| E-003_Hierarchical_ICD10 | 11.1% | 0.075 | — | — |
| E-009_Balanced_E002Init | 79.8% | 0.711 | — | — |
| **E-010_40ep_E002Init** | **83.9%** | **0.762** | **0.034** | **82.1%** |

---

## Refactoring Requirements

### R-001 — Notebook Result Logging
**Priority: High**
**Status: ✅ COMPLETE**

Every metric, result, and artifact produced by a notebook run is captured
in a structured log. `ExperimentLogger.log_results()` writes to
`outputs/experiments.json` after each notebook completes. Phase 13
`OFFICIAL_PERFORMANCE_RECORD.txt` written to each experiment's registry
directory with full metrics pulled live from scope.

---

### R-002 — Shared Notebook Utilities Module
**Priority: High**
**Status: ✅ COMPLETE**

MLflow and TensorBoard setup code consolidated into `notebooks/utils/nb_setup.py`.
All four training notebooks (02-05) updated and re-run to verify reproducibility.

**What was built:**
- `notebooks/utils/nb_setup.py` containing:
  - `setup_experiment(cfg)` — full Phase 1 in one call (~80 lines to ~15 lines)
  - `make_training_args(output_dir, cfg)` — consistent TrainingArguments factory
  - `promote_to_registry(cfg, trainer, ...)` — flat model registry promotion
  - `check_mlflow_run()` — active run guard
  - `end_mlflow_run(final_metrics)` — clean MLflow shutdown
  - `print_monitoring_urls(project_root, tb_dir)` — TensorBoard/MLflow commands
  - `_mlflow_safe_params(cfg)` — excludes internal keys, lists, warmup_ratio

**Bugs found and fixed during re-runs:**
- `max_length` missing from notebook 04 cfg dict
- `label_map.json` not saved to Stage-2 model directories in notebook 04
- `warmup_ratio` MLflow conflict — resolved via `_mlflow_safe_params()`

**Verified re-run results (all within normal training variance):**

| Notebook | Experiment | Previous | This run | Status |
|---|---|---|---|---|
| 02 | E-001 ICD-3 | 87.2% / 0.841 | 88.0% / 0.853 | ✅ |
| 03 | E-002 flat ICD-10 | 73.3% / 0.634 | 73.0% / 0.629 | ✅ |
| 04 | E-003 cold start | 11.1% / 0.075 | 11.3% / 0.075 | ✅ |
| 05 | E-009 E-002 init | 79.8% / 0.711 | 79.2% / 0.699 | ✅ |

**Note on cost:** Full notebook re-runs required ~12 hours of GPU compute.
For future refactoring verification, use a single smoke test rather than
full training runs. The re-runs confirmed reproducibility across all
experiments but were not strictly necessary to verify the utility module.

**Acceptance criteria met:**
- Phase 1 boilerplate reduced from ~80 lines to ~15 lines across all notebooks ✅
- Consistent MLflow experiment naming and param logging ✅
- `warmup_ratio` deprecation warning eliminated ✅
- All notebooks reproduce previous results within normal variance ✅

---

### R-003 — Plot/Figure Artifact Persistence
**Priority: High**
**Status: ✅ COMPLETE**

`src/plot_utils.py` with `save_figure()` saves to `outputs/visualizations/`,
logs to `outputs/run.log` and `outputs/figure_index.json`. All dashboard
and confusion matrix cells in notebooks 02-05 updated to use `save_figure()`.

---

### R-004 — MLflow as Active Query Source
**Priority: Medium**
**Status: ⏳ PENDING**

MLflow is being written to but never read from. The experiment registry
(`outputs/experiments.json`) was built manually because MLflow wasn't
being queried.

**What to build:**
- `scripts/mlflow_query.py` with `best_run()`, `compare_runs()`,
  `get_artifacts()`, `leaderboard()` functions
- `ExperimentLogger.status()` optionally pulling from MLflow

**Decision:** MLflow is source of truth for training metrics. `experiments.json`
is lightweight index. `run.log` is append-only audit trail. All three maintained.

---

### R-005 — Pydantic Input Validation for Inference
**Priority: Medium**
**Status: ⏳ PENDING**

`HierarchicalPredictor.predict()` accepts any string with no validation.

**What to build:**
- `ClinicalNoteInput` Pydantic model in `src/inference.py`
- Validators for empty notes, short notes (<20 words warning), long notes (>400 words warning)
- Backward compatible — plain string still works

---

### R-006 — EDA Notebook ICD-10 Frequency Analysis
**Priority: Medium**
**Status: ✅ COMPLETE**

Cell 23 of `01-EDA_SOAP.ipynb` — full frequency histogram with annotation
arrow for post-publication codes. Cell 29 — token pressure KDE saved via
`save_figure()`.

---

### R-007 — Flat vs Hierarchical Baseline Comparison
**Priority: Medium**
**Status: ✅ COMPLETE**

Official comparison table (all results on same original gold test set):

| Approach | Accuracy | F1 | Notes |
|---|---|---|---|
| Flat ICD-3 (E-001) | 87.2% | 0.841 | 675 classes, 30 epochs |
| Flat ICD-10 (E-002) | 73.3% | 0.634 | 1,926 classes, 40 epochs |
| Hierarchical cold start (E-003) | 11.1% | 0.075 | Fresh Stage-2 init |
| Hierarchical E-002 init 20ep (E-009) | 79.8% | 0.711 | Original gold |
| **Hierarchical E-002 init 40ep (E-010)** | **83.9%** | **0.762** | **Current best** |

---

### R-008 — Unit Tests for Core Modules
**Priority: Low**
**Status: ⏳ PENDING**

No unit tests exist. The Z override bug and `TrainingResult.get()` bug
would both have been caught immediately by tests.

**What to build:**
- `tests/test_paths.py` — verify `ExperimentPaths` resolves all conventions correctly
- `tests/test_experiment_logger.py` — verify logging writes to correct files
- `tests/test_inference_validation.py` — verify Pydantic validation (after R-005)
- `tests/test_preprocessing.py` — verify APSO flip and ICD-10 redaction

---

### R-009 — Dependency Version Audit
**Priority: Low**
**Status: 🔶 PARTIALLY COMPLETE**

sklearn version mismatch (1.1.2 vs 1.8.0) — graph reranker pickled against
old version. `pyproject.toml` `[dependency-groups] dev` format fixed.

**Remaining:**
- Rebuild graph reranker with current sklearn version or pin sklearn
- Audit remaining unpinned dependencies
- Ensure `uv.lock` is committed

---

## Implementation Order

| Order | Requirement | Status |
|---|---|---|
| 1 | R-003 (plot saving) | ✅ Complete |
| 2 | R-001 (notebook logging) | ✅ Complete |
| 3 | R-006 (EDA frequency analysis) | ✅ Complete |
| 4 | R-007 (baseline comparison) | ✅ Complete |
| 5 | R-002 (shared notebook utils) | ✅ Complete |
| 6 | R-004 (MLflow querying) | ⏳ Pending |
| 7 | R-005 (Pydantic validation) | ⏳ Pending |
| 8 | R-008 (unit tests) | ⏳ Pending |
| 9 | R-009 (dependency audit) | 🔶 Partial |

---

## Next Research Steps (beyond refactoring)

| Priority | Experiment | Expected gain | Effort |
|---|---|---|---|
| 1 | **E-011: E-010 + GraphReranker** | +1-2pp Z-chapter | Low — graph already built |
| 2 | **Z-chapter contrastive fine-tuning** | +5-10pp Z | Medium — 2 weeks |
| 3 | **Lower Z threshold to 0.5** | More Z coverage at ~85% precision | Low |
| 4 | **MIMIC-IV validation** | Reveals synthetic→real gap | Blocked on PhysioNet |

---

## What Is NOT Being Refactored (and Why)

| Item | Reason |
|---|---|
| Chapter Z contrastive fine-tuning | Research task, not refactoring |
| MIMIC-IV validation | Blocked on PhysioNet access |
| Stage-1 router retraining | E-003 Stage-1 at 98.7% routing is sufficient |
| Augmented gold pipeline | E-010 on original gold beats E-005c+graph on augmented |

---

## Notebook Re-run Status

| Notebook | Experiment | Status | Key result |
|---|---|---|---|
| 01-EDA_SOAP | EDA | ✅ Complete | Gold layer exported |
| 02-Model_ClinicalBERT_Baseline_ICD3 | E-001 | ✅ Complete | 88.0% / F1 0.853 |
| 03-Model_ClinicalBERT_Surgical_ICD10 | E-002 | ✅ Complete | 73.0% / F1 0.629 (40 epochs) |
| 04-Model_Hierarchical_ICD10 | E-003 | ✅ Complete | 11.3% / F1 0.075 (cold start) |
| 05-Model_Hierarchical_ICD10_E002Init | E-009 | ✅ Complete | 79.2% / F1 0.699 |

---

## Starting Point for a New Conversation

If starting fresh, give Claude this context:

> "I am working on the Notes-to-ICD10 project. Please read REFACTORING_PLAN.md
> in the project root. The current best model is E-010_40ep_E002Init at 83.9%
> E2E, 0.762 F1, 0.034 ECE, 82.1% Coverage@0.7. All scripts pass
> `uv run python verify_scripts.py`. Before touching any code, read
> src/paths.py, src/experiment_logger.py, and src/inference.py to understand
> the existing architecture. The shared notebook utilities are in
> notebooks/utils/nb_setup.py. The next refactoring priorities are R-004
> (MLflow query script) and R-005 (Pydantic validation for inference)."

---

*Last updated: 1 May 2026*
*Author: Refactoring session with Claude Sonnet 4.6*



# Mark R-005 complete in REFACTORING_PLAN.md
# Change: "Status: ⏳ PENDING" → "Status: ✅ COMPLETE"
# Add note: "Already implemented in src/inference.py during script layer work.
#            Verified 1 May 2026 with mock-isolated unit tests."