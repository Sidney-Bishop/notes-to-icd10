# Notes-to-ICD10 — Refactoring Plan

**Document purpose:** Structured plan for the next phase of development.
Written as a handoff document — a new conversation with Claude should be
able to pick this up and proceed without needing the full project history.

**Date:** 6 May 2026
**Current best result:** E-010_40ep_E002Init — 83.9% E2E | 0.763 F1 | 0.033 ECE | 82.1% Coverage@0.7 (95.2% acc)
**Data status:** Phase 1b locked — HF canonical + DVC + SHA256 manifest (commit 6dda8ac)

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
- `scripts/prepare_data.py` — HF-locked ingestion + CDC validation (Phase 1b)
- `scripts/generate_manifest.py` — Phase 4 SHA256 manifest generator
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

**3. All data sources must be HF-locked, never live CDC FTP.**
As of 5 May 2026, `prepare_data.py` pulls exclusively from Hugging Face Hub
(`SidneyBishop/notes-to-icd10`). This eliminates data drift, ensures reproducible
SHA256 hashes, and enables fresh clones to rebuild without external dependencies.
The canonical gold dataset is locked at 10,240 rows with validation split
9,660/495/60/25 (billable/invalid/noisy_111/placeholder_x).

---

## Experiment Registry Summary

| Experiment | E2E | F1 | ECE | Coverage@0.7 |
|---|---|---|---|---|
| E-001_Baseline_ICD3 | 86.9% | 0.843 | — | — |
| E-002_FullICD10_ClinicalBERT | 76.2% | 0.661 | — | — |
| E-003_Hierarchical_ICD10 | 11.1% | 0.075 | — | — |
| E-009_Balanced_E002Init | 79.8% | 0.711 | — | — |
| **E-010_40ep_E002Init** | **83.9%** | **0.763** | **0.033** | **82.1%** |

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
full training runs.

---

### R-003 — Plot/Figure Artifact Persistence
**Priority: High**
**Status: ✅ COMPLETE**

`src/plot_utils.py` with `save_figure()` saves to `outputs/visualizations/`,
logs to `outputs/run.log` and `outputs/figure_index.json`. All dashboard
and confusion matrix cells in notebooks 02-05 updated to use `save_figure()`.

---

### R-004 — MLflow as Active Query Source
**Priority: Low**
**Status: ⏳ PENDING**

MLflow is being written to but never read from. Low priority — `status()`
already covers the primary use case.

**What to build:**
- `scripts/mlflow_query.py` with `best_run()`, `compare_runs()`,
  `get_artifacts()`, `leaderboard()` functions

---

### R-005 — Pydantic Input Validation for Inference
**Priority: Medium**
**Status: ✅ COMPLETE**

Already implemented in `src/inference.py` during script layer work.
Verified 2 May 2026 with mock-isolated unit tests.

`ClinicalNoteInput` Pydantic model with:
- Empty/whitespace-only note → raises `ValidationError`
- Note < 20 words → `UserWarning` (still runs)
- Note > 400 words → `UserWarning` about truncation (still runs)
- Bytes input → decoded with warning
- Backward compatible — plain str still accepted

Note: `evaluate.py` suppresses these warnings when running on pre-processed
gold layer notes (`preprocessed=True`) to avoid noisy output.

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
| **Hierarchical E-002 init 40ep (E-010)** | **83.9%** | **0.763** | **Current best** |

---

### R-008 — Unit Tests for Core Modules
**Priority: Low**
**Status: ✅ COMPLETE**

94 tests across 4 files, all passing. No GPU required — heavy imports mocked.

| File | Tests | Coverage |
|---|---|---|
| `tests/test_paths.py` | 32 | `ExperimentPaths`, 3 layout conventions, all path helpers |
| `tests/test_experiment_logger.py` | 14 | `log_start`, `log_complete`, `log_results`, `log_failed`, multi-experiment |
| `tests/test_inference_validation.py` | 20 | `ClinicalNoteInput` — all validators, edge cases, backward compat |
| `tests/test_preprocessing.py` | 28 | APSO-flip, ICD-10 redaction pattern, `build_apso_note`, `redact_icd10_sections` |

Run with: `uv run pytest tests/ -v`

---

### R-009 — Dependency Version Audit
**Priority: Low**
**Status: ✅ COMPLETE**

Knowledge graph rebuilt with sklearn 1.8.0 (was 1.1.2).

Residual sklearn version warning during graph loading is from **scispacy's
internal UMLS linker** — a third-party component that bundles its own
TF-IDF vectorizer. Not fixable without a scispacy upgrade. Our own graph
pickle (`data/graph/icd10_knowledge_graph.pkl`) is clean.

---


---

### R-010 — HF-Locked Canonical Data + DVC + Manifest
**Priority: High**
**Status: ✅ COMPLETE (5 May 2026)**

Eliminated CDC FTP dependency by locking all canonical datasets to Hugging Face Hub and implementing three-layer reproducibility.

**What was built:**
- `scripts/prepare_data.py` refactored to use `hf_hub_download()` instead of FTP
  - Pulls `icd10_notes.parquet` (10,240 rows) from HF
  - Pulls `cdc_fy2026_icd10.parquet` (74,719 codes) from HF
- `scripts/generate_manifest.py` — Phase 4 manifest generator with SHA256 hashing
- `upload_to_hf.py` — utility to push canonical data to HF Hub
- DVC tracking for `data/gold/*.parquet` (`.dvc` files in git, binaries in DVC remote)
- `data/gold/MANIFEST_*.json` force-added to git (bypassing .gitignore) for audit trail

**Decisions documented:**
1. **Why HF Hub over CDC FTP:** Eliminates annual code drift (FY2026→FY2027), removes build fragility, provides CDN distribution
2. **Why DVC over git-LFS:** Supports multiple remotes without GitHub quotas, human-readable YAML pointers, pipeline-aware caching
3. **Why manifests in git:** Cryptographic provenance (SHA256) must be version-controlled alongside code for regulatory audit

**Locked artifacts (commit 6dda8ac):**
- Gold parquet: `220dafcfe6a8aa53c0a728dbf3537ed1407897f2c92050831c7ebb31c7218bc7`
- MedSynth source: `7fa03f67b113b57a5f17349c712946553b4b186e1a11f39d74e0821d02fc5ac8`
- CDC FY2026: `2433adf954c3f49296a40761b83afb98c2d61cd78ca43f335fbdd4167e5fb93d`

**Verification:**
- Fresh clone to /tmp + `dvc pull` restores exact bytes
- `python scripts/generate_manifest.py` rebuilds from HF and verifies hashes
- Zero external calls to ftp.cdc.gov

---

## Implementation Order

| Order | Requirement | Status |
|---|---|---|
| 1 | R-003 (plot saving) | ✅ Complete |
| 2 | R-001 (notebook logging) | ✅ Complete |
| 3 | R-006 (EDA frequency analysis) | ✅ Complete |
| 4 | R-007 (baseline comparison) | ✅ Complete |
| 5 | R-002 (shared notebook utils) | ✅ Complete |
| 6 | R-005 (Pydantic validation) | ✅ Complete |
| 7 | R-008 (unit tests) | ✅ Complete |
| 8 | R-009 (dependency audit) | ✅ Complete |
| 9 | **R-010 (HF-locked data + DVC)** | **✅ Complete (5 May 2026)** |
| 10 | R-004 (MLflow querying) | ⏳ Pending (low priority) |

---

## Next Research Steps (beyond refactoring)

| Priority | Experiment | Expected gain | Effort |
|---|---|---|---|
| 1 | **Z-chapter contrastive fine-tuning** | +5-10pp Z | Medium — 2 weeks |
| 2 | **Lower Z threshold to 0.5** | More Z coverage at ~85% precision | Low |
| 3 | **MIMIC-IV validation** | Reveals synthetic→real gap | Blocked on PhysioNet |
| 4 | **DVC remote setup for models** | Enable `dvc pull` for E-010 weights | Low — 1 day |

**E-011 finding:** E-010 + GraphReranker = 83.9% / F1 0.763 — identical to
E-010 alone. The graph reranker has minimal impact on E-010's well-calibrated
resolvers (most predictions already exceed the 0.7 confidence threshold).
Z-chapter improved marginally from 62.1% → 62.9%.

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
> E2E, 0.763 F1, 0.033 ECE, 82.1% Coverage@0.7. All scripts pass
> `uv run python verify_scripts.py`. All refactoring requirements are complete
> except R-004 (low priority). **Phase 1b data is locked to HF Hub
> (SidneyBishop/notes-to-icd10) with DVC + SHA256 manifests (commit 6dda8ac).**
> The unit test suite is at `tests/` — run with `uv run pytest tests/ -v`.
> To reproduce: `git clone` → `dvc pull` → `python scripts/generate_manifest.py`.
> The next research priority is Z-chapter contrastive fine-tuning or MIMIC-IV
> validation (blocked on PhysioNet access)."

---

*Last updated: 6 May 2026*
*Author: Refactoring session with Claude Sonnet 4.6 + Data Locking (R-010)*