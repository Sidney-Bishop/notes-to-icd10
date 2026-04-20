# Project Overview: MedSynth ICD-10 Coding Pipeline

This document provides an end-to-end overview of the MedSynth preprocessing pipeline, designed for a mixed audience of data scientists and business stakeholders. Each section documents a notebook in the pipeline, capturing its purpose, key transformations, and business-relevant outcomes.

---

Here are the notebooks that make up the data processing pipeline:

1. 01-EDA_SOAP.ipynb
2. 02-Model_ClinicalBERT_Baseline_ICD3.ipynb
3. 03-Model_ClinicalBERT_Surgical_ICD10.ipynb
4. 04-Model_Hierarchical_ICD10.ipynb
5. 05-Model_Hierarchical_ICD10_E002Init.ipynb

All models are located in this folder: /Users/jroche/Workspace/Python/Notes_to_ICD10_prj/notebooks

---

## 01-EDA_SOAP.ipynb

### Purpose

This notebook performs the foundational data ingestion and preparation for the clinical note to ICD-10 classification pipeline. It transforms raw synthetic medical dialogue-note pairs into a **Gold layer** dataset that is fully validated, annotated, and ready for downstream model training.

### Business Context

ICD-10 codes are the standard system for classifying medical diagnoses used worldwide for billing, reporting, and healthcare analytics. This pipeline prepares data for an automated system that reads clinical notes and predicts the appropriate ICD-10 diagnostic codes.

The notebook implements a **Zero-Trust data quality framework**: every transformation is validated against external references, every assumption is tested, and the full audit trail is preserved.

---

### Key Transformation Phases

| Phase | Name | Key Purpose | Data Quality Check |
|-------|------|-------------|-------------------|
| **Phase 0** | Infrastructure Setup | Establish project root, validate directory structure, initialize audit logging | All required directories writable; config integrity verified |
| **Phase 1a** | Ingestion & Silver Layer | Load raw MedSynth dataset, validate structural integrity, normalize ICD-10 format | All records present; no missing fields |
| **Phase 1b** | CDC FY2026 Validation | Validate ICD-10 codes against official government reference; classify by billability | Codes verified against current year standard |
| **Phase 1b.1** | Composition Audit | Analyze label distribution; verify uniform sampling design | 2,037 unique codes confirmed; minimum 5 records per code |
| **Phase 1c** | Token Pressure Audit | Measure text length vs. model context window constraints | 64.7% of notes exceed 512-token limit |
| **Phase 1d** | Raw Discovery Auditor | Visual inspection of raw data via browser interface | Human validation of data quality |
| **Phase 1e** | Pydantic Firewall | Schema validation and SOAP section isolation | 100% extraction success across all sections |
| **Phase 1f** | Surgical Signal Auditor | Visual verification of SOAP extraction quality | Human confirmation of section boundaries |
| **Phase 1g** | DuckDB Persistence | Persist validated Silver layer to persistent storage | SQL-verifiable integrity checks |
| **Phase 2a** | Decimal Restoration | Canonicalize ICD-10 codes (add decimal points) | 100% valid format after restoration |
| **Phase 2b** | Gold Layer Auditor | Visual verification of canonical code format | Human-in-the-loop validation |
| **Phase 2c** | Status Annotation | Embed CDC validation status as first-class column | Full annotation across all records |
| **Phase 3a** | APSO-Flip | Reorder clinical notes (Assessment-first) | Diagnostic signal protected from truncation |
| **Phase 3b** | Leakage Detection | Quantify explicit ICD-10 code strings in notes | 28.5% of records identified |
| **Phase 3c** | ICD-10 Redaction | Remove code leakage to prevent model "cheating" | 0 strings remaining after redaction |
| **Phase 4** | Parquet Export | Persist annotated Gold layer for downstream use | File verified; audit closed |

---

### The Gold Layer Output

After processing, the pipeline produces a **fully-annotated dataset** with 10,240 clinical records containing:

| Column Type | Description |
|-------------|-------------|
| **ID** | Unique identifier for each record, audit trail anchor |
| **Original Text** | Raw clinical note, original SOAP format, full dialogue |
| **APSO Reordered** | Assessment-first note ordering for optimal model processing |
| **Canonical ICD-10 Code** | Decimal-restored, CDC FY2026 compliant code format |
| **Code Status** | Classification: `billable` (94.3%), `noisy_111` (5.4%), or `placeholder_x` (0.24%) |
| **Extracted Sections** | Individual Subjective, Objective, Assessment, Plan sections, all redacted |
| **Token Estimates** | Pre-computed token counts for truncation-aware sampling |

---

### Key Statistics

| Metric | Value | Business Implication |
|--------|-------|---------------------|
| **Total Records** | 10,240 | Full dataset retained — no data removed |
| **Billable Leaf Codes** | 9,660 (94.3%) | High-quality training labels for production modeling |
| **Valid Parent Codes** | 555 (5.4%) | Real billing practice, retained for robustness |
| **Placeholder-X Codes** | 25 (0.24%) | Legitimate codes, retained with status annotation |
| **Token Truncation Risk** | 64.7% notes exceed 512 tokens | APSO-Flip addresses this before modeling |
| **ICD-10 Code Leakage** | 28.5% → 0% (redacted) | Model will train on clinical reasoning, not pattern-matching |
| **SOAP Extraction** | 100% success | Reliable for downstream feature engineering |

---

### Important Considerations for Downstream Use

1. **Model Input**: Use the `apso_note` column as primary input — it's Assessment-first and redacted. The `note` column contains original unredacted text for reference tasks.

2. **Label Selection**: Filter by `code_status` as appropriate:
   - **Strict mode** (billable only): 9,660 records for production-grade training
   - **Maximum data**: 10,240 records including parent codes
   - **Exclude placeholder-X**: 10,215 records if CDC description embedding is required

3. **Model Behavior**: The redaction marker `[REDACTED]` is preserved in the text — models will see this placeholder where codes were removed.

4. **Token Management**: Consider stripping markdown formatting (`**`) before tokenization to reduce context window usage.

---

### Data Quality Guarantees

| Guarantee | How It's Verified |
|-----------|-------------------|
| **No Data Loss** | Every transformation preserves record count; no filters applied |
| **Traceability** | Every record has immutable ID; full audit trail maintained |
| **Canonical Format** | ICD-10 codes validated against CDC FY2026 government standard |
| **Leakage-Free** | Automated and manual verification of zero explicit codes in input |
| **Reproducibility** | All transformations deterministic; versioned Parquet exports |

---

### Technical Architecture

```
Raw MedSynth Data (10,240 records)
         │
         ▼
    Phase 0: Infrastructure Setup
         │
         ▼
    Phase 1: Data Ingestion & Validation
         ├── Structural integrity checks
         ├── CDC FY2026 validation
         ├── Token pressure analysis
         ├── Human visual audits
         └── Persistent Silver layer
         │
         ▼
    Phase 2: Gold Layer Construction
         ├── Canonical decimal restoration
         ├── CDC-based status annotation
         └── Human verification
         │
         ▼
    Phase 3: Signal Optimization
         ├── APSO-Flip (Assessment-first ordering)
         ├── Leakage quantification (28.5% detected)
         └── Targeted redaction → 0 remaining
         │
         ▼
    Phase 4: Export
         └── Versioned Parquet artifact
```

---

## 02-Model_ClinicalBERT_Baseline_ICD3.ipynb

### Purpose
This notebook establishes the first performance baseline for the project. It tests whether a pre-trained medical transformer (`Bio_ClinicalBERT`) can accurately classify clinical notes when labels are grouped into ICD-3 categories to manage high cardinality.

### Key Implementation Details
- **Label Engineering**: Implemented **ICD-3 Stem Grouping**, collapsing 2,037 unique ICD-10 codes into ~675 categories (e.g., `M25.562` → `M25`). This increases the average sample density per class from ~5 to ~15 records.
- **Input Strategy**: Utilises the `apso_note` payload, ensuring the diagnostic "Assessment" section is at Token 0 to prevent signal loss during truncation.
- **Training Setup**: 80/10/10 stratified split (Train/Val/Test), utilising Apple Silicon MPS acceleration and MLflow for experiment tracking.

### Results & Findings
- **Performance**: The model achieved a peak **Macro F1 of ~0.76** and **Accuracy of ~0.83** at epoch 19.
- **Observation**: Performance was primarily constrained by the "long tail" of low-frequency classes (36% of categories had only 5 records total), which suppressed the Macro F1 score.
- **Conclusion**: The baseline validates that the APSO-recomposed notes provide a strong diagnostic signal and that `Bio_ClinicalBERT` is an effective backbone for this task.

---

## 03-Model_ClinicalBERT_Surgical_ICD10.ipynb

### Purpose
This notebook implements a **two-stage hierarchical classification architecture** to map clinical notes to full ICD-10 codes. It addresses the "granularity gap" and extreme label sparsity of ICD-10 by breaking the prediction into a coarse-to-fine pipeline: first predicting the ICD-10 Chapter, then the specific code within that chapter.

### Key Implementation Details
- **Hierarchical Architecture**:
    - **Stage 1 (Chapter Router)**: A 22-way classifier that predicts the ICD-10 chapter. This leverages higher data density (~440 records per chapter) to provide a stable routing signal.
    - **Stage 2 (Within-Chapter Resolver)**: A set of specialised classifiers that predict the specific leaf code within the identified chapter.
- **Experimental Evolution**:
    - **E-002 (Flat Baseline)**: A standard flat classifier predicting 1,926 codes simultaneously.
    - **E-003 (Fresh Hierarchical)**: Hierarchical approach trained from scratch.
    - **E-004a (Initialised Hierarchical)**: Hierarchical approach initialised with E-002 weights to "warm-start" the model's ICD-10 knowledge.
    - **E-005a (Extended Hierarchical)**: Targeted fine-tuning of the 6 weakest chapters to push the performance ceiling.
- **Data Engineering**:
    - Filtered to `billable` records only.
    - Implemented a **stratified 80/10/10 split by chapter**.

### Results & Findings
- **The Power of Initialisation**: The shift from E-003 (10.6% accuracy) to E-004a (65.8% accuracy) proved that initialising Stage-2 with a flat ICD-10 model provides a **6.2x improvement** in within-chapter discrimination.
- **Hierarchical vs. Flat**: The final hierarchical model (**E-004a**) achieved **65.8% End-to-End Accuracy**, outperforming the flat baseline (E-002) by **+18.9 percentage points**.
- **Performance Bottlenecks**:
    - **Chapter Z** remains the primary challenge due to the high ambiguity of administrative health codes.
    - **Chapter U** achieved 100% accuracy, while other chapters like B and C showed strong performance (>75%).
- **Conclusion**: The hierarchical architecture with E-002 initialisation is definitively superior to flat ICD-10 classification.

---

## Architectural Decisions — Script Migration & Model Strategy

*Recorded: April 2026. These decisions were made before migrating the notebook pipeline to scripts and before introducing any additional models beyond Bio_ClinicalBERT.*

---

### Decision 1: Use Case — Automated Coding with Human Review of Exceptions (Option B)

**Chosen approach:** The model codes autonomously when confidence is high, and flags records for human review when it is not.

**Architecture implication:** The system needs calibrated confidence scores alongside every prediction. Records above the confidence threshold are auto-coded and logged for audit; records below go to a human review queue.

---

### Decision 2: Label Scheme

| Experiment | Architecture | Accuracy |
|---|---|---|
| E-001 | ICD-3 flat (675 classes) | 82.7% |
| E-002 | ICD-10 flat (1,926 classes) | 46.9% |
| E-003 | ICD-10 hierarchical, fresh | 10.6% |
| E-004a | ICD-10 hierarchical, transfer | 66.9% |
| E-005a | E-004a + extended epochs | 66.9% |

**Working conclusion:** ICD-3 (82.7%) is the more defensible primary target for automated coding in production. ICD-10 is the aspirational target, viable as a second stage for high-confidence predictions.

---

### Decision 3: Model Strategy — Encoder-First, Adapter Interface Now

```
Clinical note
      │
      ▼
┌─────────────────────────────┐
│  Encoder model              │  Fast, cheap, runs locally
│  (ClinicalBERT → MedBERT)  │  Produces: top-k ICD codes + confidence
└─────────────────────────────┘
      │
      ├── High confidence ──→  Auto-code, log for audit
      │
      └── Low confidence  ──→ ┌─────────────────────────────┐
                              │  Generative model           │
                              │  (MedGemma, BioMistral)    │
                              └─────────────────────────────┘
                                      │
                                      └── Still uncertain ──→ Human review
```

---

## Project Status — April 2026

### Git Setup

| Branch | Purpose |
|---|---|
| `baseline` | Permanent snapshot — safe revert point |
| `main` | Stable trunk |
| `feature/script-layer` | Active working branch |

---

### Script Layer — Completed April 2026

#### Scripts Built

| Script | Purpose |
|---|---|
| `scripts/prepare_data.py` | Gold layer pipeline — notebook 01 equivalent |
| `scripts/train.py` | Flat and hierarchical training; `--chapters` filter for targeted retraining; `--gold-path` override for augmented data |
| `scripts/evaluate.py` | Top-1/5, Macro F1, ECE, threshold sweep, per-chapter breakdown |
| `scripts/calibrate.py` | Temperature scaling per resolver — writes `temperature.json` |
| `scripts/predict.py` | Single-note inference CLI with Use Case B routing; JSON and human-readable output modes |
| `scripts/augment.py` | Targeted synthetic data augmentation using Pydantic AI + LM Studio; incremental saves with full resume support |
| `scripts/serve.py` | FastAPI model server entrypoint |
| `src/adapters.py` | `ModelAdapter` interface, `EncoderAdapter`, `GenerativeAdapter` stub |
| `src/inference.py` | `HierarchicalPredictor` with calibrated temperatures and `preprocessed=True` flag |
| `src/server.py` | FastAPI application — loads all 19 resolvers once at startup; serves `/predict`, `/predict/batch`, `/health`, `/info` |

#### Full Experiment Results

| Experiment | Description | Notebook Target | Script Result | Status |
|---|---|---|---|---|
| **E-001_v3** | ICD-3 flat baseline (675 classes) | 82.7% acc / 0.760 F1 | **87.0% acc / 0.824 F1** | ✅ Exceeds |
| **E-002** | Full ICD-10 flat (2,037 classes) | 46.9% acc | **54.3% acc / 0.426 F1** | ✅ Exceeds |
| **E-003 Stage-1** | Chapter router (22 classes) | 95.4% acc | **98.9% acc / 0.973 F1** | ✅ Exceeds |
| **E-004a** | Hierarchical E2E (E-002 init) | 66.9% acc / 0.553 F1 | **76.6% acc / 0.674 F1** | ✅ Exceeds |
| **E-005b** | O chapter augmented resolver | — | O: 53.1% → **68.6%** (+15.5pp) | ✅ |
| **E-005c** | Merged pipeline — best result | — | **77.0% acc / 0.679 F1** | ✅ Current best |

#### Current Best Pipeline — E-005c

| Metric | Value |
|---|---|
| E2E Accuracy | **77.0%** |
| Macro F1 | **0.679** |
| ECE | **0.027** |
| Coverage @ τ=0.7 | **68.5%** |
| Accuracy on auto-coded subset | **93.6%** |

The pipeline auto-codes 68.5% of cases with 93.6% accuracy and routes the remaining 31.5% to human review, fully implementing the Use Case B design decision.

#### Per-Chapter Accuracy (E-005c)

| Chapter | Accuracy | n |
|---|---|---|
| A | 50.0% | 6 |
| B | 75.0% | 16 |
| C | 72.6% | 51 |
| D | 92.1% | 38 |
| E | 82.1% | 39 |
| F | 77.1% | 48 |
| G | 85.3% | 34 |
| H | 71.4% | 42 |
| I | 87.9% | 66 |
| J | 78.0% | 50 |
| K | 79.3% | 58 |
| L | 86.8% | 38 |
| M | 82.0% | 111 |
| N | 83.0% | 53 |
| O | **68.6%** | 51 |
| R | 82.0% | 100 |
| S | 79.5% | 44 |
| T | 87.5% | 8 |
| Z | 55.3% | 132 |

#### Model Server

The pipeline is deployed as a FastAPI server that loads all 19 resolver models once at startup and serves predictions at ~135ms per note.

```bash
uv run python scripts/serve.py
# Swagger UI: http://localhost:8000/docs
```

Endpoints: `GET /health`, `GET /info`, `POST /predict`, `POST /predict/batch` (up to 100 notes). CORS enabled.

#### Key Bugs Fixed During Migration

| Bug | Impact | Fix |
|---|---|---|
| `evaluate.py` double-preprocessing | `apso_note` re-processed, corrupting extraction — 51.5% on an 87% model | Tokenise `apso_note` directly; add `preprocessed=True` flag |
| `inference.py` model path depth | Weights at `stage2/A/model/model/` not `stage2/A/model/` | Corrected path: `hf_dir = model_dir / "model"` |
| `adapters.py` checkpoint metric | `eval_loss` saved worst-F1 checkpoint | Changed to `metric_for_best_model="macro_f1"`, `greater_is_better=True` |
| `train.py` code-filter default | `billable` excluded 580 records vs notebooks using all 10,240 | Changed default to `all` |

---

## Chapter Augmentation — April 2026

### Background

Chapters Z and O had only ~4 training records per code vs ~8 for every other chapter, caused by the stratified 80/10/10 split hitting the 5-records-per-code floor. This 2x density gap was the primary driver of their lower accuracy relative to all other chapters.

### Augmentation Pipeline

`scripts/augment.py` generates synthetic SOAP notes using **Pydantic AI** for structured, validated generation and **LM Studio** (local, free, no API costs) as the inference backend.

**Model used:** `qwen/qwen3.6-35b-a3b` (35B MoE, 3.6B active parameters) running locally on Apple M5 Max with 129GB RAM.

**Key design decisions:**
- Single note per API call — eliminates JSON truncation failures entirely
- Incremental saves after every code — safe to interrupt and resume; nothing is lost
- Schema-safe — reads gold layer Polars schema at startup and casts all new rows to match exactly
- Async parallel generation — 3 notes per code generated concurrently

**LM Studio settings required for reliable generation:**
- Context Length: 32,768 (not the default 4,096)
- Enable Thinking: OFF (prevents empty `content` responses from Qwen3's reasoning mode)

### Results

| Chapter | Codes augmented | Records added | Training density |
|---|---|---|---|
| O (Pregnancy, Childbirth) | 63 | 189 | ~4 → 8 records/code |
| Z (Factors Influencing Health) | 262 | 785 | ~4 → 8 records/code |
| **Total** | **325** | **974** | — |

Gold layer: 10,240 original + 974 synthetic = **11,214 records** in `medsynth_gold_augmented.parquet`.

### Augmentation Findings

| Chapter | E-004a baseline | E-005b augmented | Outcome |
|---|---|---|---|
| **O** | 53.1% | **68.6%** | +15.5pp — augmentation worked ✅ |
| **Z** | 55.3% | 41.7% | -13.6pp — augmentation hurt ❌ |

**Chapter O** improved significantly as expected — data volume was the limiting factor and augmentation solved it directly.

**Chapter Z** regressed. The 263 Z codes are administratively similar (screening visits, history codes, health status codes), and synthetic notes generated for these codes were too homogeneous to be discriminative. The model learned to spread probability mass across Z codes rather than discriminate between them. Chapter Z requires a fundamentally different approach — contrastive fine-tuning or retrieval augmentation — not more data of the same kind.

**Decision:** Keep augmented O resolver, revert Z to E-004a's original resolver. This is the **E-005c merged pipeline**, which is the current best result.

### Net Effect on E2E Performance

| Metric | E-004a | E-005c | Change |
|---|---|---|---|
| E2E Accuracy | 76.6% | **77.0%** | +0.4% |
| Macro F1 | 0.674 | **0.679** | +0.005 |
| ECE | 0.028 | **0.027** | improved |
| Coverage @ τ=0.7 | 68.4% | **68.5%** | stable |

---

## Next Steps — Prioritised Roadmap

### Immediate — Chapter Z: Contrastive Fine-Tuning

Data augmentation failed for Z because the codes are semantically too similar for BERT to distinguish given the training signal available. The right approach is **contrastive fine-tuning**: train the Z resolver with SimCSE or SupCon loss that explicitly pulls embeddings for similar-looking codes apart in representation space. This is a 1-2 week research effort and the highest-value remaining accuracy improvement.

### Short Term — MIMIC-IV Validation

MedSynth is synthetic — generated by GPT-4o with balanced labels. The model has learned patterns in generated clinical text, not real EHR notes. Running the existing E-005c pipeline against **MIMIC-IV** (freely available under a data use agreement at physionet.org) without retraining would immediately reveal the synthetic-to-real performance gap and which chapters are most affected. This is the most important validation step before any production consideration.

### Medium Term — GenerativeAdapter

The architecture already has the `GenerativeAdapter` stub and `HybridRouter` concept in `src/adapters.py`. When the encoder is uncertain (below threshold, small gap between top-1 and top-2 confidence), escalate to a generative model (MedGemma) that can reason about the note rather than classify it. The scaffolding is in place — implementing it is a 2-3 week project.

### Longer Term — Model Comparison

MedBERT and PubMedBERT are direct drop-in replacements for Bio_ClinicalBERT via the `EncoderAdapter` interface — a single config value change. Running E-004a with each would provide a clean model comparison at no additional architecture cost.

---

### Experiments Proposed in Notebooks (not yet implemented)

| Experiment | Proposed Change | Notes |
|---|---|---|
| **E-005b (original)** | Add dialogue as supplementary input | Could help Z — dialogue has different signal than summary note |
| **E-005c (original)** | Markdown stripping before tokenisation | Low-effort, recovers context window space |
| **E-005d** | Apply to real clinical dataset | Maps to MIMIC-IV validation above |
| **E-006** | Ensemble E-002 flat + E-005c hierarchical | E-005c already far exceeds flat baseline; marginal gain expected |