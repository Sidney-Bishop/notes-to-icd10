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
- **Label Engineering**: Implemented **ICD-3 Stem Grouping**, collapsing 2,037 unique ICD-10 codes into ~675 categories (e.g., `M25.562` $\rightarrow$ `M25`). This increases the average sample density per class from ~5 to ~15 records.
- **Input Strategy**: Utilizes the `apso_note` payload, ensuring the diagnostic "Assessment" section is at Token 0 to prevent signal loss during truncation.
- **Training Setup**: 80/10/10 stratified split (Train/Val/Test), utilizing Apple Silicon MPS acceleration and MLflow for experiment tracking.

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
    - **Stage 2 (Within-Chapter Resolver)**: A set of specialized classifiers that predict the specific leaf code within the identified chapter.
- **Experimental Evolution**:
    - **E-002 (Flat Baseline)**: A standard flat classifier predicting 1,926 codes simultaneously.
    - **E-003 (Fresh Hierarchical)**: Hierarchical approach trained from scratch.
    - **E-004a (Initialized Hierarchical)**: Hierarchical approach initialized with E-002 weights to "warm-start" the model's ICD-10 knowledge.
    - **E-005a (Extended Hierarchical)**: Targeted fine-tuning of the 6 weakest chapters to push the performance ceiling.
- **Data Engineering**: 
    - Filtered to `billable` records only.
    - Implemented a **stratified 80/10/10 split by chapter**.

### Results & Findings
- **The Power of Initialization**: The shift from E-003 (10.6% accuracy) to E-004a (65.8% accuracy) proved that initializing Stage-2 with a flat ICD-10 model provides a **6.2x improvement** in within-chapter discrimination.
- **Hierarchical vs. Flat**: The final hierarchical model (**E-004a**) achieved **65.8% End-to-End Accuracy**, outperforming the flat baseline (E-002) by **+18.9 percentage points**.
- **Performance Bottlenecks**: 
    - **Chapter Z** remains the primary challenge (**29.7% accuracy**) due to the high ambiguity of administrative health codes.
    - **Chapter U** achieved 100% accuracy, while other chapters like B and C showed strong performance (>75%).
- **Conclusion**: The hierarchical architecture with E-002 initialization is definitively superior to flat ICD-10 classification. It successfully maps clinical notes to 1,926 codes with high reliability across most chapters, despite having only ~4 training examples per code.


1. **Synthetic Data**: Generated by GPT-4o — linguistic patterns may not match real clinical notes
2. **Balanced Labels**: 5 records per code — unlike real-world imbalanced distributions  
3. **100% SOAP Extraction**: Expected for synthetic data; real EHR notes may have format variation
4. **Token Heuristics**: Uses `words × 1.3` estimate rather than actual ClinicalBERT tokenization

### Next Steps & Future Work

Following the completion of the E-005a pipeline, the following experiments are recommended to break the current performance ceiling:

| Experiment | Proposed Change | Expected Impact |
|---|---|---|
| **E-005a** | Extend Stage-2 to 30 epochs | Push Z-chapter and weak resolvers past current ceiling |
| **E-005b** | Add dialogue as supplementary input | Test if raw transcripts add signal beyond the summary note |
| **E-005c** | Markdown stripping before tokenisation | Recover context window space by removing `**` tokens |
| **E-005d** | Apply to real clinical dataset | Validate performance on non-uniform, real-world distributions |
| **E-006** | Ensemble E-002 flat + E-004a hierarchical | Combine the strengths of both architectures |


---

## Architectural Decisions — Script Migration & Model Strategy

*Recorded: April 2026. These decisions were made before migrating the notebook pipeline to scripts and before introducing any additional models beyond Bio_ClinicalBERT.*

---

### Context: Why These Decisions Were Made Now

The notebook pipeline (notebooks 01–05) was completed and validated. Before converting any notebook to a script, three fundamental architectural questions were identified that needed answering first — because the wrong answers would produce scripts that are immediately obsolete or that lock the project into a single model.

---

### Decision 1: Use Case — Automated Coding with Human Review of Exceptions (Option B)

**Chosen approach:** The model codes autonomously when confidence is high, and flags records for human review when it is not.

This was selected over two alternatives:

- **Option A (Coding assistant, top-k, human confirms everything)** — ruled out because it doesn't reduce human workload; a coder still reviews every record.
- **Option C (Hierarchical human-in-the-loop, chapter → code interactively)** — ruled out as too slow for production volume; the stage-1 chapter routing is reliable enough to run autonomously.

**Architecture implication:** The system needs calibrated confidence scores alongside every prediction — not just the top-1 code, but a principled measure of how certain the model is. Records above the confidence threshold are auto-coded and logged for audit; records below go to a human review queue. This is a different training objective than pure accuracy, and evaluation metrics must include confidence calibration, not just top-1 / Macro F1.

---

### Decision 2: Label Scheme — The Uncomfortable Finding From the Experiments

The five experiments tell a clear story with an uncomfortable implication for the ICD-10 target:

| Experiment | Architecture | Accuracy |
|---|---|---|
| E-001 | ICD-3 flat (675 classes) | 82.7% |
| E-002 | ICD-10 flat (1,926 classes) | 46.9% |
| E-003 | ICD-10 hierarchical, fresh | 10.6% |
| E-004a | ICD-10 hierarchical, transfer | 66.9% |
| E-005a | E-004a + extended epochs | 66.9% |

**66.9% on ICD-10 looks good until you ask what it means clinically.** A wrong prediction that is in the right chapter and right category (e.g. predicting E11.9 when the correct code is E11.65) is a very different failure from predicting the wrong disease entirely. The flat accuracy metric treats both failures identically.

The Z-chapter result (32.6% end-to-end accuracy) is not an edge case — administrative and circumstantial codes represent a substantial fraction of real clinical coding workload.

**Working conclusion:** ICD-3 (82.7% accuracy) is the more defensible primary target for automated coding in a production context. ICD-10 is the aspirational target, viable as a second stage for high-confidence ICD-3 predictions, or once a stronger model baseline is established. The hybrid design — high-confidence ICD-10 direct, fall back to ICD-3 + human review — is the most pragmatically useful near-term design.

---

### Decision 3: Model Strategy — Encoder-First, Adapter Interface Now

**The core problem with the current pipeline:** Bio_ClinicalBERT, the training strategy, and the label scheme are all baked together inside each notebook. Swapping models would require forking all five notebooks and manually changing model references throughout. This is not sustainable.

**The fundamental model split:**

*Encoder models (ClinicalBERT, MedBERT, PubMedBERT, clinical RoBERTa)*
Take the full input, produce a single `[CLS]` vector, attach a linear classification head, train end-to-end. Fast, memory-efficient, well-understood. The entire current pipeline is built on this pattern.

*Generative models (MedGemma, BioMistral, GPT-4o)*
Predict the next token autoregressively. Classification requires either prompting and output parsing, or constrained decoding to valid ICD code strings. Slower and more expensive, but potentially stronger at clinical reasoning due to larger pretraining corpora and more recent medical knowledge.

**These are not competing approaches — they are complementary layers:**

```
Clinical note
      │
      ▼
┌─────────────────────────────┐
│  Encoder model              │  Fast, cheap, runs locally
│  (ClinicalBERT → MedBERT   │  Produces: top-k ICD codes
│   → future encoder)        │           + confidence scores
└─────────────────────────────┘
      │
      ├── High confidence ──→  Auto-code, log for audit
      │
      └── Low confidence  ──→ ┌─────────────────────────────┐
                              │  Generative model           │  Slower, runs on
                              │  (MedGemma, BioMistral,    │  ~20-30% of cases
                              │   GPT-4o via API)          │  Produces: code +
                              │                            │  clinical rationale
                              └─────────────────────────────┘
                                      │
                                      └── Still uncertain ──→ Human review queue
```

**Chosen approach: encoder-first, with the adapter interface designed now.**

Reasons:
- The existing ClinicalBERT work provides a validated baseline to compare against
- MedBERT, PubMedBERT, and clinical RoBERTa variants are direct drop-in replacements — meaningful model comparison is available immediately by changing a single config value
- MedGemma's ICD coding capability on this specific task is unknown; encoder baselines are needed to evaluate it against anyway
- Adding a generative adapter later is straightforward if the interface is clean from the start

---

### Resulting Script Architecture

The script layer must separate concerns that the notebooks currently conflate:

```
config layer     what model, what label scheme, what hyperparameters
────────────────────────────────────────────────────────────────────
data layer       Gold layer → splits (independent of model)
────────────────────────────────────────────────────────────────────
model layer      ModelAdapter interface — any model implements this
────────────────────────────────────────────────────────────────────
training layer   flat trainer / hierarchical trainer (strategy)
────────────────────────────────────────────────────────────────────
evaluation layer top-1, top-k, chapter accuracy, confidence calibration
```

The `ModelAdapter` interface is the key abstraction. Any model — encoder or generative — implements it, so the data pipeline, routing logic, and evaluation layer never need to know which model is underneath:

```python
class ModelAdapter:
    def predict(self, note: str, top_k: int) -> PredictionResult:
        # Returns: codes, scores, confidence, metadata
        ...

class EncoderAdapter(ModelAdapter):
    # Wraps any HuggingFace encoder: ClinicalBERT, MedBERT, PubMedBERT, etc.
    ...

class GenerativeAdapter(ModelAdapter):
    # Wraps MedGemma, BioMistral, GPT-4o
    # Constrained decoding or structured output parsing
    ...

class HybridRouter:
    def __init__(self, fast: ModelAdapter, strong: ModelAdapter, threshold: float):
        ...
    def predict(self, note: str) -> PredictionResult:
        result = self.fast.predict(note)
        if result.confidence < self.threshold:
            result = self.strong.predict(note)
        return result
```

---

---

## Project Status — April 2026

### Git Setup

The project is under version control with three branches, all currently pointing at the same initial commit:

| Branch | Purpose |
|---|---|
| `baseline` | Permanent snapshot — never touch it. Safe revert point for the refactored notebook pipeline. |
| `main` | Stable trunk — receives merges from feature branches when they are ready and tested. |
| `feature/script-layer` | Active working branch — all script layer and adapter work happens here. |

### Completed

- All 5 `src/` files refactored and aligned to the notebook ground truth — `preprocessing.py`, `inference.py`, `config.py`, `data_loader.py`, `dataset.py`
- Architectural decisions documented in this file (see section above)
- Clean initial git commit with meaningful message and correct `.gitignore` — binaries, weights, and data excluded; all JSON metadata and source tracked
- `baseline` branch created as a permanent snapshot

### Next Up on `feature/script-layer`

1. **`scripts/prepare_data.py`** — notebook 01 equivalent (EDA, APSO-Flip, redaction, Gold layer export). This notebook is self-contained and ready to migrate; no architectural decisions block it.

2. **`src/adapters.py`** — the `ModelAdapter` interface and `EncoderAdapter` implementation. This is the foundation everything else builds on.

3. **`scripts/train.py`** — training script using the adapter interface; model is a config value, not a hard-coded import.

4. **`scripts/evaluate.py`** — evaluation script with confidence calibration metrics alongside accuracy and Macro F1.