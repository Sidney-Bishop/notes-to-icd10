# Project Notes: Notes to ICD-10 Pipeline

This document serves as the centralized architectural record and scientific log for the development of the ICD-10 coding pipeline. It tracks the evolution from raw data ingestion to hierarchical model design.

---

## 📓 Notebook 01: EDA & SOAP Processing (`01-EDA_SOAP.ipynb`)
**Objective:** Transform raw MedSynth synthetic dialogues into a validated, "Gold Layer" dataset ready for NLP tasks.

### Key Findings & Engineering Decisions:
- **The Truncation Problem:** Discovered that ~64.7% of clinical notes exceed Bio_ClinicalBERT's 512-token limit.
- **APSO-Flip Strategy:** Implemented a restructuring of SOAP notes to place the **Assessment** section at Token 0, ensuring critical diagnostic signals are not truncated.
- **Zero-Trust Ingestion:** Applied Pydantic schemas to validate every record and redacted explicit ICD-10 strings from text to prevent label leakage.
- **The "Noisy 111" Discovery:** Identified a subset of non-billable parent codes (5.4% of data) that create ambiguity at the full ICD-10 level but collapse cleanly at the ICD-3 level.

---

## 📓 Notebook 02: ICD-3 Baseline (`02-Model_ClinicalBERT_Baseline_ICD3.ipynb`)
**Objective:** Establish a performance floor using a simplified label space (675 ICD-3 categories).

### Key Findings & Results (M5 Max Pure Run):
- **Performance:** Achieved **82.8% Accuracy** and **0.762 Macro F1**.
- **The High-Epoch Necessity:** Proved that high-cardinality classification requires extended training; performance jumped from 0.442 (10 epochs) to 0.762 (20 epochs).
- **Top-5 Utility:** A Top-5 Accuracy of **92.0%** suggests the model is highly effective as a "suggestion engine" for human coders.
- **Hardware Benchmark:** Established a baseline runtime of **~102 minutes** on Apple M5 Max (MPS), proving the feasibility of rapid iteration.

---

## 📓 Notebook 03: Full ICD-10 Classification (`03-Model_ClinicalBERT_Surgical_ICD10.ipynb`)
**Objective:** Test the limits of a "flat" classifier by expanding to the full billable label space (1,926 codes).

### Key Findings & Results (M5 Max Pure Run):
- **The Cardinality Crash:** Accuracy dropped from 82.8% (ICD-3) to **52.3%** (Full ICD-10), and Macro F1 fell to **0.409**. This quantifies the difficulty of predicting specific leaf codes with only ~4 examples per class.
- **The "Hidden Signal":** Despite the drop in code-level accuracy, **Chapter-level accuracy remained high at 84.4%**.
- **Error Structure:** Found that **62.0% of errors are within-chapter**, meaning the model identifies the correct clinical domain but struggles with fine-grained resolution.
- **Coverage Adjustment:** Calculated a coverage-adjusted Macro F1 of **~0.82**, suggesting that per-class learning efficiency is actually higher at the full ICD-10 level than at the ICD-3 level.

---

## 🚀 Strategic Roadmap: The Path to Hierarchical Modeling

The results from E-001 and E-002 provide a definitive empirical mandate for a **Two-Stage Hierarchical Architecture**:

1.  **Stage 1 (Router):** A high-accuracy classifier to predict the ICD-10 Chapter (leveraging the ~84% accuracy found in E-002).
2.  **Stage 2 (Resolver):** Specialized classifiers for each chapter, reducing the label space from **1,926 $\rightarrow$ ~100**, thereby increasing the signal-to-noise ratio and resolving the "within-chapter" confusion identified in the confusion matrices.

**Current Status:** Baseline locked. Architecture validated. Ready for Hierarchical implementation.
