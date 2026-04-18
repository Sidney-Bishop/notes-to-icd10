

<div style="max-width: 850px; line-height: 1.6; font-family: sans-serif;">

# 📑 Table of Contents: E-003 Hierarchical ICD-10 Classification

---

### 🧠 Hierarchical Two-Stage Architecture
Experiment objective, motivation from E-002 chapter confusion analysis,
and comparison baseline.

### 🧪 Experiment Log: Scientific Record (E-003)
Official configuration and results.

### 🔬 Phase 1: Experiment Configuration
MLflow SQLite backend, Gold layer Parquet discovery, E-001 registry
path verification, Stage-1 and Stage-2 hyperparameters.

### ⚙️ Phase 1b: Environment Setup & Imports
HuggingFace local cache, MPS fallback, seed locking, Stage-1/Stage-2
output directories.

### 📥 Phase 2: Data Loading & Label Derivation
Gold layer ingestion, billable-only filter (9,660 records), chapter
label derivation (22 classes), shared train/val/test split.

### 📊 Phase 2 Observations
Chapter distribution, skip chapter decision, training examples per
chapter, Stage-1 data advantage over E-002.

### 🧭 Phase 3a: Stage-1 Setup
Tokenisation, E-001 model loading, 22-way head replacement.

### 📊 Phase 3b: Stage-1 Trainer Configuration
TrainingArguments, TensorBoard and MLflow monitoring commands.

### 🚀 Phase 3c: Stage-1 Training Ignition
10-epoch chapter router training. Best checkpoint: epoch 9.

### 📊 Stage-1 Interpretation
Training curve analysis, E-001 initialisation impact, comparison
with E-002 flat chapter accuracy.

### 📊 Phase 3b Evaluation: Stage-1 Test Set
Definitive test routing accuracy, per-chapter routing breakdown,
routing error budget for Stage-2.

### 📊 Stage-1 Test Set Interpretation
Per-chapter routing reliability, problem chapters (T, Z, S),
Stage-2 priority order.

### 🔬 Phase 4a: Stage-2 Data Preparation
Per-chapter dataset filtering, label encoders, tokenisation,
skip chapter fallback predictions.

### 📊 Phase 4b: Stage-2 Trainer Configuration
`train_chapter_resolver()` function, fresh Bio_ClinicalBERT
rationale, TensorBoard monitoring.

### 🚀 Phase 4c: Stage-2 Training Loop
19 chapter resolvers trained in priority order. All 19 saved
successfully. Weighted val accuracy: 13.6%.

### 🎯 Phase 5: End-to-End Pipeline Evaluation
Full two-stage pipeline evaluation on test set. E2E accuracy:
10.6%, Macro F1: 0.070.

### 📊 E-003 Results: Interpretation
Stage-2 failure analysis, data fragmentation diagnosis, E-004a fix.

### 🏆 Phase 6: Registry Promotion
Stage-1 router and experiment metadata saved to registry.
MLflow run closed.

---

### 🎯 Experiment Objective

Implement a hierarchical two-stage ICD-10 prediction pipeline
motivated by E-002's finding that 82.9% of predictions land in the
correct chapter while only 46.9% reach the correct specific code.

**Official E-003 Results:**

| Stage | Metric | Value |
|---|---|---|
| Stage-1 | Chapter routing accuracy | 95.3% |
| Stage-1 | Chapter Macro F1 | 0.959 |
| Stage-2 | Within-chapter accuracy | 11.1% |
| End-to-end | Accuracy | 10.6% |
| End-to-end | Macro F1 | 0.070 |

**Key finding:** Stage-1 significantly outperformed E-002's flat
chapter accuracy (+12.4pp). Stage-2 underperformed due to training
data fragmentation and fresh Bio_ClinicalBERT initialisation.
Fix: initialise Stage-2 from E-002 registry model in E-004a.

</div>




# 🧠 Hierarchical Two-Stage ICD-10 Classification (E-003)

## 🔬 Phase 1: Experiment Configuration (E-003)

## ⚙️ Phase 1b: Environment Setup & Imports (E-003)

## 📥 Phase 2: Data Loading & Label Derivation (E-003)

## 🧭 Phase 3a: Stage-1 Setup — Tokenisation & Model Initialisation

## 📊 Phase 3b: Stage-1 Trainer Configuration

## 🚀 Phase 3c: Stage-1 Training Ignition

## 📊 Phase 3d : Stage-1 Training Dashboard Capture

## 🏆 Phase 3f: Stage-1 Model Registry Promotion

## 📊 Phase 3g Evaluation: Stage-1 Test Set Performance

## 🔬 Phase 4a: Stage-2 Data Preparation

## 🔬 Stage-2 Data Preparation: Strategic Analysis

## 📊 Phase 4b: Stage-2 Trainer Configuration

## 🚀 Phase 4c: Stage-2 Training Ignition — All Chapter Resolvers

## 📊 Stage-2 Training: Post-Loop Analysis

## 📊 Phase 4c Cell 2: Stage-2 Val Results Summary

## 📊 Phase 4c Cell 3: MLflow Logging & Audit Trail

## 🎯 Phase 5: End-to-End Pipeline Evaluation

## 🏆 Phase 6: Registry Promotion (E-003)


































