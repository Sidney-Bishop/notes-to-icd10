<p align="center">
  <img src="notebooks/resources/images/notes-to-icd10-logo.png" alt="notes-to-icd10" width="700"/>
</p>

# Notes to ICD-10

[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Model](https://img.shields.io/badge/model-Bio_ClinicalBERT-green.svg)](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT)
[![Dataset](https://img.shields.io/badge/dataset-MedSynth-orange.svg)](https://huggingface.co/datasets/Ahmad0067/MedSynth)
[![Hardware](https://img.shields.io/badge/hardware-Apple%20M4%20Max-black.svg)](https://www.apple.com/mac/)

Two-stage hierarchical ICD-10 coding from clinical notes using Bio_ClinicalBERT —
**66.9% accuracy across 1,926 ICD-10 codes** from ~4 training examples per code.

---

## 🏆 Results

| Experiment | Architecture | Accuracy | Macro F1 |
|---|---|---|---|
| E-001 | ICD-3 flat, 675 classes | 82.7% | 0.760 |
| E-002 | ICD-10 flat, 1,926 classes | 46.9% | 0.352 |
| E-003 | Hierarchical, fresh BERT | 10.6% | 0.070 |
| E-004a | Hierarchical, E-002 init | 66.7% | 0.551 |
| **E-005a** | **E-004a + extended epochs** | **66.9%** | **0.553** |

**Best model (E-005a):** 66.9% top-1 accuracy, 0.553 Macro F1, 95.4% chapter
routing accuracy, 70.1% within-chapter accuracy.

---

## 🎯 Overview

This project builds an end-to-end pipeline that predicts specific ICD-10
diagnostic codes from APSO-structured clinical notes. The core finding is
that a **two-stage hierarchical architecture with transfer-learned
initialisation** substantially outperforms flat ICD-10 classification —
+20pp accuracy over the flat baseline on an extremely low-resource task.

### Key Findings

- **Flat ICD-10 classification** (E-002) achieves 46.9% — a strong baseline
  given ~4 training examples per code across 1,926 classes
- **Hierarchical architecture fails without correct initialisation** (E-003,
  10.6%) — training Stage-2 resolvers from scratch on chapter-filtered
  subsets is insufficient
- **E-002 initialisation fixes Stage-2** (E-004a, 66.7%) — fine-tuning
  existing ICD-10 representations rather than learning from scratch produces
  a 6.3x within-chapter accuracy improvement (11.1% → 69.8%)
- **Extended training yields marginal gains** (E-005a, +0.2pp) — the
  architecture reaches its ceiling on MedSynth at 20 epochs

---

## 🏗️ Architecture
```
Clinical Note (APSO format)
        │
        ▼
┌─────────────────────────┐
│     Stage-1 Router      │  Bio_ClinicalBERT (E-001 init)
│  22-way ICD-10 chapter  │  95.4% routing accuracy
│     classification      │
└─────────────────────────┘
        │
        ▼  routes to correct chapter resolver
┌─────────────────────────┐
│    Stage-2 Resolver     │  Bio_ClinicalBERT (E-002 init)
│  Within-chapter ICD-10  │  70.1% within-chapter accuracy
│      prediction         │
└─────────────────────────┘
        │
        ▼
    ICD-10 Code
```

**Stage-1** routes each note to one of 22 ICD-10 chapters (A–Z),
initialised from E-001 (ICD-3 classifier) which is already chapter-aware.

**Stage-2** resolves the specific ICD-10 code within the routed chapter.
19 dedicated resolvers fine-tuned from E-002 weights reduce the effective
label space from 1,926 to an average of ~100 classes per resolver.

**Transfer learning chain:**
```
Bio_ClinicalBERT → E-001 (ICD-3) → Stage-1 init
Bio_ClinicalBERT → E-002 (ICD-10 flat) → Stage-2 init
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11
- Apple Silicon Mac (MPS acceleration) or CUDA GPU
- ~50GB disk space for models and data
- ~16GB RAM minimum, 32GB+ recommended

### Installation
```bash
git clone https://github.com/Sidney-Bishop/notes-to-icd10.git
cd notes-to-icd10
uv sync
```

### Dataset
```python
from datasets import load_dataset
dataset = load_dataset("Ahmad0067/MedSynth")
```

### Run the Pipeline

Run notebooks in order:
```bash
uv run jupyter notebook
```

| Notebook | Experiment | Runtime |
|---|---|---|
| `01-EDA_SOAP.ipynb` | Gold layer generation | ~15 min |
| `02-Model_ClinicalBERT_Baseline_ICD3.ipynb` | E-001 ICD-3 baseline | ~2.5 hrs |
| `03-Model_ClinicalBERT_Surgical_ICD10.ipynb` | E-002 flat ICD-10 | ~2.5 hrs |
| `04-Model_Hierarchical_ICD10.ipynb` | E-003 hierarchical | ~3.5 hrs |
| `05-Model_Hierarchical_ICD10_E002Init.ipynb` | E-004a best model | ~3.5 hrs |
| `05_a-Model_Hierarchical_ICD10_E002Init.ipynb` | E-005a extended | ~1 hr |

Total training time: approximately 13–15 hours on Apple M4 Max.

### Inference
```python
from src.inference import predict

note = """
Assessment: Type 2 diabetes mellitus with hyperglycaemia.
Plan: Adjust metformin dosage, HbA1c recheck in 3 months.
Subjective: Patient reports increased thirst and frequent urination.
Objective: Fasting glucose 14.2 mmol/L, BMI 31.
"""

result = predict(note, top_k=5)
print(f"Top prediction: {result['codes'][0]} ({result['scores'][0]:.1%})")
# Top prediction: E11.65 (84.2%)
```

### Experiment Tracking
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
```

---

## 📁 Project Structure
```text
notes-to-icd10/
├── data/
│   ├── cache/              # HuggingFace model cache (gitignored)
│   ├── gold/               # Gold layer Parquet — APSO-processed
│   └── raw/                # Original MedSynth CSV (gitignored)
├── notebooks/
│   ├── 01-EDA_SOAP.ipynb
│   ├── 02-Model_ClinicalBERT_Baseline_ICD3.ipynb
│   ├── 03-Model_ClinicalBERT_Surgical_ICD10.ipynb
│   ├── 04-Model_Hierarchical_ICD10.ipynb
│   ├── 05-Model_Hierarchical_ICD10_E002Init.ipynb
│   └── 05_a-Model_Hierarchical_ICD10_E002Init.ipynb
├── outputs/
│   └── evaluations/
│       ├── registry/       # Promoted model artifacts (gitignored)
│       └── E-00*/          # Per-experiment training artifacts (gitignored)
├── src/
│   ├── config.py           # Centralised configuration + audit trail
│   ├── data_loader.py      # Gold layer ingestion utilities
│   ├── dataset.py          # HuggingFace Dataset helpers
│   ├── evaluation.py       # Metrics: Macro F1, Accuracy, Top-5
│   ├── gatekeeper.py       # Pydantic validation schemas
│   ├── inference.py        # End-to-end pipeline inference
│   └── preprocessing.py    # APSO note construction
├── artifacts.yaml          # Centralised experiment configuration
├── mlflow.db               # MLflow SQLite experiment tracker (gitignored)
├── pyproject.toml          # uv-managed dependencies
└── uv.lock
```

---

## 🔬 Methodology

### Zero-Trust Ingestion
Every record is validated against a Pydantic schema before entering
the pipeline — catching empty notes, malformed ICD-10 codes, and label
inconsistencies at ingestion time.

### APSO-Flip Preprocessing
Clinical notes are restructured so the Assessment section appears at
Token 0, preventing diagnostic evidence from being truncated by
Bio_ClinicalBERT's 512-token context window. ICD-10 strings are
redacted from note text to prevent label leakage.

### Hierarchical Decomposition
The two-stage pipeline decomposes 1,926-way ICD-10 classification into
a 22-way chapter routing problem followed by within-chapter resolution,
reducing the effective label space per resolver from 1,926 to ~100.

### Transfer Learning Chain
Rather than training from raw Bio_ClinicalBERT weights, each stage
initialises from the best available prior model — creating a progressive
transfer learning chain that accumulates ICD knowledge across experiments.

---

## ⚠️ Limitations

- **Synthetic dataset:** MedSynth uses uniform sampling (5 records per
  ICD-10 code). Real clinical code distributions are heavily skewed —
  performance on real data will differ.
- **Low-resource constraint:** ~4 training examples per ICD-10 code is
  an extremely challenging regime. Results reflect the limits of this
  constraint rather than the architecture ceiling.
- **Z-chapter difficulty:** Administrative codes (Z-chapter, 263 classes)
  achieve only 32.6% E2E accuracy due to highly similar clinical language
  across codes.
- **Apple Silicon only tested:** Training was conducted exclusively on
  Apple M4 Max with MPS. CUDA compatibility is expected but untested.

---

## 📦 Dependencies

All dependencies managed via `pyproject.toml` and `uv.lock`:
```bash
uv sync  # installs everything
```

Key libraries: `transformers`, `torch`, `polars`, `mlflow`, `pydantic`,
`scikit-learn`, `datasets`, `huggingface-hub`

---

## 📄 Citation

If you use this work, please cite the MedSynth dataset:
```bibtex
@misc{rezaie2025medsynth,
  title   = {MedSynth: Synthetic Medical Dialogue Dataset for ICD-10 Coding},
  author  = {Rezaie Mianroodi, et al.},
  year    = {2025},
  url     = {https://arxiv.org/abs/2508.01401}
}
```

---

## 💬 Issues & Suggestions

This is a personal research project. Issues and suggestions are welcome
via [GitHub Issues](https://github.com/Sidney-Bishop/notes-to-icd10/issues).

---

## 📝 License

MIT License — see [LICENSE](LICENSE) for details.

Copyright (c) 2026 Jason Roche