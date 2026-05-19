# MIMIC-IV Validation Pipeline

## Overview

This directory contains the MIMIC-IV-Note validation pipeline for the
notes-to-icd10 project. It tests whether E-010's 85.8% accuracy on
synthetic MedSynth notes generalises to real clinical discharge summaries.

## ⚠️ Restricted Data Access Required

MIMIC-IV is a credentialed dataset. You **cannot run these scripts**
without completing the PhysioNet access process:

### Step 1 — CITI Training
Complete the **"Data or Specimens Only Research"** course at:
https://about.citiprogram.org/

### Step 2 — MIMIC-IV-Note access (discharge notes)
Apply at: https://physionet.org/content/mimic-iv-note/2.2/
- Download `note/discharge.csv.gz` (~1.1GB compressed)

### Step 3 — MIMIC-IV clinical access (ICD codes)
Apply at: https://physionet.org/content/mimiciv/2.2/
- Sign the separate DUA for the clinical database
- Download `hosp/diagnoses_icd.csv.gz` (~24MB)

## Data Placement

After downloading, set environment variables in your `.env` file:

```bash
MIMIC_DISCHARGE_PATH=/path/to/discharge.csv.gz
MIMIC_DIAGNOSES_PATH=/path/to/diagnoses_icd.csv.gz
```

Or pass paths as CLI arguments (see Usage below).

## Usage

### Step 1: Prepare the medallion data layers

```bash
# Default: stratified sample of 5,000 notes (~10 minutes)
uv run python scripts/validation/validate_mimic_prepare.py

# Full run: all ~78,000 matched notes (~2-3 hours on MPS)
uv run python scripts/validation/validate_mimic_prepare.py --full-run

# Custom sample size
uv run python scripts/validation/validate_mimic_prepare.py --sample-size 10000
```

### Step 2: Run evaluation

```bash
# E-010 baseline
uv run python scripts/validation/validate_mimic_evaluate.py

# E-010 + E-014 SupCon Z hybrid (recommended)
uv run python scripts/validation/validate_mimic_evaluate.py --supcon-z
```

## Medallion Architecture

```
Bronze  raw MIMIC files registered as DuckDB views
  │
  ▼
Silver  joined discharge notes + ICD-10 primary dx
        filtered to our 1,916-code label map
        normalised ICD-10 format (A419 → A41.9)
  │
  ▼
Gold    APSO-preprocessed notes
        stratified sample by chapter
        ready for model inference
```

All intermediate data is written to `data/mimic/` (gitignored).

## What Is and Isn't Committed

| Item | Committed? |
|---|---|
| This README | ✅ Yes |
| Pipeline scripts (`.py`) | ✅ Yes |
| Raw MIMIC data | ❌ No — gitignored |
| Silver/Gold Parquet files | ❌ No — gitignored |
| Per-note predictions | ❌ No — gitignored |
| Aggregate results (`summary.json`) | ✅ Yes — no patient data |

## Citation

If you use this validation in your work, please cite:

```bibtex
@misc{johnson2023mimicivnote,
  title   = {MIMIC-IV-Note: Deidentified free-text clinical notes},
  author  = {Johnson, Alistair E.W. and others},
  year    = {2023},
  url     = {https://physionet.org/content/mimic-iv-note/2.2/},
  doi     = {10.13026/1n74-ne17}
}

@misc{johnson2023mimiciv,
  title   = {MIMIC-IV (version 2.2)},
  author  = {Johnson, Alistair and others},
  year    = {2023},
  url     = {https://physionet.org/content/mimiciv/2.2/},
  doi     = {10.13026/6mm1-ek67}
}
```
