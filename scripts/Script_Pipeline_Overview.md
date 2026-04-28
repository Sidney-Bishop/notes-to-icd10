# Notes-to-ICD10: Script Pipeline Overview

**Audience:** Developers and technical users  
**Purpose:** How to run the full ICD-10 prediction pipeline using scripts
rather than notebooks. Scripts are the production equivalent of the
notebooks — same logic, same results, no interactive cells required.  
**Last updated:** 27 April 2026  
**Current best result:** E-009_Balanced_E002Init — 79.8% accuracy, 0.711 F1

---

## The Scripts at a Glance

| Script | What it does | Notebook equivalent |
|---|---|---|
| `verify_scripts.py` | Pre-flight health check | — |
| `scripts/prepare_splits.py` | Generate deterministic train/val/test splits | Phase 2 of all notebooks |
| `scripts/train.py` | Train flat or hierarchical models | Notebooks 02, 03, 04, 05 |
| `scripts/calibrate.py` | Temperature scaling per model | Phase 6 of notebooks 04/05 |
| `scripts/evaluate.py` | Full evaluation suite | Phase 5/11 of all notebooks |
| `scripts/predict.py` | Single-note inference | Phase 12 inference check |

**Always run this first, every session:**
```bash
uv run python verify_scripts.py
```
All 9 checks must pass before any training command. This clears Python
bytecode cache, verifies all required files exist, and confirms key bug
fixes are in place.

---

## Understanding the Two Training Modes

`train.py` operates in two modes that mirror the notebook progression:

**Flat mode** (`--mode flat`) — one model predicts all codes simultaneously.
Replicates notebooks 02 (ICD-3) and 03 (flat ICD-10).

**Hierarchical mode** (`--mode hierarchical`) — two-stage pipeline.
Stage-1 routes to a chapter, Stage-2 resolves within that chapter.
Replicates notebooks 04 and 05.

---

## Full Pipeline: Reproducing E-009 (Current Best Result)

E-009 is the hierarchical pipeline with E-002 initialisation —
**79.8% test accuracy, 0.711 Macro F1** on 1,926 ICD-10 codes.

### Step 0 — Pre-flight

```bash
uv run python verify_scripts.py
```

### Step 1 — Train E-002 (flat ICD-10 baseline + Stage-2 initialiser)

E-002 serves two purposes: it is the flat baseline experiment, and its
trained weights are the starting point for E-009's Stage-2 resolvers.
If `outputs/evaluations/E-002_FullICD10_ClinicalBERT/model/` already
exists, skip this step.

```bash
uv run python scripts/train.py \
    --experiment E-002_FullICD10_ClinicalBERT \
    --label-scheme icd10 \
    --code-filter billable \
    --epochs 40 \
    --lr 2e-5 \
    --batch-size 16
```

> **Expected runtime:** ~240 minutes on Apple M5 Max (MPS)  
> **Expected result:** ~73.3% test accuracy, F1 ~0.634  
> **Why 40 epochs:** The model continues improving to epoch 28 and beyond.
> Stopping at 20 epochs costs approximately 8pp of F1.

### Step 2 — Train Stage-1 chapter router (E-003)

The Stage-1 router is trained once and reused by all hierarchical
experiments. It classifies notes into one of 22 ICD-10 chapters.
If `outputs/evaluations/E-003_Hierarchical_ICD10/stage1/model/`
already exists, skip this step.

```bash
uv run python scripts/train.py \
    --experiment E-003_Hierarchical_ICD10 \
    --mode hierarchical \
    --stage 1 \
    --code-filter billable \
    --epochs 5 \
    --lr 2e-5 \
    --batch-size 16
```

> **Expected runtime:** ~25 minutes  
> **Expected result:** ~96.4% chapter routing accuracy  
> **Why 5 epochs:** The router converges at epoch 4. Training beyond
> 5 epochs wastes compute without improving routing accuracy.

### Step 3 — Train Stage-2 resolvers with E-002 initialisation (E-009)

Each of 19 chapter resolvers is initialised from E-002 weights and
fine-tuned on chapter-filtered data. This is the key architectural
decision that produces the +6.5pp improvement over the flat baseline.

```bash
uv run python scripts/train.py \
    --experiment E-009_Balanced_E002Init \
    --mode hierarchical \
    --stage 2 \
    --code-filter billable \
    --epochs 20 \
    --lr 2e-5 \
    --batch-size 16 \
    --stage2-init outputs/evaluations/E-002_FullICD10_ClinicalBERT
```

> **Expected runtime:** ~103 minutes (19 resolvers in sequence)  
> **Expected result:** Weighted val accuracy ~83.5%, within-chapter ~82.8%  
> **Why E-002 init:** Fresh Bio_ClinicalBERT init (E-003) produced 11.1%
> E2E. E-002 init produces 79.8% E2E — the weights provide pre-learned
> ICD-10 code representations that cannot be learned from ~4 examples/code.

### Step 4 — Calibrate

Temperature scaling fits a single scalar T per model that adjusts
confidence scores without changing predictions. Calibrated confidence
is required for the auto-code threshold logic in `predict.py`.

```bash
uv run python scripts/calibrate.py \
    --experiment E-009_Balanced_E002Init \
    --stage1-experiment E-003_Hierarchical_ICD10
```

> **Expected runtime:** ~5 minutes  
> Writes `temperature.json` alongside each model directory.

### Step 5 — Evaluate

Runs the full evaluation suite on the held-out test set and writes
all metrics to `outputs/evaluations/E-009_Balanced_E002Init/eval/`.

```bash
uv run python scripts/evaluate.py \
    --experiment E-009_Balanced_E002Init \
    --mode hierarchical \
    --stage1-experiment E-003_Hierarchical_ICD10 \
    --threshold 0.7
```

> **Expected result:** ~79.8% E2E accuracy, F1 ~0.711  
> **Outputs:** `eval/summary.json`, `eval/per_class_metrics.csv`,
> `eval/predictions.parquet`, `eval/chapter_accuracy.json`

### Step 6 — Verify registry

```bash
uv run python -c "from src.experiment_logger import status; status()"
```

E-009_Balanced_E002Init should show E2E ≈ 0.798, F1 ≈ 0.711.

### Step 7 — Inference sanity check

```bash
echo "Patient presents with severe chest pain and shortness of breath.
Assessment: Acute ST-elevation myocardial infarction (STEMI).
Plan: Urgent PCI, aspirin, heparin, clopidogrel." | \
    uv run python scripts/predict.py \
        --experiment E-009_Balanced_E002Init \
        --stage1-experiment E-003_Hierarchical_ICD10 \
        --top-k 5
```

---

## Individual Experiments

### E-001 — ICD-3 Flat Baseline (Notebook 02)

Proof of concept: can Bio_ClinicalBERT learn from clinical notes?
ICD-3 has 675 categories (~12 examples each) — a tractable version
of the harder ICD-10 problem.

```bash
uv run python scripts/train.py \
    --experiment E-001_Baseline_ICD3 \
    --label-scheme icd3 \
    --epochs 30 \
    --lr 2e-5 \
    --batch-size 16

uv run python scripts/evaluate.py \
    --experiment E-001_Baseline_ICD3 \
    --mode flat
```

> **Expected:** ~87.2% test accuracy, F1 ~0.841, best epoch 28

### E-002 — Flat ICD-10 Baseline (Notebook 03)

The flat baseline — one model, all 1,926 codes, 40 epochs.
This is also the source of weights for E-009 Stage-2 initialisation.

```bash
uv run python scripts/train.py \
    --experiment E-002_FullICD10_ClinicalBERT \
    --label-scheme icd10 \
    --code-filter billable \
    --epochs 40 \
    --lr 2e-5 \
    --batch-size 16

uv run python scripts/evaluate.py \
    --experiment E-002_FullICD10_ClinicalBERT \
    --mode flat
```

> **Expected:** ~73.3% test accuracy, F1 ~0.634

### E-003 — Hierarchical Cold Start (Notebook 04)

The cold-start baseline — same two-stage architecture as E-009 but
Stage-2 initialised from fresh Bio_ClinicalBERT. Establishes that
architecture alone is not enough.

```bash
# Stage-1 only
uv run python scripts/train.py \
    --experiment E-003_Hierarchical_ICD10 \
    --mode hierarchical \
    --stage 1 \
    --code-filter billable \
    --epochs 5

# Stage-2 fresh init
uv run python scripts/train.py \
    --experiment E-003_Hierarchical_ICD10 \
    --mode hierarchical \
    --stage 2 \
    --code-filter billable \
    --epochs 20
```

> **Expected:** ~11.1% E2E accuracy — intentionally poor, this is the
> cold-start floor that demonstrates why E-002 initialisation is required.

---

## Key Flags Reference

### train.py

| Flag | Default | Notes |
|---|---|---|
| `--experiment` | `E-train` | Name for output directory and MLflow |
| `--mode` | `flat` | `flat` or `hierarchical` |
| `--stage` | `12` | `1`=Stage-1 only, `2`=Stage-2 only, `12`=both |
| `--label-scheme` | `icd10` | `icd3` (675 classes) or `icd10` (1,926 classes) |
| `--code-filter` | `all` | `all` (10,240 records) or `billable` (9,660 records) |
| `--epochs` | `10` | **Always override** — default is too low |
| `--lr` | `2e-5` | Standard Bio_ClinicalBERT fine-tuning rate |
| `--batch-size` | `8` | Use `16` to match notebooks |
| `--stage2-init` | None | Path to E-002 model dir — required for E-009 |
| `--stage1-init` | None | Path to E-001 model dir — optional for Stage-1 |
| `--chapters` | all | Limit Stage-2 to specific chapters e.g. `--chapters Z O` |
| `--dry-run` | — | Validate data loading without training |
| `--use-presplit` | — | Use splits from `prepare_splits.py` instead of generating |
| `--gold-path` | auto | Override gold layer parquet path |
| `--no-mlflow` | — | Disable MLflow logging |

### evaluate.py

| Flag | Default | Notes |
|---|---|---|
| `--experiment` | required | Must match the training experiment name |
| `--mode` | `flat` | `flat` or `hierarchical` |
| `--stage1-experiment` | `E-003_Hierarchical_ICD10` | Stage-1 source for hierarchical mode |
| `--threshold` | `0.7` | Confidence threshold for auto-code decision |
| `--sample` | None | Evaluate on a random subset (useful for quick checks) |
| `--batch-size` | `32` | Inference batch size |

### calibrate.py

| Flag | Default | Notes |
|---|---|---|
| `--experiment` | required | Stage-2 experiment name |
| `--stage1-experiment` | `E-003_Hierarchical_ICD10` | Stage-1 source |
| `--threshold` | `0.7` | Target confidence threshold |
| `--dry-run` | — | Print temperatures without writing files |

### predict.py

| Flag | Default | Notes |
|---|---|---|
| `--note` / `-n` | — | Clinical note text directly |
| `--file` / `-f` | — | Path to plain-text file |
| stdin | — | Pipe note via `echo "..." \| uv run python scripts/predict.py` |
| `--top-k` / `-k` | `5` | Number of predictions to return |
| `--threshold` / `-t` | `0.7` | Auto-code confidence threshold |
| `--experiment` | `E-004a_...` | **Override to `E-009_Balanced_E002Init`** |
| `--stage1-experiment` | `E-003_...` | Stage-1 source |
| `--json` | — | Structured JSON output instead of human-readable |

---

## Output Directory Layout

```
outputs/evaluations/
├── E-001_Baseline_ICD3/
│   ├── model/                    ← trained flat model
│   ├── label_map.json
│   ├── test_split.parquet
│   ├── train_result.json
│   └── eval/
│       ├── summary.json          ← all scalar metrics
│       ├── per_class_metrics.csv
│       └── predictions.parquet
│
├── E-002_FullICD10_ClinicalBERT/
│   └── (same structure as E-001)
│
├── E-003_Hierarchical_ICD10/
│   ├── stage1/
│   │   ├── model/               ← Stage-1 chapter router
│   │   ├── label_map.json
│   │   └── test_split.parquet
│   └── stage2/
│       ├── A/model/             ← per-chapter resolver
│       ├── B/model/
│       └── ... (19 chapters)
│
└── E-009_Balanced_E002Init/
    ├── stage1/ → (reused from E-003)
    ├── stage2/
    │   ├── A/model/             ← E-002 initialised resolver
    │   ├── B/model/
    │   └── ... (19 chapters)
    └── eval/
        ├── summary.json
        ├── chapter_accuracy.json ← per-chapter E2E breakdown
        └── predictions.parquet

outputs/
├── experiments.json             ← experiment registry
├── run.log                      ← append-only audit trail
└── mlflow.db                    ← MLflow tracking database
```

---

## Monitoring During Training

**TensorBoard** (training curves, live):
```bash
tensorboard --logdir outputs/evaluations/{experiment}/tensorboard --port 6006
```

**MLflow UI** (experiment comparison):
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
```

**Registry status** (at any time):
```bash
uv run python -c "from src.experiment_logger import status; status()"
```

---

## Differences Between Scripts and Notebooks

| Aspect | Notebooks | Scripts |
|---|---|---|
| Execution | Interactive cells, manual step-by-step | Single command, runs end-to-end |
| Output | Inline plots, cell outputs | Files in `outputs/evaluations/` |
| Retrain guard | `cfg['stage2_retrain']` flag | Re-run the command — always trains |
| Extended training | Phase 4c cell in notebook 05 | `--chapters Z K E` with lower `--lr` |
| Figure saving | `save_figure()` in each cell | Not currently implemented in scripts |
| Phase 13 record | Written at end of each notebook | Not implemented — use `evaluate.py` output |

The scripts implement the same data loading, splitting, tokenisation,
training loop, and evaluation logic as the notebooks. Results should
be reproducible to within normal training variance (±0.3pp F1) given
the same seed, epochs, and hyperparameters.

---

## Known Issues

| Issue | Impact | Notes |
|---|---|---|
| sklearn version mismatch | Warning only | Graph reranker pickled against old version — cosmetic |
| `predict.py` default `--experiment` | Wrong model used | Always pass `--experiment E-009_Balanced_E002Init` explicitly |
| `calibrate.py` default `--stage1-experiment` | Correct default | `E-003_Hierarchical_ICD10` is correct |
| Stage-2 extended training | Not in scripts | Run `train.py --stage 2 --chapters Z K E --lr 5e-6 --epochs 10` manually |

---

*Last updated: 27 April 2026*  
*All scripts pass `uv run python verify_scripts.py`*  
*Current best: E-009_Balanced_E002Init — 79.8% accuracy, 0.711 F1*
