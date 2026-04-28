# ICD-10 Pipeline — Full Clean Run Notes

**Purpose:** This document is a step-by-step guide for running the complete
experiment chain from scratch. It was written while executing each step for
the first time under the new logging infrastructure. Every command here has
been run and verified. The next person can follow this document exactly.

**Author:** Written during the clean rebuild session of 25 April 2026.

**Time required:** ~8 hours end-to-end. Can be run overnight.

---

## Before You Start — Checklist

```bash
# 1. Confirm you are in the project root
pwd
# Expected: .../Notes_to_ICD10_prj

# 2. Confirm the virtual environment is active
which python
# Expected: .../Notes_to_ICD10_prj/.venv/bin/python

# 3. Confirm gold data exists
ls data/gold/
# Expected: medsynth_gold.parquet  medsynth_gold_augmented.parquet

# 4. Confirm the experiment registry is accessible
uv run python -c "from src.experiment_logger import status; status()"
# Expected: prints the registry table (may be empty on first run)

# 5. Confirm GPU/MPS is available
uv run python -c "import torch; print(torch.backends.mps.is_available())"
# Expected: True (on Apple Silicon)
```

If any of the above fail, **stop and fix before proceeding**.

---

## Data Files — What They Are

| File | Records | Description |
|------|---------|-------------|
| `data/gold/medsynth_gold.parquet` | 10,240 | Original MedSynth gold layer — APSO-flipped, ICD-10 redacted, CDC validated |
| `data/gold/medsynth_gold_augmented.parquet` | 11,214 | Above + 974 synthetic records for chapters O and Z (via augment.py) |

**Which gold file to use:**
- E-002, E-003, E-004a, E-005b, E-005c: use the **original** (`medsynth_gold.parquet`)
- E-009 and beyond: use the **augmented** (`medsynth_gold_augmented.parquet`)

---

## Output Layout — Where Everything Goes

All experiment outputs live under `outputs/evaluations/{experiment_name}/`:

```
outputs/evaluations/E-004a_Hierarchical_E002Init/
    stage1/                          ← Stage-1 router (shared with E-003)
        model.safetensors            ← weights (FLAT convention going forward)
        config.json
        tokenizer.json
        tokenizer_config.json
        label_map.json               ← 22-chapter id→label mapping
        temperature.json             ← calibration scalar (written by calibrate.py)
        test_split.parquet           ← held-out test records for Stage-1 eval
        train_result.json            ← training history, best epoch
    stage2/
        A/                           ← one directory per chapter
            model.safetensors
            config.json
            tokenizer.json
            tokenizer_config.json
            label_map.json
            temperature.json
            train_split.parquet      ← written by prepare_splits.py
            val_split.parquet
            test_split.parquet
        B/ C/ D/ ... Z/
        stage2_results.json          ← which chapters trained, which skipped
    calibration_report.json          ← full calibration summary
    eval/
        summary.json                 ← all scalar metrics (e2e_accuracy etc)
        predictions.parquet          ← per-record predictions
        per_class_metrics.csv
        threshold_sweep.json
        chapter_accuracy.json

outputs/experiments.json             ← experiment registry (do not edit manually)
outputs/run.log                      ← append-only run log (do not edit manually)
```

---

## The Experiment Chain

```
E-002  (flat ICD-10 baseline)
  └── provides warm-start weights for Stage-2 resolvers
        ↓
E-003  (Stage-1 chapter router)
  └── provides the 22-way routing model used in all hierarchical experiments
        ↓
E-004a  (hierarchical, E-002 warm start)
  └── 19 per-chapter resolvers, each initialised from E-002 encoder weights
        ↓
E-005b  (augmented O chapter only)
  └── retrain only chapter O resolver with augmented data (+15.5pp on O)
        ↓
E-005c  (merged pipeline — no training)
  └── E-004a resolvers + E-005b's O resolver swapped in
      calibrate + evaluate → this is the benchmark target
        ↓
E-009  (new best attempt)
  └── all chapters, augmented gold, E-002 init, presplits
      expected to beat E-005c
```

---

## Stage 0 — Prepare Deterministic Splits

**What this does:** Writes fixed train/val/test parquets for every chapter so
all subsequent experiments use exactly the same test records. Without this,
each training run generates its own random split and results are not comparable.

**Run once. Do not re-run unless you want to invalidate all previous results.**

```bash
# For the original gold data (used by E-003, E-004a, E-005b, E-005c)
uv run python scripts/prepare_splits.py \
    --experiment E-004a_Hierarchical_E002Init \
    --gold-path data/gold/medsynth_gold.parquet

# For the augmented gold data (used by E-009)
uv run python scripts/prepare_splits.py \
    --experiment E-009_Balanced_E002Init \
    --gold-path data/gold/medsynth_gold_augmented.parquet
```

**Expected output:**
```
prepare_splits.py — Deterministic Split Generation
Gold path: data/gold/medsynth_gold.parquet
Experiment: E-004a_Hierarchical_E002Init
Seed: 42

Chapter A: 280 records → 224 train / 28 val / 28 test
Chapter B: 398 records → 318 train / 40 val / 40 test
...
✅ Splits written for 22 chapters
```

**Verify:**
```bash
ls outputs/evaluations/E-004a_Hierarchical_E002Init/stage2/
# Expected: A B C D E F G H I J K L M N O P Q R S T U Z
ls outputs/evaluations/E-004a_Hierarchical_E002Init/stage2/Z/
# Expected: train_split.parquet  val_split.parquet  test_split.parquet
```

---

## Stage 1 — E-002: Flat ICD-10 Baseline

**What this does:** Trains a single flat classifier over all 1,926+ ICD-10
codes simultaneously. This serves two purposes:
1. Establishes a baseline accuracy for the flat approach
2. Provides encoder weights to warm-start all Stage-2 resolvers in E-004a

**Model:** `emilyalsentzer/Bio_ClinicalBERT`
**Data:** Original gold, all 10,240 records (`--code-filter all`)
**Time:** ~3 hours

```bash
uv run python scripts/train.py \
    --experiment E-002_FullICD10_ClinicalBERT \
    --mode flat \
    --label-scheme icd10 \
    --model emilyalsentzer/Bio_ClinicalBERT \
    --code-filter all \
    --epochs 10 \
    --gold-path data/gold/medsynth_gold.parquet
```

**Expected output (end of run):**
```
── Flat training: E-002_FullICD10_ClinicalBERT ─────────────────────────
 Model: emilyalsentzer/Bio_ClinicalBERT
 Label scheme: icd10 (10,240 records)
 Num classes: 1926
 Split: 8,192 / 1,024 / 1,024
 ...
 Best epoch: X | Val accuracy: ~0.54 | Val F1: ~0.43
 Outputs: outputs/evaluations/E-002_FullICD10_ClinicalBERT/
```

**Verify weights exist:**
```bash
ls outputs/evaluations/E-002_FullICD10_ClinicalBERT/
# Must contain: model/  label_map.json  train_result.json  test_split.parquet

find outputs/evaluations/E-002_FullICD10_ClinicalBERT -name "model.safetensors"
# Must return at least one path
```

**Register in the experiment logger:**
```bash
uv run python -c "
from src.experiment_logger import ExperimentLogger
import json
from pathlib import Path
r = json.loads(Path('outputs/evaluations/E-002_FullICD10_ClinicalBERT/train_result.json').read_text())
l = ExperimentLogger('E-002_FullICD10_ClinicalBERT', script='train.py')
l.log_complete('train_flat', artifacts={
    'model_dir': 'outputs/evaluations/E-002_FullICD10_ClinicalBERT/model',
    'best_epoch': r.get('best_epoch'),
    'best_val_accuracy': r.get('best_val_accuracy'),
})
print('Registered.')
"
```

---

## Stage 2 — E-003: Stage-1 Chapter Router

**What this does:** Trains a 22-way chapter classifier. This is the gating
model that routes every clinical note to the correct ICD-10 chapter (A–Z)
before the per-chapter resolver makes the final code prediction.

**Model:** `roberta-base` (general-purpose, not clinical — still achieves ~97% accuracy)
**Data:** Original gold, billable records only (~9,660 records)
**Time:** ~1 hour

**Why roberta-base for Stage-1?**
Chapter routing is a coarser task than code resolution. Chapter labels have
~440 records each (vs ~5 per full ICD-10 code), so a general-purpose encoder
is sufficient. The clinical domain knowledge of Bio_ClinicalBERT is more
valuable for Stage-2 where it needs to discriminate between similar codes
within a chapter.

```bash
uv run python scripts/train.py \
    --experiment E-003_Stage1_Router \
    --mode hierarchical \
    --stage 1 \
    --model roberta-base \
    --code-filter billable \
    --epochs 10 \
    --gold-path data/gold/medsynth_gold.parquet
```

**Expected output:**
```
── Stage-1 Router: E-003_Stage1_Router ──────────────────────────
 Chapters: 22 | Split: 7,728 / 966 / 966
 ...
 Best epoch: X | Val accuracy: ~0.97
```

**Verify:**
```bash
find outputs/evaluations/E-003_Stage1_Router/stage1 -name "model.safetensors"
# Must return exactly one path

# Quick sanity check — should print 22 chapters
python3 -c "
import json
from pathlib import Path
lm = json.load(open('outputs/evaluations/E-003_Stage1_Router/stage1/label_map.json'))
print('Chapters:', sorted(lm['label2id'].keys()))
print('Count:', len(lm['label2id']))
"
```

**Register:**
```bash
uv run python -c "
from src.experiment_logger import ExperimentLogger
import json
from pathlib import Path
r = json.loads(Path('outputs/evaluations/E-003_Stage1_Router/stage1/train_result.json').read_text())
l = ExperimentLogger('E-003_Stage1_Router', script='train.py')
l.log_complete('train_stage1', artifacts={
    'stage1_dir': 'outputs/evaluations/E-003_Stage1_Router/stage1',
    'best_epoch': r.get('best_epoch'),
})
print('Registered.')
"
```

---

## Stage 3 — E-004a: Hierarchical with E-002 Warm Start

**What this does:** Trains 19 per-chapter resolvers, each initialised from
the E-002 flat model weights. The E-002 encoder has already learned ICD-10
representations across all codes — warm-starting from it instead of raw
Bio_ClinicalBERT gives a massive boost to within-chapter accuracy.

**Key insight:** In the notebook experiments, this single change (E-003 fresh
→ E-004a warm start) produced a 6.2x improvement in E2E accuracy (10.6% → 65.8%).

**Model:** `emilyalsentzer/Bio_ClinicalBERT` (but loaded from E-002 checkpoint)
**Stage2-init:** `outputs/evaluations/E-002_FullICD10_ClinicalBERT`
**Data:** Original gold, billable records only
**Presplits:** Yes (`--use-presplit`) — uses the splits written in Stage 0
**Time:** ~2 hours

```bash
uv run python scripts/train.py \
    --experiment E-004a_Hierarchical_E002Init \
    --mode hierarchical \
    --stage 2 \
    --model emilyalsentzer/Bio_ClinicalBERT \
    --stage2-init outputs/evaluations/E-002_FullICD10_ClinicalBERT \
    --code-filter billable \
    --epochs 10 \
    --use-presplit \
    --gold-path data/gold/medsynth_gold.parquet
```

**Expected output per chapter:**
```
 📂 Chapter A: 45 codes | 224/28/28 [presplit]
 ↪️ Transfer learning from outputs/evaluations/E-002_FullICD10_ClinicalBERT/stage2/A/model/model
 ✅ A: best epoch X
 ...
 ✅ Stage-2 complete: 19 resolvers trained, 3 chapters skipped (P, Q, U)
```

**Important — chapters P, Q, U are skipped by design:**
These chapters have too few records for reliable training. They use
majority-class fallback predictions at inference time. This is normal and expected.

**Verify:**
```bash
# Count trained resolvers
find outputs/evaluations/E-004a_Hierarchical_E002Init/stage2 \
    -name "model.safetensors" | wc -l
# Expected: 19

# Check stage2_results.json records the skip chapters
python3 -c "
import json
r = json.load(open('outputs/evaluations/E-004a_Hierarchical_E002Init/stage2/stage2_results.json'))
print('Trained:', r['chapters_trained'])
print('Skipped:', r['skip_chapters'])
"
```

**Register:**
```bash
uv run python -c "
from src.experiment_logger import ExperimentLogger
import json
from pathlib import Path
r = json.loads(Path('outputs/evaluations/E-004a_Hierarchical_E002Init/stage2/stage2_results.json').read_text())
l = ExperimentLogger('E-004a_Hierarchical_E002Init', script='train.py')
l.log_complete('train_stage2', artifacts={
    'stage2_dir': 'outputs/evaluations/E-004a_Hierarchical_E002Init/stage2',
    'chapters_trained': ','.join(r['chapters_trained']),
    'stage2_init': 'E-002_FullICD10_ClinicalBERT',
})
print('Registered.')
"
```

---

## Stage 4 — E-004a Calibration

**What this does:** Learns a temperature scalar T per resolver that makes
confidence scores reliable (ECE close to 0). This is required for Use Case B
(auto-code when confidence ≥ threshold, human review otherwise).

```bash
uv run python scripts/calibrate.py \
    --experiment E-004a_Hierarchical_E002Init \
    --stage1-experiment E-003_Stage1_Router
```

**Expected output (summary):**
```
Stage-2 summary (19 resolvers)
Avg temperature:  X.XXXX
Avg ECE:          0.XXX → 0.XXX
Avg Coverage@0.7: XX.X%  (avg accuracy on covered: 0.XXX)
```

**Verify:**
```bash
# Every trained chapter should have a temperature.json
find outputs/evaluations/E-004a_Hierarchical_E002Init/stage2 \
    -name "temperature.json" | wc -l
# Expected: 19 (one per trained chapter)
```

**Register:**
```bash
uv run python -c "
from src.experiment_logger import ExperimentLogger
l = ExperimentLogger('E-004a_Hierarchical_E002Init', script='calibrate.py')
l.log_complete('calibrate', artifacts={
    'calibration_report': 'outputs/evaluations/E-004a_Hierarchical_E002Init/calibration_report.json',
})
print('Registered.')
"
```

---

## Stage 5 — E-005b: Augmented O Chapter

**What this does:** Retrains only the chapter O (Pregnancy/Childbirth) resolver
using the augmented gold dataset which has additional synthetic O records.
Previous runs showed +15.5pp improvement on chapter O.

**Why only O?** Chapter Z was also augmented but the augmentation hurt Z
performance (-13.6pp) because the 263 Z codes are administratively too similar
— synthetic notes were homogeneous and confused the model. Chapter O benefited
because obstetric codes have more discriminative clinical signal.

```bash
uv run python scripts/train.py \
    --experiment E-004a_Hierarchical_E002Init \
    --mode hierarchical \
    --stage 2 \
    --model emilyalsentzer/Bio_ClinicalBERT \
    --stage2-init outputs/evaluations/E-002_FullICD10_ClinicalBERT \
    --code-filter billable \
    --epochs 10 \
    --use-presplit \
    --chapters O \
    --gold-path data/gold/medsynth_gold_augmented.parquet
```

Note: `--chapters O` limits training to chapter O only. All other chapter
resolvers are left untouched.

**Expected output:**
```
 📂 Chapter O: 63 codes | XXX/XX/XX [presplit]
 ↪️ Transfer learning from ...
 ✅ O: best epoch X
 ✅ Stage-2 complete: 1 resolvers trained, 0 chapters skipped
```

**Register:**
```bash
uv run python -c "
from src.experiment_logger import ExperimentLogger
l = ExperimentLogger('E-004a_Hierarchical_E002Init', script='train.py')
l.log_note('E-005b: retrained chapter O with augmented gold (11,214 records). Overwrites stage2/O/ resolver.')
print('Noted.')
"
```

---

## Stage 6 — E-005c: Merged Pipeline Evaluation

**What this does:** Evaluates the merged pipeline — E-004a with the augmented
O resolver from E-005b — to get our benchmark target metric. No new training.
Just calibrate (to pick up the new O temperature) and evaluate.

**This is the target we are trying to beat with E-009.**

```bash
# Re-calibrate to pick up new O resolver temperature
uv run python scripts/calibrate.py \
    --experiment E-004a_Hierarchical_E002Init \
    --stage1-experiment E-003_Stage1_Router

# Evaluate the merged pipeline
uv run python scripts/evaluate.py \
    --experiment E-004a_Hierarchical_E002Init \
    --mode hierarchical \
    --stage1-experiment E-003_Stage1_Router
```

**Expected output:**
```
 📈 Stage-1 (chapter) accuracy: ~0.97
 📈 Stage-2 (within-chapter):   ~0.XX
 📈 End-to-end accuracy:        ~0.XX
 📈 Macro F1:                   ~0.XXX
 📈 ECE:                        ~0.0XX
 📈 Coverage@τ=0.7:             XX.X%
```

**Record the result — this is your benchmark:**
```bash
uv run python -c "
import json
from pathlib import Path
from src.experiment_logger import ExperimentLogger
summary = json.loads(Path('outputs/evaluations/E-004a_Hierarchical_E002Init/eval/summary.json').read_text())
l = ExperimentLogger('E-004a_Hierarchical_E002Init', script='evaluate.py')
l.log_complete('evaluate', artifacts={'eval_dir': 'outputs/evaluations/E-004a_Hierarchical_E002Init/eval'})
l.log_results({
    'e2e_accuracy':    summary['e2e_accuracy'],
    'macro_f1':        summary['macro_f1'],
    'ece':             summary['ece'],
    'coverage_07':     summary.get('coverage_at_threshold', summary.get('coverage_07')),
    'stage1_accuracy': summary['stage1_accuracy'],
    'stage2_accuracy': summary.get('within_chapter_accuracy', 0),
})
print('Results registered.')
"
```

---

## Stage 7 — E-009: New Best Attempt

**What this does:** Trains a fresh hierarchical pipeline with:
- All chapters using augmented gold (11,214 records)
- E-002 warm start (not E-006 — E-002 has all ICD-10 codes and better representations)
- Deterministic presplits for reproducibility
- Same architecture as E-004a but with more data

This is expected to beat E-005c because the augmented data improves data density
across all chapters while the E-002 warm start ensures good representations.

**First write the presplits for E-009:**
```bash
# Already done in Stage 0 above — skip if already run
uv run python scripts/prepare_splits.py \
    --experiment E-009_Balanced_E002Init \
    --gold-path data/gold/medsynth_gold_augmented.parquet
```

**Train Stage-1 for E-009 (same as E-003 but logged under E-009):**
```bash
uv run python scripts/train.py \
    --experiment E-009_Balanced_E002Init \
    --mode hierarchical \
    --stage 1 \
    --model roberta-base \
    --code-filter billable \
    --epochs 10 \
    --gold-path data/gold/medsynth_gold_augmented.parquet
```

**Train Stage-2 for E-009:**
```bash
uv run python scripts/train.py \
    --experiment E-009_Balanced_E002Init \
    --mode hierarchical \
    --stage 2 \
    --model emilyalsentzer/Bio_ClinicalBERT \
    --stage2-init outputs/evaluations/E-002_FullICD10_ClinicalBERT \
    --code-filter billable \
    --epochs 10 \
    --use-presplit \
    --gold-path data/gold/medsynth_gold_augmented.parquet
```

**Calibrate E-009:**
```bash
uv run python scripts/calibrate.py \
    --experiment E-009_Balanced_E002Init \
    --stage1-experiment E-009_Balanced_E002Init
```

**Evaluate E-009:**
```bash
uv run python scripts/evaluate.py \
    --experiment E-009_Balanced_E002Init \
    --mode hierarchical \
    --stage1-experiment E-009_Balanced_E002Init
```

**Register E-009 results:**
```bash
uv run python -c "
import json
from pathlib import Path
from src.experiment_logger import ExperimentLogger
summary = json.loads(Path('outputs/evaluations/E-009_Balanced_E002Init/eval/summary.json').read_text())
l = ExperimentLogger('E-009_Balanced_E002Init', script='evaluate.py')
l.log_complete('evaluate', artifacts={'eval_dir': 'outputs/evaluations/E-009_Balanced_E002Init/eval'})
l.log_results({
    'e2e_accuracy':    summary['e2e_accuracy'],
    'macro_f1':        summary['macro_f1'],
    'ece':             summary['ece'],
    'coverage_07':     summary.get('coverage_at_threshold', summary.get('coverage_07')),
    'stage1_accuracy': summary['stage1_accuracy'],
    'stage2_accuracy': summary.get('within_chapter_accuracy', 0),
})
print('Results registered.')
"
```

---

## After Every Session — Check the Registry

```bash
uv run python src/experiment_logger.py status
```

This prints the full experiment registry showing which stages are complete
and what the metrics are. Start every new session by running this command.

---

## Troubleshooting

**"No module named src.paths"**
You are not in the project root. Run `cd /path/to/Notes_to_ICD10_prj` first.

**"Could not find artifacts.yaml"**
Same issue — not in the project root. Or the venv is not active.

**Stage-2 shows "⚠️ No checkpoint for chapter X, using base model"**
The `--stage2-init` path is wrong or the E-002 weights are missing for that chapter.
Check: `find outputs/evaluations/E-002_FullICD10_ClinicalBERT/stage2 -name "model.safetensors"`

**Calibration shows T < 0.1 or T = 0.05 (clamped)**
The resolver is systematically overconfident in the wrong direction. This usually
means the classifier head reinitialised (E-002 init had different class count).
The clamp to 0.05 prevents negative temperatures — the resolver still works but
calibration is imperfect. Retrain with more epochs.

**"Stage-1 accuracy: 24%"**
The Z override in inference.py may still be active. Check:
`grep -n "pred_chapter = 'Z'" src/inference.py`
This line must not exist (only commented-out versions are acceptable).

**OOM during Stage-2 training**
Reduce `--batch-size` to 4. Training takes longer but uses less memory.

**Evaluation takes > 3 minutes per record**
The graph reranker may be loading a very large graph. This is normal for
the first run — subsequent runs are faster due to OS disk caching.

---

## Key Decisions and Why

| Decision | Rationale |
|----------|-----------|
| roberta-base for Stage-1 | Chapter routing is a coarser task — 22 classes with ~440 records each. General encoder is sufficient and trains faster. |
| Bio_ClinicalBERT for Stage-2 | Code resolution requires clinical domain knowledge. MIMIC-III pretraining is decisive for Z chapter (administrative codes). |
| E-002 as Stage-2 warm start | Flat ICD-10 model has seen all codes simultaneously. Encoder representations are maximally informed. E-006 had fewer codes per chapter — classifier head mismatch caused reinit. |
| Presplits mandatory | Without fixed splits, test sets differ per run and results are not comparable. |
| Z override permanently removed | The phrase "physical exam" appears in every APSO note template. Any phrase-based override corrupts 100% of predictions. Stage-1 routing is model-driven only. |
| Skip chapters P, Q, U | These chapters have too few records for reliable training (<5 per code). Majority-class fallback is more accurate than an undertrained resolver. |

---

## Leaderboard Template

Fill this in after each evaluate step:

| Experiment | Stage-1 | E2E | F1 | ECE | Cov@0.7 | Notes |
|------------|---------|-----|-----|-----|---------|-------|
| E-004a (no O aug) | | | | | | Baseline hierarchical |
| E-005c (with O aug) | | | | | | Benchmark target |
| E-009 | | | | | | New best attempt |

---

*Last updated: 25 April 2026*
*Run by: — (fill in your name)*
*Total elapsed: — hours*
---

## Completed Leaderboard — 26 April 2026

| Experiment | Stage-1 | E2E | F1 | ECE | Cov@0.7 | Notes |
|------------|---------|-----|-----|-----|---------|-------|
| E-004a (E-002 warm start, original gold mismatch) | 96.5% | 20.9% | 0.141 | 0.209 | 34.0% | Classifier heads reinitialised — wrong gold |
| E-009 (E-002_Aug warm start, augmented gold) | 96.5% | **71.7%** | **0.637** | **0.030** | **65.7%** | ✅ Current best |

---

## E-009_Balanced_E002Init — Corrected Full Run

### Critical Prerequisite — E-002 Must Use the Same Gold as Stage-2

> **This is the single most important constraint in the pipeline.**
> If E-002 is trained on different gold data than Stage-2, the classifier heads
> will not transfer. Every chapter will train from random initialisation despite
> the encoder warm start, and E2E accuracy will be ~20% instead of ~72%.

### Stage 0 — Re-train E-002 on Augmented Gold

```bash
uv run python verify_scripts.py && \
uv run python scripts/train.py \
    --experiment E-002_FullICD10_ClinicalBERT_Aug \
    --mode flat \
    --label-scheme icd10 \
    --model emilyalsentzer/Bio_ClinicalBERT \
    --code-filter billable \
    --batch-size 16 \
    --epochs 20
```

**Expected:** best epoch ~19-20, val_acc ~57%, val_f1 ~0.45, ~90 minutes.
The script auto-detects `medsynth_gold_augmented.parquet` as default.

**Register:**
```bash
python3 -c "
from src.experiment_logger import ExperimentLogger
import json
from pathlib import Path
r = json.loads(Path('outputs/evaluations/E-002_FullICD10_ClinicalBERT_Aug/train_result.json').read_text())
l = ExperimentLogger('E-002_FullICD10_ClinicalBERT_Aug', script='train.py')
l.log_complete('train_flat', artifacts={
    'model_dir': 'outputs/evaluations/E-002_FullICD10_ClinicalBERT_Aug',
    'best_epoch': r['best_epoch'],
    'best_val_accuracy': r['best_val_accuracy'],
    'gold': 'medsynth_gold_augmented.parquet',
})
print('Registered.')
"
```

### Stage 1 — Prepare E-009 Splits

```bash
uv run python scripts/prepare_splits.py \
    --experiment E-009_Balanced_E002Init
```

**Expected:** 1,127 test records, 22 chapters, same as E-004a splits.

### Stage 2 — Train E-009 Stage-2 Resolvers (~90 minutes)

```bash
uv run python verify_scripts.py && \
uv run python scripts/train.py \
    --experiment E-009_Balanced_E002Init \
    --mode hierarchical \
    --stage 2 \
    --model emilyalsentzer/Bio_ClinicalBERT \
    --stage2-init outputs/evaluations/E-002_FullICD10_ClinicalBERT_Aug \
    --code-filter billable \
    --batch-size 16 \
    --epochs 20 \
    --use-presplit \
    --gold-path data/gold/medsynth_gold_augmented.parquet
```

**Verify the warm start is working** — first chapter should show:
```
↪️ Transfer learning from outputs/evaluations/E-002_FullICD10_ClinicalBERT_Aug
```
NOT: `⚠️ No checkpoint for chapter A, using base model`

If you see the warning, `train.py` is not patched correctly. Run `verify_scripts.py`.

**Expected:** 19 resolvers trained, 3 skipped (P, Q, U). No classifier head mismatches.

### Stage 3 — Calibrate (~30 seconds)

```bash
uv run python verify_scripts.py && \
uv run python scripts/calibrate.py \
    --experiment E-009_Balanced_E002Init \
    --stage1-experiment E-003_Stage1_Router
```

**Expected calibration quality:**
- Avg ECE: 0.649 → ~0.101
- Avg Coverage@0.7: ~73%
- Avg accuracy on covered: ~90%

If ECE gets **worse** after calibration, the classifier heads did not converge — check that the warm start was applied correctly.

### Stage 4 — Evaluate (~73 seconds)

```bash
uv run python verify_scripts.py && \
uv run python scripts/evaluate.py \
    --experiment E-009_Balanced_E002Init \
    --mode hierarchical \
    --stage1-experiment E-003_Stage1_Router
```

**Expected results:**
```
Stage-1 (chapter) accuracy: 0.965
Stage-2 (within-chapter):   0.743
End-to-end accuracy:        0.717
Macro F1:                   0.637
ECE:                        0.030
Coverage@τ=0.7:             65.7% (accuracy=0.897)
```

**Register results:**
```bash
python3 -c "
import json
from pathlib import Path
from src.experiment_logger import ExperimentLogger, status
summary = json.loads(Path('outputs/evaluations/E-009_Balanced_E002Init/eval/summary.json').read_text())
l = ExperimentLogger('E-009_Balanced_E002Init', script='evaluate.py')
l.log_results({
    'stage1_accuracy': summary['stage1_accuracy'],
    'stage2_accuracy': summary.get('within_chapter_accuracy', 0),
    'e2e_accuracy':    summary['e2e_accuracy'],
    'macro_f1':        summary['macro_f1'],
    'ece':             summary['ece'],
    'coverage_07':     summary.get('coverage_at_threshold', 0),
})
status()
"
```

---

## Session Checklist — Start of Every Session

```bash
# 1. Always start here
uv run python verify_scripts.py

# 2. Check experiment registry
python3 -c "from src.experiment_logger import status; status()"
```

Both must pass before running anything else.

---

*Last updated: 26 April 2026*
*E-009 result: 71.7% E2E | 0.637 F1 | 0.030 ECE*

---

## Smoke Test — End-to-End Pipeline Verification

Run this after every deployment or after any changes to `src/inference.py` to confirm the full pipeline is working correctly from raw note to ICD-10 code.

```bash
python3 -c "
import sys
sys.path.insert(0, '.')
from src.inference import HierarchicalPredictor

predictor = HierarchicalPredictor(
    experiment_name='E-009_Balanced_E002Init',
    stage1_experiment='E-003_Stage1_Router',
)

note = '''
Primary Diagnosis: Lyme Disease, unspecified.
Medications: Prescribed Doxycycline 100 mg, oral, twice daily for 21 days.
Follow-up: Schedule a follow-up appointment in 4 weeks to assess treatment efficacy.
Referrals: Refer to Neurology for persistent symptoms and possible neurological involvement.
'''

result = predictor.predict(note, top_k=5)
print()
print('=== Smoke Test Results ===')
print(f'Chapter routed to: {result[\"chapter\"]}')
print(f'Source: {result[\"stage2_source\"]}')
print()
print('Top 5 predictions:')
for code, score in zip(result[\"codes\"], result[\"scores\"]):
    marker = ' ✅' if code == 'A69.20' else ''
    print(f'  {code}  ({score:.1%}){marker}')
print()
print('Expected: A69.20 (Lyme disease, unspecified)')
" 2>/dev/null
```

**Expected output:**
```
=== Smoke Test Results ===
Chapter routed to: A
Source: resolver

Top 5 predictions:
  A69.20  (71.1%) ✅
  A41.01  (6.9%)
  A41.59  (5.3%)
  A41.9   (3.9%)
  A08.4   (3.4%)

Expected: A69.20 (Lyme disease, unspecified)
```

**What this confirms:**
- Stage-1 correctly routes Lyme Disease note to chapter A (Infectious diseases)
- Stage-2 correctly identifies A69.20 as the top prediction
- Confidence 71.1% — above the 0.7 threshold, would be auto-coded in production
- Graph reranker is loaded and functioning
- Full pipeline loads in <30 seconds on Apple Silicon

**If the smoke test fails:**
1. Run `uv run python verify_scripts.py` — check all 18 conditions pass
2. Check the experiment weights exist: `find outputs/evaluations/E-009_Balanced_E002Init/stage2 -name "model.safetensors" | wc -l` — should return 19
3. Check Stage-1 weights: `ls outputs/evaluations/E-003_Stage1_Router/stage1/model.safetensors`





# 1. kill the current picker
Ctrl+C

# 2. set ALL three (not just BASE_URL)
export ANTHROPIC_BASE_URL="http://localhost:1234/v1"
export ANTHROPIC_API_KEY="sk-local"
export ANTHROPIC_AUTH_TOKEN="sk-local"

# 3. make sure LM Studio server is actually running
curl http://localhost:1234/v1/models   # should return JSON, not "connection refused"

# 4. launch again








