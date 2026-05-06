# ICD-10 Pipeline — Full Clean Run Notes

**Purpose:** Step-by-step guide for running the complete experiment chain
from scratch. Every command has been run and verified. Follow this document
exactly to reproduce any result.

**Last updated:** 6 May 2026
**Current best:** E-010_40ep_E002Init — 83.9% E2E | 0.762 F1 | 0.034 ECE | 82.1% Coverage@0.7
**Data status:** Phase 1b locked — HF Hub + DVC (commit 6dda8ac)

---

## Before You Start — Checklist

```bash
# 1. Confirm you are in the project root
pwd
# Expected: .../Notes_to_ICD10_prj

# 2. Confirm the virtual environment is active
which python
# Expected: .../Notes_to_ICD10_prj/.venv/bin/python

# 3. Confirm gold data exists (DVC or rebuild)
ls data/gold/
# Expected: medsynth_gold_apso_*.parquet  MANIFEST_*.json
# If missing: run `dvc pull` or `python scripts/generate_manifest.py`

# 4. Confirm the experiment registry is accessible
uv run python -c "from src.experiment_logger import status; status()"
# Expected: prints the registry table

# 5. Confirm GPU/MPS is available
uv run python -c "import torch; print(torch.backends.mps.is_available())"
# Expected: True (on Apple Silicon)

# 6. Always run pre-flight before any training
uv run python verify_scripts.py
# Expected: ✅ All checks passed — safe to run training

# 7. (Optional) Verify HF data sources
uv run python -c "from huggingface_hub import hf_hub_download; print('HF Hub accessible')"
# Expected: HF Hub accessible
```

If any of the above fail, **stop and fix before proceeding**.

---

## Data Files — What They Are

| File | Records | Description | Source |
|------|---------|-------------|--------|
| `data/gold/medsynth_gold_apso_*.parquet` | 10,240 | Original gold layer — APSO-flipped, ICD-10 redacted, CDC FY2026 validated (9,660 billable + 580 non-billable) | Built from HF Hub |
| `data/gold/medsynth_gold_apso_*.parquet.dvc` | — | DVC pointer file (tracked in git) | — |
| `data/gold/MANIFEST_*.json` | — | SHA256 manifest with validation split | Generated |
| `data/gold/medsynth_gold_augmented.parquet` | 11,214 | Above + 974 synthetic records for chapters O and Z | Historic |

**Data Sources (as of 5 May 2026):**
- All canonical data now pulled from Hugging Face Hub: `SidneyBishop/notes-to-icd10`
- `icd10_notes.parquet` (10,240 rows, SHA256: 7fa03f6...)
- `cdc_fy2026_icd10.parquet` (74,719 codes, SHA256: 2433adf...)
- No external CDC FTP calls — `prepare_data.py` uses `hf_hub_download()`
- DVC tracks derived gold parquet; `dvc pull` restores exact bytes

**Which gold file to use:**

| Experiment | Gold file | Reason |
|---|---|---|
| E-001, E-002, E-003 | Original | Baseline experiments |
| E-010 (current best) | Original | Clean head transfer — E-002 and Stage-2 same gold |
| E-005c pipeline | Augmented | Historic — O chapter augmentation |

**Critical constraint:** E-002 and Stage-2 resolvers must use the **same**
gold file. If they differ, classifier heads cannot transfer and E2E collapses
to ~20%. This is enforced in `scripts/train.py`.

---

## Output Layout — Where Everything Goes

```
outputs/evaluations/E-010_40ep_E002Init/
    stage1/ → (reused from E-003_Hierarchical_ICD10)
    stage2/
        A/
            model.safetensors        ← weights
            config.json
            tokenizer.json
            tokenizer_config.json
            label_map.json           ← chapter-specific id→label mapping
            temperature.json         ← calibration scalar
            train_split.parquet
            val_split.parquet
            test_split.parquet
        B/ C/ D/ ... Z/
        stage2_results.json
    calibration_report.json
    eval/
        summary.json                 ← all scalar metrics
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
E-001  (ICD-3 flat baseline — proof of concept)
  └── 87.2% accuracy, 675 classes

E-002  (flat ICD-10, 40 epochs — CRITICAL: must be 40 epochs on original gold)
  └── 73.3% accuracy, 1,926 classes
  └── provides warm-start weights for all Stage-2 resolvers
        ↓
E-003  (Stage-1 chapter router — train once, reuse in all hierarchical experiments)
  └── 96.4–98.7% chapter routing accuracy
        ↓
E-010  (current best — hierarchical, 40-epoch E-002 init, original gold)
  └── 83.9% E2E accuracy, 0.762 F1, 82.1% Coverage@0.7 at 95.2% accuracy
```

**Why not augmented gold?** E-010 on original gold (83.9%) beats the previous
best production pipeline using augmented gold + graph reranker + Z override
(77.4%). The 40-epoch E-002 init is more valuable than augmentation.

---

## Stage 0 — Prepare Data and Deterministic Splits

**Run once. Do not re-run unless you want to invalidate all previous results.**

**Option A: Pull locked data (recommended)**
```bash
dvc pull
dvc status  # should show "up to date"
```

**Option B: Rebuild from HF Hub (verifies reproducibility)**
```bash
python scripts/generate_manifest.py
# Pulls from HF Hub, validates CDC FY2026, generates gold + SHA256 manifest
```

**Then prepare splits:**
```bash
uv run python scripts/prepare_splits.py \
    --experiment E-010_40ep_E002Init \
    --gold-path data/gold/medsynth_gold_apso_*.parquet
```

**Expected output:**
```
prepare_splits.py — Deterministic Split Generation
Chapter A: 48 records → 38 train / 5 val / 5 test
...
✅ Splits written for 22 chapters
```

**Verify:**
```bash
# Check manifest exists
ls data/gold/MANIFEST_*.json
# Check splits created
ls outputs/evaluations/E-010_40ep_E002Init/stage2/
# Expected: A B C D E F G H I J K L M N O P Q R S T U Z
```

---

## Stage 1 — E-002: Flat ICD-10 Baseline (40 epochs)

**Purpose:** Flat classifier over all 1,926 ICD-10 codes. Serves two purposes:
baseline accuracy for flat approach, and encoder weights to warm-start Stage-2.

**⚠️ Must use 40 epochs.** 20 epochs costs ~4pp E2E accuracy on Stage-2 resolvers.

```bash
uv run python verify_scripts.py && \
uv run python scripts/train.py \
    --experiment E-002_FullICD10_ClinicalBERT \
    --mode flat \
    --label-scheme icd10 \
    --model emilyalsentzer/Bio_ClinicalBERT \
    --code-filter billable \
    --batch-size 16 \
    --epochs 40
```

**Expected:** ~240 minutes. Best epoch ~38–40. Val accuracy ~76%.

**Verify:**
```bash
find outputs/evaluations/E-002_FullICD10_ClinicalBERT -name "model.safetensors"
# Must return at least one path

ls outputs/evaluations/E-002_FullICD10_ClinicalBERT/
# Must contain: model/  label_map.json  train_result.json  test_split.parquet
```

If E-002 already exists in `outputs/evaluations/registry/E-002_FullICD10_ClinicalBERT/`,
skip this step — the registry copy is the 40-epoch version.

---

## Stage 2 — E-003: Stage-1 Chapter Router

**Purpose:** 22-way chapter classifier. Routes every note to the correct
ICD-10 chapter before Stage-2 resolves the specific code.

**Train once. Reused by all hierarchical experiments.**

**Why roberta-base?** Chapter routing is a coarser task — 22 classes with
~440 records each. General encoder is sufficient and trains faster. Clinical
domain knowledge is more valuable in Stage-2 where codes are fine-grained.

```bash
uv run python verify_scripts.py && \
uv run python scripts/train.py \
    --experiment E-003_Hierarchical_ICD10 \
    --mode hierarchical \
    --stage 1 \
    --code-filter billable \
    --epochs 5
```

**Expected:** ~25 minutes. Best epoch ~4. Val accuracy ~96–97%.

**Verify:**
```bash
find outputs/evaluations/E-003_Hierarchical_ICD10/stage1 -name "model.safetensors"
# Must return exactly one path

python3 -c "
import json
lm = json.load(open('outputs/evaluations/E-003_Hierarchical_ICD10/stage1/label_map.json'))
print('Chapters:', len(lm['label2id']))
"
# Expected: 22
```

---

## Stage 3 — E-010: Stage-2 Resolvers (40-epoch E-002 init)

**Purpose:** 19 per-chapter resolvers, each initialised from the 40-epoch
E-002 encoder. This is the key step that produces the 83.9% result.

```bash
uv run python verify_scripts.py && \
uv run python scripts/train.py \
    --experiment E-010_40ep_E002Init \
    --mode hierarchical \
    --stage 2 \
    --code-filter billable \
    --stage2-init outputs/evaluations/registry/E-002_FullICD10_ClinicalBERT \
    --epochs 20
```

**Expected:** ~100 minutes (19 resolvers in sequence).

**Verify the warm start is active** — every chapter should show:
```
↪️ Transfer learning from outputs/evaluations/registry/E-002_FullICD10_ClinicalBERT/model
```
NOT: `⚠️ No checkpoint for chapter X, using base model`

If you see the warning, the `--stage2-init` path is wrong or E-002 is missing.

**Expected per chapter (classifier head mismatch is normal and expected):**
```
BertForSequenceClassification LOAD REPORT
classifier.bias   | MISMATCH | Reinit due to size mismatch - ckpt: torch.Size([1926]) vs model: torch.Size([N])
classifier.weight | MISMATCH | ...
```
The encoder (768-dimensional representations) transfers fully. Only the
final classifier layer is reinitialised — this is correct behaviour.

**Verify:**
```bash
find outputs/evaluations/E-010_40ep_E002Init/stage2 \
    -name "model.safetensors" | wc -l
# Expected: 19
```

**Chapters P, Q, U are skipped by design** — too few records for reliable
training. They use majority-class fallback predictions at inference.

---

## Stage 4 — Calibrate E-010

**Purpose:** Fit temperature scalar T per resolver so confidence scores
are reliable for auto-code threshold decisions.

**Prerequisite:** If `label_map.json` or `test_split.parquet` are missing
from `E-003_Hierarchical_ICD10/stage1/`, run the fix first:

```bash
# Fix 1: generate label_map.json from chapter_mapping.json
python3 -c "
import json
from pathlib import Path
with open('outputs/evaluations/E-003_Hierarchical_ICD10/stage1/chapter_mapping.json') as f:
    ch_map = json.load(f)
label_map = {
    'label2id': ch_map['chapter2id'],
    'id2label': {str(v): k for k, v in ch_map['chapter2id'].items()},
    'num_labels': ch_map['num_chapters'],
    'label_scheme': 'chapter'
}
with open('outputs/evaluations/E-003_Hierarchical_ICD10/stage1/label_map.json', 'w') as f:
    json.dump(label_map, f, indent=4)
print('Written:', len(label_map['id2label']), 'chapters')
"

# Fix 2: regenerate test_split.parquet from gold layer
uv run python -c "
import polars as pl
from pathlib import Path
from sklearn.model_selection import train_test_split

gold_path = sorted(Path('data/gold').glob('medsynth_gold_apso_*.parquet'))[-1]
df = pl.read_parquet(gold_path).filter(pl.col('code_status') == 'billable')
df = df.with_columns(pl.col('standard_icd10').str.slice(0, 1).alias('chapter_label'))
chapters = sorted(df['chapter_label'].unique().to_list())
chapter2id = {ch: i for i, ch in enumerate(chapters)}
df = df.with_columns(
    pl.col('chapter_label').replace(list(chapter2id.keys()), [chapter2id[k] for k in chapter2id])
      .cast(pl.Int64).alias('chapter_id')
)
df_pd = df.select(['id','apso_note','standard_icd10','chapter_label','chapter_id']).to_pandas()
_, temp = train_test_split(df_pd, test_size=0.2, stratify=df_pd['chapter_id'], random_state=42)
_, test = train_test_split(temp, test_size=0.5, random_state=42)
out = Path('outputs/evaluations/E-003_Hierarchical_ICD10/stage1/test_split.parquet')
pl.from_pandas(test).write_parquet(out)
print(f'Written: {len(test)} records')
"
```

Then calibrate:

```bash
uv run python scripts/calibrate.py \
    --experiment E-010_40ep_E002Init \
    --stage1-experiment E-003_Hierarchical_ICD10
```

**Expected summary:**
```
Avg temperature:  ~0.28
Avg ECE:          0.665 → 0.115
Avg Coverage@0.7: 80.5%  (avg accuracy on covered: 90.7%)
```

If ECE gets **worse** after calibration, the classifier heads did not converge
properly — retrain with more epochs or check the warm start path.

**Verify:**
```bash
find outputs/evaluations/E-010_40ep_E002Init/stage2 \
    -name "temperature.json" | wc -l
# Expected: 19
```

---

## Stage 5 — Evaluate E-010

```bash
uv run python scripts/evaluate.py \
    --experiment E-010_40ep_E002Init \
    --mode hierarchical \
    --stage1-experiment E-003_Hierarchical_ICD10 \
    --threshold 0.7
```

**Expected results:**
```
📈 Stage-1 (chapter) accuracy: 0.987
📈 Stage-2 (within-chapter):   0.850
📈 End-to-end accuracy:        0.839
📈 Macro F1:                   0.762
📈 ECE:                        0.034
📈 Coverage@τ=0.7:             82.1% (accuracy=0.952)
```

**Register results:**
```bash
uv run python -c "
from src.experiment_logger import ExperimentLogger, status
el = ExperimentLogger('E-010_40ep_E002Init', script='scripts/train.py')
el.log_results({
    'e2e_accuracy':    0.839,
    'macro_f1':        0.762,
    'stage1_accuracy': 0.987,
    'within_chapter':  0.850,
    'ece':             0.034,
    'coverage_07':     0.821,
})
status()
"
```

---

## Smoke Test — End-to-End Pipeline Verification

Run after any changes to `src/inference.py` or after deploying a new model.

```bash
uv run python -c "
import sys
sys.path.insert(0, '.')
from src.inference import HierarchicalPredictor

predictor = HierarchicalPredictor(
    experiment_name='E-010_40ep_E002Init',
    stage1_experiment='E-003_Hierarchical_ICD10',
)

note = '''
Primary Diagnosis: Lyme Disease, unspecified.
Medications: Prescribed Doxycycline 100mg, oral, twice daily for 21 days.
Follow-up: Schedule a follow-up appointment in 4 weeks.
Referrals: Refer to Neurology for persistent symptoms.
'''

result = predictor.predict(note, top_k=5)
print()
print('=== Smoke Test Results ===')
print(f'Chapter routed to: {result[\"chapter\"]}')
print(f'Source: {result[\"stage2_source\"]}')
print()
print('Top 5 predictions:')
for code, score in zip(result['codes'], result['scores']):
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
  ...

Expected: A69.20 (Lyme disease, unspecified)
```

**If the smoke test fails:**
1. Run `uv run python verify_scripts.py` — all checks must pass
2. Check Stage-2 weights: `find outputs/evaluations/E-010_40ep_E002Init/stage2 -name "model.safetensors" | wc -l` — should return 19
3. Check Stage-1 weights: `ls outputs/evaluations/E-003_Hierarchical_ICD10/stage1/model/model.safetensors`

---

## Session Checklist — Start of Every Session

```bash
# 1. Always start here
uv run python verify_scripts.py

# 2. Check experiment registry
uv run python -c "from src.experiment_logger import status; status()"
```

Both must pass before running anything else.

---

## Key Decisions and Why

| Decision | Rationale |
|---|---|
| roberta-base for Stage-1 | Coarser task — 22 classes, ~440 records each. General encoder sufficient. |
| Bio_ClinicalBERT for Stage-2 | Code resolution needs clinical domain knowledge. MIMIC-III pretraining decisive. |
| 40 epochs for E-002 | Model still improving at epoch 20. +4.1pp E2E accuracy vs 20 epochs. |
| Original gold for E-010 | Full head transfer (E-002 and Stage-2 same code space). Augmented gold causes head mismatch. |
| Presplits mandatory | Without fixed splits, test sets differ per run and results are not comparable. |
| Z override permanently removed | "physical exam" appears in every APSO template — phrase override corrupts 100% of predictions. |
| Skip chapters P, Q, U | Too few records for reliable training. Majority-class fallback is more accurate. |
| **HF Hub + DVC for data (May 2026)** | **Eliminates CDC FTP drift, enables byte-identical reproduction, provides SHA256 audit trail** |

---

## Troubleshooting

**"No module named src.paths"**
Not in project root. Run `cd .../Notes_to_ICD10_prj`.

**"FileNotFoundError: data/gold/medsynth_gold_apso_*.parquet"**
Gold data not pulled. Run `dvc pull` or `python scripts/generate_manifest.py` to rebuild from HF Hub.

**"Could not find artifacts.yaml"**
Same issue — not in project root, or venv not active.

**"hf_hub_download failed"**
No internet or HF Hub down. Check connection, or use `dvc pull` if data already cached in DVC remote.

**Stage-2 shows "⚠️ No checkpoint for chapter X, using base model"**
`--stage2-init` path wrong or E-002 weights missing.
Check: `find outputs/evaluations/registry/E-002_FullICD10_ClinicalBERT -name "model.safetensors"`

**Calibration shows T < 0.1 (clamped to 0.05)**
Resolver is overconfident in wrong direction — head reinit issue.
Retrain with more epochs or verify warm start was applied.

**ECE gets worse after calibration**
Classifier heads did not converge — the warm start may not have transferred.
Verify `↪️ Transfer learning from ...` appeared during Stage-2 training.

**Coverage@0.7 = 0%**
Model is uncalibrated — temperature.json files are missing or stale.
Re-run calibration step.

**OOM during Stage-2 training**
Reduce `--batch-size` to 8 or 4.

---

## Full Leaderboard — 6 May 2026 (Phase 1b locked)

| Rank | Experiment | Stage-1 | E2E | F1 | ECE | Cov@0.7 | Cov Acc |
|---|---|---|---|---|---|---|---|
| 🥇 | **E-010_40ep_E002Init** | **98.7%** | **83.9%** | **0.762** | **0.034** | **82.1%** | **95.2%** |
| 🥈 | E-009_Balanced_E002Init | 96.4% | 79.8% | 0.711 | — | — | — |
| 🥉 | E-005c + Graph + Override | 97.0% | 77.4% | 0.679 | 0.027 | 68.5% | 93.6% |
| 4th | E-002 flat | — | 73.3% | 0.634 | — | — | — |
| 5th | E-008_Balanced | — | 34.2% | 0.249 | — | — | — |
| 6th | E-006_Hierarchical_Clean | — | 23.8% | 0.160 | — | — | — |
| 7th | E-004a_Hierarchical_E002Init | — | 20.9% | 0.141 | — | — | — |
| 8th | E-003_Hierarchical_ICD10 | — | 11.1% | 0.075 | — | — | — |

**The single most important architectural insights:**

> 1. Train E-002 on the **same gold dataset** as Stage-2 — head mismatch
>    collapses E2E from ~80% to ~20%.
>
> 2. Train E-002 for **40 epochs**, not 20 — costs +4.1pp E2E accuracy.

---

*Last updated: 6 May 2026*
*Current best: E-010_40ep_E002Init — 83.9% E2E | 0.762 F1 | 0.034 ECE*
*Data: HF-locked + DVC (commit 6dda8ac)*