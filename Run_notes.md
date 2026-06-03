# ICD-10 Pipeline — Full Clean Run Notes

**Purpose:** Step-by-step guide for running the complete experiment chain
from scratch. Every command has been run and verified. Follow this document
exactly to reproduce any result.

**Last updated:** 3 June 2026
**Headline result:** two end-to-end runs differing only in redaction depth —
code-only (E-009) E2E 0.849 vs code+description (E-016) E2E 0.567. The 28.2-point
gap is diagnosis-description leakage. 0.567 is the de-leaked number; 0.849 is the
leaky baseline.
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
# Expected: medsynth_gold_apso.parquet  MANIFEST_*.json
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
| `data/gold/medsynth_gold_apso.parquet` | 10,240 | Original gold layer — APSO-flipped, ICD-10 *code* redacted (diagnosis description retained), CDC FY2026 validated (9,660 billable + 580 non-billable) | Built from HF Hub |
| `data/gold/medsynth_gold_apso_deleaked.parquet` | 10,240 | De-leaked gold — ICD-10 code *and* its associated diagnosis description redacted (situation 2) | Built with `prepare_data.py --redact-descriptions` |
| `data/gold/medsynth_gold_apso.parquet.dvc` | — | DVC pointer file (tracked in git) | — |
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
| E-001, E-002, E-003 | Original | Baseline experiments (code-only regime) |
| E-009 (code-only baseline) | Original | Leaky baseline — diagnosis description retained |
| E-016 (de-leaked) | De-leaked | Situation 2 — code + description redacted |
| E-005c pipeline | Augmented | Historic — O chapter augmentation |

**Critical constraint:** E-002 and Stage-2 resolvers must use the **same**
gold file. If they differ, classifier heads cannot transfer and E2E collapses
to ~20%. This is enforced in `scripts/train.py`.

---

## Output Layout — Where Everything Goes

```
outputs/evaluations/E-016_Deleaked_FullRebuild/
    stage1/ → (trained under this experiment; de-leaked regime)
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

E-002  (flat ICD-10, 40 epochs — CRITICAL: must be 40 epochs, same gold as Stage-2)
  └── 73.3% accuracy, 1,926 classes
  └── provides warm-start weights for all Stage-2 resolvers
        ↓
Stage-1 chapter router (22-way; Bio_ClinicalBERT — trained per experiment)
  └── 94–98% chapter routing accuracy
        ↓
End-to-end hierarchical run (Stage-1 router + Stage-2 resolvers, E-002 init)
  ├── E-009  code-only regime (diagnosis description retained) → E2E 0.849  [leaky baseline]
  └── E-016  de-leaked regime (code + description redacted)    → E2E 0.567  [clean]
```

**The headline finding:** the same end-to-end pipeline scores 0.849 when the
diagnosis description is left in the note and 0.567 when it is removed. The
28.2-point drop is the leakage, not a change in the model or recipe — only the
redaction depth of the data differs.

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

**For the de-leaked gold (situation 2):**
```bash
uv run python scripts/prepare_data.py --redact-descriptions
# Writes data/gold/medsynth_gold_apso_deleaked.parquet
# Flag OFF = byte-identical to the code-only (0.849) pipeline
```

**Then prepare splits** (substitute the experiment name you are running):
```bash
uv run python scripts/prepare_splits.py \
    --experiment E-016_Deleaked_FullRebuild \
    --gold-path data/gold/medsynth_gold_apso_deleaked.parquet
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
ls outputs/evaluations/E-016_Deleaked_FullRebuild/stage2/
# Expected: A B C D E F G H I J K L M N O P Q R S T U Z
```

---

---

## Stage 0b — Build Knowledge Graph (~5 min)

**Required before calibration and evaluation.** The graph reranker used by
`evaluate.py` and `src/inference.py` depends on `data/graph/icd10_knowledge_graph.pkl`.
Without it, `evaluate.py` will fail with `FileNotFoundError`.

**Run once per gold layer** — if you regenerate the gold parquet, rebuild the graph.

```bash
uv run python scripts/build_graph.py
```

**Expected output:**
```
Graph: 6,837 nodes, 258,954 edges
Codes:   1,926
Concepts:4,889
✅ Graph complete
```

**Verify:**
```bash
ls data/graph/icd10_knowledge_graph.pkl
# Expected: file exists, ~50MB
```

---

## Stage 1 — E-002: Flat ICD-10 Baseline (40 epochs)

**Purpose:** Flat classifier over all 1,926 ICD-10 codes. Serves two purposes:
baseline accuracy for flat approach, and encoder weights to warm-start Stage-2.

**⚠️ Must use 40 epochs.** The model is still improving at epoch 20; 40 epochs
produces richer encoder representations for the Stage-2 warm start.

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

## Stage 2 — Stage-1 Chapter Router

**Purpose:** 22-way chapter classifier. Routes every note to the correct
ICD-10 chapter before Stage-2 resolves the specific code.

**Train per experiment.** For a clean end-to-end run the router must be trained
on the *same* gold regime as Stage-2 — a router trained on code-only text but
evaluated on de-leaked notes suffers a train/serve mismatch (Stage-1 accuracy
collapsed from ~0.95 to ~0.76 when this was violated). Train Stage-1 under the
experiment you are running, on the matching gold.

**Model:** Bio_ClinicalBERT (confirmed from E-003's saved config — `model_type`
bert, vocab 28996). The router uses the same clinical encoder as Stage-2; the
chapter-routing task benefits from clinical domain knowledge just as code
resolution does.

```bash
uv run python verify_scripts.py && \
uv run python scripts/train.py \
    --experiment E-016_Deleaked_FullRebuild \
    --mode hierarchical \
    --stage 1 \
    --model emilyalsentzer/Bio_ClinicalBERT \
    --code-filter billable \
    --gold-path data/gold/medsynth_gold_apso_deleaked.parquet \
    --epochs 5
```

**Expected:** ~25 minutes. Best epoch ~4. Val accuracy ~94–97%.

**Verify:**
```bash
find outputs/evaluations/E-016_Deleaked_FullRebuild/stage1 -name "model.safetensors"
# Must return exactly one path

python3 -c "
import json
lm = json.load(open('outputs/evaluations/E-016_Deleaked_FullRebuild/stage1/label_map.json'))
print('Chapters:', len(lm['label2id']))
"
# Expected: 22
```

---

## Stage 3 — Stage-2 Resolvers (40-epoch E-002 init)

**Purpose:** 19 per-chapter resolvers, each initialised from the 40-epoch
E-002 encoder. Trained on the same gold regime as Stage-1.

```bash
uv run python verify_scripts.py && \
uv run python scripts/train.py \
    --experiment E-016_Deleaked_FullRebuild \
    --mode hierarchical \
    --stage 2 \
    --code-filter billable \
    --stage2-init outputs/evaluations/registry/E-002_FullICD10_ClinicalBERT \
    --gold-path data/gold/medsynth_gold_apso_deleaked.parquet \
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
find outputs/evaluations/E-016_Deleaked_FullRebuild/stage2 \
    -name "model.safetensors" | wc -l
# Expected: 19
```

**Chapters P, Q, U are skipped by design** — too few records for reliable
training. They use majority-class fallback predictions at inference.

> **Tip:** Stages 2 and 3, plus calibration and evaluation, can be run in one
> orchestrated command via `run_experiment.py --train-stage1` (this is how E-016
> was produced — see the end-to-end block in the README). Running them
> individually as above is equivalent and easier to debug.

---

## Stage 4 — Calibrate

**Purpose:** Fit temperature scalar T per resolver so confidence scores
are reliable for auto-code threshold decisions.

**Prerequisite:** If `label_map.json` or `test_split.parquet` are missing
from the Stage-1 directory, run the fix first:

```bash
# Fix 1: generate label_map.json from chapter_mapping.json
python3 -c "
import json
from pathlib import Path
stage1 = 'outputs/evaluations/E-016_Deleaked_FullRebuild/stage1'
with open(f'{stage1}/chapter_mapping.json') as f:
    ch_map = json.load(f)
label_map = {
    'label2id': ch_map['chapter2id'],
    'id2label': {str(v): k for k, v in ch_map['chapter2id'].items()},
    'num_labels': ch_map['num_chapters'],
    'label_scheme': 'chapter'
}
with open(f'{stage1}/label_map.json', 'w') as f:
    json.dump(label_map, f, indent=4)
print('Written:', len(label_map['id2label']), 'chapters')
"

# Fix 2: regenerate test_split.parquet from the gold layer (use the matching gold)
uv run python -c "
import polars as pl
from pathlib import Path
from sklearn.model_selection import train_test_split

gold_path = Path('data/gold') / 'medsynth_gold_apso_deleaked.parquet'
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
out = Path('outputs/evaluations/E-016_Deleaked_FullRebuild/stage1/test_split.parquet')
pl.from_pandas(test).write_parquet(out)
print(f'Written: {len(test)} records')
"
```

Then calibrate (Stage-1 experiment must match where Stage-1 was trained):

```bash
uv run python scripts/calibrate.py \
    --experiment E-016_Deleaked_FullRebuild \
    --stage1-experiment E-016_Deleaked_FullRebuild
```

**Expected summary (de-leaked regime, E-016):**
```
Avg temperature:  ~0.34
Avg ECE:          0.413 → 0.196
Avg Coverage@0.7: ~54%   (avg accuracy on covered: ~0.79)
```

If ECE gets **worse** after calibration, the classifier heads did not converge
properly — retrain with more epochs or check the warm start path.

**Verify:**
```bash
find outputs/evaluations/E-016_Deleaked_FullRebuild/stage2 \
    -name "temperature.json" | wc -l
# Expected: 19
```

---

## Stage 5 — Evaluate

```bash
uv run python scripts/evaluate.py \
    --experiment E-016_Deleaked_FullRebuild \
    --mode hierarchical \
    --stage1-experiment E-016_Deleaked_FullRebuild \
    --threshold 0.7
```

**Expected results — situation 2 (de-leaked, E-016):**
```
📈 Stage-1 (chapter) accuracy: 0.948
📈 Stage-2 (within-chapter):   0.598
📈 End-to-end accuracy:        0.567
📈 Macro F1:                   0.446
📈 ECE:                        0.0703
📈 Coverage@τ=0.7:             48.2% (accuracy=0.818)
```

**For comparison — situation 1 (code-only, E-009, leaky baseline):**
```
📈 Stage-1 (chapter) accuracy: 0.984
📈 Stage-2 (within-chapter):   0.863
📈 End-to-end accuracy:        0.849
📈 Macro F1:                   0.774
📈 ECE:                        0.024
📈 Coverage@τ=0.7:             81.2% (accuracy=0.952)
```

The difference between the two is the quantified diagnosis-description leakage.

**Register results:**
```bash
uv run python -c "
from src.experiment_logger import ExperimentLogger, status
el = ExperimentLogger('E-016_Deleaked_FullRebuild', script='scripts/run_experiment.py')
el.log_results({
    'e2e_accuracy':    0.567,
    'macro_f1':        0.446,
    'stage1_accuracy': 0.948,
    'within_chapter':  0.598,
    'ece':             0.0703,
    'coverage_07':     0.482,
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
    experiment_name='E-016_Deleaked_FullRebuild',
    stage1_experiment='E-016_Deleaked_FullRebuild',
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

> Note: this smoke-test note states the diagnosis in plain text ("Lyme Disease"),
> so it exercises the inference path, not the de-leaked regime. It is a wiring
> check, not a leakage-free accuracy measurement.

**If the smoke test fails:**
1. Run `uv run python verify_scripts.py` — all checks must pass
2. Check Stage-2 weights: `find outputs/evaluations/E-016_Deleaked_FullRebuild/stage2 -name "model.safetensors" | wc -l` — should return 19
3. Check Stage-1 weights: `ls outputs/evaluations/E-016_Deleaked_FullRebuild/stage1/model/model.safetensors`

---


---

## Disk Management — Reclaim Checkpoint Storage

Training checkpoints accumulate during the pipeline (~3.6GB per resolver,
~150-270GB for a full end-to-end run). After training completes
successfully, use the cleanup script to reclaim disk space.

**The cleanup script is safe** — it only removes `checkpoint-N/` and
`checkpoints/` directories. All final model weights, label maps, calibration
temperatures, and evaluation results are preserved.

```bash
# Preview what would be deleted (no changes made)
uv run python scripts/cleanup.py --dry-run

# Clean all experiments
uv run python scripts/cleanup.py

# Keep current best, clean everything else
uv run python scripts/cleanup.py --keep E-016_Deleaked_FullRebuild

# Clean one experiment only
uv run python scripts/cleanup.py --experiment E-003_Hierarchical_ICD10
```

**Expected output:**
```
✅ Deleted 158 checkpoint directories
   Freed 266.8 GB
```

**Verify after cleanup:**
```bash
du -sh outputs/evaluations/*/  | sort -rh
# Each experiment should now be 7-10GB (model weights only)
```

> **When to run:** After every full pipeline run before committing results.
> Add to your session checklist alongside `verify_scripts.py`.

---

## Session Checklist — Start of Every Session

```bash
# 1. Always start here
uv run python verify_scripts.py

# 2. Check experiment registry
uv run python -c "from src.experiment_logger import status; status()"

# 3. (After training) Reclaim checkpoint storage
uv run python scripts/cleanup.py --dry-run  # preview
uv run python scripts/cleanup.py            # delete
```

Steps 1 and 2 must pass before running anything else.
Step 3 is optional — run after any training to reclaim disk space.

---

## Key Decisions and Why

| Decision | Rationale |
|---|---|
| Bio_ClinicalBERT for Stage-1 | Confirmed from E-003's saved config (model_type bert, vocab 28996). Chapter routing benefits from clinical domain knowledge; the router shares the clinical encoder with Stage-2. |
| Bio_ClinicalBERT for Stage-2 | Code resolution needs clinical domain knowledge. MIMIC-III pretraining decisive. |
| 40 epochs for E-002 | Model still improving at epoch 20; 40 epochs aids convergence and produces a richer encoder for the Stage-2 warm start. |
| Same gold for E-002 and Stage-2 | Full head transfer (same code space). Mismatched gold causes head mismatch and E2E collapse. |
| Stage-1 trained on the same regime as Stage-2 | A router trained on code-only text but served de-leaked notes mismatches (Stage-1 acc 0.95→0.76). Train both stages on the same gold. |
| Presplits mandatory | Without fixed splits, test sets differ per run and results are not comparable. |
| Z override permanently removed | "physical exam" appears in every APSO template — phrase override corrupts 100% of predictions. |
| Skip chapters P, Q, U | Too few records for reliable training. Majority-class fallback is more accurate. |
| **HF Hub + DVC for data (May 2026)** | **Eliminates CDC FTP drift, enables byte-identical reproduction, provides SHA256 audit trail** |

---

## Troubleshooting

**"No module named src.paths"**
Not in project root. Run `cd .../Notes_to_ICD10_prj`.

**"FileNotFoundError: data/gold/medsynth_gold_apso.parquet"**
Gold data not pulled. Run `dvc pull` or `python scripts/generate_manifest.py` to rebuild from HF Hub.

**"Could not find artifacts.yaml"**
Same issue — not in project root, or venv not active.

**"hf_hub_download failed"**
No internet or HF Hub down. Check connection, or use `dvc pull` if data already cached in DVC remote.

**Stage-2 shows "⚠️ No checkpoint for chapter X, using base model"**
`--stage2-init` path wrong or E-002 weights missing.
Check: `find outputs/evaluations/registry/E-002_FullICD10_ClinicalBERT -name "model.safetensors"`

**Stage-1 chapter accuracy much lower at eval than at calibration**
Train/serve regime mismatch — Stage-1 was trained on a different gold regime than
the notes it is evaluated on. Retrain Stage-1 on the same gold as Stage-2.

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

## Leaderboard — End-to-End Runs (3 June 2026)

The two rows that matter are the same pipeline under the two redaction regimes.
Earlier experiments are architecture history (all code-only regime) and are not
directly comparable to the de-leaked number.

| Run | Regime | Stage-1 | E2E | F1 | ECE | Cov@0.7 | Cov Acc |
|---|---|---|---|---|---|---|---|
| **E-009_Balanced_E002Init** | code-only (leaky baseline) | 98.4% | **84.9%** | 0.774 | 0.024 | 81.2% | 95.2% |
| **E-016_Deleaked_FullRebuild** | code + description (clean) | 94.8% | **56.7%** | 0.446 | 0.070 | 48.2% | 81.8% |
| E-015_E009_Deleaked | de-leaked, Stage-1 reused (lower bound) | 75.8% | 48.2% | 0.368 | 0.131 | 47.5% | — |
| E-005c + Graph + Override | code-only (historic) | 97.0% | 77.4% | 0.679 | 0.027 | 68.5% | 93.6% |
| E-002 flat | code-only (historic) | — | 73.3% | 0.634 | — | — | — |

*E-015 is retained as the intermediate step that exposed the Stage-1 train/serve
mismatch: reusing a code-only-trained router on de-leaked notes dropped Stage-1 to
0.758 and E2E to 0.482; retraining Stage-1 on de-leaked data (E-016) recovered it to
0.948 / 0.567.*

**The single most important findings:**

> 1. **Diagnosis-description leakage inflates the headline by ~33%** — the same
>    pipeline scores 0.849 code-only vs 0.567 de-leaked. Report 0.567 as the clean
>    number, 0.849 as the leaky baseline.
>
> 2. Train E-002 on the **same gold dataset** as Stage-2 — head mismatch collapses
>    E2E from ~80% to ~20%.
>
> 3. Train **both stages on the same redaction regime** — a code-only router served
>    de-leaked notes mismatches and understates true accuracy.

---

*Last updated: 3 June 2026*
*Headline: code-only (E-009) E2E 0.849 vs de-leaked (E-016) E2E 0.567 — the gap is diagnosis-description leakage*
*Data: HF-locked + DVC (commit 6dda8ac)*
