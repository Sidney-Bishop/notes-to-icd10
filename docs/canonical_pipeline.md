# Canonical Pipeline — Notes_to_ICD10

*Authoritative, source-derived description of the training pipeline. Every
experiment name, init source, and hyperparameter below was read directly from
the notebook `cfg` dicts (notebooks 02–05) and the scripts, not inferred. Where
this document and older docs/README text disagree, this document is correct and
the older text is drift to be corrected.*

*Established: 2026-06-01. Naming basis: notebook source of truth.*

---

## 1. Naming truth (read this first)

The notebooks define exactly these experiments. These names are canonical:

| ID | `experiment_name` (canonical) | Notebook | What it is |
|----|-------------------------------|----------|------------|
| E-001 | `E-001_Baseline_ICD3` | 02 | ICD-3 flat baseline, 675 classes |
| E-002 | `E-002_FullICD10_ClinicalBERT` | 03 | Flat ICD-10, 1,926 billable classes |
| E-003 | `E-003_Hierarchical_ICD10` | 04 | Hierarchical, **cold-start** Stage-2 (the failure: 12.7%). Also the notebook that trains the **Stage-1 chapter router**. |
| E-009 | `E-009_Balanced_E002Init` | 05 | Hierarchical, Stage-2 from E-002 init. **The best/canonical hierarchical model.** |

**`E-010_40ep_E002Init` is NOT a canonical experiment.** No notebook defines it.
It appears only in the README headline and in run directories created outside the
notebooks. It is naming drift and is to be deprecated and deleted. The canonical
name for the best hierarchical model is **`E-009_Balanced_E002Init`** (notebook
05, `cfg["experiment_name"]`).

**Stage-1 is trained once and reused.** Notebook 04 (E-003) trains the 22-way
chapter router. Notebook 05 (E-009) *loads* that router from the E-003 registry
and does not retrain it (`cfg["stage1_source"] = "E-003_Hierarchical_ICD10"`,
and cell 19: "Stage-1 was trained in notebook 04 … No retraining needed").
Therefore in calibrate/evaluate, `--stage1-experiment` is
**`E-003_Hierarchical_ICD10`** — the experiment that owns the router — not the
stage-2 experiment.

---

## 2. The transfer-learning chain (verified from `cfg` dicts)

```
Bio_ClinicalBERT (emilyalsentzer/Bio_ClinicalBERT)
   │
   ├──> E-001_Baseline_ICD3            (notebook 02; 30 epochs, lr 2e-5, batch 16)
   │        │
   │        └──> Stage-1 chapter router, trained in E-003 (notebook 04)
   │                  init from E-001; 5 epochs, lr 2e-5, batch 16
   │                  → registry/E-003_Hierarchical_ICD10/stage1/model/
   │
   └──> E-002_FullICD10_ClinicalBERT   (notebook 03; 40 epochs, lr 2e-5, batch 16)
            │
            └──> Stage-2 resolvers, trained in E-009 (notebook 05)
                      init from E-002 (ignore_mismatched_sizes=True replaces the
                      1,926-way head with each chapter's head)
                      20 epochs, lr 2e-5, batch 16, warmup_ratio 0.1
                      → E-009_Balanced_E002Init/stage2/{chapter}/model/
```

Shared across all stages: `max_length` 512, `weight_decay` 0.01, `seed` 42,
`code_status_filter` "billable", payload column `apso_note`.

Skip chapters (too few classes for a resolver; Stage-1 prediction used directly
as the code via a fallback): **U, P, Q**. Trainable chapters: **19**.
Stage-2 priority order (notebook 05): Z, R, T, S, B, K, M, I, L, D, E, G, O, N,
J, F, H, C, A.

---

## 3. Canonical run order (scripts)

This is the headless-script equivalent of running notebooks 01→05 in order, plus
the knowledge-graph build the reranker depends on. Every argument is given
explicitly — do not rely on CLI defaults, several of which do not match the
trained regime (see §4).

```bash
# ─────────────────────────────────────────────────────────────────────────────
# 0. Pre-flight (optional but recommended)
#    Checks files exist, D007 layout fixes present, loggers wired, HF-not-FTP.
#    NOTE: its DVC-pointer checks ([3d]) FAIL on a fresh clone lacking .dvc files
#    (see open question Q2). That failure does not block the pipeline.
uv run python verify_scripts.py

# ─────────────────────────────────────────────────────────────────────────────
# 1. Gold layer — pull raw from HF, SHA256-verify, write the gold parquet.
#    Entry point. No pre-step. Flags: --no-duckdb, --offline, --dry-run only.
uv run python scripts/prepare_data.py
#    → data/gold/medsynth_gold_apso.parquet

# ─────────────────────────────────────────────────────────────────────────────
# 2. Deterministic splits (filter-then-split on billable; seed 42).
uv run python scripts/prepare_splits.py \
    --experiment E-009_Balanced_E002Init \
    --gold-path data/gold/medsynth_gold_apso.parquet \
    --code-filter billable

# ─────────────────────────────────────────────────────────────────────────────
# 3. Knowledge graph (REQUIRED before evaluate — the reranker reads it).
#    Pass --gold-path explicitly: the default looks for the wrong filename
#    (medsynth_gold_augmented.parquet) and otherwise picks "latest by sort".
#    Needs scispacy + en_ner_bc5cdr_md + UMLS linker (download on first run).
uv run python scripts/build_graph.py \
    --gold-path data/gold/medsynth_gold_apso.parquet
#    → data/graph/icd10_knowledge_graph.pkl + two JSON indices

# ─────────────────────────────────────────────────────────────────────────────
# 4. E-001 — ICD-3 baseline (notebook 02 regime). Init for the Stage-1 router.
#    30 epochs, lr 2e-5, batch 16. code-filter ALL (verified nb02 cell 16: the
#    billable filter is commented out; audit trail records code_status_filter "all").
uv run python scripts/train.py \
    --experiment E-001_Baseline_ICD3 \
    --mode flat --label-scheme icd3 \
    --model emilyalsentzer/Bio_ClinicalBERT \
    --code-filter all --batch-size 16 --epochs 30

# ─────────────────────────────────────────────────────────────────────────────
# 5. E-002 — flat ICD-10 (notebook 03 regime). Init for the Stage-2 resolvers.
#    40 epochs (cfg: "consistent with E-001 final run"), lr 2e-5, batch 16.
uv run python scripts/train.py \
    --experiment E-002_FullICD10_ClinicalBERT \
    --mode flat --label-scheme icd10 \
    --model emilyalsentzer/Bio_ClinicalBERT \
    --code-filter billable --batch-size 16 --epochs 40

# ─────────────────────────────────────────────────────────────────────────────
# 6. E-003 Stage-1 — 22-way chapter router (notebook 04 regime).
#    MUST init from E-001 (else cold-starts from base BERT). 5 epochs, batch 16.
#    Trained ONCE; reused by E-009 via --stage1-experiment in calibrate/evaluate.
uv run python scripts/train.py \
    --experiment E-003_Hierarchical_ICD10 \
    --mode hierarchical --stage 1 \
    --stage1-init outputs/evaluations/E-001_Baseline_ICD3/model \
    --code-filter billable --epochs 5 --batch-size 16

# ─────────────────────────────────────────────────────────────────────────────
# 7. E-009 Stage-2 — 19 per-chapter resolvers, warm-started from E-002.
#    20 epochs, lr 2e-5, batch 16, warmup 0.1. Skips U/P/Q.
uv run python scripts/train.py \
    --experiment E-009_Balanced_E002Init \
    --mode hierarchical --stage 2 \
    --stage2-init outputs/evaluations/E-002_FullICD10_ClinicalBERT \
    --code-filter billable --epochs 20 --batch-size 16 --use-presplit

# ─────────────────────────────────────────────────────────────────────────────
# 8. Calibrate — temperature scaling for Stage-1 + 19 resolvers.
#    --stage1-experiment is E-003 (where the router lives), NOT E-009.
uv run python scripts/calibrate.py \
    --experiment E-009_Balanced_E002Init \
    --stage1-experiment E-003_Hierarchical_ICD10 \
    --threshold 0.7

# ─────────────────────────────────────────────────────────────────────────────
# 9. Evaluate — end-to-end hierarchical on the test split.
#    Same --stage1-experiment E-003 as calibrate.
uv run python scripts/evaluate.py \
    --experiment E-009_Balanced_E002Init \
    --mode hierarchical \
    --stage1-experiment E-003_Hierarchical_ICD10 \
    --threshold 0.7
```

---

## 4. Gotchas (each cost real debugging time)

- **`--stage1-experiment` is `E-003_Hierarchical_ICD10`, not the stage-2 name.**
  The router is owned by E-003 and reused. The script *default* is already
  `E-003_Hierarchical_ICD10` — correct — but only works if E-003's Stage-1 was
  trained cleanly (step 6). A prior on-disk E-003 Stage-1 was left in the D007
  split layout (config/tokenizer but no `model.safetensors`); retraining via
  step 6 replaces it with a complete, loadable `stage1/model/`.
- **`build_graph.py` is required, and its `--gold-path` default is wrong.** Pass
  `--gold-path data/gold/medsynth_gold_apso.parquet` explicitly. Without the
  graph, evaluate's reranker has nothing to read.
- **`--code-filter` differs by experiment — verified from each notebook's own cfg:**
  E-001 uses **`all`** (10,240 records; notebook 02 cell 16 keeps the billable
  filter commented out and its audit trail records `code_status_filter: "all"`).
  E-002 uses **`billable`** (9,660; notebook 03 cfg `code_status_filter: "billable"`,
  matching `_filter_gold`'s own docstring). The `train.py` argparse help
  ("`all` matches E-001/E-002") is misleading — it is correct for E-001 only.
  Stages 1/2 (E-003/E-009) use `billable`.
- **`--batch-size` must be passed explicitly.** The CLI default is 8; every
  notebook cfg uses 16. Omitting it trains at half the intended batch size.
- **`--stage2-init` points at the experiment ROOT, not its `model/` dir.** The
  code tries `{init}/model` as a candidate and appends the subdir itself; passing
  `.../E-002_FullICD10_ClinicalBERT/model` resolves to `.../model/model`, fails
  all candidates, and silently cold-starts each resolver from base BERT (the
  E-003 12.7% failure mode). Correct: `.../E-002_FullICD10_ClinicalBERT`.
- **`--use-presplit` is Stage-2 only.** Flat mode (E-001/E-002) and Stage-1
  ignore it and self-split via `_split_dataframe` (stratified, seed 42). For
  Stage-2 it requires `prepare_splits.py` to have run under the SAME experiment
  (E-009) so the per-chapter `stage2/{ch}/*_split.parquet` files exist; otherwise
  it silently falls through to an internal split.
- **`--epochs` must be passed explicitly.** The `train.py` CLI default is 10
  (artifacts.yaml says 3); neither matches any trained regime. Canonical: E-001
  = 30, E-002 = 40, Stage-1 = 5, Stage-2 = 20.
- **E-002 is 40 epochs.** (A prior session ran 30 and observed convergence by
  epoch 27; the source regime is 40, "consistent with E-001 final run." Use 40
  for a source-faithful reproduction.)
- **Headline number is provisional under current redaction (D005).** Gold redacts
  ICD-10 code strings but retains semantic diagnosis labels, biasing accuracy
  upward via residual leakage. The first publishable number requires
  semantic-label redaction (open question Q8).

---

## 5. Reference numbers (from notebook source / logs)

| Experiment | Task | Accuracy (logged) | Notes |
|---|---|---|---|
| E-001 | ICD-3, 675 classes | 87.6% | not billable; not directly comparable |
| E-002 | flat ICD-10, 1,926 | 73.0% | flat baseline; chapter acc ~91.2% |
| E-003 | hierarchical cold start | 12.7% (e2e 0.111 logged) | right architecture, wrong Stage-2 init |
| **E-009** | **hierarchical, E-002 init** | **e2e 0.798 logged** (overview cites 77.2%) | **canonical best** |

Note the within-E-009 discrepancy (logged 0.798 vs overview 77.2%): these come
from different runs/measurement points of the same notebook recipe over time, and
the README's "83.9%" headline came from a later un-named run (the E-010 drift).
The authoritative E-009 number is whatever a fresh canonical run (§3) produces;
until then, treat 0.798 (the notebook's own logged e2e) as the reference and the
README's 83.9% as unverified drift.

---

## 6. Out of scope here (verify before documenting as canonical)

Real entry points that exist but are NOT part of this core path and were not
traced: the SupCon/hybrid Z-chapter variants (E-014 etc.), ModernBERT trials
(E-012/E-013), MIMIC-IV validation, `serve.py`. Read their argparse + I/O before
adding any to this document.
