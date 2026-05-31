# Architecture

Current description of how the system is built. **State file — overwrite freely**
as the refactor changes the system. Describes what *is*, never what was.

> Status of this document: drafted 2026-05-31 from a read of `src/` and
> `scripts/` in the uploaded snapshot. Items marked **(unconfirmed)** were not
> verified against a live run and should be checked before relying on them.

## What the system does

Predicts billable ICD-10 diagnostic codes from clinical notes (APSO-structured:
Assessment, Plan, Subjective, Objective). The model is a two-stage hierarchical
classifier built on `emilyalsentzer/Bio_ClinicalBERT`. Trained and evaluated on
the synthetic MedSynth dataset; not validated on real clinical data (see
`charter.md` for scope, `open_questions.md` for the synthetic→real gap).

## Pipeline stages (data → models → inference)

```
HF Hub (SidneyBishop/notes-to-icd10)
   │  prepare_data.py  (SHA256-locked download)
   ▼
silver  ──Pydantic gatekeeper──▶  gold  (medsynth_gold_apso.parquet)
   │  prepare_splits.py  (per-chapter stratified 80/10/10, seed 42)
   ▼
per-chapter train/val/test splits  ──▶  outputs/evaluations/<EXP>/stage2/<CH>/*.parquet
   │  train.py
   ├─ flat mode:          single classifier (E-001 ICD-3, E-002 flat ICD-10)
   └─ hierarchical mode:
        Stage-1: 22-way chapter router
        Stage-2: per-chapter resolvers, warm-started from E-002 weights
   │  calibrate.py  (temperature scaling per model → temperature.json)
   ▼
inference (src/inference.py HierarchicalPredictor)
   Stage-1 routes → Stage-2 resolves → T-scaled softmax → confidence gate (τ=0.7)
   low-confidence / Z-chapter → GraphReranker (UMLS graph + Z-phrase dictionary)
   │  evaluate.py  (reuses HierarchicalPredictor; assembles test set from
   ▼                 per-chapter test_split.parquet files)
outputs/evaluations/<EXP>/eval/summary.json  (+ calibration, sweep, predictions)
```

## Data layer (Medallion architecture)

- **Source:** HF dataset `SidneyBishop/notes-to-icd10`, downloaded by
  `prepare_data.py` with SHA256 verification against hardcoded expected hashes
  (`icd10_notes.parquet`, `cdc_fy2026_icd10.parquet`). A hash mismatch raises
  immediately rather than continuing on unexpected data.
- **Phased build** (`prepare_data.py`): 1a ingest → 1b CDC validation (classifies
  each code billable / non_billable_parent / placeholder_x / invalid) → 1e
  Pydantic firewall (`src/gatekeeper.py`) → 1g DuckDB silver vault → 2a decimal
  restoration → 2c status annotation → 3a APSO-flip → 3b leakage detection → 3c
  ICD-10 redaction → 4 export gold.
- **Gold artifact:** `data/gold/medsynth_gold_apso.parquet`. Canonical record
  count and code count have two reported figures (9,660/1,926 vs 9,578/1,914) —
  see `open_questions.md` Q1.
- **Config authority:** `src/config.py` `ArtifactConfig` singleton resolves all
  paths from `artifacts.yaml`; provides Polars I/O and JSONL audit logging.
- **Versioning:** gold + reference parquets are DVC-tracked. **(unconfirmed:**
  the DVC remote is a local-filesystem store and a fresh clone could not pull all
  files in testing on 2026-05-31 — see `journal.md` / `open_questions.md`.)

## Preprocessing (`src/preprocessing.py`)

- **APSO-Flip:** reorders SOAP sections to Assessment-first so diagnostic content
  survives Bio_ClinicalBERT's 512-token truncation. Applied identically at
  training and inference time.
- **ICD-10 redaction:** regex-removes code strings and `(ICD-10: …)` parentheticals
  from note text to prevent label leakage.
- Two interfaces: DataFrame-level (Polars, used by the gold pipeline) and
  single-string (`prepare_inference_input`, used at runtime).

## Model layer

- **Abstraction:** `src/adapters.py` `EncoderAdapter` (concrete) behind a
  `ModelAdapter` interface, so the encoder is a config value — swappable without
  touching training code. A `GenerativeAdapter` stub documents a future LLM path.
- **Stage-1 router:** 22-way chapter classifier (chapter = first letter of code).
- **Stage-2 resolvers:** one classifier per chapter; warm-started from the
  40-epoch flat E-002 model. Verified in `outputs/experiments.json`:
  `E-010_40ep_E002Init` has `stage2_init: outputs/evaluations/E-002_FullICD10_ClinicalBERT`,
  while E-002 itself trained from `stage2_init: none`. This warm start is the
  project's central finding — see `decisions.md`.
- **Skip chapters:** P, Q, U have no trained resolver (too few records);
  `train.py` records fallback default codes for them in `stage2_results.json`.
- **Calibration:** `calibrate.py` fits a single temperature T per model by
  minimising NLL on the held-out split (Guo et al. 2017); written to
  `temperature.json`, read at inference load time.

## Inference (`src/inference.py`)

- `HierarchicalPredictor` loads Stage-1 + all Stage-2 models + temperatures once.
- Path resolution via `src/paths.py` `ExperimentPaths`, which auto-detects three
  historical on-disk layout conventions (FLAT / SINGLE / NESTED).
- Input validated by `ClinicalNoteInput` (Pydantic): empty → error; <20 words →
  warn; >400 words → truncation warn.
- **GraphReranker** (`src/graph_reranker.py`) fires only when top confidence <0.7
  or chapter == Z; combines a UMLS knowledge-graph affinity score with a
  high-precision Z-phrase dictionary. Found to have minimal impact on the
  well-calibrated E-010 model.

## Evaluation (`scripts/evaluate.py`)

- Reuses the production `HierarchicalPredictor`, so evaluation mirrors deployment
  exactly.
- **Test set is assembled from the per-chapter `test_split.parquet` files** — so
  the split-generation step (`prepare_splits.py`) directly determines what is
  evaluated. (This coupling is why a split-filter bug propagated into reported
  numbers — see `decisions.md` D001.)
- Outputs `summary.json` with: n_test, stage1_accuracy, within_chapter_accuracy,
  e2e_accuracy, macro_f1, ece, coverage_at_threshold, accuracy_at_threshold,
  per-chapter accuracy.

> **Current on-disk state (verified 2026-05-31) — split and eval are NOT yet
> reconciled.** The E-010 `summary.json` on disk is the historical run:
> n_test = **966**, e2e = **0.8385**, macro_f1 = 0.7628, ECE = 0.0329,
> stage1 = 0.9855, timestamped 2026-05-06. Meanwhile `prepare_splits.py` was
> re-run with the billable filter (D001) and now produces **971** test records —
> but `evaluate.py` has NOT been re-run against the 971 split. So: **971 is the
> current split regime; 966/83.9% is the last *evaluated* regime; no 971-based
> accuracy number exists yet.** Do not quote an 83.9% figure as belonging to the
> 971 regime — they are different runs. Reconciling this (re-run evaluate on the
> 971 split) is a tracked next step (`status.md`). Lowest chapter: Z at 0.6288
> (n=132), the target of the E-014 SupCon-Z work.

## Experiment tracking & conventions

- **Experiment naming:** `outputs/evaluations/E-xxx_<name>/` with `stage1/`,
  `stage2/<CH>/`, `eval/`, `calibration_report.json`.
- **Logging:** `src/experiment_logger.py` writes `outputs/run.log` and
  `outputs/experiments.json`. MLflow is also written (sqlite backend).
- **Environment:** Python managed by `uv` (`uv.lock`); invoke via `uv run …`.
- **Hardware / device:** `src/inference.py` (lines ~168-173, ~504-509) does
  explicit backend auto-detection in priority order **MPS → CUDA → CPU**. The
  CUDA path exists and is real code; it is simply second in priority behind MPS.
  Developed and run on Apple Silicon (M5 Max, 128 GB), so MPS is the active
  backend; the CUDA path is exercised only on CUDA hardware (verified present in
  code 2026-05-31, not run on CUDA here).

## Auxiliary scripts (not on the core path)

`build_graph.py` (builds the UMLS reranker graph), `train_supcon_z.py` +
`evaluate_hybrid.py` (Z-chapter supervised-contrastive fine-tuning),
`augment.py`, `cluster_analysis.py`, `validation/validate_mimic_*.py` (MIMIC-IV
real-data validation), `cleanup.py` (reclaim checkpoint disk), `generate_manifest.py`
(SHA256 manifest).

**Serving (live, verified 2026-05-31).** Two files: `scripts/serve.py` is the
launcher (`main()` → `uvicorn.run()`, loads models once at startup); the FastAPI
app, routes, and config endpoints live in `src/server.py`. Not a stub — a working
serving layer.
