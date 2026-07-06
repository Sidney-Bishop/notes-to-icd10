# Notes to ICD-10

Hierarchical ICD-10 coding of synthetic clinical notes with an honest,
verifiable evaluation methodology. This repository accompanies a study whose
central finding is that a **description-level leak** in a widely-used synthetic
benchmark inflated headline accuracy by ~30% and **reversed the ranking of
model backbones** — a cautionary result for how clinical-NLP systems are
evaluated on synthetic data.

[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Model](https://img.shields.io/badge/model-Clinical_ModernBERT-green.svg)](https://huggingface.co/Simonlee711/Clinical_ModernBERT)
[![Dataset](https://img.shields.io/badge/dataset-SidneyBishop%2Fnotes--to--icd10-orange.svg)](https://huggingface.co/datasets/SidneyBishop/notes-to-icd10)
[![DVC](https://img.shields.io/badge/data-DVC-blueviolet.svg)](https://dvc.org)

---

## 🏆 Headline results — the backbone ranking inverted

On the original benchmark (code-only redaction, "leaky"), Bio_ClinicalBERT
appeared to dominate. Removing description leakage — a second redaction pass
that strips the human-readable ICD-10 description strings, not just the codes —
reverses the ranking: **Clinical ModernBERT beats Bio_ClinicalBERT by ~18
points end-to-end**, and it does so on 17 of 19 chapters. Same architecture,
same splits, same hyperparameters — only the redaction level differs.

### De-leaked, hierarchical, 966-record content-addressed test set

| Model | E2E accuracy | Macro F1 | ECE | Coverage@0.7 | Precision on covered |
|---|---|---|---|---|---|
| **Clinical ModernBERT (E-024)** | **76.9%** | **0.672** | **0.039** | **73.7%** | **90.0%** |
| Bio_ClinicalBERT (E-021)         | 59.2%     | 0.477     | 0.088     | 55.3%     | 83.7%     |
| BioClinical ModernBERT (E-026)   | 39.8%     | 0.289     | 0.049     | 18.3%     | —         |

Stage-1 routers are near-identical across backbones (~95%), so the entire
difference is in Stage-2 within-chapter code resolution — exactly where the
leaked description signal operated.

### Leakage inflated and inverted

| Regime | Bio_ClinicalBERT | Clinical ModernBERT | Ranking |
|---|---|---|---|
| Leaky (code-only redaction)          | 0.849 | 0.488 | ClinicalBERT ≫ ModernBERT |
| De-leaked (code + description)       | **0.592** | **0.769** | **ModernBERT ≫ ClinicalBERT** |

The leaked description signal was exploited best by the older model,
manufacturing a backbone conclusion that is the opposite of the truth on clean
data. A previously-reported +21.2 pp SupCon improvement on the Z-chapter
shrinks to +0.5 pp at the system level once de-leaked — the "improvement" was
overwhelmingly artifact.

Full results and the leakage analysis are in
[`publications/notes_to_icd10/`](publications/notes_to_icd10/) (Quarto).

---

## 🎯 What this is

Two-stage hierarchical ICD-10 classifier on **synthetic** MedSynth clinical
notes (MIMIC-IV is used only to quantify the synthetic→real domain gap; never
for training). Predicts billable ICD-10 codes across ~1,926 classes from
APSO-structured notes.

**Architecture:**

- **Stage 1** — 22-way chapter router (Bio_ClinicalBERT or Clinical ModernBERT),
  trained once and reused.
- **Stage 2** — 19 per-chapter resolvers (P/Q/U skipped), **warm-started from a
  flat ICD-10 classifier**. Warm-starting is load-bearing: cold-started
  Stage-2 collapses to ~11% E2E. This lesson survives de-leaking.
- **Calibration** — temperature scaling per model; the τ = 0.7 confidence gate
  drives selective automation. Low-confidence or Z-chapter notes go through a
  graph-affinity reranker.

**Regimes:**

- **Leaky** gold — ICD-10 code strings redacted; descriptions retained. Path
  `data/gold/medsynth_gold_apso.parquet`. This is what the original benchmark
  uses.
- **De-leaked** gold — codes AND descriptions redacted. Path
  `data/gold/medsynth_gold_apso_deleaked.parquet`. This is the corrected
  benchmark. A residual ~18.6% overlap remains (see the paper's dataset
  section), so de-leaked numbers still slightly over-estimate a perfectly
  clean ceiling.

---

## 🚀 Quick start

### Prerequisites

- Python 3.12 (`.python-version` pins it)
- [uv](https://github.com/astral-sh/uv) for dependency management
- Apple Silicon with MPS, or a CUDA GPU (MPS is the actively-exercised path)
- ~20 GB disk for a single full run; ~140 GB if you plan to run both leaky
  and de-leaked rebuilds side-by-side

### Install

```bash
git clone https://github.com/Sidney-Bishop/notes-to-icd10.git
cd notes-to-icd10
uv sync
```

### Get the data

```bash
# Option A — pull the locked artefacts via DVC (recommended for reproduction)
dvc pull

# Option B — rebuild from Hugging Face (verifies the SHA256 chain from source)
uv run python scripts/prepare_data.py                          # leaky gold
uv run python scripts/prepare_data.py --redact-descriptions    # de-leaked gold
```

Both golds are needed if you want to run both regimes. The dataset is
[`SidneyBishop/notes-to-icd10`](https://huggingface.co/datasets/SidneyBishop/notes-to-icd10)
(MedSynth + CDC FY2026), SHA256-verified against pinned hashes.

### Verify the environment

```bash
uv run python verify_scripts.py
```

Runs 11 integrity checks (data provenance, DVC pointers, ExperimentLogger
wiring, path resolution, etc.). Must print **`✅ All checks passed — safe to
run training`** before any real training run. The orchestrator (below) runs it
as a mandatory pre-launch gate.

---

## 🏃 Running the pipeline

There are three ways to run this repository, in increasing complexity. Pick
the one that matches what you're trying to do.

### 1. Reproduce the paper — full rebuild orchestrator (recommended)

The full paper pipeline (flat pre-train × 3 backbones → hierarchical routers +
resolvers × 3 backbones → SupCon Z chain → MIMIC-IV validation) runs as a
single tested orchestrator. It supports two presets — one per redaction
regime — that differ **only** in the gold file and the experiment-name series;
every hyperparameter is identical, so the two runs are a clean A/B on
leakage.

**Always dry-run first** (validates the whole chain without spending GPU
time):

```bash
uv run python scripts/run_full_deleaked_rebuild.py --preset deleaked --dry-run
uv run python scripts/run_full_deleaked_rebuild.py --preset leaky    --dry-run
```

Each dry-run should end with `RUN COMPLETE: 12 OK / 0 FAILED / 0 SKIPPED`.

Then run for real (each takes ~16 hours on an M-series Mac):

```bash
uv run python scripts/run_full_deleaked_rebuild.py --preset leaky      # E-04x_Leaky series
uv run python scripts/run_full_deleaked_rebuild.py --preset deleaked   # E-05x_Deleaked series
```

**What happens:**

- A rundir is created at `~/full_rebuild_runs/<timestamp>/` for logs.
- `verify_scripts.py` runs first as a hard gate; if it fails, the rebuild
  aborts before any training.
- Twelve phases execute in order, each streaming stdout/stderr to both the
  console and a per-phase logfile.
- Postflight checks each phase's expected outputs exist before the next
  phase starts.
- Experiment artefacts land under `outputs/evaluations/<experiment>/`.

**Why two presets, not one:** the two runs are the paper's A/B on redaction
level. They use fresh, parallel experiment-name series (`E-04x_*_Leaky` and
`E-05x_*_Deleaked`) so neither overwrites the other on disk, and every
non-gold/non-name hyperparameter is identical. A regression test
(`tests/test_orchestration_output_paths.py::TestTwoPresetsLaunchSafety`)
guards this invariant — if any hyperparameter ever drifts between presets,
the test fails.

### 2. Reproduce a single experiment — individual scripts

To reproduce one experiment (e.g. E-002 flat pre-train, or Stage-2 alone),
run the scripts directly. This is the manual path the orchestrator drives
under the hood, useful for debugging one stage or trying a variant.

The full step-by-step guide with verification commands, expected runtimes,
and the individual `train.py` / `calibrate.py` / `evaluate.py` invocations
is in **[`Run_notes.md`](Run_notes.md)** (single source of truth for
individual-script reproduction).

Read `docs/canonical_pipeline.md` for the definitive source-verified pipeline
(experiment names, init sources, hyperparameters, gotchas). Its §4 "Gotchas"
each cost real debugging time — worth reading before running anything by
hand.

### 3. Explore in notebooks

Notebooks 01-05 are the canonical source for the pipeline recipe (per
`docs/canonical_pipeline.md`) and are useful for interactive work:

```bash
uv run jupyter notebook
```

| Notebook | Experiment | Runtime |
|---|---|---|
| `01_prepare_data.ipynb` | Data prep | ~5 min |
| `02_flat_encoder.ipynb` | E-002 flat ICD-10 | ~4 hr |
| `03_stage1_router.ipynb` | Stage-1 chapter router | ~25 min |
| `04_stage2_resolvers.ipynb` | Stage-2 warm-started resolvers | ~100 min |
| `05_calibrate_evaluate.ipynb` | Calibration + eval | ~10 min |

Notebooks and scripts converge on the same models; the scripts are the
headless equivalents used by the orchestrator.

---

## 🔒 Reproducibility

Every ingredient of a training run is pinned:

- **Data** — HF dataset SHA256-verified at download; DVC-locked gold parquets.
- **Split** — content-addressed (dataframe sorted by a stable key before
  splitting), seed 42; every model sees the same 966-record test set, so
  cross-model comparisons are exact.
- **RNG** — `_set_all_seeds()` seeds Python, NumPy, and PyTorch (+ CUDA/MPS)
  at the start of every training entry point (`train_flat`,
  `train_hierarchical_stage1`, `train_hierarchical_stage2`), before any model
  instantiation or DataLoader shuffle. Reproducible up to GPU/MPS
  non-deterministic reduction kernels (~0.2-0.3 pp residual).
- **Environment** — `uv.lock` pins every dependency; `.python-version` pins
  the interpreter.
- **Gate** — `verify_scripts.py` runs 11 integrity checks before any full
  rebuild.

Reported figures are **single training runs on a fixed content-addressed
split**, not multi-seed means. The seeded pipeline makes a multi-seed study
straightforward as future work; the current single-run figures land close
enough that the ~18-point backbone inversion is well outside plausible
run-to-run variance. See `publications/notes_to_icd10/sections/08_limitations.qmd`
for the full disclosure.

---

## 🗂 Repository layout

```
Notes_to_ICD10_prj/
├── README.md                          ← you are here
├── Run_notes.md                       ← detailed step-by-step run guide
├── verify_scripts.py                  ← mandatory pre-launch integrity gate
├── src/                               ← library modules
│   ├── adapters.py                    (encoder abstraction; Trainer-based training)
│   ├── config.py                      (path resolution via artifacts.yaml)
│   ├── orchestration.py               (phase specs, RunConfig presets)
│   ├── inference.py                   (HierarchicalPredictor — deployment path)
│   ├── graph_reranker.py              (UMLS graph reranker for low-confidence)
│   └── ...
├── scripts/                           ← entry points
│   ├── prepare_data.py                (build leaky and de-leaked golds)
│   ├── prepare_splits.py              (per-chapter splits — SupCon path only)
│   ├── train.py                       (flat + Stage-1 + Stage-2 training)
│   ├── calibrate.py                   (temperature scaling)
│   ├── evaluate.py                    (E2E eval on the sorted test split)
│   ├── run_full_deleaked_rebuild.py   (the two-preset orchestrator CLI)
│   └── validation/
│       └── validate_mimic_evaluate.py (MIMIC-IV real-world eval)
├── tests/                             ← 185 tests (pytest)
├── docs/                              ← project documentation
│   ├── PROJECT_BRIEF.md               (fresh-Claude orientation doc)
│   ├── charter.md, philosophy.md      (intent + doc conventions)
│   ├── architecture.md                (system architecture)
│   ├── canonical_pipeline.md          (definitive pipeline recipe + gotchas)
│   ├── decisions.md                   (D001-D017, append-only)
│   ├── journal.md, open_questions.md, backlog.md
│   └── model_drift_mitigation.md      (how PROJECT_BRIEF is maintained)
├── notebooks/                         ← notebooks 01-05 (canonical recipe)
├── publications/notes_to_icd10/       ← the Quarto paper
├── data/                              ← DVC-tracked; gold parquets + MANIFESTs
└── outputs/                           ← experiment artefacts (not committed)
```

---

## 🧪 Tests

```bash
uv run pytest -q
```

185 tests, ~5 seconds. Covers split determinism, orchestrator phase specs,
output-path collisions, per-preset A/B invariants, training seeding, path
resolution, and more.

---

## 🔬 Methodology highlights

- **Zero-trust data ingestion** — every source (HF dataset, CDC codes) is
  SHA256-verified against pinned hashes at download. A `Gatekeeper` Pydantic
  firewall discards records that fail schema/billability checks.
- **APSO-Flip preprocessing** — clinical notes are reordered
  Assessment-first so the diagnostic content survives the 512-token
  transformer window. Applied identically at train and inference.
- **Description-level redaction** — the v5 deterministic redactor removes
  ICD-10 code description phrases (not just the codes) from the assessment
  text, producing the de-leaked gold. The residual ~18.6% partial-overlap
  is documented, not glossed over.
- **Content-addressed splits** — the train/val/test partition is a
  deterministic function of `(row content, seed)`, so every model is
  evaluated on the same test records. This is what makes cross-backbone
  comparison exact.
- **MIMIC-IV validation** — 4,877 real discharge summaries used only to
  quantify domain shift. Both leaky (~12%) and de-leaked (~8%) models
  collapse on real notes; the synthetic→real gap is domain-shift dominated,
  independent of leakage.

---

## ⚠️ Limitations

- Synthetic training data. Real-world deployment collapses to near-floor
  performance on MIMIC-IV — domain adaptation is required.
- ~18.6% residual description overlap remains in the de-leaked gold.
  De-leaked numbers slightly over-estimate a perfectly clean ceiling.
- Single-label prediction (top-1 code). Real clinical notes routinely carry
  multiple diagnoses; multi-label prediction is future work.
- Single training runs, not multi-seed means. The seeding infrastructure
  makes multi-seed feasible; not done for this paper.
- Clinical ModernBERT is used at `max_length=512` for a fair A/B against
  ClinicalBERT. Its native context (up to 8,192 tokens) is not exercised,
  though MedSynth notes rarely exceed 512 tokens anyway.

Full limitations in `publications/notes_to_icd10/sections/08_limitations.qmd`.

---

## 📦 Dependencies

Managed via `uv` and pinned in `uv.lock`. Key packages:

- `transformers`, `torch`, `datasets` — model + training stack
- `polars`, `duckdb` — data processing (silver → gold)
- `mlflow` — experiment tracking (via `src/experiment_logger.py`)
- `pydantic` — schema validation
- `dvc` — data version control
- `pytest` — test harness

---

## 📄 Citation

```bibtex
@misc{roche2026notes_to_icd10,
  title     = {Description-level leakage in synthetic clinical benchmarks:
               a cautionary tale from ICD-10 coding},
  author    = {Roche, Jason},
  year      = {2026},
  note      = {In preparation},
  url       = {https://github.com/Sidney-Bishop/notes-to-icd10}
}
```

The dataset:

```bibtex
@misc{roche2026notes_to_icd10_dataset,
  title  = {notes-to-icd10: MedSynth + CDC FY2026 for ICD-10 coding research},
  author = {Roche, Jason},
  year   = {2026},
  url    = {https://huggingface.co/datasets/SidneyBishop/notes-to-icd10}
}
```

---

## 💬 Issues & suggestions

Open an issue on
[GitHub](https://github.com/Sidney-Bishop/notes-to-icd10/issues) — bug
reports, questions about the leakage methodology, or replication issues are
all welcome.

---

## 📝 License

MIT — see [LICENSE](LICENSE).
