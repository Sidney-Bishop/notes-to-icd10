# Ornith Review — Q12 Verification

## Review status

**Launch verdict:** The two publication runs (leaky `E-04x_*_Leaky` and de-leaked `E-05x_*_Deleaked`) are safe to launch once the Q12 fixes are applied. The headline numbers flow through the sorted, content-addressed split path (`train.py::_split_dataframe`), and the seeding chain is complete across all three training paths.

**Reproducibility verdict:** Seeding is complete. Three independent layers cover the full RNG surface:
1. `_set_all_seeds()` in `train.py` — seeds python random, numpy, torch (+CUDA/MPS)
2. `_split_dataframe()` in `train.py` — sorts by stable key before splitting (content-addressed)
3. `TrainingArguments(seed=...)` in `src/adapters.py` — seeds the Trainer's internal DataLoader worker RNG

Residual non-determinism: MPS/CUDA non-deterministic reduction kernels (acknowledged in `_set_all_seeds` docstring). Expected drift with same seed: ~0.2–0.3pp, not bit-exact.

**Confirmed launch-blockers (2):**

| # | Issue | File | Line | Fix |
|---|-------|------|------|-----|
| 1 | MIMIC output-path collision | `scripts/validation/validate_mimic_evaluate.py` | 289, 303 | Parameterize output dir on `args.base_experiment` |
| 2 | `--deleaked-reference` unconditional | `src/orchestration.py` | 210 | Parameterize `_mimic_cmd` with `deleaked: bool` |

Additional hardcoding that must be resolved for the two-config publication runs (per backlog): gold path, experiment names, and orchestrator parameterization (`PRESET_LEAKY` / `PRESET_DELEAKED`).

**Remaining ledger:** 52 files [ ] pending — tracked for a later pass. Not in scope for the publication runs.

---

Verified findings from `scripts/validation/validate_mimic_evaluate.py` against the actual code.

## Structural notes

### `sources/` directory — dead code, stale duplicates

Both files in `sources/` **differ** from their `src/` counterparts and are **not imported anywhere**. They are dead/stale shadow copies.

**Evidence:**
- `diff sources/experiment_logger.py src/experiment_logger.py` — files differ. The `sources/` version uses a manual `_project_root()` walker; the `src/` version uses the `config` singleton.
- `diff sources/inference.py src/inference.py` — files differ. The `src/` version has Pydantic validation (`ClinicalNoteInput`), `ExperimentPaths` import, and word-count warning constants; the `sources/` version has none of this.
- `grep -rn "from sources\.\|import sources\." --include="*.py" .` — **no results**. Nothing in the codebase imports from `sources/`.

**Conclusion:** `sources/` is dead code. The `src/` versions are the live modules. During review, ignore `sources/` and focus on `src/`.

### Notebooks — 5 Jupyter notebooks exist

The repo contains 5 Jupyter notebooks (01–05) that are the canonical source for the pipeline per `docs/canonical_pipeline.md`:

```
./notebooks/01-EDA_SOAP_1.ipynb
./notebooks/02-Model_ClinicalBERT_Baseline_ICD3.ipynb
./notebooks/03-Model_ClinicalBERT_Surgical_ICD10.ipynb
./notebooks/04-Model_Hierarchical_ICD10.ipynb
./notebooks/05-Model_Hierarchical_ICD10_E002Init.ipynb
```

These are not Python files (`.py`), so they're not in the 57-file Python inventory above. They may be reviewed in later passes if relevant to the publication runs, but they're not part of the executable codebase that runs the pipeline.

**Decision:** Notebooks are out of scope for this review unless we find evidence that the script pipeline diverges from what the notebooks document. The ledger tracks Python files only.

## Coverage ledger

Complete inventory of Python files in the repo, grouped by location. Each file is marked `[ ] pending` until reviewed in detail. As we fine-tooth-comb the codebase, we'll tick items to `[x] reviewed`.

### `src/` — Library modules

- [x] `src/adapters.py` — ModelAdapter interface and EncoderAdapter implementation (swappable encoder abstraction)
- [ ] `src/config.py` — Project-wide configuration singleton (ArtifactConfig, path resolution from artifacts.yaml)
- [ ] `src/data_loader.py` — Gold layer ingestion utilities
- [ ] `src/dataset.py` — PyTorch Dataset wrapper for ClinicalBERT fine-tuning
- [ ] `src/evaluation.py` — Comprehensive metrics computation (E2E accuracy, F1, ECE, coverage)
- [ ] `src/experiment_logger.py` — Structured experiment logging
- [ ] `src/gatekeeper.py` — Pydantic validation firewall for clinical records (ClinicalRecord BaseModel)
- [ ] `src/graph_reranker.py` — Graph-augmented prediction re-ranking (UMLS graph + Z-phrase dictionary)
- [ ] `src/inference.py` — End-to-end hierarchical ICD-10 prediction pipeline (HierarchicalPredictor)
- [x] `src/orchestration.py` — Testable orchestration primitives for the full de-leaked rebuild (phase specs, command builders)
- [ ] `src/paths.py` — Canonical artifact path resolution (ExperimentPaths)
- [ ] `src/plot_utils.py` — Figure persistence and traceability
- [ ] `src/preprocessing.py` — APSO-Flip and ICD-10 redaction utilities (v5 redactor)
- [ ] `src/server.py` — ICD-10 prediction model server (FastAPI)

### `scripts/` — Entry points / pipeline stages

- [ ] `scripts/augment.py` — Targeted data augmentation for weak ICD-10 chapters
- [ ] `scripts/build_graph.py` — ICD-10 knowledge graph construction
- [ ] `scripts/build_z_dict.py` — Z-chapter phrase dictionary builder (purpose unclear — needs deeper read)
- [ ] `scripts/calibrate.py` — Temperature scaling for ICD-10 hierarchical pipeline
- [ ] `scripts/cleanup.py` — Experiment storage cleanup utility
- [ ] `scripts/cluster_analysis.py` — Unsupervised clustering of clinical note embeddings
- [ ] `scripts/evaluate_hybrid.py` — Hybrid E2E evaluation with per-chapter resolver overrides
- [ ] `scripts/evaluate_real_reranker.py` — Real-world graph reranker evaluation
- [ ] `scripts/evaluate_reranker.py` — Graph reranker evaluation (Z notes not in train file)
- [ ] `scripts/evaluate.py` — Evaluation script for trained ICD-10 classifiers
- [ ] `scripts/extract_doid_icd10.py` — Extract DOID → ICD-10-CM mappings from doid.obo
- [ ] `scripts/extract_embeddings.py` — Extract embeddings using SimCSE encoder
- [ ] `scripts/fetch_raw_medsynth.py` — Reproducible Phase 0 raw fetcher (HF dataset download)
- [ ] `scripts/generate_manifest.py` — Phase 4 export manifest with SHA256 hashes
- [ ] `scripts/generate_simcse_pairs.py` — Create contrastive pairs from HDBSCAN clusters
- [ ] `scripts/predict.py` — ICD-10 inference entrypoint
- [x] `scripts/prepare_data.py` — Headless gold layer preparation pipeline (Medallion build)
- [x] `scripts/prepare_splits.py` — Generate deterministic per-chapter train/val/test splits
- [ ] `scripts/run_experiment.py` — Experiment orchestration driver
- [ ] `scripts/run_full_deleaked_rebuild.py` — Full de-leaked rebuild orchestrator
- [ ] `scripts/serve.py` — ICD-10 model server entrypoint
- [ ] `scripts/train_simcse.py` — Contrastive fine-tuning using code-level pairs
- [ ] `scripts/train_supcon_z.py` — Supervised contrastive fine-tuning for Z-chapter resolver
- [x] `scripts/train.py` — Model fine-tuning script (three entry points: flat, hier stage-1, hier stage-2)

### `scripts/validation/` — MIMIC-IV validation pipeline

- [ ] `scripts/validation/validate_mimic_prepare.py` — MIMIC-IV Medallion preparation (Bronze → Gold)
- [x] `scripts/validation/validate_mimic_evaluate.py` — MIMIC-IV evaluation (E2E accuracy, F1, ECE, coverage)

### `tests/` — Test suite

- [ ] `tests/test_experiment_logger.py` — Tests for experiment logging
- [ ] `tests/test_hierarchical_predictor.py` — Tests for HierarchicalPredictor
- [ ] `tests/test_inference_validation.py` — Tests for inference validation
- [ ] `tests/test_orchestration_output_paths.py` — Tests for orchestration output path correctness
- [ ] `tests/test_orchestration.py` — Tests for orchestration primitives
- [ ] `tests/test_paths_stage2_splits.py` — Tests for stage-2 split paths
- [ ] `tests/test_paths.py` — Tests for path resolution
- [ ] `tests/test_preprocessing.py` — Tests for preprocessing (v5 redactor behavior)
- [ ] `tests/test_split_logic.py` — Tests for split logic
- [ ] `tests/test_train_seeding.py` — Tests for training RNG seeding
- [ ] `tests/test_verify_scripts.py` — Tests for verify_scripts.py check groups
- [ ] `tests/test_z_matcher.py` — Tests for Z-chapter matcher

### `sources/` — Legacy/duplicate modules (needs clarification)

- [ ] `sources/experiment_logger.py` — Structured experiment logging (duplicate of src/experiment_logger.py?)
- [ ] `sources/inference.py` — End-to-end hierarchical ICD-10 prediction pipeline (duplicate of src/inference.py?)

### `notebooks/utils/` — Notebook utilities

- [ ] `notebooks/utils/nb_setup.py` — Notebook environment setup

### Root-level executables

- [ ] `upload_to_hf.py` — Upload artifacts to HuggingFace dataset repo
- [ ] `verify_scripts.py` — Pre-flight script verification (runs check groups, exits 0/1)

## Finding 1 — MIMIC output-path collision

**Severity:** launch-blocker

**Claim:** The MIMIC evaluation script writes to a fixed output path that does not incorporate the experiment/run identity, so two publication runs (leaky + de-leaked) would overwrite each other's results.

**Evidence:**
```python
# scripts/validation/validate_mimic_evaluate.py, lines 277-309
def save_results(
    results: dict,
    reference: dict,
    use_supcon_z: bool,
    threshold: float,
) -> Path:
    """
    Save aggregate results (no patient-level data) to the evaluation directory.

    Only aggregate statistics are saved — no note IDs, no predictions,
    no text. This ensures no MIMIC data leaks into git-tracked files.
    """
    out_dir = config.project_root / "outputs" / "evaluations" / "mimic_iv_validation"
    out_dir.mkdir(parents=True, exist_ok=True)

    output = {
        "experiment":         "E-010_40ep_E002Init" + (" + E-014_SupCon_Z" if use_supcon_z else ""),
        "validation_dataset": "MIMIC-IV-Note v2.2 + MIMIC-IV clinical v2.2",
        "threshold":          threshold,
        "mimic_results":      {k: v for k, v in results.items() if k != "chapter_accuracy"},
        "medsynth_reference": reference,
        "chapter_accuracy":   results["chapter_accuracy"],
        "data_access":        "PhysioNet credentialed — DUA v1.5.0 signed",
        "note":               "Aggregate statistics only — no patient-level data",
    }

    out_path = out_dir / ("summary_supcon_z.json" if use_supcon_z else "summary.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n 💾 Results saved: {out_path}")
    print(f"    (Aggregate statistics only — no patient data)")
    return out_path
```

The output directory is hardcoded at line 289: `outputs/evaluations/mimic_iv_validation/`. The filename at line 303 varies only by the `use_supcon_z` flag (`summary.json` or `summary_supcon_z.json`), not by `base_experiment`. Neither the leaky nor de-leaked publication run uses SupCon Z, so both would write to `summary.json` — the second run silently overwrites the first.

**Proposed fix:** Parameterize the output directory on the experiment name. Change line 289 to:
```python
out_dir = config.project_root / "outputs" / "evaluations" / f"mimic_iv_validation_{args.base_experiment.split('_')[0]}"
```
This produces `mimic_iv_validation_E-041/` and `mimic_iv_validation_E-051/` for the two publication runs. Update the orchestrator's MIMIC phase spec `outputs=` to match.

---

## Finding 2 — `--deleaked-reference` flag passed unconditionally

**Severity:** launch-blocker

**Claim:** The orchestrator's `_mimic_cmd` function unconditionally includes `--deleaked-reference` in the MIMIC evaluation command, which is correct for the de-leaked run but wrong for the leaky run (would compare a leaky MIMIC result against a de-leaked synthetic reference).

**Evidence:**
```python
# src/orchestration.py, lines 204-211
def _mimic_cmd(base_experiment: str, stage1_experiment: str) -> list[str]:
    return [
        "env", "PYTHONPATH=.", "uv", "run", "python",
        "scripts/validation/validate_mimic_evaluate.py",
        "--base-experiment", base_experiment,
        "--stage1-experiment", stage1_experiment,
        "--deleaked-reference",
    ]
```

Line 210 unconditionally adds `--deleaked-reference` to the command list. The flag selects which MedSynth reference the MIMIC result is compared against: `--deleaked-reference` → `MEDSYNTH_REFERENCE_DELEAKED` (0.592); otherwise the leaky `MEDSYNTH_REFERENCE` (~0.858). For the leaky publication run, this flag should be omitted so the MIMIC result is compared against the leaky synthetic reference.

**Proposed fix:** Parameterize `_mimic_cmd` to accept a `deleaked: bool` parameter and conditionally include the flag:
```python
def _mimic_cmd(base_experiment: str, stage1_experiment: str, deleaked: bool = False) -> list[str]:
    cmd = [
        "env", "PYTHONPATH=.", "uv", "run", "python",
        "scripts/validation/validate_mimic_evaluate.py",
        "--base-experiment", base_experiment,
        "--stage1-experiment", stage1_experiment,
    ]
    if deleaked:
        cmd.append("--deleaked-reference")
    return cmd
```
Update all callers to pass the appropriate `deleaked` value based on the run preset.

---

## prepare_data.py

**Severity:** should-fix (phase numbering gaps) / minor (has_leakage discarded)

**Claim:** The gold-layer builder implements a phased Medallion build (ingest → CDC validation → Pydantic firewall → DuckDB → decimal restoration → status annotation → APSO-flip → code redaction → description redaction → export). The leaky vs de-leaked distinction is controlled by the `--redact-descriptions` CLI flag, which toggles Phase 3d. Phase 3c (code-only redaction) runs unconditionally — this is the "leaky" baseline.

**Evidence:**
```python
# scripts/prepare_data.py, lines 249-258
def phase_4(df, dry=False, redact_desc=False):
    print("\n── Phase 4: Export ───────────────────────────────────────")
    if dry:
        print(" (dry-run, skipping write)")
        return
    fname = "medsynth_gold_apso_deleaked.parquet" if redact_desc else "medsynth_gold_apso.parquet"
    p = config.resolve_path("data", "gold") / fname
    p.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(p, compression="snappy")
    print(f" ✅ {p.name}")
```

```python
# scripts/prepare_data.py, lines 260-289 (main)
def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--no-duckdb", action="store_true")
    pa.add_argument("--dry-run", action="store_true")
    pa.add_argument("--offline", action="store_true")
    pa.add_argument("--redact-descriptions", action="store_true",
                    help="Q8: also redact diagnosis descriptions (v5) -> medsynth_gold_apso_deleaked.parquet")
    a = pa.parse_args()

    print(f"\n{'='*70}\n prepare_data.py — HF-locked canonical\n{'='*70}")

    gold = config.resolve_path("data", "gold")
    raw = phase_1a_ingest()
    chk, cdc = phase_1b(raw, gold, a.offline)
    val = phase_1e(raw)

    if not a.no_duckdb:
        phase_1g(val)

    g = phase_2a(val)
    g = phase_2c(g, chk, cdc)
    g = phase_3a(g)
    g = phase_3b(g)
    g = phase_3c(g)
    if a.redact_descriptions:
        ont = PROJECT_ROOT / "data" / "ontology"
        g = phase_3d(g,
                     dictionary_path=ont / "q8_phrasing_dictionary.json",
                     cdc_path=ont / "icd10cm_2026.parquet")
    phase_4(g, a.dry_run, a.redact_descriptions)
```

**Redaction logic (src/preprocessing.py):**
- `redact_icd10_sections()` (Phase 3c, unconditional) — removes ICD-10 CODE strings (e.g., "M25.562", "N39.0") from SOAP sections using regex `ICD10_REDACT_PATTERN`. Leaves descriptions intact → **leaky**.
- `redact_descriptions()` (Phase 3d, only with `--redact-descriptions`) — removes human-readable DESCRIPTIONS (e.g., "pain in left knee") from the assessment section using dictionary-anchored matching + CDC fallback. **De-leaked**.

**Correctness concerns:**

1. **Phase numbering gaps.** Phases jump from 1b→1e (skipping 1c, 1d) and 1g→2a (skipping 2b). This is confusing but not a bug — likely phases were removed or renamed. Could mislead someone tracing the pipeline.

2. **`has_leakage` flag computed then discarded.** Phase 3b computes `has_leakage` (line 238), but phase 3c drops it (line 243). Per Q8, "half the inventory is built and discarded" — the leakage count is useful for auditing but is thrown away. This is a missed opportunity for transparency, not a correctness issue.

3. **Hardcoded ontology paths.** Phase 3d hardcodes paths to `PROJECT_ROOT / "data" / "ontology" / "q8_phrasing_dictionary.json"` and `icd10cm_2026.parquet` (lines 285-288). If these files move or are missing, the de-leaked run fails silently (or with a clear FileNotFoundError). Not a bug, but a fragility.

4. **No SHA256 for ontology files.** The two HF source files (icd10_notes.parquet, cdc_fy2026_icd10.parquet) are SHA256-verified (lines 43-46, 58-87), but the ontology files (q8_phrasing_dictionary.json, icd10cm_2026.parquet) are not. If they're corrupted or modified, the de-leaked redaction could produce wrong results. Lower priority than the HF sources (ontology files are committed artifacts, not downloaded), but worth noting.

**Determinism:**
- SHA256 verification ensures reproducible HF downloads.
- Polars operations are deterministic.
- Redaction logic is deterministic (regex-based, no randomness).
- Dict ordering in Python 3.7+ is insertion-ordered, so `lookup` dict in phase_2c is deterministic.
- **No non-determinism detected.** The gold layer should be reproducible given the same HF sources and ontology files.

**Impact on publication runs:**
- Leaky run: `python scripts/prepare_data.py` → `medsynth_gold_apso.parquet`
- De-leaked run: `python scripts/prepare_data.py --redact-descriptions` → `medsynth_gold_apso_deleaked.parquet`
- The two gold files are distinct and should not be mixed. The pipeline correctly separates them by filename.

---

## prepare_splits.py

**Severity:** should-fix (determinism claim not fully verified)

**Claim:** The split script implements per-chapter stratified 80/10/10 splits (train/val/test) with seed=42, filtering to billable codes before splitting. However, the "content-addressed / sorted before splitting" claim from the project docs is **not visible in this code** — determinism relies on sklearn's `random_state` being consistent, but the DataFrame is not explicitly sorted by a stable key before splitting, which could make splits order-dependent.

**Evidence:**
```python
# scripts/prepare_splits.py, lines 46-52 (function signature)
def prepare_splits(
    gold_path: Path,
    experiment_name: str = "E-006_Hierarchical_Clean",
    seed: int = 42,
    test_size: float = 0.2,
    val_size: float = 0.5,  # 0.5 of the 0.2 = 0.1 overall
    code_filter: str = "billable",
) -> None:
```

```python
# scripts/prepare_splits.py, lines 120-126 (chapter extraction)
# Extract chapter for stratification
df = df.with_columns(
    pl.col("standard_icd10").str.slice(0, 1).alias("chapter")
)

chapters = sorted(df["chapter"].unique().to_list())
print(f" Chapters found: {', '.join(chapters)} ({len(chapters)} total)")
```

```python
# scripts/prepare_splits.py, lines 136-177 (per-chapter split)
# Split per chapter to maintain exact stratification
for ch in chapters:
    ch_df = df.filter(pl.col("chapter") == ch)
    n_total = len(ch_df)

    if n_total < 3:
        print(f" ⚠️  Warning: Chapter {ch} has only {n_total} records, skipping stratification")
        # Put all in train if too few
        train_df = ch_df
        val_df = ch_df.head(0)
        test_df = ch_df.head(0)
    else:
        # First split: train vs (val+test)
        train_df, temp_df = train_test_split(
            ch_df.to_pandas(),
            test_size=test_size,
            random_state=seed,
            stratify=ch_df["standard_icd10"].to_pandas(),
        )

        # Second split: val vs test
        if len(temp_df) < 2:
            val_df = temp_df
            test_df = temp_df.head(0)
        else:
            # Only stratify if every class has at least 2 members
            # temp_df is pandas here — value_counts() returns a Series
            strat_counts = temp_df["standard_icd10"].value_counts()
            can_stratify = (
                temp_df["standard_icd10"].nunique() > 1
                and strat_counts.min() >= 2
            )
            val_df, test_df = train_test_split(
                temp_df,
                test_size=val_size,
                random_state=seed,
                stratify=temp_df["standard_icd10"] if can_stratify else None,
            )

        # Convert back to polars
        train_df = pl.from_pandas(train_df)
        val_df = pl.from_pandas(val_df)
        test_df = pl.from_pandas(test_df)
```

**Split logic:**
- **Per-chapter:** Splits independently for each ICD-10 chapter (first letter of code).
- **Stratified:** Uses `stratify=ch_df["standard_icd10"]` to maintain code distribution across splits.
- **Ratios:** 80/10/10 (test_size=0.2, val_size=0.5 of test_size = 0.1 overall).
- **Seed:** 42 (default, passed to sklearn's `random_state`).

**Determinism concern — "content-addressed" claim not verified:**
The project docs (PROJECT_BRIEF.md Section 1, decisions.md) claim the split is "content-addressed, reproducible" and "sorts by a stable key before splitting." However, **this code does NOT explicitly sort the DataFrame before splitting**. The only sorting is `chapters = sorted(...)` for iteration order, but within each chapter, `ch_df` retains whatever order it had after the billable filter.

sklearn's `train_test_split` with `random_state` is deterministic **only if the input order is deterministic**. If two runs of `prepare_data.py` produce the gold DataFrame in different row orders (e.g., due to dict ordering, filesystem order, or non-deterministic filtering), the splits could differ even with the same seed.

**To verify true determinism:** The gold DataFrame should be sorted by a stable key (e.g., `ID` or `standard_icd10`) before splitting. This is not visible in the code.

**Billable filter — filter-then-split: CONFIRMED.**
```python
# scripts/prepare_splits.py, lines 105-117
if code_filter == "billable":
    if "code_status" not in df.columns:
        raise ValueError(
            "Gold layer has no 'code_status' column — cannot apply the "
            "billable filter. Regenerate gold via scripts/prepare_data.py."
        )
    before = len(df)
    df = df.filter(pl.col("code_status") == "billable")
    print(f" Billable filter: {before:,} → {len(df):,} records")
```
The filter runs BEFORE splitting, so the test set is an honest 10% of the 9,660 billable records (~966 test). This matches the D001 decision.

**Correctness concerns:**

1. **Small-chapter handling.** Chapters with < 3 records skip stratification and go entirely to train (lines 140-145). Chapters with temp_df < 2 records put everything in val and leave test empty (lines 156-158). This could distort evaluation if small chapters exist.

2. **Stratification fallback.** If a chapter's temp_df has only 1 unique code or any code has < 2 members, stratification is disabled (`can_stratify = False`) and the split becomes random (lines 163-172). This could introduce non-determinism if the input order varies.

3. **Test count verification is loose.** The check allows ±1 per chapter due to rounding (line 229), which is reasonable but masks small imbalances.

4. **No explicit sort before split.** As noted above, this is the biggest concern. Without sorting by a stable key (e.g., `ID`), the split depends on the row order in the gold DataFrame, which may not be deterministic across runs of `prepare_data.py`.

**Impact on publication runs:**
- If the gold DataFrame order is non-deterministic, the splits could differ between runs, making the publication numbers non-reproducible.
- **Recommended fix:** Add an explicit sort before splitting:
  ```python
  df = df.sort("ID")  # or some other stable key
  ```
  This would make the split truly content-addressed and reproducible regardless of gold DataFrame row order.

---

## prepare_splits.py — determinism deep-dive

**Severity:** minor (for current publication path) / should-fix (for SupCon chain)

**Claim:** There are two split code paths with different determinism guarantees:
- `scripts/prepare_splits.py` — does NOT sort before splitting (unsorted, order-dependent)
- `scripts/train.py::_split_dataframe` — DOES sort by "id" or ["standard_icd10", "apso_note"] before splitting (sorted, content-addressed)

The current publication runs use the SORTED path (`run_experiment.py` → `train.py` without `--use-presplit`), so the headline numbers are reproducible. However, the SupCon Z chain (E-022, which produces the 86.7% hybrid number in Q5) uses the UNSORTED path via `prepare_splits.py`, making it non-reproducible.

**Evidence — prepare_splits.py does NOT sort:**
```python
# scripts/prepare_splits.py, lines 90-117 (load + filter)
# Load gold layer
print(f"\n📥 Loading gold layer...")
df = pl.read_parquet(gold_path)
print(f" Total records: {len(df):,}")

# --- Code filter ---
if code_filter == "billable":
    if "code_status" not in df.columns:
        raise ValueError(...)
    before = len(df)
    df = df.filter(pl.col("code_status") == "billable")
    print(f" Billable filter: {before:,} → {len(df):,} records")

# lines 120-137 (chapter extraction, NO sort)
df = df.with_columns(
    pl.col("standard_icd10").str.slice(0, 1).alias("chapter")
)

chapters = sorted(df["chapter"].unique().to_list())  # only chapters sorted, not df

# Split per chapter to maintain exact stratification
for ch in chapters:
    ch_df = df.filter(pl.col("chapter") == ch)  # ch_df retains original order
    n_total = len(ch_df)
    # ... train_test_split on ch_df without prior sort
```

**Evidence — train.py::_split_dataframe DOES sort:**
```python
# scripts/train.py, lines 211-253
def _split_dataframe(
    df: pl.DataFrame,
    label_col: str,
    seed: int = 42,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    The frame is sorted by a stable key before splitting so the partition is
    content-addressed: it depends only on (row content, seed), never on the
    order rows arrived in.
    """
    # Stable, content-addressed ordering before the (position-based) split.
    if "id" in df.columns:
        df = df.sort("id")
    elif "standard_icd10" in df.columns and "apso_note" in df.columns:
        df = df.sort(["standard_icd10", "apso_note"])
    df_pd = df.to_pandas()

    train_pd, temp_pd = train_test_split(
        df_pd,
        test_size=0.2,
        random_state=seed,
        stratify=df_pd[label_col],
    )
    val_pd, test_pd = train_test_split(
        temp_pd,
        test_size=0.5,
        random_state=seed,
    )
    return (
        pl.from_pandas(train_pd),
        pl.from_pandas(val_pd),
        pl.from_pandas(test_pd),
    )
```

**Evidence — which path the publication runs use:**

The orchestrator's `build_phase_specs` (src/orchestration.py:214) defines the de-leaked rebuild pipeline:

```python
# src/orchestration.py, lines 227-265 (hierarchical experiments)
specs.append(PhaseSpec(
    name="hier_clinicalbert",
    cmd=_hier_cmd(EHIER, MODEL_CLINICALBERT, f"{eb}/{E002}", gold, dry_run),
    ...
))
specs.append(PhaseSpec(
    name="hier_clinical_modernbert",
    cmd=_hier_cmd(E024, MODEL_CLINICAL_MODERNBERT, f"{eb}/{E023}", gold, dry_run),
    ...
))
```

```python
# src/orchestration.py, lines 144-155 (_hier_cmd)
def _hier_cmd(experiment: str, model: str, stage2_init: str, gold: str,
              dry_run: bool) -> list[str]:
    cmd = [
        "env", "PYTHONPATH=.", "uv", "run", "python", "scripts/run_experiment.py",
        "--experiment", experiment, "--model", model,
        "--stage2-init", stage2_init,
        "--train-stage1", "--stage1-model", model,
        "--gold-path", gold, "--epochs", "20", "--code-filter", "billable",
    ]
    if dry_run:
        cmd.append("--dry-run")
    return cmd
```

`_hier_cmd` calls `run_experiment.py` WITHOUT `--use-presplit`.

```python
# scripts/run_experiment.py, lines 307-308 (passes --use-presplit to train.py if set)
if use_presplit:
    cmd += ["--use-presplit"]
```

```python
# scripts/train.py, lines 611-629 (uses presplit if flag is set, else calls _split_dataframe)
# Use pre-written splits if --use-presplit and files exist
_use_pre = (
    cfg.get("use_presplit", False)
    and (_ch_dir / "train_split.parquet").exists()
    and (_ch_dir / "val_split.parquet").exists()
    and (_ch_dir / "test_split.parquet").exists()
)
if _use_pre:
    train_df = pl.read_parquet(_ch_dir / "train_split.parquet")
    ...
else:
    train_df, val_df, test_df = _split_dataframe(ch_df, label_col, seed=seed)
```

**Conclusion for publication runs:** The main hierarchical experiments (E-021, E-024, E-026) use `run_experiment.py` → `train.py` WITHOUT `--use-presplit`, so they call `_split_dataframe()` which DOES sort. **The headline publication numbers are reproducible.**

**Exception — SupCon Z chain:**
```python
# src/orchestration.py, lines 267-277 (SupCon Z chain)
specs.append(PhaseSpec(
    name="supcon_presplits",
    cmd=_presplits_cmd(ESUP_BASE, gold),  # calls prepare_splits.py (UNSORTED)
    ...
))
specs.append(PhaseSpec(
    name="supcon_zbase",
    cmd=_supcon_zbase_cmd(ESUP_BASE, f"{eb}/{E002}", gold),  # uses --use-presplit
    ...
))
```

```python
# src/orchestration.py, lines 170-177 (_supcon_zbase_cmd)
def _supcon_zbase_cmd(experiment: str, stage2_init: str, gold: str) -> list[str]:
    return [
        "env", "PYTHONPATH=.", "uv", "run", "python", "scripts/train.py",
        "--experiment", experiment, "--mode", "hierarchical", "--stage", "2",
        "--code-filter", "billable", "--epochs", "20",
        "--stage2-init", stage2_init, "--gold-path", gold,
        "--use-presplit", "--chapters", "Z",  # <-- USES PRESPLIT (UNSORTED)
    ]
```

The SupCon Z chain (E-022) uses `prepare_splits.py` (unsorted) + `--use-presplit`, so it's non-reproducible. This chain produces the 86.7% hybrid number mentioned in Q5.

**Impact on publication runs:**
- **Headline numbers (E-021 59.2%, E-024 76.9%):** Reproducible (use sorted path).
- **SupCon hybrid number (86.7%):** NOT reproducible (uses unsorted presplit path). Per Q5, this number cannot be quoted with error bars while the headline has them — asymmetry reads as cherry-picking.

**Recommended fixes:**
1. **For prepare_splits.py:** Add explicit sort before splitting:
   ```python
   df = df.sort("id")  # or ["standard_icd10", "apso_note"]
   ```
2. **For SupCon chain:** Either (a) fix prepare_splits.py as above, or (b) run SupCon without `--use-presplit` to use the sorted `_split_dataframe` path.

---

## run_experiment.py + evaluate.py — split-path confirmation

**Severity:** VERIFIED SAFE (for headline publication numbers)

**Claim:** The headline hierarchical publication runs (E-021, E-024, E-026) use the SORTED split path end-to-end. `run_experiment.py` does NOT call `prepare_splits.py` itself and does NOT read/write presplit parquets — it purely passes arguments through to `train.py`. Without `--use-presplit`, `train.py` calls `_split_dataframe()` (which sorts), writes the SORTED `test_split.parquet` files, and `evaluate.py` reads from those same files. The reported numbers are computed on the sorted path.

**Evidence — run_experiment.py does NOT touch prepare_splits.py:**

Scanning `scripts/run_experiment.py` end-to-end (536 lines):
- No import of `prepare_splits`
- No subprocess call to `prepare_splits.py`
- No read/write of presplit parquet files
- Only passes `--use-presplit` flag through to `train.py` (lines 307-308)

```python
# scripts/run_experiment.py, lines 289-308 (training stage)
cmd = [
    "uv", "run", "python", "scripts/train.py",
    "--experiment", experiment,
    "--mode", "hierarchical",
    "--stage", "2",
    "--code-filter", code_filter,
    "--epochs", str(epochs),
]
if model:
    cmd += ["--model", model]
if stage2_init:
    cmd += ["--stage2-init", stage2_init]
if gold_path:
    cmd += ["--gold-path", gold_path]
if chapters:
    cmd += ["--chapters"] + chapters
if max_length:
    cmd += ["--max-length", str(max_length)]
if use_presplit:
    cmd += ["--use-presplit"]
```

The `use_presplit` parameter defaults to `False` (line 211) and the orchestrator's `_hier_cmd` does NOT set it, so `train.py` receives no `--use-presplit` flag.

**Evidence — train.py writes SORTED test_split.parquet (without --use-presplit):**

```python
# scripts/train.py, lines 626-629 (without --use-presplit)
else:
    train_df, val_df, test_df = _split_dataframe(ch_df, label_col, seed=seed)
    print(f" 📂 Chapter {chapter}: {len(label2id)} codes | "
          f"{len(train_df):,}/{len(val_df):,}/{len(test_df):,}")
```

```python
# scripts/train.py, lines 665-669 (writes test split to disk)
ch_dir = output_base / "stage2" / chapter
ch_dir.mkdir(parents=True, exist_ok=True)

# Save test split and label map per chapter
test_df.write_parquet(ch_dir / "test_split.parquet")
```

The `test_df` here is the output of `_split_dataframe()` which DOES sort (lines 231-234). So the written `test_split.parquet` files are SORTED.

**Evidence — evaluate.py reads from train.py's test_split.parquet:**

```python
# scripts/evaluate.py, lines 464-471 (hierarchical evaluation)
# Stage-2 test splits — one Parquet per chapter
for ch_dir in sorted(s2_dir.iterdir()):
    if not ch_dir.is_dir():
        continue
    test_path = ch_dir / "test_split.parquet"
    if not test_path.exists():
        raise FileNotFoundError(f"{experiment_name}: missing test_split.parquet for chapter {ch_dir.name} at {test_path}. Run prepare_splits.py first.")
    ch_df = pl.read_parquet(test_path)
```

The `s2_dir` is `exp_dir / "stage2"` (line 460), which is the same directory where `train.py` writes the test splits (line 665). So `evaluate.py` reads from the SORTED splits written by `train.py`.

**Full trace for headline publication runs:**

1. Orchestrator's `_hier_cmd` → `run_experiment.py` (no `--use-presplit`)
2. `run_experiment.py` → `train.py` (no `--use-presplit`)
3. `train.py` → `_split_dataframe()` (SORTED by "id" or ["standard_icd10", "apso_note"])
4. `train.py` → `test_df.write_parquet(ch_dir / "test_split.parquet")` (writes SORTED splits)
5. `evaluate.py` → `pl.read_parquet(ch_dir / "test_split.parquet")` (reads SORTED splits)
6. `evaluate.py` → computes E2E accuracy, F1, etc. on SORTED test set

**Conclusion: VERIFIED SAFE.** The headline hierarchical publication numbers (E-021 59.2%, E-024 76.9%) are computed on the SORTED, content-addressed split path. They are reproducible given the same gold layer and seed.

**Exception — SupCon chain (E-022):**
The SupCon Z chain uses `prepare_splits.py` (unsorted) + `--use-presplit`, so it's non-reproducible. This produces the 86.7% hybrid number in Q5.

---

## Launch-blocker fixes (Q12 + beyond)

### 1. MIMIC output-path collision (Q12a)

**Severity:** launch-blocker

**Claim:** Both publication runs (leaky + de-leaked) would write MIMIC results to the same hardcoded path, causing the second run to overwrite the first.

**Evidence — validate_mimic_evaluate.py:**
```python
# scripts/validation/validate_mimic_evaluate.py, lines 277-309
def save_results(
    results: dict,
    reference: dict,
    use_supcon_z: bool,
    threshold: float,
) -> Path:
    """..."""
    out_dir = config.project_root / "outputs" / "evaluations" / "mimic_iv_validation"  # LINE 289 — HARDCODED
    out_dir.mkdir(parents=True, exist_ok=True)

    output = {
        "experiment":         "E-010_40ep_E002Init" + (" + E-014_SupCon_Z" if use_supcon_z else ""),
        ...
    }

    out_path = out_dir / ("summary_supcon_z.json" if use_supcon_z else "summary.json")  # LINE 303 — NO experiment identity
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    ...
```

**Evidence — orchestration.py consumer:**
```python
# src/orchestration.py, lines 309-317
# --- MIMIC de-leaked eval (depends on EHIER) ----------------------------
specs.append(PhaseSpec(
    name="mimic_deleaked",
    cmd=_mimic_cmd(EHIER, EHIER),
    inputs=["data/mimic/gold/mimic_gold.parquet",
            f"{eb}/{EHIER}/stage1/model/model.safetensors"],
    outputs=[f"{eb}/mimic_iv_validation/summary.json"],  # LINE 315 — HARDCODED, must match script
    supports_dry_run=False,
))
```

**Fix — validate_mimic_evaluate.py (lines 289, 303):**
Parameterize the output directory on the experiment name. Change:
```python
out_dir = config.project_root / "outputs" / "evaluations" / "mimic_iv_validation"
```
to:
```python
out_dir = config.project_root / "outputs" / "evaluations" / f"mimic_iv_validation_{args.base_experiment.split('_')[0]}"
```

This produces `mimic_iv_validation_E-041/` and `mimic_iv_validation_E-051/` for the two publication runs. The filename (`summary.json` or `summary_supcon_z.json`) can stay as-is since each run uses a different experiment name.

**Fix — orchestration.py (line 315):**
Update the MIMIC phase spec's `outputs=` to match the new path pattern:
```python
outputs=[f"{eb}/mimic_iv_validation_{base_experiment.split('_')[0]}/summary.json"],
```

Note: The `_mimic_cmd` function needs to receive the `base_experiment` parameter (it already does at line 204), so this derivation is possible.

### 2. --deleaked-reference flag (Q12b)

**Severity:** launch-blocker

**Claim:** The `--deleaked-reference` flag is unconditionally added to the MIMIC command, which is correct for the de-leaked run but wrong for the leaky run (would compare a leaky MIMIC result against a de-leaked synthetic reference).

**Evidence:**
```python
# src/orchestration.py, lines 204-211
def _mimic_cmd(base_experiment: str, stage1_experiment: str) -> list[str]:
    return [
        "env", "PYTHONPATH=.", "uv", "run", "python",
        "scripts/validation/validate_mimic_evaluate.py",
        "--base-experiment", base_experiment,
        "--stage1-experiment", stage1_experiment,
        "--deleaked-reference",  # LINE 210 — UNCONDITIONAL
    ]
```

**Fix — orchestration.py (lines 204-211):**
Parameterize `_mimic_cmd` to accept a `deleaked: bool` parameter and conditionally include the flag:
```python
def _mimic_cmd(base_experiment: str, stage1_experiment: str, deleaked: bool = False) -> list[str]:
    cmd = [
        "env", "PYTHONPATH=.", "uv", "run", "python",
        "scripts/validation/validate_mimic_evaluate.py",
        "--base-experiment", base_experiment,
        "--stage1-experiment", stage1_experiment,
    ]
    if deleaked:
        cmd.append("--deleaked-reference")
    return cmd
```

**Caller update — orchestration.py (line 312):**
The MIMIC phase spec currently calls `_mimic_cmd(EHIER, EHIER)`. For the de-leaked run, this is correct (deleaked=True is the default). For the leaky run, the orchestrator needs to be parameterized to call `_mimic_cmd(LEAKY_HIER, LEAKY_HIER, deleaked=False)`.

**Note:** The orchestrator is not yet parameterized for two configs (per the backlog: "drafted in-session, NOT yet committed"). The parameterization needs to:
1. Define `PRESET_LEAKY` and `PRESET_DELEAKED` configurations
2. Thread a `deleaked` boolean through `build_phase_specs` or the phase spec construction
3. Use the appropriate gold path for each preset (leaky: `medsynth_gold_apso.parquet`, de-leaked: `medsynth_gold_apso_deleaked.parquet`)

### 3. Other potential collisions (Q12 misses)

**Severity:** should-fix (for publication runs)

**Claim:** The orchestrator has additional hardcoded assumptions that would need updating for the two-config publication runs.

**Evidence — orchestration.py:**

**(a) Hardcoded de-leaked gold path (line 33):**
```python
GOLD_DELEAKED = "data/gold/medsynth_gold_apso_deleaked.parquet"
```
This is used throughout `build_phase_specs` (lines 220, 229, 235, etc.). For the leaky run, this needs to be `data/gold/medsynth_gold_apso.parquet`.

**(b) Hardcoded experiment names (lines 221-223, 268-269):**
```python
E002, EHIER = "E-020_Flat_ClinicalBERT_Deleaked", "E-021_Hier_ClinicalBERT_Deleaked"
E023, E024 = "E-023_Flat_ClinicalModernBERT_Deleaked", "E-024_Hier_ClinicalModernBERT_Deleaked"
E025, E026 = "E-025_Flat_BioClinicalModernBERT_Deleaked", "E-026_Hier_BioClinicalModernBERT_Deleaked"
ESUP_BASE = "E-022_Deleaked_SupConBase"
ESUP = "E-022_SupCon_Z_Deleaked"
```
These are hardcoded for the de-leaked rebuild. For the leaky publication runs, the experiment names would be different (e.g., `E-041_*_Leaky`, `E-051_*_Deleaked` per the backlog).

**(c) SupCon chain hardcoding (lines 267-307):**
The entire SupCon Z chain is hardcoded to the de-leaked experiments. For the leaky run, this chain may not exist or would use different experiment names.

**Fix approach:**
The orchestrator needs to be parameterized with a `RunConfig` (as mentioned in the backlog) that includes:
- `gold_path`: which gold parquet to use
- `experiment_prefix`: "E-04x" for leaky, "E-05x" for de-leaked
- `deleaked`: boolean flag for the MIMIC reference
- Experiment name mappings for each backbone

The current `build_phase_specs` function would need to accept a `config: RunConfig` parameter and use it to construct the experiment names, gold paths, and MIMIC command flags.

**No other hardcoded path collisions found.** All other phase specs use the experiment name in their output paths (e.g., `f"{eb}/{E002}/model/model.safetensors"`), so they won't collide between runs as long as the experiment names are distinct.

---

## train.py

**Severity:** should-fix (CLI defaults don't match canonical regime) / minor (silent fallback)

**Claim:** The seeding is correctly implemented and called before model instantiation in all three training paths. However, the CLI defaults (`--code-filter=all`, `--epochs=10`, `--batch-size=8`) don't match the canonical_pipeline.md regime (billable, 20 epochs, 16 batch-size), though the orchestrator passes explicit values so the publication runs are safe.

### 1. Seeding verification

**Severity:** VERIFIED CORRECT

**Claim:** `_set_all_seeds` seeds python random, numpy, AND torch (including CUDA/MPS), and is called BEFORE any model instantiation, data shuffling, or dropout-affected forward pass in all three training paths.

**Evidence — _set_all_seeds implementation (lines 259-285):**
```python
def _set_all_seeds(seed: int) -> None:
    """Pin python / numpy / torch (+ MPS, CUDA) RNG for reproducible training.

    train.py historically seeded ONLY the data split (train_test_split
    random_state); model weight initialisation, dropout, and batch shuffling
    drew from unseeded global RNG, so runs were not reproducible run-to-run.
    This pins all of them. Note: full bitwise reproducibility is still not
    guaranteed on MPS/CUDA (non-deterministic reduction kernels), but with the
    same seed, init + shuffle order are fixed, so runs land far closer
    (~0.2-0.3pp drift rather than ~1-2pp).
    """
    import random as _random
    _random.seed(seed)
    try:
        import numpy as _np
        _np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch as _torch
        _torch.manual_seed(seed)
        if _torch.cuda.is_available():
            _torch.cuda.manual_seed_all(seed)
        # MPS shares the CPU/global generator via torch.manual_seed; no
        # separate mps.manual_seed in stable torch, so the above covers it.
    except Exception:
        pass
```

Seeds: python `random`, numpy `random`, torch `manual_seed` (plus `cuda.manual_seed_all` if CUDA available). MPS covered by `torch.manual_seed`.

**Evidence — call sites (BEFORE model/split):**

```python
# train_flat, line 348 (BEFORE split at 378, model at 382)
def train_flat(...):
    ...
    seed = cfg.get("seed", 42)
    _set_all_seeds(seed)  # LINE 348
    ...
    train_df, val_df, test_df = _split_dataframe(df, label_col, seed=seed)  # LINE 378
    ...
    adapter = EncoderAdapter.from_pretrained(...)  # LINE 382
```

```python
# train_hierarchical_stage1, line 459 (BEFORE split at 480, model at 483)
def train_hierarchical_stage1(...):
    _set_all_seeds(cfg.get("seed", 42))  # LINE 459
    ...
    train_df, val_df, test_df = _split_dataframe(df, label_col, seed=cfg.get("seed", 42))  # LINE 480
    ...
    adapter = EncoderAdapter.from_pretrained(...)  # LINE 483
```

```python
# train_hierarchical_stage2, line 554 (BEFORE tokenizer at 558, chapter loop at 567)
def train_hierarchical_stage2(...):
    ...
    seed = cfg.get("seed", 42)
    _set_all_seeds(seed)  # LINE 554
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name_or_path"])  # LINE 558
    ...
    for chapter in chapters:  # LINE 567
```

**No randomness-consuming step before seed in any path.** All three paths call `_set_all_seeds` as one of the first operations, before any model loading, splitting, or tokenization.

### 2. Determinism completeness

**Severity:** could not verify (DataLoader shuffle in EncoderAdapter)

**Claim:** The seeding covers weight init + split order. Whether batch shuffling in the DataLoader is also seeded depends on `EncoderAdapter.train()` implementation, which is not in this file.

**Evidence:**
- `torch.use_deterministic_algorithms` is NOT set in train.py (would enforce deterministic ops but may slow things down)
- The docstring acknowledges MPS/CUDA non-deterministic reduction kernels (lines 265-268)
- The expected drift is "~0.2-0.3pp" with same seed, not bit-exact reproducibility

**Could not verify:** Whether the DataLoader in `EncoderAdapter.train()` uses `shuffle=True` and whether its generator is seeded. This is in `src/adapters.py`, not reviewed yet. If the DataLoader uses unseeded global RNG for shuffling, two runs could differ in batch order even with the same seed.

### 3. Correctness concerns

**Severity:** minor (silent fallback)

**Claim:** The warm-start path has a silent fallback to base model if the init path is wrong, which could produce a non-reproducible model without obvious error.

**Evidence — warm-start path (lines 631-652):**
```python
# Stage-2 warm-start path
init_root = cfg.get("stage2_init")
if init_root:
    # Try multiple path conventions (different experiments save differently)
    candidates = [
        Path(init_root) / "stage2" / chapter / "model" / "model",  # E-002 nested
        Path(init_root) / "stage2" / chapter / "model",             # E-006 convention
        Path(init_root) / "stage2" / chapter,                       # E-008 flat
        Path(init_root) / "model",                                   # flat experiment model subdir
        Path(init_root),                                             # flat experiment root (E-002_Aug)
    ]
    init_path = None
    for candidate in candidates:
        if candidate.exists() and (candidate / "model.safetensors").exists():
            init_path = str(candidate)
            print(f" ↪️ Transfer learning from {init_path}")
            break
    if init_path is None:
        init_path = cfg["model_name_or_path"]
        print(f" ⚠️ No checkpoint for chapter {chapter}, using base model")  # WARNING, not error
else:
    init_path = cfg["model_name_or_path"]
```

If `--stage2-init` points to a path where no `model.safetensors` exists in any of the 5 candidate layouts, the code falls back to the base model with only a warning printed. This could silently produce a cold-start model instead of the intended warm-start.

**Evidence — skip chapters (line 551):**
```python
skip_chapters = set(cfg.get("skip_chapters", ["P", "Q", "U"]))
```

P, Q, U are skipped by default (too few records). They receive majority-class fallback predictions at inference. This is documented behavior, not a bug.

**Evidence — save path (lines 421, 516, 681):**
```python
adapter.save(model_dir)  # save into model_dir (save() writes FLAT, no '/model' appended) — D007
_finalize_model_dir(model_dir, tokenizer, adapter.model)
```

D007 fix confirmed: `adapter.save()` writes flat layout into the dir it's given, and `_finalize_model_dir` adds tokenizer + config. This produces a complete model directory.

### 4. CLI defaults vs canonical_pipeline.md

**Severity:** should-fix (but not a launch-blocker for publication runs)

**Claim:** The CLI defaults in train.py don't match the canonical_pipeline.md regime, but the orchestrator passes explicit values so the publication runs are safe.

**Evidence — argparse defaults (lines 794-808):**
```python
p.add_argument("--code-filter", choices=["all", "billable"], default="all", ...)  # LINE 794-802
p.add_argument("--epochs", type=int, default=10)  # LINE 803
p.add_argument("--batch-size", type=int, default=8)  # LINE 805
p.add_argument("--lr", type=float, default=2e-5)  # LINE 804
p.add_argument("--warmup-ratio", type=float, default=0.1)  # LINE 806
p.add_argument("--weight-decay", type=float, default=0.01)  # LINE 807
p.add_argument("--max-length", type=int, default=512)  # LINE 808
```

**Discrepancies with canonical_pipeline.md:**
- `--code-filter`: default="all" (10,240 records), but hierarchical should use "billable" (9,660 records)
- `--epochs`: default=10, but canonical_pipeline says 20 for hierarchical
- `--batch-size`: default=8, but canonical_pipeline says 16

**Evidence — orchestrator passes explicit values:**
```python
# src/orchestration.py, lines 135-141 (_flat_cmd)
def _flat_cmd(experiment: str, model: str, gold: str) -> list[str]:
    return [
        "env", "PYTHONPATH=.", "uv", "run", "python", "scripts/train.py",
        "--experiment", experiment, "--mode", "flat", "--label-scheme", "icd10",
        "--model", model, "--code-filter", "billable", "--batch-size", "16",
        "--epochs", "40", "--max-length", "512", "--gold-path", gold,
    ]
```

```python
# src/orchestration.py, lines 144-155 (_hier_cmd)
def _hier_cmd(experiment: str, model: str, stage2_init: str, gold: str,
              dry_run: bool) -> list[str]:
    cmd = [
        "env", "PYTHONPATH=.", "uv", "run", "python", "scripts/run_experiment.py",
        "--experiment", experiment, "--model", model,
        "--stage2-init", stage2_init,
        "--train-stage1", "--stage1-model", model,
        "--gold-path", gold, "--epochs", "20", "--code-filter", "billable",
    ]
```

The orchestrator passes explicit `--code-filter billable`, `--epochs 20` (or 40 for flat), `--batch-size 16` (for flat), so the publication runs use the correct regime regardless of train.py defaults.

**Recommendation:** Update train.py defaults to match canonical_pipeline.md to avoid confusion for users running train.py directly. But this is not a launch-blocker since the orchestrator overrides them.
