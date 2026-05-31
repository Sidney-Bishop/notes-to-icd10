# Decisions Log

Methodology decisions and their evidence. Each entry records *what* was decided,
*why*, the *evidence* it rests on, and any *open items* — so the reasoning
survives independently of the person who made it.

---

## 2026-05-31 — Canonical test split: billable-only, filtered at split time

### Finding

The reported end-to-end test-set size disagreed across the repository's
artifacts, and the disagreement traced to a real pipeline bug, not loose
transcription:

| Source | Test N | E-010 E2E | Provenance |
|---|---|---|---|
| Committed `summary.json` (HEAD, 30 Apr) | 966 | 83.9% | git HEAD at investigation time; historical split-then-filter |
| Working-tree `summary.json` (10 May) | 972 | 85.4% | preserved on branch `snapshot/2026-05-10-eval`, commit `9a73669` |
| `prepare_splits.py` on full gold (31 May) | 1,030 | — | live run during this investigation; unfiltered bug |
| **`prepare_splits.py --code-filter billable` (31 May, this fix)** | **971** | — | **canonical: filter-then-split on 9,660 billable, seed 42** |

**Root cause.** The billable filter (`code_status == "billable"`) was applied
in `scripts/train.py` (`_filter_gold`, ~line 159) but **not** in
`scripts/prepare_splits.py`, which split whatever gold it was handed. So the
test-set size depended on *which stage did the filtering*:

- `prepare_splits.py` on the full 10,240-record gold → 1,030 test
- `train.py --code-filter billable` → filtered to 9,660 → 966 test

The two stages were out of sync. Different historical runs filtered at
different points, producing the 966 / 972 / 1,030 spread.

**Aggravating factor.** `prepare_splits.py`'s self-check computed
`expected_test = int(len(df) * test_size * val_size)` against the *unfiltered*
`len(df)` (10,240 → expected 1,024) and printed "✅ matches" at 1,030. The
guardrail validated the splitter against the wrong population, which is why the
bug went unnoticed.

### Evidence

- `scripts/train.py:154` — comment: `'billable' — 9,660 records (notebooks 02-05 default for hierarchical)`
- `scripts/train.py:159-160` — the filter the split stage was missing
- `scripts/train.py:754` — CLI help: `Use 'billable' (9,660 records) for hierarchical experiments`
- `scripts/build_graph.py:136` — knowledge graph also built on billable-only
- `publications/notes_to_icd10/sections/03_dataset.qmd:28` — "**9,660 records** across **1,926** [codes]"
- `scripts/prepare_splits.py` (pre-fix) — no `code_status` / billable handling anywhere

### Decision

1. **Billable-only is the canonical regime.** Every reported model was trained
   on the 9,660 billable records; the test set is a stratified 10% of those
   (~966). This matches `train.py` and the dataset documentation.

2. **Filter at split time (filter-then-split), not split-then-filter.** The
   billable filter is now applied in `prepare_splits.py` *before* the
   per-chapter stratified split, so the test set is an honest stratified 10% of
   the 9,660 billable records. The alternative (split the full 10,240, then let
   training drop non-billable) reproduces the historical accident but yields a
   test set that is "the billable subset of an all-codes split" — harder to
   defend and not what the documentation claims.

3. **Consequence accepted.** Filter-then-split is not byte-identical to the
   historical committed run. The verified test count is **971** (filter-then-
   split on 9,660 billable, seed 42), versus the historical **966**
   (split-then-filter). This +5-record delta is the supersession made concrete:
   it comes entirely from where the billable filter is applied relative to the
   stratified split. The prior regime is preserved at branch
   `snapshot/2026-05-10-eval` (commit `9a73669`); the new 971 regime supersedes
   it and is what all future reproductions report.

4. **Self-check fixed.** The `expected_test` validation now runs against the
   filtered population, so it can no longer rubber-stamp an unfiltered split.

### Implementation

`scripts/prepare_splits.py`:
- New `--code-filter {all,billable}` argument, **default `billable`**.
- Filter applied immediately after gold load, before chapter extraction.
- Mirrors `train.py:_filter_gold` exactly (same column, same values, same error).
- Guards: raises if `code_status` column is absent, or on an unknown filter value.

Reproduce the canonical split:

```bash
uv run python scripts/prepare_splits.py \
    --experiment E-010_40ep_E002Init \
    --gold-path data/gold/medsynth_gold_apso.parquet \
    --code-filter billable
```

Expected: total test = **971** (verified 31 May 2026). Note this is *not* 966:
`int(9660 × 0.2 × 0.5) = 966` is the un-rounded ideal, but the split is computed
**per chapter** and each chapter's test count is independently rounded, so the
22 per-chapter roundings sum to 971. The split is deterministic at seed 42 —
the same command always yields 971. Verified breakdown:
train / val / test = 7,728 / 961 / 971 = 9,660.

### Open items

- [x] **Confirmed the regenerated test N is 971** (verified 31 May 2026 on
      branch `fix/splits-billable-filter`; filter reported `10,240 → 9,660`,
      total test 971). Replaces the historical 966.
- [ ] **Freeze the split.** The `stage2/{CH}/*_split.parquet` files are
      currently gitignored, so the split is not version-controlled — the deeper
      reason it could drift. Decide whether to commit them or DVC-track them so
      the exact test set is pinned.
- [ ] **Reconcile dataset counts.** `03_dataset.qmd` carries two figures:
      9,660 records / 1,926 codes (reported) vs 9,578 records / 1,914 codes
      (post-Pydantic-validation, line 104). Decide which is canonical and state
      the relationship explicitly rather than leaving both.
- [ ] **Then** run the multi-seed reproduction on the frozen 971 split; numbers
      are only comparable once the split is fixed.

---