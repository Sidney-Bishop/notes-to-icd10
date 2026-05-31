# Open Questions

Unresolved tensions carried forward. Monotonic IDs (Q1, Q2, …). Resolve by adding
a `Status: RESOLVED — see Dxxx` line, never by deleting.

---

## Q1 — Dataset record/code count: 9,660/1,926 vs 9,578/1,914?

`03_dataset.qmd` reports both: 9,660 records / 1,926 codes as the headline, and
9,578 records / 1,914 codes as "post-Pydantic-validation." The relationship (the
gatekeeper drops ~82 records / ~12 codes) is implied but not stated cleanly, and
the reported figures should be made internally consistent.

**Decide:** which count is canonical for the paper, and state the gatekeeper delta
explicitly rather than leaving two numbers.

**Status:** OPEN.

---

## Q2 — DVC remote is not portable; fresh clone cannot pull data

The committed DVC default remote is a placeholder (`.dvc/CONFIGURE_LOCAL`); the
real remote is a local-filesystem store recorded only in gitignored
`.dvc/config.local`. A fresh clone therefore cannot reproduce the data without
manual reconfiguration — which contradicts the README's "clone → dvc pull →
byte-identical" reproducibility claim.

**Decide:** stand up a genuinely shareable DVC remote (e.g. backed by the HF
storage already used for the canonical dataset) and push all tracked artifacts,
so a clean clone can pull everything. Then replace the `CONFIGURE_LOCAL`
placeholder with documented setup steps.

**Status:** OPEN. (Found 2026-05-31, see journal.)

---

## Q3 — Orphaned DVC artifact `data/ontology/icd10cm_2026.parquet`

This file is DVC-tracked (pointer committed) but its bytes were never pushed to
the remote, and no code in `scripts/` or `src/` references it. It may be a leftover
from a refactored-out validation step.

**Decide:** either push it to the remote (if it should exist) or `dvc remove` the
pointer (if it's genuinely dead). Don't leave a tracked file that breaks
`dvc pull` and has no consumer.

**Status:** OPEN.

---

## Q4 — Environment drift vs the paper's basis

A fresh `uv sync` resolves to newer libraries (transformers 5.9, torch 2.12;
sklearn historically drifted 1.1.2→1.8.0, scispacy churn noted) than the stack
that produced the documented results. A clean reproduction might land outside the
documented 84–87% range due to environment, not method.

**Decide:** whether to pin `uv.lock` to the paper's versions before the
reproduction run, and gate seed 1 against the documented range (halt + pin if it
misses) rather than silently adopting a regressed number.

**Status:** OPEN.

---

## Q5 — Headline figure for the publication: 85.8% (E-010) vs 86.7% (E-014 hybrid)?

E-010 mean-of-4 (~85.8%) and the E-014 SupCon-Z hybrid (~86.7%) are both quoted as
"best." If 86.7% appears near the headline it cannot be a single run while 85.8%
has error bars — that asymmetry reads as cherry-picking.

**Decide:** either run E-014 across the same seeds so both carry ± std, or demote
86.7% to a clearly-labelled preliminary result and headline 85.8% ± std.

**Status:** OPEN.

---

## Q6 — Synthetic→real generalisation gap

E-010 reports ~85.8% on synthetic MedSynth but ~12% E2E on real MIMIC-IV
discharge summaries (per publication drafts). The gap is attributed to domain
shift, not architecture, but MIMIC-IV validation completeness depends on
PhysioNet access.

**Decide:** how prominently to report the gap and whether any
domain-adaptation/fine-tuning on real data is in scope (per charter, training on
real data is currently out of scope).

**Status:** OPEN.

---

## Q7 — **BLOCKING:** E-003 stage-1 model is unloadable and unbacked

The Stage-1 chapter router (experiment E-003) cannot be loaded as it sits on
disk, which blocks ALL hierarchical evaluation (every hierarchical run needs the
stage-1 router).

**State (verified 2026-05-31):**
- Weights at `stage1/model.safetensors`; config + tokenizer at `stage1/model/` —
  split across two directories, so no single directory is a complete model.
- `_find_model_dir` returns the top-level dir (has weights, no tokenizer) →
  `AutoTokenizer.from_pretrained` fails.
- Weights are gitignored (`.gitignore:78`) and NOT in DVC — they exist only on
  this one disk, backed up nowhere.

**Decided path (see D004):** do not force a load by moving files. **Retrain
stage-1 cleanly**, then back it up (DVC) before relying on it. A freshly retrained
stage-1 with a known-good, single-directory layout is the trustworthy fix;
file-shuffling an unbacked artifact of uncertain provenance is not.

**Also worth fixing** (secondary, not the blocker): `_find_model_dir` accepts a
directory as "the model" on `model.safetensors` alone, without requiring
`config.json` + tokenizer in the same dir. Even after retraining, hardening this
to require a *complete* model directory would prevent a future split-layout from
silently mis-resolving. Track separately if pursued.

**Status:** OPEN — BLOCKING. Blocks the 971-regime evaluation (status #0).
