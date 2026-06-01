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

## Q2 — DVC remote plumbing is broken (but the raw data IS reproducible from HF)

**Refined 2026-05-31.** Earlier framing ("fresh clone cannot reproduce the data")
was too broad. The accurate picture:

- The **raw canonical inputs ARE published and reproducible**: the HF dataset
  `SidneyBishop/notes-to-icd10` holds `data/medsynth/icd10_notes.parquet` (19.8 MB)
  and `data/reference/cdc_fy2026_icd10.parquet` (1.4 MB). Both verified locally
  against `prepare_data.py`'s pinned SHA256s — byte-for-byte canonical.
- Gold is NOT published (by design — D006); it's regenerated from the raw.
- The actual gap is the **DVC remote configuration**: the committed default remote
  is the placeholder `.dvc/CONFIGURE_LOCAL`; the real remote lives in gitignored
  `.dvc/config.local`, which a fresh clone doesn't get. So `dvc pull` fails on a
  clean clone until manually reconfigured — but `prepare_data.py` falls back to
  `hf_hub_download`, so the data is still obtainable without DVC.

**Decide:** stand up a portable DVC remote (or document the HF-direct path as the
primary route, since `prepare_data.py` already supports it) and replace the
`CONFIGURE_LOCAL` placeholder with real setup steps, so a clean clone reproduces
without manual DVC surgery.

**Status:** OPEN — but narrower than first thought. Data reproducibility is intact
via HF + `prepare_data.py`; only the DVC convenience layer needs fixing.

---

## Q3 — `data/ontology/icd10cm_2026.parquet` missing from DVC remote; no confirmed consumer

`dvc status` reports this file "not in cache" — DVC-tracked (pointer committed)
but bytes never pushed to the remote. It still exists in the original working dir.

**Verified 2026-05-31 (do not re-confuse with the file below):**
- The reference file the pipeline actually READS for ICD-10/CDC validation is
  **`cdc_fy2026_icd10.parquet`** (loaded in EDA notebook Phase 1b:
  `config.resolve_path("data","gold") / "cdc_fy2026_icd10.parquet"`). That file is
  DVC-tracked, present, and pulls fine — no problem.
- **`icd10cm_2026.parquet`** (the missing one) is NOT referenced by name in the EDA
  notebook or in any script under `scripts/` or `src/` (grep verified). It has no
  confirmed code consumer in what has been searched. (Notebooks 02–05 not yet
  searched — see caveat.)
- The two files have near-identical names and were conflated earlier in this
  session; they are different artifacts. `cdc_fy2026_icd10` = the live reference;
  `icd10cm_2026` = the missing, unreferenced one.

**Decide:** confirm whether `icd10cm_2026.parquet` is read by notebooks 02–05 or
any path not yet searched. If a real consumer exists → push it to the DVC remote.
If genuinely unreferenced → `dvc remove` the pointer so it stops breaking
`dvc pull`. Do not assume it is dead until 02–05 are checked.

**Status:** OPEN. (Mislabelled "orphan", then over-corrected to "is read"; this is
the verified middle: no consumer found in searched files, unconfirmed in 02–05.)

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

## Q7 — RESOLVED (2026-05-31, via D007): stage-1 model now loadable

**Status: RESOLVED — see D007.**

Root cause was `train.py` calling `adapter.save(<parent>)` in all three training
paths, splitting weights from config/tokenizer. Fixed (D007) to
`adapter.save(model_dir)`. Verified by gate run (stage-1 only, ~31 min):

- On disk: NO `model.safetensors` stranded at `stage1/` top level; `stage1/model/`
  contains config.json + tokenizer.json + tokenizer_config.json + model.safetensors
  + label_map.json — one complete model directory.
- Load test: `AutoTokenizer.from_pretrained` + `AutoModelForSequenceClassification
  .from_pretrained` on `stage1/model/` both succeed, 22 labels. This is the exact
  operation that previously threw "Couldn't instantiate the backend tokenizer."

Stage-1 router trained healthily from base Bio_ClinicalBERT: best epoch 3,
val_acc 0.920, macro_f1 0.933 (final epoch 6: val_acc 0.935). Note: trained from
base, not E-001-initialised, so routing is slightly below the historical ~96%
(expected; E-001 init can be added for the full run if desired).

---

## Q8 — Implement semantic-label redaction (the unimplemented cell-52 proposal)

The EDA notebook advocates redacting semantic diagnosis labels from the note text
but never implements it (see D005). The current canonical gold retains labels like
"pain in left knee" in the model input — residual label leakage that inflates
accuracy and would not exist in real clinical notes (a likely contributor to the
synthetic→real gap, Q6).

**Decide / do:** implement semantic-label redaction in `preprocessing.py` (strip
the diagnosis description and the `ICD-10:` / `Description:` / `Diagnosis:`
scaffolding, and ideally the `[REDACTED]` markers), regenerate gold, re-run the
full pipeline, and report the accuracy **delta** vs the code-only regime (D005).
That delta quantifies how much retained labels were inflating results.

**Status:** OPEN. Gates the first *publishable* number (the D005 retrain is only a
provisional reproducibility check).

## Q9 — graph reranker uses sklearn artifacts pickled under old version (2026-06-01)

During the E-010 hierarchical evaluate, the graph reranker loaded TfidfVectorizer
and TfidfTransformer pickled under sklearn 1.1.2, now running under 1.8.0
("InconsistentVersionWarning ... use at your own risk"). The run completed and
produced E2E 0.838, but the reranker's contribution may be subtly affected by the
version mismatch. This is a concrete instance of the env drift flagged in Q4.

**To resolve:** either re-fit the TF-IDF artifacts under the pinned sklearn, or pin
sklearn to a version compatible with the stored pickles, then re-run evaluate and
confirm the number is stable. Cheap to check; matters for a publishable number.

**Status:** VERIFIED BENIGN (2026-06-01). scispacy 0.5.5 pins no sklearn version; a --dry-run behavioural check showed the UMLS linker produces correct concepts under sklearn 1.8.0 (M25.562 → C0030193 "Pain" @0.97). The InconsistentVersionWarning is precautionary/cosmetic, not breakage. Optional future tidy: pin sklearn to silence the warning, but no correctness issue. Downgraded from low-medium to cosmetic.
