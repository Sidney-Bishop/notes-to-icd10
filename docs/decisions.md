# Decisions

Append-only. Monotonic IDs (D001, D002, …). Never edit a past entry; supersede it
with a new ID that references it. Shape: Context / Decision / Rationale /
Trade-offs / When to revisit.

> These entries begin on 2026-05-31, the day the documentation system was
> established. Decisions made earlier in the project's life are referenced as
> background within these entries but are NOT logged as if recorded at the time —
> doing so would fabricate a record. See `philosophy.md`.

---

## D001 — Apply the billable filter inside `prepare_splits.py` (filter-then-split)

**Date:** 2026-05-31

**Context.** Reported test-set size disagreed across artifacts: 966 (committed
eval), 972 (a later working-tree eval), and 1,030 (a fresh `prepare_splits.py`
run). Root cause: the billable filter (`code_status == "billable"`) lived only in
`train.py`, not in `prepare_splits.py`, so the test set depended on which stage
filtered. The split script's own self-check validated against the *unfiltered*
population and silently passed at 1,030.

**Decision.** Add a `--code-filter {all,billable}` argument to
`prepare_splits.py`, default `billable`, applied *before* the per-chapter split
(filter-then-split). Fix the self-check to validate against the filtered count.

**Rationale.** Filter-then-split yields an honest stratified 10% of the 9,660
billable records — the regime every reported experiment uses (per `train.py` and
the dataset docs). It keeps the split and training stages in lockstep so the
discrepancy cannot recur.

**Trade-offs.** Not byte-identical to the historical committed run. Verified test
count is **971** (filter-then-split), vs the historical **966**
(split-then-filter). The +5 delta is the cost of a defensible, reproducible
split; the prior regime is preserved at branch `snapshot/2026-05-10-eval`
(commit `9a73669`).

**When to revisit.** If a reviewer requires exact continuity with the historical
966 number, or if the split strategy changes (e.g. different stratification).

---

## D002 — Canonical regime is billable-only, test N = 971

**Date:** 2026-05-31

**Context.** Following D001, the project needs one canonical test set so multi-seed
results are comparable.

**Decision.** The canonical evaluation regime is billable-only (9,660 records),
filter-then-split at seed 42, producing a deterministic test set of **971**
records. All future reported numbers use this regime.

**Rationale.** Verified by running the fixed `prepare_splits.py` on the locked
gold: filter reported `10,240 → 9,660`, totals train/val/test = 7,728/961/971.
Deterministic at seed 42. The 966-vs-971 difference is per-chapter rounding, not
randomness (`int(9660×0.2×0.5)=966` is the un-rounded ideal; 22 per-chapter
roundings sum to 971).

**Trade-offs.** Supersedes the historical 966; documentation that quoted 966 must
be updated to 971. Numbers from a fresh reproduction will not exactly equal the
old single-run figures.

**When to revisit.** If the gold dataset's record count is reconciled (see
`open_questions.md`) in a way that changes the split.

---

## D003 — Establish the `docs/` convention; no retrospective fabrication

**Date:** 2026-05-31

**Context.** The project's prior lack of documented intent forced re-derivation of
decisions from code and artifacts. (This is the proximate reason for the current
refactor.)

**Decision.** Adopt the eight-file documentation convention described in
`philosophy.md` (seven `docs/` files + root README). Append-only logs start now;
no back-dated entries for earlier choices.

**Rationale.** A trustworthy record requires that recorded fact be distinguishable
from reconstruction. Fabricating history to look complete would defeat the
purpose.

**Trade-offs.** Earlier decisions (hierarchical-over-flat, ClinicalBERT-over-
alternatives, the 40-epoch warm start, DVC-over-git-LFS) are real and important
but are referenced as background rather than logged as D-entries, because they
predate this record. They can be promoted to formal D-entries later *if* re-made
or re-affirmed during the refactor.

**When to revisit.** Never rescinds; amend the convention via a new D-ID.
