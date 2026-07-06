# Model Drift Mitigation: Persistent On-Disk Project Brief

**Audience:** A Claude instance (or other long-context LLM) that has been working on this project for an extended period and is approaching or has crossed context compaction boundaries.
**Purpose:** Implement a persistent orientation document on disk that survives compaction by not living in any LLM's context. This is the same solution applied to `docs/PROJECT_BRIEF.md` in this project, and to other projects (tst_llm, Cedant) before it.
**Effort:** ~1-2 hours, split across foundation-doc inventory and synthesis.
**Output:** A committed `docs/PROJECT_BRIEF.md` at ~500-700 lines, depending on how much in-flight state there is to capture.

---

## 1. Why this matters (the problem)

If you are reading this, you are likely the assistant the user is asking to implement or refresh this solution. Before you do, understand the problem you are solving.

Long conversations between an LLM and a user accumulate context that the LLM uses for orientation: which files exist where, which decisions were made and why, what conventions the project follows, what's currently in flight, what's been tried and ruled out. For this project specifically, that means: the two-gold leakage story, the D014-D017 de-leaked findings, the canonical-pipeline gotchas (the `--stage2-init` root-vs-model trap, the `E-003` router ownership), the two-publication-runs plan, and the user's working discipline. This context is what lets the LLM behave like a continuing collaborator rather than a confused stranger.

**The failure mode:** when the conversation exceeds the LLM's context window, the platform summarises earlier portions into a compacted form. The compaction captures roughly what happened, but the specific texture of recent decisions, the exact state of in-flight work, and the user's conventions all drift in ways neither the LLM nor the user notices immediately. Symptoms seen in this project's kind of work:

- The LLM re-investigates the pipeline structure for work it planned earlier the same session.
- The LLM re-asks which gold file is which, or which experiment name is canonical (E-009 vs the deprecated E-010).
- The LLM forgets discipline rules (explicit-file commits, no terminal-flooding, one-step-at-a-time) and reintroduces patterns the project has explicitly moved away from.
- The LLM treats settled results (the de-leaked numbers, the backbone inversion) as more uncertain than they are, and re-derives them.

The user observes this as the LLM getting visibly worse over a long conversation, or losing important context between sessions.

**The root cause:** the project's documentation is fine. What's unreliable is the LLM's *working memory of where the documentation lives and what's in it*. Documentation an LLM has to rediscover from scratch each conversation costs context tokens to use.

**The fix:** a persistent on-disk **project brief** — a single canonical orientation document, committed at `docs/PROJECT_BRIEF.md`, that any LLM instance can read at the start of a session to reconstruct the project's mental model in ~5-8 minutes without any conversation history.

The brief is not a replacement for the foundation docs (charter, philosophy, architecture, status, canonical_pipeline, decisions, journal, open_questions, backlog). It is the **map** that points to the **territory**, with a reading order, working conventions, and load-bearing constraints called out explicitly.

---

## 2. The solution (concept)

A project brief is a single Markdown file with these properties:

1. **It lives in the repository, not in any LLM's context.** Committed to disk, version-controlled, survives compaction by definition.
2. **It is the single document a fresh LLM should read first.** Other documents are pointed to from it, in a reading order.
3. **It is honest about what it does not capture.** A "What this does NOT capture" section (Section 9 in the template) enumerates inferred-but-not-verified claims and points to foundation docs as authoritative.
4. **It has an append-only Revision History at the bottom.** Each new version records what changed and why.
5. **It is readable in ~5-8 minutes for orientation**, while supporting deeper reading for specific tasks via the Documentation Map.

The brief mirrors this project's own **state-vs-history discipline** (`philosophy.md`): state sections (0-10) get overwritten when reality changes; the Revision History (final section) is append-only. The brief is itself a state+history hybrid — which is exactly why it fits a project that already thinks in those terms.

---

## 3. When to write or refresh a project brief

**Strong signals that the time is now:**

- The user has noticed you forgetting things between sessions.
- The conversation summary in your current context is several days old and its texture is thin.
- The user is repeatedly re-explaining project conventions or decisions you should already know.
- You catch yourself re-asking which gold is which, or re-deriving a settled de-leaked number, or re-reading the pipeline you ran yesterday.
- You are approaching context compaction (you may notice earlier turns disappearing from working memory).

**Weak signals (consider but not urgent):**

- A long session has been productive but feels like it's accumulating cognitive load.
- The project has recently added significant complexity (new decisions, a new experiment series, a new in-flight plan) that a fresh instance wouldn't infer from the code alone.

**Counter-signals (probably don't need a brief):**

- The change is small enough that the existing `status.md` + recent journal entries already cover it.

**The honest threshold:** if you have answered the same project-context question more than once across sessions, or the user has expressed frustration at re-explaining, write or refresh the brief.

---

## 4. Process (in order)

### Step 1 — Inventory the documentation surface [10-15 minutes]

Know what you are working with before planning. For this project:

```bash
ls -la docs/
wc -l docs/*.md
ls -la *.md
grep -E "^## D[0-9]" docs/decisions.md      # decision headers (D001-D0NN)
grep -E "^## Q[0-9]" docs/open_questions.md  # question headers (Q1-QNN)
head -40 docs/charter.md                     # intent + non-negotiables
cat docs/philosophy.md                        # the docs conventions (read in full)
cat docs/status.md                            # current work state
```

**Output:** a list of every documentation file with line counts and structural shape (headings, D-IDs, Q-IDs). You should be able to answer "what files exist and what is each for?" from memory afterwards.

### Step 2 — Tier the documents by reading priority [5 minutes]

- **Foundation files (read in full):** charter, philosophy, architecture, status, canonical_pipeline. Small (< ~250 lines each), stable, concept-defining.
- **Long history files (read structurally):** decisions.md (834 lines), journal.md (747 lines) — read via heading grep + the most recent entries. For this project, **read D014-D017 in full** — they carry the de-leaked findings that are the current headline.
- **Reference/fluid files:** backlog, open_questions — read structure + open items.

### Step 3 — Read the foundation files in full [20-40 minutes]

If you have a local LLM available (e.g. `claude -p` against a local endpoint), delegate the reading. Otherwise read them yourself, in full if < ~300 lines, else structure + selected sections. **Output:** foundation-doc content in context, plus extracted structure from the large history docs.

### Step 4 — Draft v0.1 from current context [30-45 minutes]

Ship a v0.1 explicitly labelled with its gaps rather than waiting for perfection. Use the template in Section 5. Fill each section from: foundation docs (step 3), conversation history (in-flight context), the file inventory (step 1), and conventions you've observed but not seen written down.

**Mark inferred claims explicitly.** For this project the highest-risk inferences are: anything about the two publication runs (they may not have happened yet), the exact state of uncommitted work, and any number not traceable to a decision or a `summary.json`. When in doubt, write "unconfirmed: …" — the project's own `philosophy.md` demands this.

### Step 5 — Cross-check v0.1 against sources, write v1.0 [30-60 minutes]

Treat every inferred claim with suspicion. Re-read the foundation files and verify each claim. In this project the classic drift traps are:

- **Stale state docs.** `status.md` and `canonical_pipeline.md` are dated 2026-05-31/06-01 and predate the de-leaking; `decisions.md` runs to D017. If your brief's current-state section quotes status.md, it is stale — use the decisions log.
- **Naming drift.** E-010 is deprecated naming; E-009 is canonical for the best original hierarchical model. The de-leaked rebuild used the E-02x series.
- **Number provenance.** 0.849 is the leaky provisional number; 0.592 / 0.769 are the de-leaked headline. Never present a leaky number as de-leaked or vice-versa.

Write a v1.0 that tightens or corrects each inferred claim. v1.0 is the version that gets committed.

### Step 6 — Add Revision History and Gaps sections [10 minutes]

The Revision History (Section 11) is append-only — never edit prior entries, only add. Each entry records: version + date, author attribution, what changed and why, verification status (which claims verified against sources, which inferred), and cross-references to commits. The Gaps section (Section 9) records what the brief does not claim to capture authoritatively.

### Step 7 — Commit it to the repository [5 minutes]

The entire value depends on the brief surviving compaction by living on disk. Commit to `docs/PROJECT_BRIEF.md` (mirrors the existing `docs/` convention). Commit message should be substantive — describe what the brief is for, what it covers, and the method. Per this project's discipline: **commit the file by explicit name** (no `git add .`), and update docs in the same spirit the code changes are committed.

---

## 5. Template structure

Adapt section names to the project's vocabulary, but preserve the shape.

```
# <Project Name> — Project Brief

**Kind:** Persistent orientation document for fresh Claude instances.
**Update style:** Append-only Revision History at the bottom. State sections overwritten when materially changed.
**Read this if:** You are a Claude instance picking up this project cold.

## 0. If you are a fresh Claude reading this
   - Reading order (this brief first, then charter → philosophy → decisions → recent journal)
   - A staleness note if state docs lag the decisions log
   - Estimated time to functional context

## 1. Project at a glance
   - Compact table: what / hardware / stack / architecture / data / stage / headline finding / conventions / current work state

## 2. Detailed overview
   - What this project is (2-3 paragraphs) / what it explicitly is not / the shape of the work

## 3. Core architecture or organising principle
   - The project's most important structural finding (here: two-stage hierarchical + warm-started Stage-2; and the de-leaked backbone inversion)

## 4. Documentation map
   - Every doc with lines / purpose / when-to-read, split by state vs history vs fluid
   - A "what to read for which task" matrix

## 5. Architecture (essentials)
   - Data layer, model/pipeline layer, orchestration — essentials only; point to architecture.md + canonical_pipeline.md

## 6. Current state (with date)
   - What's settled (the de-leaked findings, D014-D017) and what's in flight (the two publication runs)
   - Blocking issues called out
   - Top priorities

## 7. Important constraints (load-bearing — don't break by accident)
   - The leakage story, the pipeline gotchas, reproducibility/seeding, naming, zsh/tooling traps

## 8. Working conventions
   - The docs discipline (state-vs-history), and how the user actually works (one-step-at-a-time, explicit-file commits, no terminal-flooding)

## 9. What this document does NOT capture
   - Honest limitations; inferred-not-verified claims; pointers to authoritative sources

## 10. Maintenance instructions
   - When to refresh (D018+, runs complete, structure changes); when not to; how

## 11. Revision history
   - Append-only; v1.0 — date — author — method — verification status
```

---

## 6. Things that are easy to get wrong

### Over-claiming completeness

The temptation is to make the brief feel authoritative by minimising the Gaps section. Resist it. `philosophy.md` is explicit: "a fabricated decision or a guessed-at architecture detail poisons the whole system." A brief that says "these specific claims are inferred, the foundation docs are authoritative" is more useful than one that pretends to be complete.

### Treating the brief as a replacement for foundation docs

The brief is the map; the foundation docs are the territory. Do not summarise `canonical_pipeline.md`'s gotchas so thoroughly that a future LLM feels it can skip reading them before running training — those gotchas each cost real debugging time and the brief only names them.

### Quoting stale state docs as current

This project's specific hazard: `status.md` / `canonical_pipeline.md` predate the de-leaking. If the brief's current-state section leans on them, it will report a provisional pre-de-leaked picture as if current. Use `decisions.md` (D014-D017) for the current headline and say explicitly that the state docs are stale on this point.

### Refreshing too often

Don't edit the brief for every journal entry — that drifts it from "stable orientation map" to "rolling status update," a job `status.md` does better. Refresh when a decision closes, the publication runs complete, the doc structure changes, or the project enters a new stage.

### Editing prior Revision History entries

Append-only by discipline. Never edit prior entries even if later wrong — add a new entry that corrects the prior one. The path of revisions is itself information. (This is the same supersede-don't-edit rule the project applies to `decisions.md`.)

### Missing the cross-document referencing convention

The project uses D-IDs (D001…D017) and Q-IDs (Q1…Q11), monotonic and permanent. The brief should reference decisions and questions by ID, and respect the supersede-don't-edit rule.

---

## 7. Practical note: how this project's brief was first written (2026-06-04)

**Project state at the time:** A long publication-prep session that had crossed a context-compaction boundary — the de-leaked rebuild was complete and committed, the paper had been rewritten across ten sections, the orchestrator's output-path bug fixed and tested, and the training pipeline had just been given RNG seeding. The user was mid-way through planning two final publication runs (leaky + de-leaked) from the seeded pipeline. The user asked for a project brief (and this companion methodology guide) modelled on the versions from another project.

**Method:**

1. Inventoried `docs/` via shell (`ls -la`, `wc -l`, decision/question header greps).
2. Read the nine foundation docs in full (charter, philosophy, architecture, status, canonical_pipeline, backlog, open_questions, plus decisions D001-D017 headers and journal structure).
3. Cross-checked the de-leaked numbers against the session's committed work (D015-D017) and flagged the status.md staleness by date comparison.
4. Synthesised `docs/PROJECT_BRIEF.md` (~v1.0) using the Section 5 template, with in-flight work (the two publication runs, the unresolved MIMIC output-path collision) explicitly marked as not-yet-committed in the Gaps section.
5. Committed both documents to `docs/`.

**Outcome:** A brief that the next Claude instance can read in ~5-8 minutes for functional context, versus 30+ minutes to reconstruct from the foundation docs alone — and, importantly, one that correctly points at `decisions.md` rather than the stale `status.md` for the current headline.

---

## 8. A note on what this approach does and does not solve

**It solves:** lost context across compaction boundaries within a long session; onboarding cost for a fresh Claude picking up cold; discoverability of the documentation surface; the user's frustration at re-explaining project context.

**It does not solve:** foundation docs being stale or wrong (the brief points to them as authoritative; if they're wrong it inherits the wrongness — maintain them, and note known staleness like status.md's); real disagreements between the LLM's behaviour and the user's intent (the brief captures conventions, not values); the need for occasional refresh (durable, not permanent).

**It is not magic.** It is a discipline — the same state-vs-history discipline the project already applies to its `docs/` folder, extended to a single orientation document. The value comes from doing it consistently.

---

## 9. If you are the LLM implementing this

You have context for this project that the next instance will not: the feel of the leakage story, why the backbone inversion matters, which constraints are load-bearing, how the user actually works. Trust those instincts — they come from your accumulated session context.

But be honest about what you don't know. The Gaps section is the most important section in the brief. A brief that over-claims is worse than no brief. For this project specifically, be scrupulous about: the provenance of every number (leaky vs de-leaked), whether the publication runs have actually happened, and which work is committed versus drafted-in-session. Mark inferred claims. Point at `decisions.md` and the `summary.json` files as authoritative.

Read this document, then write (or refresh) the brief using the Section 5 template. Commit it by explicit name. Tell the user what you committed and where.

That's all.

---

**Document version:** v1.0
**Created:** 2026-06-04
**Based on the discipline applied to:** tst_llm (Apple Silicon LLM benchmarking) and Cedant (reinsurance RAG underwriting copilot), adapted to Notes_to_ICD10 (hierarchical ICD-10 classification, publication-prep).
**Author:** Claude (Anthropic), adapting the persistent-on-disk-brief discipline to this project during the 2026-06-04 publication-prep session.
