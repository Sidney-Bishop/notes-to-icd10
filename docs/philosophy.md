# Documentation Philosophy

This file explains how the `docs/` folder works, so anyone opening it — an
outside collaborator, a future agent, or future-you in six months — can use and
maintain it without tribal knowledge. Read this first.

## Why this exists

This project previously accumulated decisions, results, and findings in scattered
places — commit messages, notebook cells, model `config.json` files, and prose
docs that drifted out of sync. The cost was concrete: reconstructing *what was
done and why* required re-running code and diffing artifacts, because the intent
was never written down in one trustworthy place. This folder fixes that. It is a
first-class deliverable, updated in the same commits as the code that motivates
it, never reconstructed afterward.

## The two kinds of file

**State files — overwrite freely. They describe what is true *now*.**
- `architecture.md` — how the system is currently built.
- `status.md` — where the work currently stands (Done / In-progress / Blocked / Next).

**Append-only files — never edit past entries. They are a historical record.**
- `decisions.md` — choices someone might later question (IDs: D001, D002, …).
- `journal.md` — dated log of what actually happened (bugs, surprises, dead ends).
- `open_questions.md` — unresolved tensions (IDs: Q1, Q2, …); resolve by adding a
  status line pointing to the D-ID that settled it, never by deleting.
- `backlog.md` — agreed-but-not-in-flight work items, `- [ ]` checkboxes.

`charter.md` is a near-static state file: what the project is for, who it's for,
what success looks like, what's out of scope. Dated; changed rarely.

## The routing rule (apply strictly)

| If it is… | It goes in… |
|---|---|
| true now (system shape) | `architecture.md` |
| true now (work state) | `status.md` |
| a choice someone will question later | `decisions.md` (D-ID) |
| something that happened | `journal.md` (dated) |
| not yet decided | `open_questions.md` (Q-ID) |
| queued work | `backlog.md` |
| what the project is / its scope | `charter.md` |
| how this doc system works | `philosophy.md` (this file) |

When in doubt: *facts → state files; reasoning → decisions; events → journal.*

## Conventions

- **D-IDs and Q-IDs are monotonic and permanent.** Never renumber. Supersede an
  old decision with a new D-ID that references it; don't rewrite the old one.
- **Decision entry shape:** Context / Decision / Rationale / Trade-offs / When to revisit.
- **Date every append-only entry** (ISO `YYYY-MM-DD`).
- **No retrospective fabrication.** Do not invent D-IDs or journal entries for
  choices made before they were recorded. The one legitimate exception in this
  project: decisions genuinely made *during* the session that created this folder
  (2026-05-31) are recorded with that real date — they are contemporaneous, not
  reconstructed. Everything earlier is referenced as background, not logged as if
  we were there to record it.
- **Update docs in the same commit as the code change** that motivates them, so
  the two never drift.

## A note on trust

The value of these files is exactly their trustworthiness. A fabricated decision
or a guessed-at architecture detail poisons the whole system — a reader can no
longer tell recorded fact from reconstruction. Where something is inferred rather
than verified, it is marked as such in the file. Prefer "unconfirmed: …" over a
confident statement you cannot stand behind.
