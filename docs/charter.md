# Charter

*Dated 2026-05-31. Near-static state file — change rarely, and date any change.*

## What this project is

A research system that predicts billable ICD-10 diagnostic codes from clinical
notes, using a two-stage hierarchical classifier (chapter router → per-chapter
resolver) built on Bio_ClinicalBERT, trained on the synthetic MedSynth dataset.

## Who it is for

The author's research, intended to support a publication. Secondary audience: any
researcher reproducing or extending the approach from the public repo and HF
dataset.

## What success looks like

- A reproducible pipeline: a clean clone rebuilds byte-identical data and produces
  reported metrics within documented variance.
- A defensible headline result on the full billable ICD-10 code set (~1,926
  codes) in the low-resource regime (~4 training examples per code), reported with
  error bars across seeds.
- An honest account of limitations — especially the synthetic→real gap.

## Explicitly out of scope

- Live clinical deployment / coding decisions on real patients. This is a research
  prototype, not a certified coding tool.
- Real-data training. MIMIC-IV is used only for validation/quantifying the domain
  gap, not as a training source (subject to PhysioNet access).
- Beating state-of-the-art at any cost — the goal is a clear, reproducible,
  honestly-reported result, not a leaderboard number.

## Non-negotiables

- **Reproducibility over convenience:** data locked to HF + SHA256; derived
  artifacts versioned; splits frozen and deterministic.
- **No label leakage:** ICD-10 strings redacted from note text.
- **Honest reporting:** numbers come from a single generated source of truth, not
  hand-transcription; discrepancies are documented, not smoothed over.
