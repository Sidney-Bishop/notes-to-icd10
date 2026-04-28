# Notes-to-ICD10: Pipeline Overview

**Audience:** Business stakeholders and non-technical readers  
**Purpose:** Plain-language explanation of what this project does, why it is hard, and how we approached it  
**Last updated:** 27 April 2026  
**Current best result:** 79.8% accuracy on 1,926 ICD-10 codes

---

## What Are We Trying to Do?

Every time a patient is seen by a doctor, the clinical encounter must be assigned one or more **ICD-10 codes** — a standardised international classification system used for billing, research, and public health reporting. A code like `M25.561` means "Pain in right knee." There are approximately **70,000 ICD-10 codes** in total, of which **1,926 are clinically billable** in our dataset.

Currently, assigning these codes requires a trained human medical coder to read the clinical note and select the correct code. This is time-consuming, expensive, and error-prone. The goal of this project is to train an AI model to **read a clinical note and predict the correct ICD-10 code automatically**.

---

## The Dataset: MedSynth

Our training data is a synthetic clinical dataset called **MedSynth**, containing **10,240 clinical notes** across 1,926 unique billable ICD-10 codes.

Each note is written in **APSO format** — a clinical documentation style that leads with the **Assessment** (the doctor's diagnosis) followed by the Plan, then Subjective and Objective findings. This ordering matters because it puts the most diagnostically useful information at the beginning of the note, within the AI model's attention window.

### The Fundamental Data Problem

This dataset has a severe constraint: **approximately 5 clinical notes per ICD-10 code**. To put that in perspective:

| Scenario | Examples available |
|---|---|
| Teaching a child to recognise a dog | Thousands of photos |
| Training ImageNet (standard AI benchmark) | 1,000+ per category |
| **Our ICD-10 dataset** | **~5 per code** |

A typical machine learning task requires hundreds or thousands of examples per category to train reliably. We have 5. This is the central challenge the entire project is designed to address — and every modelling decision in the notebooks flows from it.

---

## The ICD-10 Code Hierarchy: Understanding the Structure

ICD-10 codes have a natural four-level hierarchical structure. Understanding this is essential to following the project's logic:

```
Level 1 — Chapter    (1 char):   M        Musculoskeletal diseases      22 groups
Level 2 — Category   (3 chars):  M25      Other joint disorders        675 groups  ← ICD-3
Level 3 — Subcategory(4-5 chars):M25.5    Pain in joint
Level 4 — Full code  (5-7 chars):M25.561  Pain in right knee         1,926 codes   ← ICD-10
```

**ICD-3** refers to the first 3 characters of any code — the "category" level. There are **675 unique ICD-3 categories** in our dataset. Multiple full ICD-10 codes collapse into the same ICD-3 group: `M25.561` (right knee), `M25.562` (left knee), and `M25.50` (unspecified joint) all belong to `M25`.

**ICD chapters** are coarser still — just the single leading letter. All codes beginning with `M` belong to the musculoskeletal chapter, regardless of whether they are `M25`, `M47`, or `M79`. There are only **22 chapters** in our dataset.

The relationship between these levels matters because it defines two different ways of building a smarter prediction system — and we tested both.

---

## Why Is This Hard?

### Problem 1: Too Many Categories, Too Little Data

With 1,926 codes and only ~5 examples each, a model trained directly on full ICD-10 codes is like asking someone to identify 1,926 different dog breeds having seen only 5 photos of each. The model simply cannot learn reliable boundaries between similar codes.

### Problem 2: Codes That Look Almost Identical

Many ICD-10 codes describe clinically near-identical situations:
- `M25.561` — Pain in right knee
- `M25.562` — Pain in left knee
- `M25.569` — Pain in unspecified knee

The clinical note may not explicitly state laterality, making even experienced human coders uncertain. An AI model seeing only 5 examples of each has almost no chance of learning this distinction reliably.

### Problem 3: Administrative Code Ambiguity (The Z-Chapter Problem)

Z-codes cover routine health encounters, screenings, and administrative health histories. Their clinical notes are deliberately generic — a routine check-up reads very similarly regardless of which specific Z-code applies. This makes Chapter Z the hardest chapter to predict accurately in every single experiment we ran, and it remains the primary unsolved challenge in the current system.

### Problem 4: The Context Window

Bio_ClinicalBERT — the AI model we use — can only read 512 "tokens" (roughly words) at a time. Many clinical notes are longer than this. If the most diagnostically useful information appears late in a long note, the model may never read it. This is why the APSO note ordering (Assessment first) is critical — it ensures the diagnosis appears within the model's reading window.

---

## Our Approach: A Progressive Strategy

We did not jump straight to predicting full ICD-10 codes. Instead, we built up progressively across five notebooks, learning from each step and using each result as the foundation for the next.

---

## Notebook 01 — Understanding the Data

**What this notebook does:** Before building any model, we analysed the dataset to understand its structure, the distribution of codes, and the preprocessing required.

**Key findings:**
- 64.7% of clinical notes exceed the 512-token limit — the APSO reordering is essential to ensure the diagnosis is read first
- 28.5% of notes contained explicit ICD-10 code strings that had to be removed before training (otherwise the model would simply read the answer rather than learn to diagnose)
- The dataset contains exactly 5 records per code — the uniform sampling design is both a constraint and a guarantee of balance

**Output:** A "Gold layer" dataset — cleaned, reordered, and validated — that all downstream notebooks use as their starting point.

---

## Notebook 02 — Can We Even Learn from These Notes?
### Experiment E-001 | ICD-3 Prediction | 675 Categories

**The question:** Before attempting the hard problem, can an AI model learn *anything* meaningful from these clinical notes?

**The approach:** We deliberately made the problem easier by predicting at **ICD-3 resolution** — 675 categories instead of 1,926 codes. With ~15 training examples per ICD-3 category (vs ~5 per ICD-10 code), the learning problem is three times more tractable.

**The model:** Bio_ClinicalBERT — a version of the BERT language model pre-trained specifically on clinical text. We fine-tuned it to predict one of 675 ICD-3 categories.

**Training:** 30 epochs (~150 minutes on Apple M5 Max). A key finding from this notebook: the model continued improving all the way to epoch 28 before plateauing. This confirmed that high-cardinality clinical classification requires significantly more training than standard benchmarks — 10 epochs would have missed 8 percentage points of F1.

**The result:**

| Metric | Value | What it means |
|---|---|---|
| Test Accuracy | 87.2% | Correct ICD-3 category 87 times in 100 |
| Macro F1 | 0.841 | Consistent performance across all 675 categories |
| Top-5 Accuracy | 93.4% | Correct category in top 5 predictions 93% of the time |

**Why this matters:** This was a crucial proof of concept. The clinical notes contain learnable diagnostic signal, the preprocessing works, and Bio_ClinicalBERT is the right model family. Without this confirmation, everything that follows would be built on unproven foundations.

> **Important:** ICD-3 prediction is **not** the goal of the project. ICD-3 codes are not billable. This notebook exists solely to prove the approach works before tackling the harder problem.

**A revealing test:** When we fed the model a note describing a heart attack (STEMI) written in conventional note order (not APSO order), the model's top prediction was Z95 — "presence of cardiac implants." Why? Because the note's treatment plan mentioned "urgent PCI" (a procedure that results in a coronary stent), and the model weighted the plan section heavily. The correct code (I21 — acute myocardial infarction) appeared at rank 4. This directly validated the APSO reordering strategy: in production, notes must be reordered before prediction.

---

## Notebook 03 — The Hard Problem: Direct ICD-10 Prediction
### Experiment E-002 | Flat ICD-10 | 1,926 Codes

**The question:** What happens if we simply try to predict all 1,926 ICD-10 codes directly with a single model?

**The approach:** Same Bio_ClinicalBERT architecture, now with a 1,926-way classification head. Same training data, same notes, harder problem. We trained for 40 epochs to ensure the model had every opportunity to reach its ceiling.

**The result:**

| Metric | Value |
|---|---|
| Test Accuracy | 73.3% |
| Macro F1 | 0.634 |
| Top-5 Accuracy | 87.6% |

73.3% accuracy on a 1,926-way classification task with only 4 training examples per code is genuinely strong. But it establishes the **flat baseline** — the best a single model can do when asked to simultaneously distinguish all 1,926 codes.

**The critical insight from chapter-level analysis:** The model achieves **91.2% accuracy** at predicting which of 22 clinical chapters a note belongs to, but only 73.3% at the full code level. This 17.9 percentage point gap tells us exactly what the model struggles with: it knows a patient has a musculoskeletal condition; it struggles to pinpoint exactly which of 222 musculoskeletal codes applies.

Furthermore, 67.1% of the model's errors stay within the correct clinical chapter — wrong specific code, right domain. Only 8.8% of predictions cross into a completely different clinical territory. The model has learned the broad structure of ICD-10 almost perfectly. The remaining challenge is **within-chapter precision**.

This finding directly defines the opportunity for the hierarchical approach.

---

## Notebook 04 — The Hierarchical Approach, First Attempt
### Experiment E-003 | Two-Stage Hierarchical | Cold Start

**The question:** Can we do better than the flat model by splitting the prediction into two dedicated stages — one to identify the clinical domain, one to resolve the specific code?

### Understanding the Architecture (Important Clarification)

A natural intuition — and a very reasonable one — is that the two stages would be: (1) predict the ICD-3 category, then (2) predict the specific ICD-10 code within that category. This would mean Stage 2 only has to choose between 2–5 codes per group, a very easy task.

**We chose a different decomposition.** Stage 1 predicts at the **chapter level** (22 chapters), not the ICD-3 level (675 categories). Here is why:

1. **Routing errors are catastrophic.** If Stage 1 sends a note to the wrong resolver, Stage 2 cannot recover — the correct code is simply not available in the wrong resolver's label space. A 675-way ICD-3 classifier with only ~15 examples per category would make significantly more routing errors than a 22-way chapter classifier with ~440 examples per chapter.

2. **E-001 already solves ICD-3.** Notebook 02 trains a dedicated 675-way ICD-3 classifier achieving 87.2% accuracy. If ICD-3 routing were the goal, E-001 is the answer. The hierarchical notebooks explore a structurally different decomposition.

The tradeoff is real: routing at chapter level leaves Stage 2 with ~100 codes to resolve per chapter (rather than ~3 per ICD-3 group). But the higher routing reliability justifies this.

```
Clinical Note
     │
     ▼
┌─────────────────────────────────────────┐
│  STAGE 1: Chapter Router                │
│  22-way classifier                      │
│  "Which of 22 clinical chapters?"       │
│  e.g. → Chapter M (Musculoskeletal)     │
│  Accuracy: 96.4%                        │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│  STAGE 2: Chapter Resolver              │
│  Within-chapter classifier (~100 codes) │
│  "Which specific M-code?"               │
│  e.g. → M25.561 (Pain in right knee)   │
│  19 separate resolvers, one per chapter │
└─────────────────────────────────────────┘
```

**Stage 1 training:** The chapter router was initialised from E-001's weights (the ICD-3 classifier), since E-001 had already learned to associate clinical language with broad diagnostic groupings. It achieved **96.4% chapter routing accuracy** in just 5 epochs — exceeding the flat model's implicit chapter accuracy of 91.2%. A key operational finding: the router converges at epoch 4; training beyond 5 epochs wastes compute.

**Stage 2 training — what went wrong:**

Stage 2 models were initialised from **fresh Bio_ClinicalBERT weights** — the base model with no ICD-10 knowledge whatsoever. Each of 19 resolvers had to learn ICD-10 code distinctions from scratch, using only the ~440 records in its chapter (~4 per code).

**End-to-end result: 11.1% accuracy** — far worse than the flat baseline of 73.3%.

The architecture was correct. The initialisation was wrong.

The most telling evidence: Chapter M had 888 training records — far above any data scarcity threshold — yet achieved only 5.8% accuracy across 222 codes. More data did not help. The problem was that fresh Bio_ClinicalBERT has never seen an ICD-10 code and cannot learn to distinguish `M25.561` from `M25.562` from 4 training examples, no matter how well-designed the architecture is. It needs a starting point that already understands ICD-10.

---

## Notebook 05 — The Hierarchical Approach, Correctly Initialised
### Experiment E-009 | Two-Stage Hierarchical | E-002 Initialisation

**The question:** What if Stage 2 starts from the flat ICD-10 model (E-002) instead of fresh weights?

**The insight:** E-002 spent 40 epochs learning to distinguish all 1,926 ICD-10 codes. It already "knows" that `M25.561` and `M25.562` are similar but different. If each Stage 2 resolver starts from these weights rather than from scratch, it only needs to **sharpen** its within-chapter discrimination — not learn ICD-10 from zero.

**The architecture is identical to Notebook 04:**
- Stage 1: same 22-way chapter router reused from Notebook 04 — no retraining
- Stage 2: 19 chapter resolvers, each initialised from E-002 weights, fine-tuned on chapter-filtered data

**Stage 2 training results (before test evaluation):**

| Chapter | Val Accuracy | Notes |
|---|---|---|
| T (Injuries/Poisoning) | 100% | Perfect — was 0% in E-003 |
| A (Infectious Diseases) | 100% | Perfect — was 0% in E-003 |
| N (Genitourinary) | 94.1% | |
| B, I, C, R, S ... | 87–92% | Strong across the board |
| Z (Administrative) | 59.5% | Still the hardest chapter |
| **Weighted average** | **83.5%** | vs 13.4% in E-003 |

The E-002 initialisation delivered a **70 percentage point improvement** in within-chapter accuracy. Chapter A — which had completely failed (0%) in E-003 — achieved perfect accuracy. The difference is entirely explained by the starting weights.

**End-to-end result:**

| Approach | Accuracy | vs Flat Baseline |
|---|---|---|
| Flat ICD-10 (E-002, Notebook 03) | 73.3% | baseline |
| Hierarchical cold start (E-003, Notebook 04) | 11.1% | −62.2pp |
| **Hierarchical E-002 init (E-009, Notebook 05)** | **79.8%** | **+6.5pp** ✅ |

**The hierarchical approach definitively beats the flat baseline** — 79.8% vs 73.3%, a +6.5 percentage point improvement. This is the first time in the project that the hierarchical architecture has clearly outperformed a single flat model.

**Extended training:** We also ran additional training for the 6 weakest chapter resolvers (Z, K, E, H, T, S) for 10 more epochs at a lower learning rate. The result: essentially no change (±0.008 F1 across all 6 chapters). The models had already reached their ceiling. Phase 2 weights are the final resolvers.

---

## The Complete Picture

| Notebook | Experiment | Task | Accuracy | Key Learning |
|---|---|---|---|---|
| 01 | EDA | Dataset understanding | — | APSO ordering is critical; 5 examples per code is the constraint |
| 02 | E-001 | ICD-3 (675 categories) | 87.2% | Clinical notes are learnable; 30 epochs needed |
| 03 | E-002 | Flat ICD-10 (1,926 codes) | 73.3% | Sets the flat baseline; 91.2% chapter accuracy |
| 04 | E-003 | Hierarchical, fresh init | 11.1% | Right architecture, wrong initialisation |
| 05 | E-009 | Hierarchical, E-002 init | **79.8%** | **Beats flat baseline; E-002 init is the key** |

---

## What 79.8% Accuracy Means in Practice

On the held-out test set of 966 clinical notes that the model had never seen:

- **771 notes** (79.8%) — exactly correct ICD-10 code predicted
- **160 notes** (16.6%) — routed to the correct chapter, wrong specific code within it
- **35 notes** (3.6%) — routed to the wrong chapter entirely

**Top-5 accuracy is 87–99% across most chapters.** This means the correct ICD-10 code appears in the model's top 5 predictions for the vast majority of notes. In a clinical coding assistance workflow — where a human coder selects from a ranked shortlist rather than a full 1,926-code catalogue — the correct code would be surfaced for virtually every patient encounter.

---

## What Remains to Be Solved

### The Z-Chapter Problem (Primary Target)

Administrative codes (Chapter Z) achieved only 52.9% end-to-end accuracy. These codes cover routine health encounters with deliberately generic clinical language — by design, a routine check-up note reads similarly regardless of the specific Z-code. This is a structural challenge, not a training deficiency. Targeted approaches (contrastive learning, data augmentation specific to Z-codes) are the next avenue.

### The Data Constraint

MedSynth's uniform 5-examples-per-code design does not reflect real-world clinical data, where common conditions have thousands of examples and rare ones have very few. The entire project has been constrained by this artificial uniformity. Validating on a real clinical dataset — such as MIMIC-IV from PhysioNet — is the next major step, pending data access approval.

### The Remaining Gap

The +6.5pp improvement from hierarchical over flat is meaningful but modest. The theoretical ceiling (based on Stage 1 routing accuracy of 96.4% × the within-chapter target of 80.4%) suggests the architecture could reach approximately 77–80% E2E with better Stage 2 performance. The current 79.8% result is already near this ceiling, with Z-chapter being the primary remaining lever.

---

## Glossary

| Term | Plain English |
|---|---|
| ICD-10 | International Classification of Diseases, 10th revision — the global standard for medical diagnosis codes |
| ICD-3 | The first 3 characters of an ICD-10 code — a coarser grouping (675 categories vs 1,926 codes) |
| ICD Chapter | The first letter of an ICD-10 code — the broadest grouping (22 chapters) |
| APSO | Assessment, Plan, Subjective, Objective — the note ordering that puts the diagnosis first |
| Bio_ClinicalBERT | An AI language model pre-trained on millions of clinical notes |
| Macro F1 | A performance metric that weights all categories equally regardless of how common they are |
| Flat model | A single AI model that predicts all 1,926 codes simultaneously |
| Hierarchical model | A two-stage AI system: Stage 1 routes to a chapter, Stage 2 resolves within that chapter |
| Stage 1 / Chapter Router | The first-stage classifier that assigns a note to one of 22 clinical chapters |
| Stage 2 / Chapter Resolver | One of 19 second-stage classifiers, each specialised in one clinical chapter |
| Cold start | Initialising Stage 2 from fresh weights with no prior ICD-10 knowledge |
| E-002 initialisation | Initialising Stage 2 from the trained flat model — the key architectural decision |

---

*This document reflects the state of the project as of 27 April 2026.*  
*Current best result: E-009 — 79.8% accuracy, 0.711 Macro F1 on 1,926 ICD-10 codes.*  
*All five notebooks have been reviewed, re-run, and documented.*