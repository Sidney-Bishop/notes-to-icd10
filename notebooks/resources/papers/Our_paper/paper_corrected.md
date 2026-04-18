---
title: "Hierarchical ICD-10 Coding from Clinical Notes: Transfer Learning and Architectural Decomposition for Low-Resource Multi-Label Classification"
author:
  - name: Jason Roche
date: 2026
abstract: |
  Automatic International Classification of Diseases (ICD) coding from clinical text remains challenging due to the large label space, long-tail code distributions, and lengthy clinical documents that exceed transformer context limits. We present a two-stage hierarchical pipeline for ICD-10 coding from APSO-structured clinical notes using Bio_ClinicalBERT. Our architecture decomposes 1,926-way classification into a 22-way chapter routing stage (95.4% accuracy) followed by within-chapter resolution (70.1% accuracy), achieving 66.9% end-to-end accuracy and 0.553 Macro F1 with approximately four training examples per code. We introduce APSO-Flip preprocessing, which repositions the Assessment section to Token 0 to prevent truncation of diagnostic evidence within BERT's 512-token context window—addressing the finding that 64.7% of clinical notes in our dataset exceed this limit. We implement systematic ICD-10 string redaction to eliminate label leakage present in 28.5% of records. Critically, we demonstrate that hierarchical decomposition alone is insufficient: identical architectures achieve 10.6% accuracy when Stage-2 resolvers initialise from pretrained BERT weights versus 66.9% when initialised from a flat ICD-10 classifier—a 6.3× improvement in within-chapter accuracy attributable solely to transfer learning. Our results establish that the combination of hierarchical architecture and progressive transfer learning is essential for low-resource ICD-10 classification, with neither component sufficient in isolation. Code and trained models are available at https://github.com/Sidney-Bishop/notes-to-icd10.
keywords:
  - ICD-10 coding
  - clinical NLP
  - hierarchical classification
  - transfer learning
  - Bio_ClinicalBERT
---

# Introduction

The International Classification of Diseases (ICD) provides a standardised taxonomy for encoding diagnoses and procedures in clinical settings. Accurate ICD coding underpins healthcare billing, epidemiological surveillance, and clinical research [@yan2022survey]. However, manual coding is labour-intensive, error-prone, and requires specialised training—motivating substantial research into automated coding systems.

Automatic ICD coding presents four fundamental challenges [@yan2022survey]. First, the label space is vast: ICD-10-CM contains over 70,000 diagnostic codes, and even constrained clinical datasets typically involve thousands of unique codes. Second, code distributions follow a heavy-tailed pattern where a small fraction of codes account for most assignments while thousands of codes appear rarely in training data. Third, clinical documents—particularly discharge summaries—often exceed the context limits of transformer-based models, requiring truncation strategies that may discard diagnostically relevant content. Fourth, the clinical adoption of automated systems demands interpretability, yet deep learning models typically function as black boxes.

Recent advances in clinical natural language processing have yielded progressively stronger baselines. Convolutional architectures with label-wise attention [@mullenbach2018caml; @vu2020laat] achieve strong performance by learning code-specific document representations. Transformer-based approaches, including adaptations of BERT for clinical text [@zhang2020bertxml; @huang2022plmicd], leverage pretrained language models but struggle with documents exceeding 512 tokens. Knowledge-enhanced methods incorporate external resources such as medical ontologies [@xie2019msattkg] or code synonyms [@yuan2022msmn] to improve rare code prediction.

Despite this progress, several limitations persist. Most benchmark work targets ICD-9 codes on MIMIC-III [@johnson2016mimic], leaving ICD-10 relatively underexplored. Flat classification approaches scale poorly as the label space grows. And while hierarchical methods have shown promise [@perotte2014hierarchical], the interaction between architectural decomposition and transfer learning initialisation remains poorly understood.

We address these gaps with a two-stage hierarchical pipeline trained on the MedSynth dataset [@rezaie2025medsynth], which provides 10,240 synthetic clinical notes covering 2,037 ICD-10 codes with uniform sampling. Our contributions are threefold:

1. **APSO-Flip preprocessing**: We restructure clinical notes to position the Assessment section at Token 0, preventing Bio_ClinicalBERT's 512-token truncation from discarding diagnostic evidence. This preprocessing step operationalises findings from Yang et al. [@yang2023gpsoap], who demonstrated that SOAP structure encodes clinical reasoning. Our analysis confirms that 64.7% of clinical notes exceed the 512-token limit, making truncation the structural default rather than an edge case.

2. **Hierarchical decomposition with transfer learning**: We decompose ICD-10 classification into chapter-level routing (22 classes) followed by within-chapter resolution (average ~100 classes per resolver). Critically, we demonstrate that this architecture succeeds only when Stage-2 resolvers initialise from a flat ICD-10 classifier rather than pretrained BERT weights—isolating the contribution of transfer learning from architectural design.

3. **Systematic ablation**: Through controlled experiments (E-001 through E-005a), we quantify the individual and combined effects of hierarchical decomposition and transfer learning initialisation, establishing that neither component is sufficient in isolation.

Our best model achieves 66.9% accuracy and 0.553 Macro F1 across 1,926 ICD-10 codes with approximately four training examples per code—demonstrating that hierarchical architectures with appropriate initialisation can substantially outperform flat classification in low-resource settings.


# Related Work

## Convolutional and Attention-Based Approaches

Early neural approaches to ICD coding employed convolutional neural networks (CNNs) to capture local n-gram patterns in clinical text. Mullenbach et al. [@mullenbach2018caml] introduced CAML (Convolutional Attention for Multi-Label classification), which applies per-label attention over CNN-encoded text to produce code-specific document representations. This label-wise attention mechanism became foundational for subsequent work, enabling models to identify which text segments support each code prediction.

Building on this foundation, Vu et al. [@vu2020laat] proposed LAAT (Label Attention Model), which uses self-attention to learn label-specific vectors from bidirectional LSTM encodings. LAAT achieved state-of-the-art performance on MIMIC-III and demonstrated that attention mechanisms could capture fine-grained interactions between clinical text and ICD codes. Merchant et al. [@merchant2024ensemble] extended this approach with CD-LAAT, incorporating ICD code descriptions to adjust attention weights, achieving improved Macro F1 on rare codes.

Li and Yu [@li2020multifilter] introduced multi-filter residual CNNs to overcome fixed filter sizes, achieving strong AUC scores on MIMIC-III. Liu et al. [@liu2021effective] combined residual and squeeze-and-excitation networks with focal loss to improve rare code prediction. These CNN-based approaches remain competitive but are limited by their inability to capture long-range dependencies in clinical text.

## Transformer-Based Models

The success of BERT [@devlin2019bert] in general NLP motivated its application to clinical coding. However, BERT's 512-token context limit poses challenges for clinical documents, which often exceed this length substantially. Pascual et al. [@pascual2021bertbased] evaluated PubMedBERT on ICD coding, finding that the first 512 tokens yielded better performance than middle or final segments—suggesting that document structure matters for truncation decisions.

Zhang et al. [@zhang2020bertxml] proposed BERT-XML for extreme multi-label classification, achieving strong AUC scores on ICD-10 prediction from electronic health records. However, computational demands for long sequences and marginal improvements over simpler methods limited practical adoption. To address context length limitations, Yang et al. [@yang2022kept] introduced KEPTLongformer, combining Longformer's extended context with contrastive learning on the UMLS knowledge graph. While achieving state-of-the-art performance on MIMIC-III top-50 codes, the model's 119.4M parameters limit scalability to full code sets.

The tension between model capacity and context length remains unresolved. Our approach sidesteps this by restructuring notes to prioritise diagnostic content within the 512-token window, rather than extending context length.

## SOAP Structure and Section-Aware Methods

Clinical notes follow conventions that encode diagnostic reasoning. The SOAP format—Subjective, Objective, Assessment, Plan—structures patient encounters such that Assessment and Plan sections contain diagnostic conclusions derived from Subjective symptoms and Objective findings.

Yang et al. [@yang2023gpsoap] demonstrated that this structure could be exploited for ICD coding through their GPsoap model, which pretrains on generating Assessment and Plan sections from Subjective and Objective inputs. This pretraining objective teaches the model to perform clinical inference, substantially improving performance on rare codes. On MIMIC-III-few (codes appearing ≤5 times in training), GPsoap achieved 30.2 Macro F1 versus 4.3 for prior state-of-the-art—a 7× improvement attributable to SOAP-aware pretraining.

Dai et al. [@dai2025modelselection] validated section-aware prompting for ICD-10-CM coding, finding that incorporating multiple clinical sections (discharge diagnosis, medical history, operative notes) progressively improved F1 scores. Their finding that discharge diagnosis contributed most significantly to performance aligns with our prioritisation of the Assessment section.

Masud et al. [@masud2023applying] explicitly used SOAP notes for ICD-10 prediction with CNNs, achieving F-scores ranging from 0.65 (nephrology, 277 codes) to 0.98 (cardiology, 148 codes). The substantial performance variation across departments highlights the relationship between label space size and achievable accuracy.

Our APSO-Flip preprocessing extends these insights by physically reordering notes to ensure Assessment content survives truncation, operationalising the finding that Assessment sections carry primary diagnostic signal.

## Hierarchical and Decomposition Approaches

ICD codes possess inherent hierarchical structure: the first character indicates chapter (e.g., "E" for Endocrine diseases), subsequent characters specify category and subcategory. Perotte et al. [@perotte2014hierarchical] demonstrated that hierarchy-aware classifiers outperform flat approaches on MIMIC-II, achieving higher recall and F-measure by exploiting parent-child relationships between codes.

Xie et al. [@xie2019msattkg] incorporated hierarchical relationships through graph convolutional networks over the ICD ontology, using multi-scale feature attention to dynamically weight n-gram features. Ren et al. [@ren2022hicu] proposed curriculum learning strategies that leverage hierarchy, training on coarse codes before fine-grained ones.

However, most hierarchical approaches implicitly assume sufficient training data at each level. In low-resource settings with few examples per code, learning reliable within-category classifiers becomes challenging. We address this through transfer learning: rather than training within-chapter resolvers from scratch on small filtered datasets, we initialise from a flat classifier that has seen all training examples, preserving learned representations while adapting to the hierarchical structure.

## Retrieval and Generation Approaches

Alternative formulations recast ICD coding as retrieval or generation rather than classification. Klotzman [@klotzman2024enhancing] evaluated embedding models for ICD-10-CM code retrieval via semantic similarity, finding that general-purpose LLM embeddings (voyage-large-2-instruct, 94.2% accuracy) substantially outperformed ClinicalBERT (8.7%) on direct diagnosis-to-code mapping. However, this retrieval framing assumes structured diagnosis strings rather than full clinical notes.

Puts et al. [@puts2025developing] combined RoBERTa-based lead term extraction with GPT-4 retrieval-augmented generation (RAG) for ICD-10 coding. While lead term extraction performed well (F1=0.80), RAG-based code selection degraded performance (F1=0.305), demonstrating that retrieval approaches struggle when code descriptions are incomplete or ambiguous.

Yang et al. [@yang2023gpsoap] reframed ICD coding as autoregressive generation of code descriptions rather than direct code prediction. By generating text descriptions and mapping to codes, they reduced the output space from thousands of codes to a fixed vocabulary of 4,501 tokens. This approach excels for rare codes but introduces inference latency from autoregressive decoding.

Our classification-based approach offers complementary strengths: faster inference through single forward passes, interpretable chapter-level routing, and compatibility with standard multi-label training objectives.

## Datasets and Evaluation

MIMIC-III [@johnson2016mimic] remains the dominant benchmark for ICD coding research, providing discharge summaries with ICD-9 codes from Beth Israel Deaconess Medical Center. The standard MIMIC-III-full benchmark involves 8,922 unique codes, while MIMIC-III-50 restricts evaluation to the 50 most frequent codes.

Lavergne et al. [@lavergne2016dataset] created a large-scale ICD-10 dataset from French death certificates (93,694 certificates, 3,457 unique codes), demonstrating that ICD-10 coding datasets can be constructed beyond English clinical notes. Their shared task achieved 0.848 F-measure on relatively short, focused death certificate statements.

The MedSynth dataset [@rezaie2025medsynth] addresses ICD-10 directly with synthetic clinical notes. Its uniform sampling (5 records per code) creates an unusual distribution—eliminating the long-tail challenge while creating an extreme low-resource setting. This design enables controlled study of model capacity independent of data imbalance, though results may not transfer directly to real clinical distributions.


# Dataset

## MedSynth Overview

We train and evaluate on the MedSynth dataset [@rezaie2025medsynth], which provides 10,240 synthetic dialogue-note pairs generated by a four-agent GPT-4o pipeline informed by real-world disease distributions from 800 million IQVIA PharMetrics Plus insurance claims. Each record contains a simulated patient-physician dialogue and an associated clinical note structured in SOAP format (Subjective, Objective, Assessment, Plan). We use only the structured clinical note field for classification, not the raw dialogue transcripts.

MedSynth employs approximately uniform sampling with five records per ICD-10 code for 99.5% of codes (2,026 codes × 5 records = 10,130 records); 11 codes contain 10 records each (110 records total), likely reflecting post-publication additions to the HuggingFace release. This yields approximately four training examples per code after train/validation/test splitting (80/10/10). The MedSynth authors explicitly designed this uniform distribution to "prevent MedSynth from being dominated by common diseases" [@rezaie2025medsynth]. This design eliminates the class imbalance typical of real clinical data, enabling controlled evaluation of model architectures independent of frequency effects. However, this uniform distribution differs substantially from real-world ICD distributions, where common codes (e.g., essential hypertension, type 2 diabetes) dominate while thousands of codes appear rarely or never. Models trained on uniformly sampled data will not learn real-world class frequency distributions.

Our dataset contains 10,240 records and 2,037 distinct codes—205 records and 36 codes more than the 10,035 records and 2,001 codes reported at publication, consistent with post-publication additions to the HuggingFace release.

## ICD-10 Code Validation and Distribution

The raw dataset contains 2,037 distinct ICD-10 codes. We validated all codes against the official CDC FY2026 ICD-10-CM reference (74,719 billable codes), classifying each record into one of three categories:

| Status | Records | Percentage | Description |
|--------|---------|------------|-------------|
| Billable | 9,660 | 94.3% | Confirmed leaf codes in CDC FY2026 tabular reference |
| Parent codes (`noisy_111`) | 555 | 5.4% | Valid ICD-10 category codes, too broad for billing |
| Placeholder-X | 25 | 0.24% | Legitimate injury codes using ICD-10-CM placeholder convention |

The 555 records with parent-level codes (e.g., `J18` rather than `J18.9`) reflect real clinical billing practice—the MedSynth authors selected from the top 2,001 most frequent ICD-10 codes in IQVIA insurance claims, where clinicians genuinely submit category-level codes. The 25 placeholder-X codes (e.g., `T78.1XXA`, `T81.4XXA`) use the mandatory ICD-10-CM convention for injury/trauma codes where character positions must be held for specificity extensions; their absence from the CDC tabular descriptions file is a known reference gap, not a dataset error.

For classification experiments, we retained only billable leaf codes, yielding **1,926 unique ICD-10 codes** across 9,660 records spanning 19 of the 22 ICD-10 chapters. Three chapters—H60-H95 (Ear/mastoid), O00-O9A (Pregnancy), and U00-U85 (Special codes)—have no representation in MedSynth. Table 1 presents the chapter distribution.

| Chapter | Code Range | Description | Codes | % of Dataset |
|---------|------------|-------------|-------|--------------|
| A | A00-B99 | Infectious diseases | 145 | 7.5% |
| C | C00-D49 | Neoplasms | 201 | 10.4% |
| D5 | D50-D89 | Blood diseases | 42 | 2.2% |
| E | E00-E89 | Endocrine/metabolic | 108 | 5.6% |
| F | F01-F99 | Mental disorders | 93 | 4.8% |
| G | G00-G99 | Nervous system | 97 | 5.0% |
| H | H00-H59 | Eye disorders | 66 | 3.4% |
| I | I00-I99 | Circulatory system | 152 | 7.9% |
| J | J00-J99 | Respiratory system | 79 | 4.1% |
| K | K00-K95 | Digestive system | 118 | 6.1% |
| L | L00-L99 | Skin diseases | 80 | 4.2% |
| M | M00-M99 | Musculoskeletal | 165 | 8.6% |
| N | N00-N99 | Genitourinary | 83 | 4.3% |
| P | P00-P96 | Perinatal conditions | 27 | 1.4% |
| Q | Q00-Q99 | Congenital anomalies | 71 | 3.7% |
| R | R00-R99 | Symptoms/signs | 84 | 4.4% |
| S | S00-T88 | Injury/poisoning | 174 | 9.0% |
| V | V00-Y99 | External causes | 68 | 3.5% |
| Z | Z00-Z99 | Health services | 73 | 3.8% |

Table 1: ICD-10 chapter distribution in MedSynth after filtering to billable codes (n=1,926). Chapters are identified by their first character code.

The largest chapters (C: Neoplasms, S: Injury, M: Musculoskeletal) contain 165-201 codes, while the smallest (P: Perinatal) contains only 27. This variation affects within-chapter resolver complexity and contributes to performance differences across chapters.

## SOAP Structure and Extraction

MedSynth notes follow SOAP structure with clearly demarcated sections:

- **Subjective**: Patient-reported symptoms, history, and concerns
- **Objective**: Physical examination findings, vital signs, laboratory results
- **Assessment**: Diagnostic impressions and clinical reasoning
- **Plan**: Treatment recommendations, follow-up instructions

The Assessment section typically contains explicit diagnostic statements that directly correspond to assigned ICD-10 codes. This structure motivates our APSO-Flip preprocessing, which prioritises Assessment content for transformer encoding.

SOAP section extraction achieved **100% success** across all 10,240 records using regex-based parsing. This structural consistency is a feature of the MedSynth generation pipeline—the Note Writer Agent was explicitly instructed to produce SOAP-formatted notes, and the Note Polisher Agent enforced correct section placement. This extraction rate should not be assumed to generalise to real EHR clinical notes, which exhibit greater structural variability including missing sections, non-standard headers, and free-form narratives.

## Token Distribution and Truncation Analysis

Bio_ClinicalBERT's 512-token context window necessitates truncation for longer clinical notes. We quantified truncation risk using a conservative `words × 1.3` token estimation heuristic calibrated against medical text:

| Field | Average Tokens | Records Exceeding 512 |
|-------|----------------|----------------------|
| Clinical Note | 565 | 64.7% |
| Dialogue | 932 | 100.0% |

The clinical note distribution peaks just above the 512-token threshold, meaning the majority of notes lose tail content under standard left-to-right truncation. This confirms that **truncation is the structural default for this dataset, not an edge case**. The finding motivates our APSO-Flip preprocessing: in standard SOAP ordering, the Assessment section appears third (after Subjective and Objective), placing diagnostic content in the truncation zone for most records.

The `words × 1.3` heuristic is conservative for Bio_ClinicalBERT specifically, which uses WordPiece tokenisation and tends to produce more tokens per word than the BPE tokenisers used to report counts in the MedSynth paper. True truncation rates are likely higher than reported here.

## Label Leakage Detection and Redaction

The MedSynth Note Writer Agent embedded ICD-10 code strings directly in clinical narratives, typically within the Assessment section (e.g., "Diagnosis: Pain in left knee (ICD-10: M25.562)"). This constitutes data leakage for any classification task—a model trained on unredacted notes can predict labels by pattern-matching the code string rather than reasoning from clinical language.

We detected explicit ICD-10 code strings using a regex pattern anchored on the ICD-10 structural convention (leading letter followed by exactly two digits), which correctly discriminates codes from common medical abbreviations (MCL, CBC, TSH, HPI) that share superficially similar alphanumeric structure. The pattern covers raw format (`M25562`), canonical format (`M25.562`), 3-character parent codes (`J18`), and placeholder-X codes (`T78.1XXA`).

| Section | Records with Leakage | Percentage |
|---------|---------------------|------------|
| Assessment | 2,855 | 27.9% |
| Plan | 238 | 2.3% |
| Objective | 89 | 0.9% |
| Subjective | 60 | 0.6% |
| **Total unique records** | **2,923** | **28.5%** |

Table 2: Distribution of ICD-10 code string leakage across SOAP sections. Section percentages sum to more than 28.5% because some records contain leakage in multiple sections.

The Assessment section accounts for 97.7% of all leaking records, confirming it as the primary leakage vector. The remaining 71.5% of records (7,317) contained no explicit ICD-10 code strings.

We replaced all detected code strings with `[REDACTED]` markers rather than deleting them, preserving sentence structure and making redaction auditable:

> *"Diagnosis: Pain in left knee (ICD-10: M25.562)"*  
> becomes:  
> *"Diagnosis: Pain in left knee (ICD-10: [REDACTED])"*

Post-redaction verification confirmed **zero remaining ICD-10 code strings** across all SOAP sections. The original unredacted notes are preserved in a separate column for audit traceability.

## Preprocessing and Quality Control

We implement zero-trust data ingestion with Pydantic schema validation. Each record is validated for non-empty note text, valid ICD-10 code format (letter followed by digits with optional decimal), presence of all SOAP sections, and code existence in the CDC FY2026 reference taxonomy. Records failing validation are excluded.

ICD-10 codes in the source data are stored without decimal points (e.g., `M25562`), reflecting the IQVIA PharMetrics Plus insurance claims format. We restore canonical CDC format by inserting a decimal after the third character (e.g., `M25562` → `M25.562`). All 10,240 codes passed structural validation (3-8 characters, leading alpha, alphanumeric content).

Two records contained empty dialogue fields (0.02%); these were retained with sentinel imputation (`[NO_TRANSCRIPT_AVAILABLE]`) rather than dropped. Clinical notes were complete for all records.

The final dataset comprises 9,660 records across 1,926 ICD-10 codes after filtering to billable codes only. For ICD-10 experiments (E-002 onwards), we use an 80/10/10 split yielding 7,728 training, 966 validation, and 966 test examples. For the ICD-3 baseline (E-001), we retain all 10,240 records to maximise training signal, yielding 8,192 training, 1,024 validation, and 1,024 test examples across 675 ICD-3 categories.


# Methodology

## APSO-Flip Preprocessing

Bio_ClinicalBERT's 512-token context window necessitates truncation for longer clinical notes. Standard left-to-right truncation discards content beyond position 512, potentially removing diagnostically critical information.

Clinical notes in SOAP format place Assessment and Plan sections after Subjective and Objective content. In longer notes, this ordering risks truncating the Assessment section—precisely the content most relevant for ICD coding. Yang et al. [@yang2023gpsoap] demonstrated that Assessment sections encode diagnostic conclusions that can be inferred from Subjective and Objective content, establishing their primacy for coding tasks. Our analysis confirms that 64.7% of MedSynth notes exceed the 512-token limit, with the Assessment section positioned in the truncation zone for most records.

We introduce APSO-Flip preprocessing, which reorders notes to position Assessment at Token 0:

```
Original SOAP order:  Subjective → Objective → Assessment → Plan
APSO-Flip order:      Assessment → Plan → Subjective → Objective
```

This reordering ensures that diagnostic content survives truncation while preserving all original text. The transformation is deterministic and reversible, introducing no information loss for notes within the context window. Records exceeding 512 tokens lose their Subjective and Objective tail content—clinically less critical for classification—but retain the coded diagnosis in the Assessment section.

Implementation uses pre-extracted SOAP section columns from the Pydantic-based parser (100% extraction success) rather than re-running regex on raw notes, ensuring consistency with validated extraction. The recomposed `apso_note` serves as the canonical model input, while the original SOAP-ordered note is preserved in a separate column for audit traceability.

The APSO-Flip does not reduce token count—the same content is present, just reordered. The transformation protects the Assessment from truncation by moving it to the front; it does not compress the note. After the flip, 64.8% of APSO notes still exceed 512 tokens, but the diagnostic signal is now guaranteed to survive truncation.

## Two-Stage Hierarchical Architecture

Our pipeline decomposes ICD-10 classification into two stages that exploit the code taxonomy's inherent structure.

### Stage 1: Chapter Router

The Stage-1 router classifies notes into one of 22 ICD-10 chapters based on the first character of the code (A, B, C, ..., Z). This reduces the initial classification from 1,926 classes to 22, enabling more reliable predictions with limited training data.

Architecture:
- **Encoder**: Bio_ClinicalBERT (110M parameters)
- **Classifier**: Linear layer mapping [CLS] embedding to 22 classes
- **Initialisation**: Fine-tuned from E-001 (ICD-3 classifier, see Section 5)
- **Training**: Cross-entropy loss, AdamW optimiser, linear warmup with cosine decay

The router achieves 95.4% accuracy on chapter prediction, providing reliable routing for Stage-2 resolution.

### Stage 2: Within-Chapter Resolvers

Stage-2 comprises 19 independent resolvers, one per represented ICD-10 chapter (3 chapters—H60-H95 (Ear/mastoid), O00-O9A (Pregnancy), and U00-U85 (Special codes)—have no representation in MedSynth). Each resolver performs classification within its chapter's code subset, reducing the effective label space from 1,926 to an average of ~100 codes per resolver.

Architecture (per resolver):
- **Encoder**: Bio_ClinicalBERT (110M parameters)
- **Classifier**: Linear layer mapping [CLS] embedding to chapter-specific code count
- **Initialisation**: Fine-tuned from E-002 (flat ICD-10 classifier)
- **Training**: Cross-entropy loss on chapter-filtered training subset

During inference, the Stage-1 prediction routes each note to its corresponding Stage-2 resolver, which outputs the final ICD-10 code prediction.

### Transfer Learning Chain

A critical finding of this work is that resolver initialisation determines hierarchical pipeline success. We establish a progressive transfer learning chain:

```
Bio_ClinicalBERT (pretrained)
        ↓
    E-001: ICD-3 classifier (675 classes, chapter-aware)
        ↓
    Stage-1 Router initialisation
        
Bio_ClinicalBERT (pretrained)
        ↓
    E-002: Flat ICD-10 classifier (1,926 classes)
        ↓
    Stage-2 Resolver initialisation (×19 resolvers)
```

This chain preserves learned representations across experiments. E-001 learns chapter-level distinctions useful for routing. E-002 learns fine-grained ICD-10 code representations from all training examples, providing Stage-2 resolvers with initialisation superior to random or pretrained weights.

## Training Configuration

All experiments use consistent hyperparameters:

- **Optimiser**: AdamW with weight decay 0.01
- **Learning rate**: 2e-5 with linear warmup (10% of steps) and linear decay to zero
- **Batch size**: 16
- **Epochs**: 20 for E-001; varies for subsequent experiments (see individual descriptions)
- **Maximum sequence length**: 512 tokens
- **Hardware**: Apple M4 Max with MPS acceleration, 128GB RAM (~18.5 samples/sec)

Training employs checkpoint selection based on validation Macro F1, with the best-performing checkpoint selected for final evaluation. All experiments are tracked in MLflow with SQLite backend for reproducibility. The classification head adds minimal parameters relative to the 110M pretrained base: a dropout layer (p=0.1) followed by a linear projection from the 768-dimensional [CLS] embedding to the target label space (518K parameters for 675-class ICD-3, 1.48M for 1,926-class ICD-10—less than 1.5% of total model size).

## Evaluation Metrics

We report:

- **Accuracy**: Top-1 exact match between predicted and true ICD-10 codes
- **Macro F1**: Unweighted mean of per-class F1 scores, treating all codes equally regardless of frequency
- **Top-5 Accuracy**: Whether the ground truth label appears in the model's top 5 predictions—a clinical utility metric simulating real-world coding assistance where a ranked list of candidate codes is presented to human coders
- **Chapter Routing Accuracy**: Stage-1 classification accuracy (hierarchical models only)
- **Within-Chapter Accuracy**: Stage-2 accuracy conditioned on correct routing (hierarchical models only)

Macro F1 is particularly important for evaluating rare code performance, as it weights all classes equally rather than being dominated by frequent codes. However, Macro F1 should be interpreted as a conservative lower bound: due to MedSynth's uniform sampling design, not all classes appear in evaluation splits. In E-001, only 489 of 675 ICD-3 classes (72.4%) appear in the test set; absent classes contribute F1 = 0.0 to the macro average, suppressing the reported metric.


# Experiments

We conduct a systematic series of experiments to isolate the contributions of hierarchical decomposition and transfer learning initialisation.

## E-001: ICD-3 Baseline

**Objective**: Establish a coarse-grained baseline and create initialisation weights for Stage-1 routing.

**Setup**: Flat classification of clinical notes into ICD-3 categories (first three characters of ICD-10 code, e.g., "E11" for Type 2 diabetes). This yields 675 classes with approximately 15 training examples per class (minimum 5, maximum 130). All 10,240 records are used, with an 80/10/10 stratified split yielding 8,192 training, 1,024 validation, and 1,024 test examples. Note that due to uniform sampling constraints, the validation/test split is random rather than stratified—245 of 675 ICD-3 categories have only 5 records total, insufficient for sklearn's stratified splitting requirement of ≥2 per class.

**Results**: 83.0% validation accuracy, 0.763 Macro F1, 91.5% Top-5 accuracy (20 epochs, best checkpoint at epoch 19). Test set performance closely matched validation (82.6% accuracy, 0.747 Macro F1, 92.0% Top-5 accuracy), confirming generalization rather than overfitting to the validation set during checkpoint selection.

**Training dynamics**: Extended training proved critical. Table 3a shows the progression:

| Epochs | Accuracy | Macro F1 | Top-5 Acc. | Improvement |
|--------|----------|----------|------------|-------------|
| 3 | 18.0% | 0.042 | 38.9% | — |
| 10 | 61.8% | 0.442 | 83.9% | 10.5× F1 |
| 20 | 83.0% | 0.763 | 91.5% | 18× F1 vs epoch 3 |

Table 3a: E-001 training progression across epochs.

The model spends early epochs learning broad ICD chapter distinctions, middle epochs resolving within-chapter family differences, and late epochs fine-tuning boundary cases between semantically adjacent categories. Stopping at 10 epochs would have significantly underestimated model capacity.

**Significance**: Strong ICD-3 performance demonstrates that Bio_ClinicalBERT can extract meaningful diagnostic signal from APSO-ordered clinical notes. Confusion matrix analysis on the test set revealed that 7 of the 10 most confused ICD-3 families were Z-chapter administrative codes (Z00, Z01, Z12, Z13, Z30, Z76, Z86)—codes sharing similar clinical language around routine visits, screenings, and health service encounters. The model confuses Z00 (General Examination) with Z01 (Special Examination) but does not confuse musculoskeletal codes with psychiatric codes. This semantic adjacency pattern—errors clustering within clinical chapters rather than across them—indicates the model learns broad chapter structure but struggles with fine-grained within-chapter discrimination, directly motivating our hierarchical decomposition. The trained encoder provides chapter-aware initialisation for Stage-1.

## E-002: Flat ICD-10 Classification

**Objective**: Establish the full ICD-10 baseline and create initialisation weights for Stage-2 resolvers.

**Setup**: Direct classification into 1,926 ICD-10 codes using Bio_ClinicalBERT with a single linear classifier head.

**Results**: 46.9% accuracy, 0.352 Macro F1.

**Significance**: This represents a strong baseline given the extreme low-resource setting (~4 examples per code). The model must distinguish among 1,926 classes with minimal supervision—substantially harder than ICD-3 classification. The trained encoder captures fine-grained ICD-10 distinctions that inform Stage-2 initialisation.

## E-003: Hierarchical with Fresh BERT Initialisation

**Objective**: Evaluate hierarchical decomposition with standard pretrained initialisation.

**Setup**: Two-stage pipeline where Stage-1 initialises from E-001 but Stage-2 resolvers initialise from pretrained Bio_ClinicalBERT weights (not E-002). Each resolver trains only on its chapter-filtered subset.

**Results**: 10.6% accuracy, 0.070 Macro F1.

**Significance**: Catastrophic failure despite the architectural decomposition. Within-chapter accuracy drops to 11.1%, indicating that resolvers cannot learn meaningful distinctions when training from scratch on small chapter subsets (averaging ~500 examples per resolver across ~100 codes, yielding ~5 examples per code—even more constrained than the full dataset).

## E-004a: Hierarchical with E-002 Initialisation

**Objective**: Evaluate hierarchical decomposition with transfer-learned initialisation.

**Setup**: Identical architecture to E-003, but Stage-2 resolvers initialise from E-002 weights rather than pretrained Bio_ClinicalBERT.

**Results**: 66.7% accuracy, 0.551 Macro F1.

**Significance**: The same architecture that achieved 10.6% in E-003 now achieves 66.7%—a **56.1 percentage point improvement** attributable solely to initialisation. Within-chapter accuracy improves from 11.1% to 69.8% (6.3× improvement). This result establishes that transfer learning is not merely helpful but essential for hierarchical ICD-10 coding in low-resource settings.

## E-005a: Extended Training

**Objective**: Evaluate whether additional training epochs improve E-004a performance.

**Setup**: E-004a architecture with training extended from 10 to 20 epochs.

**Results**: 66.9% accuracy, 0.553 Macro F1.

**Significance**: Marginal improvement (+0.2pp accuracy) indicates that E-004a has reached its performance ceiling on MedSynth at the current architecture scale. Further gains likely require architectural modifications, additional data, or external knowledge sources.

## Summary of Results

| Experiment | Architecture | Initialisation | Val Accuracy | Test Accuracy | Macro F1 | Top-5 Acc. |
|------------|--------------|----------------|--------------|---------------|----------|------------|
| E-001 | Flat ICD-3 (675) | Bio_ClinicalBERT | 83.0% | 82.6% | 0.763 / 0.747 | 91.5% / 92.0% |
| E-002 | Flat ICD-10 (1,926) | Bio_ClinicalBERT | 46.9% | — | 0.352 | — |
| E-003 | Hierarchical | Stage-2: Bio_ClinicalBERT | 10.6% | — | 0.070 | — |
| E-004a | Hierarchical | Stage-2: E-002 | 66.7% | — | 0.551 | — |
| E-005a | Hierarchical (extended) | Stage-2: E-002 | 66.9% | — | 0.553 | — |

Table 3b: Summary of experimental results across all configurations. For E-001, validation/test metrics are shown separated by "/".

The contrast between E-003 and E-004a provides a controlled ablation: identical architectures with different initialisation yield 56.1pp accuracy difference. This isolates transfer learning as the critical factor, demonstrating that hierarchical decomposition is necessary but not sufficient for low-resource ICD-10 coding.


# Results

## Overall Performance

Our best model (E-005a) achieves 66.9% top-1 accuracy and 0.553 Macro F1 across 1,926 ICD-10 codes. The hierarchical pipeline outperforms flat classification (E-002) by 20.0 percentage points in accuracy and 0.201 in Macro F1, demonstrating substantial gains from architectural decomposition combined with transfer learning.

Stage-1 routing achieves 95.4% accuracy across 22 chapters, indicating that chapter-level classification is highly reliable. Stage-2 within-chapter resolution achieves 70.1% accuracy, representing the primary source of end-to-end error.

## Chapter-Level Analysis

Performance varies substantially across ICD-10 chapters, reflecting differences in chapter size, clinical language distinctiveness, and code similarity within chapters.

| Chapter | Codes | Routing Acc. | Within-Ch. Acc. | E2E Acc. |
|---------|-------|--------------|-----------------|----------|
| A (Infectious) | 145 | 97.2% | 72.3% | 70.3% |
| C (Neoplasms) | 201 | 98.1% | 68.4% | 67.1% |
| E (Endocrine) | 108 | 96.8% | 74.2% | 71.8% |
| I (Circulatory) | 152 | 97.5% | 71.8% | 70.0% |
| M (Musculoskeletal) | 165 | 94.2% | 69.1% | 65.1% |
| S (Injury) | 174 | 93.8% | 67.5% | 63.3% |
| Z (Health services) | 73 | 88.4% | 36.9% | 32.6% |

Table 4: Chapter-level performance breakdown for selected chapters. E2E Acc. = Routing Acc. × Within-Ch. Acc.

The Z-chapter (Factors influencing health status and contact with health services) presents particular difficulty, achieving only 32.6% end-to-end accuracy despite containing relatively few codes (73). This chapter covers administrative and circumstantial codes (e.g., "encounter for screening," "personal history of") that share similar clinical language patterns, making fine-grained discrimination challenging.

In contrast, chapters with clinically distinctive presentations (A: Infectious diseases, E: Endocrine/metabolic) achieve strong performance exceeding 70% accuracy. The variation suggests that within-chapter code similarity, rather than chapter size alone, determines classification difficulty.

## Error Analysis

We analyse errors from the E-005a model to identify systematic failure modes.

### Routing Errors (4.6% of test set)

Stage-1 misrouting occurs most frequently between clinically related chapters:

- **I ↔ R**: Circulatory symptoms misclassified as general symptoms/signs
- **M ↔ S**: Musculoskeletal conditions confused with injury codes
- **F ↔ G**: Mental disorders confused with nervous system conditions

These confusions reflect genuine clinical overlap—chest pain (R07) may indicate circulatory disease (I-chapter) or non-cardiac causes (R-chapter)—rather than model failure per se.

### Within-Chapter Errors (29.9% of correctly routed samples)

Within-chapter errors cluster in three patterns:

1. **Specificity errors**: Correct disease category but wrong specificity level (e.g., E11.9 "Type 2 diabetes without complications" predicted instead of E11.65 "Type 2 diabetes with hyperglycemia")

2. **Laterality/location errors**: Correct condition but wrong anatomical specification (e.g., M54.5 "Low back pain" vs M54.6 "Pain in thoracic spine")

3. **Z-chapter ambiguity**: Administrative codes with minimal distinguishing clinical features (e.g., Z23 "Encounter for immunization" vs Z00.00 "General adult medical examination")

The prevalence of specificity errors suggests that finer-grained distinctions require either more training data or architectural modifications that explicitly model code hierarchies beyond the chapter level.

## Comparison to Baselines

Direct comparison to prior work is complicated by differences in datasets (MIMIC-III vs MedSynth), code systems (ICD-9 vs ICD-10), and evaluation protocols. We contextualise our results against reported benchmarks, noting that direct performance comparison is inappropriate due to these fundamental differences:

| Model | Dataset | Codes | Macro F1 |
|-------|---------|-------|----------|
| CAML [@mullenbach2018caml] | MIMIC-III-full | 8,922 | 0.088 |
| LAAT [@vu2020laat] | MIMIC-III-full | 8,922 | 0.099 |
| JointLAAT [@vu2020laat] | MIMIC-III-full | 8,922 | 0.107 |
| CD-LAAT [@merchant2024ensemble] | MIMIC-III-full | 8,922 | 0.120 |
| GPsoap [@yang2023gpsoap] | MIMIC-III-full | 4,075 | 0.134 |
| **Ours (E-005a)** | MedSynth | 1,926 | **0.553** |

Table 5: Macro F1 comparison across ICD coding methods. Direct comparison is limited by dataset and code system differences; this table contextualises rather than ranks methods.

Our substantially higher Macro F1 (0.553 vs 0.088-0.134) reflects MedSynth's uniform sampling (5 records per code), which eliminates the long-tail distribution that depresses Macro F1 on MIMIC-III where thousands of codes appear rarely or never. Real clinical datasets present orders of magnitude more codes with severely imbalanced frequencies. The comparison contextualises our results within the broader literature but should not be interpreted as demonstrating superior methodology.


# Discussion

## The Critical Role of Transfer Learning

Our central finding is that hierarchical decomposition alone is insufficient for low-resource ICD-10 coding. The E-003 vs E-004a comparison provides a controlled ablation: identical two-stage architectures achieve 10.6% vs 66.9% accuracy depending solely on Stage-2 initialisation. This 6.3× improvement in within-chapter accuracy (11.1% → 70.1%) establishes that transfer learning is not merely beneficial but essential.

The failure mode of E-003 is instructive. When Stage-2 resolvers train from pretrained Bio_ClinicalBERT weights, they must learn ICD-10 distinctions from scratch using only chapter-filtered subsets. With approximately 500 training examples per resolver spread across ~100 codes, each code receives ~5 examples—an even more extreme low-resource setting than the full dataset. The resolvers cannot converge to meaningful decision boundaries.

E-002 initialisation resolves this by providing resolvers with encoder weights that already capture ICD-10 distinctions learned from the full training set. Fine-tuning adapts these representations to within-chapter discrimination while preserving the semantic understanding acquired from all training examples. The resolver's task shifts from learning ICD representations from scratch to refining pre-learned representations for chapter-specific classification.

This finding has implications for hierarchical classification in low-resource settings generally. Decomposition strategies that partition training data into smaller subsets risk undermining the sample efficiency gains they aim to achieve. Transfer learning from models trained on the full dataset provides a mechanism to preserve representation quality across the decomposition.

## APSO-Flip and Section Prioritisation

Our APSO-Flip preprocessing operationalises findings from Yang et al. [@yang2023gpsoap], who demonstrated that SOAP structure encodes clinical reasoning. By positioning Assessment content at Token 0, we ensure that diagnostic conclusions survive Bio_ClinicalBERT's 512-token truncation—critical given our finding that 64.7% of notes exceed this limit.

The effectiveness of this approach depends on Assessment sections containing sufficient diagnostic signal—an assumption validated by the SOAP documentation standard, which specifies that Assessment should contain "the clinician's interpretation of the patient's condition." For notes where Assessment is brief or uninformative, APSO-Flip provides less benefit.

Alternative approaches to context limitation include hierarchical attention over document segments [@dai2022transformerbased], Longformer-style sparse attention [@yang2022kept], or section-specific encoding with late fusion. These approaches trade computational cost for information retention. APSO-Flip offers a zero-cost preprocessing alternative when Assessment content is reliably informative.

Training dynamics provide empirical validation of the APSO-Flip strategy: smooth, monotonic loss descent without divergence suggests the Assessment-first ordering provides consistent, high-density diagnostic signal within the 512-token window. Noisy or truncated inputs would manifest as erratic gradients and slower loss reduction.

## Discrimination vs Representation

E-001 results establish that the fundamental challenge in ICD coding is fine-grained discrimination rather than clinical representation. Bio_ClinicalBERT successfully extracts diagnostic signal from APSO-ordered notes—achieving 92.0% Top-5 accuracy on the test set across 675 ICD-3 categories with only ~15 training examples per class—but struggles to resolve semantically adjacent codes when clinical language overlaps.

Confusion matrix analysis on the E-001 test set revealed this pattern concretely: 7 of the 10 most confused ICD-3 families were Z-chapter administrative codes (Z00, Z01, Z12, Z13, Z30, Z76, Z86). These codes share similar clinical language—routine visits, screenings, and health service encounters—making fine-grained discrimination particularly challenging. The model confuses Z00 (General Examination) with Z01 (Special Examination)—neighbouring administrative codes with similar wellness visit language—but does not confuse musculoskeletal codes with psychiatric or respiratory codes.

Qualitative inference testing reinforced these findings. On a cardiac case (STEMI presentation with ST elevation, elevated troponin, and chest pain radiating to the left arm), a 10-epoch model failed to surface I21 (Acute myocardial infarction) in its top-5 predictions, instead predicting M79 (soft tissue pain). The 20-epoch model correctly placed I21 at rank 5, with all top-5 predictions being cardiac-related codes (Z86, Z95, I48, R07, I21). This demonstrates that extended training improves both accuracy and clinical coherence—the model learns to route cardiac presentations to cardiac codes even when it cannot resolve the specific diagnosis.

This semantic adjacency pattern—errors clustering within clinical chapters rather than across them—indicates that the model learns broad categorical structure but lacks fine-grained within-chapter discrimination. The finding directly motivated our hierarchical decomposition: separating chapter-level routing (where the model already excels) from within-chapter resolution (where focused training is needed). The two-stage architecture operationalises this insight by giving Stage-2 resolvers a focused discrimination task within a reduced label space.

## Limitations

Several limitations constrain interpretation of our results:

**Synthetic data generation bias**: A fundamental limitation arises from the synthetic data generation process. The MedSynth Note Writer Agent was conditioned on the target ICD-10 code during generation, meaning clinical narratives were written *from* the label rather than the label being inferred *from* independent clinical observations. While we redact explicit code strings (28.5% of records), the diagnosis descriptions themselves (e.g., "Pain in left knee") remain in the text. These descriptions may provide stronger and more consistent signal than would appear in real clinical notes, where the same underlying condition might be described with greater lexical variability, ambiguity, or indirection. A real clinician documenting knee pain might write "patient reports discomfort in the lateral aspect of the left knee exacerbated by stair climbing" without ever using the phrase "pain in left knee"—yet both would map to the same ICD-10 code. This represents a potential source of inflated model performance that future work should investigate through more aggressive redaction strategies or evaluation on real clinical data.

**Uniform sampling**: MedSynth's five-records-per-code design eliminates class imbalance, creating an atypical evaluation setting. Real ICD distributions follow heavy-tailed patterns where common codes dominate and rare codes may have zero training examples. Our Macro F1 of 0.553 reflects balanced per-code performance but would likely degrade substantially on imbalanced real-world distributions. Models trained on uniformly sampled data will not learn real-world class frequency priors.

**`[REDACTED]` marker learnability**: The `[REDACTED]` marker itself appears in 28.5% of records and may become a learnable signal. Models with sufficient capacity could associate `[REDACTED]` with specific codes via surrounding context (e.g., "Pain in left knee (ICD-10: [REDACTED])" strongly suggests M25.562), partially circumventing the redaction's intent. Final evaluation should ideally use real clinical data without any redaction artifacts.

**Dialogue leakage not assessed**: The dialogue column has not been scanned or redacted. The MedSynth Dialogue Polisher Agent was designed to ensure all note content—including ICD-10 codes—appears in the dialogue. If dialogue is used as model input in future work, a separate leakage detection pass is required before training.

**Markdown formatting artifacts**: The `apso_note` column contains markdown formatting (`**` bold markers) from the Note Writer Agent's output. These tokens consume context window space without adding clinical signal. Future preprocessing should strip markdown before tokenisation to maximise clinical token density within the 512-token window.

**Limited code coverage**: Our 1,926 codes represent a fraction of the full ICD-10-CM taxonomy (>70,000 codes). Scaling to the complete code set introduces challenges in model capacity, training efficiency, and rare code handling that our experiments do not address.

**Evaluation class coverage**: MedSynth's uniform sampling (5 records per code) combined with stratification constraints means not all classes appear in evaluation splits. In E-001, only 489 of 675 ICD-3 classes (72.4%) appear in the test set; absent classes contribute F1 = 0.0 to the macro average. The reported Macro F1 values are therefore conservative lower bounds—true performance on seen classes is substantially higher (approximately 0.85 adjusted for coverage in E-001).

**Single dataset evaluation**: Without cross-dataset validation, we cannot assess generalisation to other clinical note formats, patient populations, or institutional documentation practices.

**Z-chapter difficulty**: Administrative codes (Z00-Z99) achieve only 32.6% accuracy, indicating that our approach struggles with codes distinguished primarily by administrative context rather than clinical presentation. Incorporating structured metadata (visit type, encounter reason) could address this limitation.

**SOAP extraction generalisability**: Our 100% SOAP extraction success rate reflects the structural consistency of synthetically generated notes. Real EHR clinical notes exhibit greater structural variability including missing sections, non-standard headers, and free-form narratives. The extraction patterns validated here would require substantial adaptation for real clinical deployment.

## Comparison to Alternative Approaches

Our classification-based hierarchical approach offers trade-offs relative to alternatives:

**Vs generation approaches** [@yang2023gpsoap]: Classification provides faster inference (single forward pass vs autoregressive decoding) and deterministic outputs but cannot generate codes unseen during training. For closed code sets like ICD-10, classification is appropriate; for evolving taxonomies, generation offers flexibility.

**Vs retrieval approaches** [@klotzman2024enhancing]: Classification learns task-specific representations end-to-end, while retrieval relies on pretrained embeddings. Klotzman's finding that general LLM embeddings outperform ClinicalBERT for retrieval suggests that larger pretrained models could benefit classification as well.

**Vs flat classification with label attention** [@vu2020laat; @merchant2024ensemble]: Label attention approaches handle all codes simultaneously, potentially capturing code co-occurrence patterns that our chapter-isolated resolvers miss. However, attention over 1,926 labels is computationally expensive and may dilute focus on chapter-specific distinctions.

**Vs knowledge-enhanced approaches** [@xie2019msattkg; @yuan2022msmn]: Incorporating ICD ontology structure or code synonyms could improve our resolver performance, particularly for rare codes. Our architecture is compatible with knowledge injection at both routing and resolution stages.

## Future Directions

Several directions could extend this work:

1. **Real clinical data evaluation**: Validating on MIMIC-IV or other clinical datasets with authentic documentation and ICD-10 codes is essential to assess whether findings transfer from synthetic to real clinical text.

2. **Aggressive redaction strategies**: Investigating whether masking diagnosis descriptions entirely (not just code strings) better simulates the inference challenge of real clinical coding, where diagnoses must be inferred from symptoms and findings rather than stated explicitly.

3. **Knowledge graph integration**: Incorporating ICD-10 hierarchy structure beyond chapter level, potentially using graph neural networks over the full code taxonomy.

4. **Multi-level hierarchy**: Extending beyond two stages to three or more levels (chapter → category → subcategory → full code), with careful management of the training data partitioning problem identified in E-003.

5. **Ensemble with retrieval**: Combining classification confidence with embedding similarity for uncertain predictions.

6. **Larger language models**: Evaluating whether larger clinical language models (GatorTron, BioGPT) provide better initialisation for hierarchical decomposition.


# Conclusion

We presented a two-stage hierarchical pipeline for ICD-10 coding from clinical notes, achieving 66.9% accuracy and 0.553 Macro F1 across 1,926 codes with approximately four training examples per code. Our approach combines hierarchical decomposition—routing through 22 ICD-10 chapters before within-chapter resolution—with transfer learning initialisation from flat classifiers trained on the full dataset.

The critical finding is that these components are jointly necessary: hierarchical architecture without transfer learning achieves only 10.6% accuracy (E-003), while the same architecture with E-002 initialisation achieves 66.9% (E-004a). This 56.1 percentage point improvement from initialisation alone establishes transfer learning as essential for hierarchical classification in low-resource settings.

Our APSO-Flip preprocessing ensures that Assessment sections—containing primary diagnostic content—survive transformer truncation, addressing the finding that 64.7% of clinical notes exceed Bio_ClinicalBERT's 512-token limit. Systematic redaction of ICD-10 code strings (present in 28.5% of records) prevents trivial label leakage while preserving clinical narrative structure.

Several important caveats apply. The synthetic MedSynth dataset's uniform sampling and generation process create evaluation conditions that differ substantially from real clinical coding. Diagnosis descriptions generated from labels may inflate performance relative to real notes where diagnoses must be inferred. The `[REDACTED]` markers and markdown artifacts introduce additional synthetic elements that would not appear in authentic clinical text. These limitations motivate validation on real clinical datasets as a priority for future work.

The combination of architectural decomposition, transfer learning, and section-aware preprocessing provides a template for clinical NLP tasks requiring fine-grained classification with limited supervision. The code and trained models are available at https://github.com/Sidney-Bishop/notes-to-icd10 to support reproducibility and extension.
