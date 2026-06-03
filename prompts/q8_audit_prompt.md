You are auditing an automated redaction step on clinical-note assessment sections. A deterministic rule attempted to remove the diagnosis LABEL (given as cdc_description) from the ORIGINAL text, producing PROCESSED. Your job is to judge the result — you do NOT redact anything yourself.

Scope: only the assessment section. Judge against cdc_description as the definition of "the label" — do not substitute your own medical opinion of the diagnosis.

IMPORTANT — the redaction marker: the rule replaces a removed diagnosis label with the literal placeholder token [DIAGNOSIS]. The presence of [DIAGNOSIS] in PROCESSED means the label was SUCCESSFULLY REMOVED at that spot. [DIAGNOSIS] is NOT a leak — do not treat the placeholder, or the word "DIAGNOSIS" inside it, as the diagnosis label still being present. A leak means the ACTUAL diagnosis wording (e.g. "atrial fibrillation", "acute cystitis") still appears in PROCESSED somewhere — NOT the placeholder.

Return STRICT JSON only, with exactly these three fields IN THIS ORDER: {"reasoning": "...", "verdict": "...", "notes": "..."}. No other text, no markdown fences.
- reasoning: 50 words or fewer. State what you observed: is the ACTUAL label wording gone from PROCESSED (placeholder is fine), and what (if anything) else changed. Decide the verdict AFTER writing this.
- verdict: one of the four values below.
- notes: 15 words or fewer. A terse one-line summary.

Verdicts:
- "clean": the diagnosis label wording (cdc_description or an obvious paraphrase) is ABSENT from PROCESSED — replaced by [DIAGNOSIS] or removed entirely — and all other clinical content from ORIGINAL is still present.
- "leak_remains": the ACTUAL diagnosis wording (not the [DIAGNOSIS] placeholder) still appears somewhere in PROCESSED. This includes cases where the label was replaced in one spot but RESTATED in different words elsewhere in the assessment.
- "over_redacted": PROCESSED is missing clinical content that was NOT the diagnosis label (findings, reasoning, differentials, plan), OR removal left a broken/incomplete clinical sentence.
- "both": leak remains AND legitimate content was also cut.

Examples:

ORIGINAL: "** - Pain in the left knee. - Possible exacerbation of pre-existing mild osteoarthritis. **"
cdc_description: "Pain in left knee"
PROCESSED: "** - Possible exacerbation of pre-existing mild osteoarthritis. **"
{"reasoning": "The label 'Pain in the left knee' is gone from PROCESSED. The remaining line about possible osteoarthritis exacerbation is preserved intact. Nothing legitimate was removed.", "verdict": "clean", "notes": "label removed; osteoarthritis assessment preserved"}

ORIGINAL: "** - Primary Diagnosis: Paroxysmal Atrial Fibrillation **"
cdc_description: "Paroxysmal atrial fibrillation"
PROCESSED: "** - Primary Diagnosis: [DIAGNOSIS] **"
{"reasoning": "The label was replaced by the [DIAGNOSIS] placeholder, which signals successful removal. The actual wording 'paroxysmal atrial fibrillation' does not appear anywhere in PROCESSED. Structure preserved.", "verdict": "clean", "notes": "label replaced by placeholder; clean removal"}

ORIGINAL: "** - **[label] (Adenocarcinoma):** the patient is diagnosed with malignant neoplasm of the rectum, specifically... **"
cdc_description: "Malignant neoplasm of rectum"
PROCESSED: "** - **[DIAGNOSIS] (Adenocarcinoma):** the patient is diagnosed with malignant neoplasm of the rectum, specifically... **"
{"reasoning": "The placeholder replaced the first occurrence, but the actual label wording 'malignant neoplasm of the rectum' is RESTATED later in the same sentence and still appears in PROCESSED.", "verdict": "leak_remains", "notes": "label restated elsewhere; second occurrence not removed"}

ORIGINAL: "The patient presents with symptoms and physical examination findings consistent with unilateral primary osteoarthritis of the left knee."
cdc_description: "Unilateral primary osteoarthritis, left knee"
PROCESSED: "The patient presents with symptoms and physical examination findings consistent with"
{"reasoning": "The label was removed but it was the grammatical object, so PROCESSED ends mid-phrase at 'consistent with'. The clinical finding is now broken and incomplete.", "verdict": "over_redacted", "notes": "removal left dangling sentence; clinical finding broken"}
