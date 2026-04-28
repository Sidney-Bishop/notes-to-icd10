#!/usr/bin/env python3
"""
Extract DOID → ICD-10-CM mappings from doid.obo
Output: data/ontology/doid_icd10cm.json
"""
import re
from pathlib import Path
import json
from collections import defaultdict

obo_path = Path("data/ontology/doid.obo")
out_path = Path("data/ontology/doid_icd10cm.json")

doid_to_icds = defaultdict(list)
current_id = None

with open(obo_path, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line.startswith("id: DOID:"):
            current_id = line.split()[1]
        elif line.startswith("xref: ICD10CM:"):
            # format: xref: ICD10CM:Z00.00
            icd = line.split("ICD10CM:")[1].split()[0]
            if current_id:
                doid_to_icds[current_id].append(icd)

# flatten to icd → list of DOIDs (for reverse lookup)
icd_to_doids = defaultdict(list)
for doid, icds in doid_to_icds.items():
    for icd in icds:
        icd_to_doids[icd].append(doid)

# also build synonym map for fast text lookup
synonym_to_icd = {}
with open(obo_path, encoding="utf-8") as f:
    current_id = None
    name = None
    synonyms = []
    for line in f:
        if line.startswith("id: DOID:"):
            # save previous
            if current_id and name:
                for syn in [name] + synonyms:
                    syn_clean = syn.lower()
                    if current_id in doid_to_icds:
                        for icd in doid_to_icds[current_id]:
                            synonym_to_icd[syn_clean] = icd
            current_id = line.strip().split()[1]
            name = None
            synonyms = []
        elif line.startswith("name: "):
            name = line[6:].strip()
        elif line.startswith("synonym: "):
            # synonym: "annual wellness" EXACT []
            m = re.match(r'synonym: "(.+?)"', line)
            if m:
                synonyms.append(m.group(1))

# save
output = {
    "doid_to_icd10cm": dict(doid_to_icds),
    "icd10cm_to_doid": dict(icd_to_doids),
    "synonym_to_icd10cm": synonym_to_icd,
    "stats": {
        "doids_with_icd": len(doid_to_icds),
        "unique_icd10cm": len(icd_to_doids),
        "synonyms_mapped": len(synonym_to_icd)
    }
}

out_path.write_text(json.dumps(output, indent=2))
print(f"✓ Extracted {output['stats']['unique_icd10cm']} ICD-10-CM codes")
print(f"✓ Mapped {output['stats']['synonyms_mapped']} synonyms")
print(f"✓ Saved to {out_path}")