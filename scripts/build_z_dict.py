import requests, json, time
from pathlib import Path

# Start with your actual failing phrases + top Z codes from your data
phrases = [
    "general adult medical examination without abnormal findings",
    "general adult medical examination with abnormal findings",
    "encounter for general adult medical examination",
    "annual wellness exam",
    "annual wellness",
    "routine physical exam",
    "routine physical",
    "wellness visit",
    "health check",
    "preventive visit",
    "well child visit",
    "health supervision of infant",
    "preoperative examination",
    "pre-op clearance",
    "gastrostomy status",
    "gastrostomy tube check",
    "g-tube check",
    "encounter for attention to gastrostomy",
    "personal history of",
    "family history of",
    "screening for",
]

z_dict = {}
for phrase in phrases:
    try:
        url = f"https://clinicaltables.nlm.nih.gov/api/icd10cm/v3/search?terms={phrase}&maxList=3"
        r = requests.get(url, timeout=5).json()
        if r[0] > 0:
            # r[3] is [[code, title],...]
            code = r[3][0][0]
            title = r[3][0][1]
            z_dict[phrase.lower()] = {"code": code, "title": title}
            print(f"✓ {phrase:55} → {code} ({title[:40]})")
        else:
            print(f"✗ {phrase}")
    except Exception as e:
        print(f"Error {phrase}: {e}")
    time.sleep(0.15)

out = Path("data/ontology/z_dictionary.json")
out.write_text(json.dumps(z_dict, indent=2))
print(f"\nSaved {len(z_dict)} mappings to {out}")