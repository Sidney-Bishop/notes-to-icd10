from huggingface_hub import HfApi, create_repo
from pathlib import Path

REPO_ID = "SidneyBishop/notes-to-icd10"   # <-- your actual username
api = HfApi()
create_repo(REPO_ID, repo_type="dataset", exist_ok=True, private=False)

files = {
    "data/medsynth/icd10_notes.parquet": Path("data/medsynth/icd10_notes.parquet"),
    "data/reference/cdc_fy2026_icd10.parquet": Path("data/gold/cdc_fy2026_icd10.parquet"),
}

for hf_path, local in files.items():
    api.upload_file(
        path_or_fileobj=str(local),
        path_in_repo=hf_path,
        repo_id=REPO_ID,
        repo_type="dataset",
        commit_message=f"Add {local.name}"
    )
    print(f"✓ {local.name}")

print(f"Done: https://huggingface.co/datasets/{REPO_ID}")
