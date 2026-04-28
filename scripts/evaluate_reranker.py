import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import polars as pl
from src.graph_reranker import GraphReranker

df = pl.read_parquet('data/gold/medsynth_gold_augmented.parquet')
# take Z notes not in your train file
train_ids = set(pl.read_parquet('data/gold/medsynth_train.parquet')['id'].to_list())
val = df.filter(
    pl.col('standard_icd10').str.starts_with('Z') &
    (~pl.col('id').is_in(train_ids))
).sample(200, seed=42)

rr = GraphReranker(graph_dir=Path('data/graph/train_only'))
rr.load()

rescued = 0
for row in val.iter_rows(named=True):
    true = row['standard_icd10']
    cands = [(true,0.55),('Z00.00',0.52),('Z01.818',0.50),('Z23',0.48),('Z79.899',0.45)]
    out = rr.rerank(row['apso_note'], cands)
    if out[0].code == true:
        rescued += 1

print(f'Graph kept correct code on top: {rescued}/200')