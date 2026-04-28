#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))



#!/usr/bin/env python3
"""
evaluate_real_reranker.py — Real-world graph reranker evaluation

Uses the actual HierarchicalPredictor (with Stage-1 + Stage-2) to generate
candidates, then measures how often the graph rescues low-confidence predictions.
"""

import polars as pl
from pathlib import Path
from tqdm import tqdm
from src.inference import HierarchicalPredictor
from src.preprocessing import prepare_inference_input

def main():
    print("Loading predictor...")
    predictor = HierarchicalPredictor()

    # Load validation data - adjust path if needed
    gold_path = Path("data/gold/medsynth_gold_augmented.parquet")
    df = pl.read_parquet(gold_path)

    # Filter to Z-chapter for focused evaluation
    z_df = df.filter(pl.col("standard_icd10").str.starts_with("Z")).head(500)
    print(f"Evaluating on {len(z_df)} Z-chapter notes")

    results = []

    for row in tqdm(z_df.iter_rows(named=True), total=len(z_df)):
        true_code = row["standard_icd10"]
        note = row["apso_note"]

        # Get raw Stage-2 scores (before graph)
        text = prepare_inference_input(note)

        # Run through predictor to get baseline
        # We need to bypass the graph to see baseline confidence
        s1_inputs = predictor.stage1_tokenizer(text, truncation=True, padding="max_length",
                                               max_length=512, return_tensors="pt")
        s1_inputs = {k: v.to(predictor.device) for k, v in s1_inputs.items()
                     if k in ["input_ids", "attention_mask"]}

        import torch
        with torch.no_grad():
            s1_logits = predictor.stage1_model(**s1_inputs).logits
        s1_logits = s1_logits / predictor.stage1_temperature
        pred_chapter = predictor.id2chapter[int(torch.argmax(s1_logits).item())]

        if pred_chapter!= "Z" or "Z" not in predictor.stage2_models:
            continue

        # Stage-2 baseline
        ch_model = predictor.stage2_models["Z"]
        ch_tokenizer = predictor.stage2_tokenizers["Z"]
        s2_inputs = ch_tokenizer(text, truncation=True, padding="max_length",
                                 max_length=512, return_tensors="pt")
        s2_inputs = {k: v.to(predictor.device) for k, v in s2_inputs.items()
                     if k in ["input_ids", "attention_mask"]}

        with torch.no_grad():
            s2_logits = ch_model(**s2_inputs).logits
            T = predictor.stage2_temperatures.get("Z", 1.0)
            import numpy as np
            s2_probs = torch.softmax(s2_logits / T, dim=-1).cpu().numpy()[0]

        top_idx = np.argsort(s2_probs)[::-1][:5]
        codes = [predictor.stage2_id2label["Z"][int(i)] for i in top_idx]
        scores = [float(s2_probs[i]) for i in top_idx]

        baseline_top = codes[0]
        baseline_conf = scores[0]
        baseline_correct = (baseline_top == true_code)

        # Apply graph reranking if low confidence
        graph_rescued = False
        graph_correct = False
        combined_score = 0.0

        if baseline_conf < 0.7:
            candidates = list(zip(codes, scores))
            reranked = predictor.reranker.rerank(text, candidates)
            if reranked:
                graph_top = reranked[0].code
                combined_score = reranked[0].combined_score
                graph_correct = (graph_top == true_code)
                # Our current threshold
                graph_rescued = (combined_score >= 0.08 and graph_correct)

        results.append({
            "true_code": true_code,
            "baseline_top": baseline_top,
            "baseline_conf": baseline_conf,
            "baseline_correct": baseline_correct,
            "graph_rescued": graph_rescued,
            "graph_correct": graph_correct,
            "combined_score": combined_score,
        })

    # Analyze results
    import pandas as pd
    df_results = pd.DataFrame(results)

    low_conf = df_results[df_results["baseline_conf"] < 0.7]
    total_low = len(low_conf)
    rescued = low_conf["graph_rescued"].sum()
    baseline_correct_low = low_conf["baseline_correct"].sum()

    print("\n" + "="*60)
    print("REAL RERANKER EVALUATION RESULTS")
    print("="*60)
    print(f"Total Z notes evaluated: {len(df_results)}")
    print(f"Low-confidence (<0.7): {total_low} ({total_low/len(df_results)*100:.1f}%)")
    print(f"Baseline correct in low-conf: {baseline_correct_low} ({baseline_correct_low/total_low*100:.1f}%)")
    print(f"Graph rescued (≥0.08 + correct): {rescued} ({rescued/total_low*100:.1f}%)")
    print(f"Net gain: +{rescued - baseline_correct_low} correct predictions")
    print("="*60)

    # Save detailed results
    output_path = Path("outputs/evaluations/real_reranker_results.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(output_path, index=False)
    print(f"\nDetailed results saved to: {output_path}")

if __name__ == "__main__":
    main()