#!/usr/bin/env python3
"""
build_graph.py — ICD-10 Knowledge Graph Construction
=====================================================
Builds a knowledge graph that links:
  - Clinical concepts (UMLS CUIs) extracted from MedSynth notes
  - ICD-10 codes and their hierarchical relationships
  - Co-occurrence patterns between codes
  - Code confusability edges (codes that share many concepts)

The graph is used at inference time to re-rank low-confidence
predictions by querying concept-to-code affinities.

Output
------
    data/graph/icd10_knowledge_graph.pkl   — NetworkX graph
    data/graph/code_concept_index.json     — code → top UMLS CUIs
    data/graph/concept_icd_index.json      — CUI → ICD-10 codes

Usage
-----
    uv run python scripts/build_graph.py
    uv run python scripts/build_graph.py --chapters Z O --sample 500
    uv run python scripts/build_graph.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import config  # noqa: E402

GRAPH_DIR = PROJECT_ROOT / "data" / "graph"
BANNER = "=" * 62


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

def clean_note(text: str) -> str:
    """Strip markdown formatting and redaction markers for NER."""
    text = re.sub(r'\*+', '', text)           # remove **bold**
    text = re.sub(r'\[REDACTED\]', '', text)  # remove redaction markers
    text = re.sub(r'\n+', ' ', text)          # flatten newlines
    text = re.sub(r'\s+', ' ', text)          # normalise whitespace
    return text.strip()


# ---------------------------------------------------------------------------
# NER + UMLS linking
# ---------------------------------------------------------------------------

def load_nlp():
    """Load scispacy NER pipeline with UMLS entity linker."""
    import spacy
    import scispacy  # noqa: F401
    from scispacy.linking import EntityLinker  # noqa: F401

    print("   Loading NER model (en_ner_bc5cdr_md)...")
    nlp = spacy.load("en_ner_bc5cdr_md")

    print("   Loading UMLS linker (cached after first run)...")
    nlp.add_pipe(
        "scispacy_linker",
        config={"resolve_abbreviations": True, "linker_name": "umls"},
    )
    return nlp


def extract_concepts(
    nlp,
    text: str,
    min_score: float = 0.85,
    max_per_entity: int = 3,
) -> list[dict]:
    """
    Extract UMLS concepts from clinical text.
    Returns list of {cui, name, score, entity_text} dicts.
    """
    doc = nlp(clean_note(text)[:1000])  # cap at 1000 chars for speed
    linker = nlp.get_pipe("scispacy_linker")
    concepts = []
    seen_cuis = set()

    for ent in doc.ents:
        for cui, score in ent._.kb_ents[:max_per_entity]:
            if score < min_score or cui in seen_cuis:
                continue
            entity_info = linker.kb.cui_to_entity[cui]
            concepts.append({
                "cui":         cui,
                "name":        entity_info.canonical_name,
                "score":       round(float(score), 3),
                "entity_text": ent.text,
                "label":       ent.label_,
            })
            seen_cuis.add(cui)

    return concepts


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph(
    df: pl.DataFrame,
    nlp,
    chapters: list[str] | None,
    sample: int | None,
) -> tuple:
    """
    Build the knowledge graph from the gold layer.

    Returns:
        G               — NetworkX DiGraph
        code_concepts   — {icd10_code: {cui: weight}}
        concept_codes   — {cui: {icd10_code: weight}}
    """
    import networkx as nx

    # Filter to billable records
    billable = df.filter(pl.col("code_status") == "billable")

    if chapters:
        billable = billable.filter(
            pl.col("standard_icd10").str.slice(0, 1).is_in(chapters)
        )

    if sample:
        billable = billable.sample(min(sample, len(billable)), seed=42)

    codes = billable["standard_icd10"].unique().to_list()
    print(f"   Processing {len(billable):,} records across {len(codes)} codes...")

    # --- Step 1: Extract concepts per code ---
    code_concepts: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    concept_codes: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))

    t0 = time.time()
    for i, row in enumerate(billable.iter_rows(named=True)):
        code = row["standard_icd10"]
        text = row["apso_note"] or row["assessment"] or ""
        if not text.strip():
            continue

        concepts = extract_concepts(nlp, text)
        for c in concepts:
            code_concepts[code][c["cui"]] += c["score"]
            concept_codes[c["cui"]][code] += c["score"]

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            remaining = (len(billable) - i - 1) / rate
            print(f"   [{i+1:,}/{len(billable):,}] "
                  f"{rate:.1f} notes/s | "
                  f"~{remaining/60:.1f} min remaining")

    print(f"\n   Extracted concepts from {len(billable):,} notes in "
          f"{(time.time()-t0)/60:.1f} min")
    print(f"   Unique codes with concepts: {len(code_concepts)}")
    print(f"   Unique UMLS concepts found: {len(concept_codes)}")

    # --- Step 2: Build NetworkX graph ---
    G = nx.DiGraph()

    # Add ICD-10 code nodes
    for code in codes:
        chapter = code[0]
        G.add_node(code, node_type="icd10", chapter=chapter)

    # Add ICD-10 hierarchy edges (parent → child)
    for code in codes:
        # Add 3-char parent if it exists and differs
        parent = code[:3]
        if parent != code and parent in codes:
            G.add_edge(parent, code, edge_type="hierarchy", weight=1.0)
        # Add chapter node
        chapter = code[0]
        if not G.has_node(chapter):
            G.add_node(chapter, node_type="chapter")
        G.add_edge(chapter, code, edge_type="chapter_member", weight=1.0)

    # Add concept nodes and concept→code edges
    for cui, code_weights in concept_codes.items():
        G.add_node(cui, node_type="concept")
        for code, weight in code_weights.items():
            # Normalise weight by number of records for this code
            n_records = len(billable.filter(pl.col("standard_icd10") == code))
            normalised = weight / max(n_records, 1)
            G.add_edge(cui, code,
                       edge_type="concept_to_code",
                       weight=round(normalised, 4))
            G.add_edge(code, cui,
                       edge_type="code_to_concept",
                       weight=round(normalised, 4))

    # Add code confusability edges (codes that share top concepts)
    print("\n   Computing code confusability edges...")
    codes_with_concepts = list(code_concepts.keys())
    confusability_added = 0
    for i, code_a in enumerate(codes_with_concepts):
        top_a = set(list(sorted(
            code_concepts[code_a], key=code_concepts[code_a].get, reverse=True
        ))[:10])

        for code_b in codes_with_concepts[i+1:]:
            if code_a[0] != code_b[0]:  # only within same chapter
                continue
            top_b = set(list(sorted(
                code_concepts[code_b], key=code_concepts[code_b].get, reverse=True
            ))[:10])

            overlap = len(top_a & top_b)
            if overlap >= 3:
                similarity = overlap / len(top_a | top_b)
                G.add_edge(code_a, code_b,
                           edge_type="confusable",
                           weight=round(similarity, 3))
                G.add_edge(code_b, code_a,
                           edge_type="confusable",
                           weight=round(similarity, 3))
                confusability_added += 1

    print(f"   Added {confusability_added} confusability edges")
    print(f"   Graph: {G.number_of_nodes()} nodes, "
          f"{G.number_of_edges()} edges")

    return G, dict(code_concepts), dict(concept_codes)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build ICD-10 knowledge graph from MedSynth + UMLS.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--gold-path", type=Path, default=None,
                        help="Gold layer parquet (default: medsynth_gold_augmented.parquet)")
    parser.add_argument("--chapters", nargs="+", default=None,
                        help="Limit to specific chapters (e.g. --chapters Z O). "
                             "Default: all chapters.")
    parser.add_argument("--sample", type=int, default=None,
                        help="Process a random sample of N records (for testing).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Load data and NLP pipeline but don't build graph.")
    args = parser.parse_args()

    print(f"\n{BANNER}")
    print(f"  build_graph.py — ICD-10 Knowledge Graph")
    print(f"  Chapters:  {', '.join(args.chapters) if args.chapters else 'all'}")
    print(f"  Sample:    {args.sample or 'all records'}")
    print(f"  Dry-run:   {args.dry_run}")
    print(BANNER)

    # Resolve gold path
    if args.gold_path:
        gold_path = args.gold_path
    else:
        gold_dir = config.resolve_path("data", "gold")
        augmented = gold_dir / "medsynth_gold_augmented.parquet"
        gold_path = augmented if augmented.exists() else sorted(
            gold_dir.glob("*.parquet"))[-1]

    print(f"\n   Loading gold layer: {gold_path.name}")
    df = pl.read_parquet(gold_path)
    print(f"   Loaded: {len(df):,} records")

    print("\n   Loading NLP pipeline...")
    nlp = load_nlp()
    print("   ✅ NLP pipeline ready")

    if args.dry_run:
        # Test on 3 records
        sample = df.filter(pl.col("code_status") == "billable").head(3)
        for row in sample.iter_rows(named=True):
            concepts = extract_concepts(nlp, row["apso_note"] or "")
            print(f"\n   {row['standard_icd10']}: {len(concepts)} concepts")
            for c in concepts[:3]:
                print(f"     {c['cui']} | {c['name']} | {c['score']:.2f}")
        print("\n[DRY RUN complete]")
        return

    # Build graph
    print("\n── Building graph ──────────────────────────────────────")
    G, code_concepts, concept_codes = build_graph(
        df, nlp, args.chapters, args.sample
    )

    # Save outputs
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)

    graph_path = GRAPH_DIR / "icd10_knowledge_graph.pkl"
    with open(graph_path, "wb") as f:
        pickle.dump(G, f)
    print(f"\n   💾 Graph saved: {graph_path}")

    # Save indices as JSON (top-20 concepts per code)
    code_idx = {
        code: dict(sorted(cuis.items(), key=lambda x: x[1], reverse=True)[:20])
        for code, cuis in code_concepts.items()
    }
    concept_idx = {
        cui: dict(sorted(codes.items(), key=lambda x: x[1], reverse=True)[:20])
        for cui, codes in concept_codes.items()
    }

    with open(GRAPH_DIR / "code_concept_index.json", "w") as f:
        json.dump(code_idx, f, indent=2)
    with open(GRAPH_DIR / "concept_icd_index.json", "w") as f:
        json.dump(concept_idx, f, indent=2)

    print(f"   💾 Indices saved: {GRAPH_DIR}")
    print(f"\n{BANNER}")
    print(f"  ✅ Graph complete")
    print(f"  Nodes:   {G.number_of_nodes():,}")
    print(f"  Edges:   {G.number_of_edges():,}")
    print(f"  Codes:   {len(code_concepts)}")
    print(f"  Concepts:{len(concept_codes)}")
    print(BANNER)
    print(f"""
Next — test the graph reranker:

  uv run python scripts/build_graph.py --dry-run

Then integrate with inference:
  src/graph_reranker.py
""")


if __name__ == "__main__":
    main()