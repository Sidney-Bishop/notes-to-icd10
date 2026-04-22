"""
src/graph_reranker.py — Graph-Augmented Prediction Re-ranking

Re-ranks low-confidence encoder predictions using:
1. ICD-10 knowledge graph + UMLS concepts
2. High-precision phrase dictionary (for Z-codes and other administrative codes)
"""

from __future__ import annotations

import json
import logging
import pickle
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GRAPH_DIR = PROJECT_ROOT / "data" / "graph"
Z_DICT_PATH = PROJECT_ROOT / "data" / "ontology" / "z_dictionary.json"

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RerankedPrediction:
    code: str
    encoder_score: float
    graph_score: float
    combined_score: float
    matched_concepts: List[str] = field(default_factory=list)
    is_graph_boosted: bool = False

# ---------------------------------------------------------------------------
# Z-Phrase Matcher (high-precision dictionary)
# ---------------------------------------------------------------------------

class ZPhraseMatcher:
    """Fast, deterministic phrase → ICD-10 code matcher with word boundaries."""

    def __init__(self, dict_path: Path = Z_DICT_PATH):
        self.phrases: Dict[str, str] = {}
        self._patterns: List[Tuple[re.Pattern, str, str]] = []

        if not dict_path.exists():
            logger.warning(f"Z-dictionary not found at {dict_path}")
            return

        try:
            data = json.loads(dict_path.read_text())
            for phrase, meta in data.items():
                code = meta["code"]
                norm_phrase = phrase.lower().strip()
                self.phrases[norm_phrase] = code
                # Compile word-boundary pattern for robust matching
                pattern = re.compile(r'\b' + re.escape(norm_phrase) + r'\b')
                self._patterns.append((pattern, norm_phrase, code))
            logger.info(f"Z-dictionary loaded: {len(self.phrases)} phrases")
        except Exception as e:
            logger.error(f"Failed to load Z-dictionary: {e}")
            self.phrases = {}
            self._patterns = []

    def match(self, text: str) -> Tuple[str, str] | None:
        """Return (phrase, code) for first match, or None."""
        text_lower = text.lower()
        for pattern, phrase, code in self._patterns:
            if pattern.search(text_lower):
                return phrase, code
        return None

# ---------------------------------------------------------------------------
# GraphReranker
# ---------------------------------------------------------------------------

class GraphReranker:
    """
    Re-ranks low-confidence predictions using graph affinity and phrase dictionary.

    Parameters

    graph_weight : float
        Weight for graph score (0-1). Default 0.1 keeps encoder dominant.
    min_encoder_score : float
        Rerank only if top-1 < this threshold.
    min_concepts : int
        Minimum UMLS concepts required for graph scoring.
    z_boost : float
        Additive boost for dictionary matches. Default 0.18.
    enable_z_injection : bool
        If True, inject dictionary code when not in top-k candidates.
    """

    def __init__(
        self,
        graph_weight: float = 0.1,
        min_encoder_score: float = 0.7,
        min_concepts: int = 2,
        graph_dir: Path = GRAPH_DIR,
        z_boost: float = 0.18,
        enable_z_injection: bool = True,
    ):
        self.graph_weight = graph_weight
        self.min_encoder_score = min_encoder_score
        self.min_concepts = min_concepts
        self.graph_dir = Path(graph_dir)
        self.z_boost = z_boost
        self.enable_z_injection = enable_z_injection

        self._graph = None
        self._code_idx = None
        self._concept_idx = None
        self._nlp = None
        self._linker = None
        self._loaded = False
        self._z_matcher = ZPhraseMatcher()

    # ------------------------------------------------------------------
    # Lazy loading
    # ------------------------------------------------------------------

    def load(self) -> None:
        if self._loaded:
            return

        graph_path = self.graph_dir / "icd10_knowledge_graph.pkl"
        code_path = self.graph_dir / "code_concept_index.json"
        concept_path = self.graph_dir / "concept_icd_index.json"

        if not graph_path.exists():
            raise FileNotFoundError(
                f"Knowledge graph not found at {graph_path}. "
                f"Run: uv run python scripts/build_graph.py"
            )

        t0 = time.time()
        with open(graph_path, "rb") as f:
            self._graph = pickle.load(f)
        with open(code_path) as f:
            self._code_idx = json.load(f)
        with open(concept_path) as f:
            self._concept_idx = json.load(f)

        import spacy
        import scispacy # noqa: F401
        from scispacy.linking import EntityLinker # noqa: F401

        self._nlp = spacy.load("en_ner_bc5cdr_md")
        self._nlp.add_pipe(
            "scispacy_linker",
            config={"resolve_abbreviations": True, "linker_name": "umls"},
        )
        self._linker = self._nlp.get_pipe("scispacy_linker")
        self._loaded = True

        logger.info(
            f"GraphReranker loaded in {time.time()-t0:.1f}s "
            f"({self._graph.number_of_nodes()} nodes)"
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _clean(text: str) -> str:
        text = re.sub(r'\*+', '', text)
        text = re.sub(r'\[REDACTED\]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def _extract_concepts(self, text: str, min_score: float = 0.85) -> Dict[str, float]:
        doc = self._nlp(self._clean(text)[:1000])
        concepts: Dict[str, float] = {}
        for ent in doc.ents:
            for cui, score in ent._.kb_ents[:3]:
                if score >= min_score:
                    concepts[cui] = max(concepts.get(cui, 0.0), float(score))
        return concepts

    def _graph_affinity(self, code: str, note_concepts: Dict[str, float]) -> Tuple[float, List[str]]:
        if code not in self._code_idx or not note_concepts:
            return 0.0, []

        code_concepts = self._code_idx[code]
        matched = set(note_concepts) & set(code_concepts)
        if not matched:
            return 0.0, []

        score = sum(note_concepts[c] * code_concepts[c] for c in matched)
        normalised = score / max(len(code_concepts), 1)

        names = []
        for cui in list(matched)[:5]:
            try:
                names.append(self._linker.kb.cui_to_entity[cui].canonical_name)
            except KeyError:
                names.append(cui)
        return round(normalised, 4), names

    # ------------------------------------------------------------------
    # Re-ranking
    # ------------------------------------------------------------------

    def rerank(
        self,
        note: str,
        candidates: List[Tuple[str, float]],
    ) -> List[RerankedPrediction]:
        if not self._loaded:
            self.load()

        # 1. Check dictionary
        z_match = self._z_matcher.match(note)
        z_code = z_phrase = None
        if z_match:
            z_phrase, z_code = z_match

        # 2. Prepare candidate list (copy, don't mutate input)
        cand_list = list(candidates)
        if self.enable_z_injection and z_code and not any(c == z_code for c, _ in cand_list):
            cand_list.append((z_code, 0.01)) # minimal encoder score
            cand_list.sort(key=lambda x: x[1], reverse=True)

        # 3. Extract concepts
        note_concepts = self._extract_concepts(note)
        if len(note_concepts) < self.min_concepts and not z_code:
            return [
                RerankedPrediction(c, s, 0.0, s, [], False)
                for c, s in cand_list
            ]

        # 4. Score each candidate
        results = []
        for code, enc_score in cand_list:
            graph_score, matched = self._graph_affinity(code, note_concepts)
            combined = (1 - self.graph_weight) * enc_score + self.graph_weight * graph_score

            is_boosted = False
            if z_code and code == z_code:
                combined += self.z_boost
                is_boosted = True
                matched = matched + [f"Z-DICT:{z_phrase}"]

            results.append(RerankedPrediction(
                code=code,
                encoder_score=round(enc_score, 4),
                graph_score=round(graph_score, 4),
                combined_score=min(round(combined, 4), 1.0),
                matched_concepts=matched,
                is_graph_boosted=is_boosted or graph_score > enc_score,
            ))

        results.sort(key=lambda r: r.combined_score, reverse=True)
        return results

    def should_rerank(self, top1_confidence: float) -> bool:
        return top1_confidence < self.min_encoder_score

    def explain(self, note: str, code: str) -> dict:
        if not self._loaded:
            self.load()
        concepts = self._extract_concepts(note)
        affinity, matched = self._graph_affinity(code, concepts)
        return {
            "code": code,
            "graph_affinity": affinity,
            "matched_concepts": matched,
            "note_concept_count": len(concepts),
        }