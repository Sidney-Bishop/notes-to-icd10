"""
inference.py — End-to-end hierarchical ICD-10 prediction pipeline.

Implements the two-stage inference pipeline established in notebook
05-Model_Hierarchical_ICD10_E002Init.ipynb (Phase 5):

    Stage-1: 22-way chapter router (loaded from E-003 registry)
        → predicts ICD-10 chapter (first character of code)

    Stage-2: Per-chapter resolver (loaded from E-004a/E-005a registry)
        → predicts full ICD-10 code within the routed chapter

Preprocessing mirrors the training pipeline exactly:
    ``prepare_inference_input()`` from src.preprocessing applies
    APSO-Flip and ICD-10 redaction before tokenisation.

Public API

    predict(note, top_k=5, experiment_name="E-005a_...")
        Single-note end-to-end prediction. Returns codes + scores.

    HierarchicalPredictor
        Class-based interface for repeated inference — loads all models
        once and reuses them across calls.
"""

import json
import warnings
import torch
import numpy as np
from pathlib import Path
from typing import Optional, Union
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from pydantic import BaseModel, field_validator, model_validator
from src.config import config
from src.paths import ExperimentPaths
from src.preprocessing import prepare_inference_input
from src.graph_reranker import GraphReranker


# ---------------------------------------------------------------------------
# R-005: Input validation
# ---------------------------------------------------------------------------

_MIN_WORDS_WARNING = 20    # fewer than this → warn, still run
_MAX_WORDS_WARNING = 400   # more than this → warn about truncation, still run


class ClinicalNoteInput(BaseModel):
    """
    Validates and normalises a clinical note before inference.

    Accepts either a raw string or a pre-validated instance.
    Backward compatible — HierarchicalPredictor.predict() accepts
    both plain str and ClinicalNoteInput.

    Validation rules
    ----------------
    - Note must be a non-empty string after stripping whitespace
    - Note must be valid UTF-8 (non-UTF-8 bytes are sanitised with a warning)
    - Notes shorter than 20 words produce a reliability warning
    - Notes longer than 400 words produce a truncation warning
      (Bio_ClinicalBERT silently truncates to 512 tokens)
    """

    note: str
    preprocessed: bool = False

    @field_validator("note", mode="before")
    @classmethod
    def sanitise_encoding(cls, v: str) -> str:
        """Sanitise non-UTF-8 characters with a warning."""
        if isinstance(v, bytes):
            v = v.decode("utf-8", errors="replace")
            warnings.warn(
                "ClinicalNoteInput: note was bytes, decoded with UTF-8 "
                "(replacement characters used for invalid bytes).",
                UserWarning,
                stacklevel=4,
            )
        return v

    @field_validator("note")
    @classmethod
    def note_must_be_non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError(
                "Clinical note cannot be empty. "
                "Provide a non-empty APSO-structured clinical note."
            )
        return v.strip()

    @model_validator(mode="after")
    def check_length(self) -> "ClinicalNoteInput":
        n_words = len(self.note.split())
        if n_words < _MIN_WORDS_WARNING:
            warnings.warn(
                f"ClinicalNoteInput: note is very short ({n_words} words). "
                f"Predictions may be unreliable. "
                f"A typical APSO note has at least {_MIN_WORDS_WARNING} words.",
                UserWarning,
                stacklevel=4,
            )
        elif n_words > _MAX_WORDS_WARNING:
            warnings.warn(
                f"ClinicalNoteInput: note is long ({n_words} words). "
                f"Bio_ClinicalBERT will silently truncate to 512 tokens — "
                f"content beyond ~400 words may be lost. "
                f"Consider using APSO ordering to ensure Assessment appears first.",
                UserWarning,
                stacklevel=4,
            )
        return self

# ---------------------------------------------------------------------------
# Registry path helper — kept for ClinicalPredictor (legacy flat models)
# ---------------------------------------------------------------------------

def _registry_base() -> Path:
    return config.resolve_path("outputs", "evaluations") / "registry"

# ---------------------------------------------------------------------------
# Chapters that have no trained resolver — use fallback prediction
# (mirrors SKIP_CHAPTERS from notebooks 04 and 05)
# ---------------------------------------------------------------------------
_SKIP_CHAPTERS = {"U", "P", "Q"}


class HierarchicalPredictor:
    """
    End-to-end two-stage ICD-10 predictor.

    Loads all models once on construction and reuses them across
    ``predict()`` calls — suitable for batch inference or interactive demos.

    Path resolution is delegated to ``ExperimentPaths`` (src/paths.py),
    which handles all three historical layout conventions automatically
    (FLAT / SINGLE / NESTED).

    Parameters

    experiment_name : str
        Name of the hierarchical experiment whose Stage-2 resolvers to load.
        Defaults to the best model (E-010).
    stage1_experiment : str
        Name of the experiment providing the Stage-1 router.
        Defaults to E-003 (Stage-1 is shared across E-003/E-004a/E-005a).
    device : str or None
        Torch device string. Auto-detects MPS → CUDA → CPU if None.
    use_reranker : bool
        Load and apply the graph reranker. Set False to skip graph loading
        for faster startup when reranking is not needed (e.g. evaluation
        runs on E-010, where E-011 showed minimal reranker impact).
    """

    def __init__(
        self,
        experiment_name: str = "E-005a_Hierarchical_ICD10_Extended",
        stage1_experiment: str = "E-003_Hierarchical_ICD10",
        device: Optional[str] = None,
        use_reranker: bool = True,
    ) -> None:

        # --- Device ---
        if device is not None:
            self.device = torch.device(device)
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        print(f"🔧 Device: {self.device}")

        # --- Path resolution ---
        paths = ExperimentPaths(experiment_name, stage1_experiment)

        # --- Stage-1 ---
        stage1_path = paths.stage1_model_dir()
        if stage1_path is None or not stage1_path.exists():
            raise FileNotFoundError(
                f"Stage-1 model not found under {paths._s1_base}\n"
                f"Run notebook 04 through Phase 6 first."
            )
        print(f"📥 Loading Stage-1 router from {stage1_experiment}...")
        self.stage1_tokenizer = AutoTokenizer.from_pretrained(str(stage1_path))
        self.stage1_model = AutoModelForSequenceClassification.from_pretrained(
            str(stage1_path)
        ).to(self.device)
        self.stage1_model.eval()

        # Load temperature (1.0 = uncalibrated)
        self.stage1_temperature = 1.0
        s1_temp_path = paths.stage1_temperature_existing()
        if s1_temp_path is not None:
            with open(s1_temp_path) as f:
                self.stage1_temperature = json.load(f).get("temperature", 1.0)
            print(f" 🌡 Stage-1 temperature: {self.stage1_temperature:.4f}")

        # Build chapter id→label maps from Stage-1 model config
        self.id2chapter: dict[int, str] = {
            int(k): v
            for k, v in self.stage1_model.config.id2label.items()
        }
        self.chapter2id: dict[str, int] = {
            v: int(k) for k, v in self.id2chapter.items()
        }
        print(f" ✅ Stage-1: {len(self.id2chapter)} chapters")

        # --- Stage-2 ---
        stage2_base = paths.stage2_base
        if not stage2_base.exists():
            raise FileNotFoundError(
                f"Stage-2 directory not found at {stage2_base}\n"
                f"Run notebook 05 through Phase 6 first."
            )

        print(f"📥 Loading Stage-2 resolvers from {experiment_name}...")
        self.stage2_models: dict[str, AutoModelForSequenceClassification] = {}
        self.stage2_tokenizers: dict[str, AutoTokenizer] = {}
        self.stage2_id2label: dict[str, dict[int, str]] = {}
        self.stage2_temperatures: dict[str, float] = {}

        chapters_found = 0
        for ch_dir in sorted(stage2_base.iterdir()):
            if not ch_dir.is_dir():
                continue
            ch = ch_dir.name
            hf_dir = paths.stage2_model_dir(ch)
            label_map_path = paths.stage2_label_map(ch)

            if hf_dir is None or label_map_path is None:
                continue

            with open(label_map_path) as f:
                lmap = json.load(f)

            ch_model = AutoModelForSequenceClassification.from_pretrained(
                str(hf_dir)
            ).to(self.device)
            ch_model.eval()

            self.stage2_models[ch] = ch_model
            self.stage2_tokenizers[ch] = AutoTokenizer.from_pretrained(str(hf_dir))
            self.stage2_id2label[ch] = {
                int(k): v for k, v in lmap["id2label"].items()
            }
            # Load temperature (defaults to 1.0 if not calibrated yet)
            temp_path = paths.stage2_temperature_existing(ch)
            if temp_path is not None:
                with open(temp_path) as f:
                    self.stage2_temperatures[ch] = json.load(f).get("temperature", 1.0)
            else:
                self.stage2_temperatures[ch] = 1.0
            chapters_found += 1

        print(f" ✅ Stage-2: {chapters_found} resolvers loaded")

        # --- Skip-chapter fallbacks ---
        # Loaded from stage2_results.json saved during training
        self.skip_chapter_defaults: dict[str, Optional[str]] = {}
        results_path = paths.stage2_results()
        if results_path.exists():
            with open(results_path) as f:
                results = json.load(f)
            self.skip_chapter_defaults = results.get("skip_chapters", {})
        else:
            # Populate with None — will return "UNKNOWN" for skip chapters
            for ch in _SKIP_CHAPTERS:
                self.skip_chapter_defaults[ch] = None

        # --- Graph reranker (optional) ---
        # E-011 showed the reranker has minimal impact on E-010's well-calibrated
        # resolvers. Set use_reranker=False to skip graph loading for faster startup.
        self.use_reranker = use_reranker
        if use_reranker:
            print("📥 Loading graph reranker...")
            self.reranker = GraphReranker(graph_dir=Path("data/graph"))
            self.reranker.load()
        else:
            self.reranker = None
            print("⏭  Graph reranker skipped (use_reranker=False)")

        print(f"⚡ Predictor ready")

    # -----------------------------------------------------------------------

    def predict(
        self,
        note: Union[str, ClinicalNoteInput],
        top_k: int = 5,
        preprocessed: bool = False,
    ) -> dict:
        """
        Predict ICD-10 codes for a single clinical note.

        Applies APSO-Flip + ICD-10 redaction preprocessing, routes through
        Stage-1, then resolves via the appropriate Stage-2 chapter resolver.

        Parameters

        note : str or ClinicalNoteInput
            Raw clinical note in SOAP format, or a pre-validated
            ClinicalNoteInput instance. Plain strings are accepted for
            backward compatibility and are validated automatically.
        top_k : int
            Number of top predictions to return. Applies only to the Stage-2
            resolver output. Stage-1 always returns the single top chapter.
        preprocessed : bool
            Set True if the note has already been APSO-flipped and redacted
            (e.g. evaluation directly on gold layer apso_note column).

        Returns

        dict with keys:
            codes : list[str] — ICD-10 codes, highest confidence first
            scores : list[float] — Corresponding softmax probabilities
            chapter : str — Predicted ICD-10 chapter (Stage-1 output)
            stage2_source : str — "resolver", "fallback", or "fallback_no_model"

        Raises

        pydantic.ValidationError
            If note is empty or not a string.
        """
        # Validate and normalise input — accepts both str and ClinicalNoteInput
        if not isinstance(note, ClinicalNoteInput):
            note = ClinicalNoteInput(note=note, preprocessed=preprocessed)

        # Preprocessing — mirrors training pipeline exactly.
        # If the caller has already preprocessed the text (e.g. evaluation on
        # Gold layer apso_note which is already APSO-flipped and redacted),
        # pass preprocessed=True to skip this step and avoid double-processing.
        text = note.note if note.preprocessed else prepare_inference_input(note.note)

        # --- Stage-1: chapter routing ---
        s1_inputs = self.stage1_tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt",
        )
        # Drop token_type_ids if the tokenizer doesn't produce it (RoBERTa-based variants).
        # Mirror the pattern in adapters.py: tokenizer.model_input_names is authoritative.
        _s1_allowed = set(self.stage1_tokenizer.model_input_names)
        s1_inputs = {
            k: v.to(self.device)
            for k, v in s1_inputs.items()
            if k in _s1_allowed
        }

        with torch.no_grad():
            s1_logits = self.stage1_model(**s1_inputs).logits
        s1_logits = s1_logits / self.stage1_temperature

        pred_chapter_id = int(torch.argmax(s1_logits, dim=-1).item())
        pred_chapter = self.id2chapter.get(pred_chapter_id, "UNKNOWN")

        # --- Stage-2: within-chapter resolution ---
        if pred_chapter in _SKIP_CHAPTERS:
            fallback_code = self.skip_chapter_defaults.get(pred_chapter, "UNKNOWN")
            return {
                "codes": [fallback_code or "UNKNOWN"],
                "scores": [1.0],
                "chapter": pred_chapter,
                "stage2_source": "fallback",
            }

        if pred_chapter not in self.stage2_models:
            return {
                "codes": ["UNKNOWN"],
                "scores": [1.0],
                "chapter": pred_chapter,
                "stage2_source": "fallback_no_model",
            }

        ch_model = self.stage2_models[pred_chapter]
        ch_tokenizer = self.stage2_tokenizers[pred_chapter]
        ch_id2label = self.stage2_id2label[pred_chapter]

        s2_inputs = ch_tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt",
        )
        _s2_allowed = set(ch_tokenizer.model_input_names)
        s2_inputs = {
            k: v.to(self.device)
            for k, v in s2_inputs.items()
            if k in _s2_allowed
        }

        with torch.no_grad():
            s2_logits = ch_model(**s2_inputs).logits
            T = self.stage2_temperatures.get(pred_chapter, 1.0)
            s2_probs = torch.softmax(s2_logits / T, dim=-1).cpu().numpy()[0]

        top_k_actual = min(top_k, len(ch_id2label))
        top_indices = np.argsort(s2_probs)[::-1][:top_k_actual]

        codes = [ch_id2label.get(int(i), "UNKNOWN") for i in top_indices]
        scores = [float(s2_probs[i]) for i in top_indices]

        # --- Graph-augmented reranking for low-confidence predictions ---
        top_confidence = scores[0] if scores else 0.0
        should_rerank = self.use_reranker and (
            (top_confidence < 0.7) or (pred_chapter == "Z")
        )
        if should_rerank:
            candidates = list(zip(codes, scores))
            reranked = self.reranker.rerank(text, candidates)

            threshold = 0.05 if pred_chapter == "Z" else 0.35
            if reranked and reranked[0].combined_score >= threshold:
                codes = [r.code for r in reranked]
                scores = [r.combined_score for r in reranked]
                return {
                    "codes": codes,
                    "scores": scores,
                    "chapter": pred_chapter,
                    "stage2_source": "graph_reranked",
                }

        return {
            "codes": codes,
            "scores": scores,
            "chapter": pred_chapter,
            "stage2_source": "resolver",
        }

# ---------------------------------------------------------------------------
# Convenience function — matches the README's documented API exactly
# ---------------------------------------------------------------------------

# Module-level cache so repeated predict() calls don't reload all models.
# Keyed by (experiment_name, stage1_experiment) so different experiments
# each get their own cached instance.
_PREDICTOR_CACHE: dict[tuple[str, str], "HierarchicalPredictor"] = {}


def predict(
    note: str,
    top_k: int = 5,
    experiment_name: str = "E-005a_Hierarchical_ICD10_Extended",
    stage1_experiment: str = "E-003_Hierarchical_ICD10",
) -> dict:
    """
    End-to-end ICD-10 prediction for a single clinical note.

    The underlying ``HierarchicalPredictor`` is cached after the first call
    so subsequent calls with the same experiment arguments reuse loaded models
    rather than reloading from disk. For a fresh load (e.g. after model
    weights change), instantiate ``HierarchicalPredictor`` directly.

    Parameters

    note : str
        Raw clinical note in SOAP format.
    top_k : int
        Number of top ICD-10 predictions to return.
    experiment_name : str
        Hierarchical experiment whose Stage-2 resolvers to use.
    stage1_experiment : str
        Experiment providing the Stage-1 router.

    Returns

    dict with keys: codes, scores, chapter, stage2_source

    Example

    >>> from src.inference import predict
    >>> result = predict(note, top_k=5)
    >>> print(f"Top prediction: {result['codes'][0]} ({result['scores'][0]:.1%})")
    Top prediction: E11.65 (84.2%)
    """
    cache_key = (experiment_name, stage1_experiment)
    if cache_key not in _PREDICTOR_CACHE:
        _PREDICTOR_CACHE[cache_key] = HierarchicalPredictor(
            experiment_name=experiment_name,
            stage1_experiment=stage1_experiment,
        )
    return _PREDICTOR_CACHE[cache_key].predict(note, top_k=top_k)

# ---------------------------------------------------------------------------
# Legacy single-model API — kept for backward compatibility with any
# code that imports predict_icd3 from notebooks 01-02
# ---------------------------------------------------------------------------

class ClinicalPredictor:
    """
    Single-model predictor for flat classifiers (E-001, E-002).

    Used by notebooks 01–03. For end-to-end hierarchical inference,
    use ``HierarchicalPredictor`` or ``predict()`` instead.
    """

    def __init__(self, experiment_name: str = "E-001_Baseline_ICD3") -> None:
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        model_path = (
            _registry_base() / experiment_name / "model"
        )
        if not model_path.exists():
            raise FileNotFoundError(
                f"No registry model found at {model_path}\n"
                f"Run Phase 10 (model registry promotion) in the relevant notebook first."
            )
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        self.model = AutoModelForSequenceClassification.from_pretrained(
            str(model_path)
        ).to(self.device)
        self.model.eval()
        print(f"✅ Model loaded: {experiment_name} | device: {self.device} | "
              f"labels: {self.model.config.num_labels}")

    def predict(self, text: str, top_k: int = 5) -> dict:
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding="max_length",
        ).to(self.device)

        with torch.no_grad():
            probs = torch.softmax(
                self.model(**inputs).logits, dim=1
            ).cpu().numpy()[0]

        top_idx = np.argsort(probs)[::-1][:top_k]
        return {
            "codes": [self.model.config.id2label[int(i)] for i in top_idx],
            "scores": [float(probs[i]) for i in top_idx],
        }

def predict_icd3(
    text: str,
    top_k: int = 5,
    experiment_name: str = "E-001_Baseline_ICD3",
) -> dict:
    """Flat single-model prediction. Legacy API for notebooks 01–02."""
    return ClinicalPredictor(experiment_name=experiment_name).predict(text, top_k=top_k)