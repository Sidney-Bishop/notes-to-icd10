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
----------
    predict(note, top_k=5, experiment_name="E-005a_...")
        Single-note end-to-end prediction. Returns codes + scores.

    HierarchicalPredictor
        Class-based interface for repeated inference — loads all models
        once and reuses them across calls.
"""

import json
import torch
import numpy as np
from pathlib import Path
from typing import Optional
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.config import config
from src.preprocessing import prepare_inference_input

# ---------------------------------------------------------------------------
# Registry path helpers — mirrors the path logic from the notebooks
# ---------------------------------------------------------------------------

def _registry_base() -> Path:
    return config.resolve_path("outputs", "evaluations") / "registry"


def _stage1_model_path(stage1_experiment: str = "E-003_Hierarchical_ICD10") -> Path:
    """
    Stage-1 router lives under the E-003 experiment directory (not registry),
    at: outputs/evaluations/{E003_name}/stage1/model/
    """
    return (
        config.resolve_path("outputs", "evaluations")
        / stage1_experiment
        / "stage1"
        / "model"
    )


def _stage2_dir(experiment_name: str) -> Path:
    """
    Stage-2 resolvers live under the experiment's own eval directory:
    outputs/evaluations/{experiment_name}/stage2/
    """
    return (
        config.resolve_path("outputs", "evaluations")
        / experiment_name
        / "stage2"
    )


# ---------------------------------------------------------------------------
# Chapters that have no trained resolver — use fallback prediction
# (mirrors SKIP_CHAPTERS from notebooks 04 and 05)
# ---------------------------------------------------------------------------
_SKIP_CHAPTERS = {"U", "P", "Q"}

# Default fallback codes for skip chapters, populated lazily from registry
_SKIP_CHAPTER_DEFAULTS_FILE = "skip_chapter_defaults.json"


class HierarchicalPredictor:
    """
    End-to-end two-stage ICD-10 predictor.

    Loads all models once on construction and reuses them across
    ``predict()`` calls — suitable for batch inference or interactive demos.

    Registry layout expected on disk
    ---------------------------------
    Stage-1 (router):
        outputs/evaluations/{stage1_experiment}/stage1/model/

    Stage-2 (resolvers, one per chapter):
        outputs/evaluations/{experiment_name}/stage2/{chapter}/model/
        outputs/evaluations/{experiment_name}/stage2/{chapter}/model/label_map.json
        outputs/evaluations/{experiment_name}/stage2/stage2_results.json
            → contains ``skip_chapters`` key with fallback predictions

    Parameters
    ----------
    experiment_name : str
        Name of the hierarchical experiment whose Stage-2 resolvers to load.
        Defaults to the best model (E-005a).
    stage1_experiment : str
        Name of the experiment providing the Stage-1 router.
        Defaults to E-003 (Stage-1 is shared across E-003/E-004a/E-005a).
    device : str or None
        Torch device string. Auto-detects MPS → CUDA → CPU if None.
    """

    def __init__(
        self,
        experiment_name: str = "E-005a_Hierarchical_ICD10_Extended",
        stage1_experiment: str = "E-003_Hierarchical_ICD10",
        device: Optional[str] = None,
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

        # --- Stage-1 ---
        stage1_path = _stage1_model_path(stage1_experiment)
        if not stage1_path.exists():
            raise FileNotFoundError(
                f"Stage-1 model not found at {stage1_path}\n"
                f"Run notebook 04 through Phase 6 first."
            )
        print(f"📥 Loading Stage-1 router from {stage1_experiment}...")
        self.stage1_tokenizer = AutoTokenizer.from_pretrained(str(stage1_path))
        self.stage1_model = AutoModelForSequenceClassification.from_pretrained(
            str(stage1_path)
        ).to(self.device)
        self.stage1_model.eval()

        # Build chapter id→label maps from Stage-1 model config
        self.id2chapter: dict[int, str] = {
            int(k): v
            for k, v in self.stage1_model.config.id2label.items()
        }
        self.chapter2id: dict[str, int] = {
            v: int(k) for k, v in self.id2chapter.items()
        }
        print(f"   ✅ Stage-1: {len(self.id2chapter)} chapters")

        # --- Stage-2 ---
        stage2_base = _stage2_dir(experiment_name)
        if not stage2_base.exists():
            raise FileNotFoundError(
                f"Stage-2 directory not found at {stage2_base}\n"
                f"Run notebook 05 through Phase 6 first."
            )

        print(f"📥 Loading Stage-2 resolvers from {experiment_name}...")
        self.stage2_models: dict[str, AutoModelForSequenceClassification] = {}
        self.stage2_tokenizers: dict[str, AutoTokenizer] = {}
        self.stage2_id2label: dict[str, dict[int, str]] = {}

        chapters_found = 0
        for ch_dir in sorted(stage2_base.iterdir()):
            if not ch_dir.is_dir():
                continue
            ch = ch_dir.name
            model_dir = ch_dir / "model"
            label_map_path = model_dir / "label_map.json"

            if not model_dir.exists() or not label_map_path.exists():
                continue

            with open(label_map_path) as f:
                lmap = json.load(f)

            ch_model = AutoModelForSequenceClassification.from_pretrained(
                str(model_dir)
            ).to(self.device)
            ch_model.eval()

            self.stage2_models[ch] = ch_model
            self.stage2_tokenizers[ch] = AutoTokenizer.from_pretrained(
                str(model_dir)
            )
            self.stage2_id2label[ch] = {
                int(k): v for k, v in lmap["id2label"].items()
            }
            chapters_found += 1

        print(f"   ✅ Stage-2: {chapters_found} resolvers loaded")

        # --- Skip-chapter fallbacks ---
        # Loaded from stage2_results.json saved during training
        self.skip_chapter_defaults: dict[str, Optional[str]] = {}
        results_path = stage2_base / "stage2_results.json"
        if results_path.exists():
            with open(results_path) as f:
                results = json.load(f)
            self.skip_chapter_defaults = results.get("skip_chapters", {})
        else:
            # Populate with None — will return "UNKNOWN" for skip chapters
            for ch in _SKIP_CHAPTERS:
                self.skip_chapter_defaults[ch] = None

        print(f"⚡ Predictor ready")

    # -----------------------------------------------------------------------

    def predict(self, note: str, top_k: int = 5) -> dict:
        """
        Predict ICD-10 codes for a single clinical note.

        Applies APSO-Flip + ICD-10 redaction preprocessing, routes through
        Stage-1, then resolves via the appropriate Stage-2 chapter resolver.

        Parameters
        ----------
        note : str
            Raw clinical note in SOAP format.
        top_k : int
            Number of top predictions to return. Applies only to the Stage-2
            resolver output. Stage-1 always returns the single top chapter.

        Returns
        -------
        dict with keys:
            codes  : list[str]  — ICD-10 codes, highest confidence first
            scores : list[float] — Corresponding softmax probabilities
            chapter : str       — Predicted ICD-10 chapter (Stage-1 output)
            stage2_source : str — "resolver", "fallback", or "fallback_no_model"
        """
        # Preprocessing — mirrors training pipeline exactly
        text = prepare_inference_input(note)

        # --- Stage-1: chapter routing ---
        s1_inputs = self.stage1_tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt",
        )
        # Drop token_type_ids if model doesn't use them (RoBERTa-based variants)
        s1_inputs = {
            k: v.to(self.device)
            for k, v in s1_inputs.items()
            if k in ["input_ids", "attention_mask", "token_type_ids"]
            and k in self.stage1_model.config.to_dict().get("architectures", [""])[0].lower()
            or k in ["input_ids", "attention_mask"]
        }

        with torch.no_grad():
            s1_logits = self.stage1_model(**s1_inputs).logits
        pred_chapter_id = int(torch.argmax(s1_logits, dim=-1).item())
        pred_chapter = self.id2chapter.get(pred_chapter_id, "UNKNOWN")

        # --- Stage-2: within-chapter resolution ---
        if pred_chapter in _SKIP_CHAPTERS:
            fallback_code = self.skip_chapter_defaults.get(pred_chapter, "UNKNOWN")
            return {
                "codes":        [fallback_code or "UNKNOWN"],
                "scores":       [1.0],
                "chapter":      pred_chapter,
                "stage2_source": "fallback",
            }

        if pred_chapter not in self.stage2_models:
            return {
                "codes":        ["UNKNOWN"],
                "scores":       [1.0],
                "chapter":      pred_chapter,
                "stage2_source": "fallback_no_model",
            }

        ch_model     = self.stage2_models[pred_chapter]
        ch_tokenizer = self.stage2_tokenizers[pred_chapter]
        ch_id2label  = self.stage2_id2label[pred_chapter]

        s2_inputs = ch_tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt",
        )
        s2_inputs = {
            k: v.to(self.device)
            for k, v in s2_inputs.items()
            if k in ["input_ids", "attention_mask", "token_type_ids"]
        }

        with torch.no_grad():
            s2_logits = ch_model(**s2_inputs).logits
            s2_probs  = torch.softmax(s2_logits, dim=-1).cpu().numpy()[0]

        top_k_actual = min(top_k, len(ch_id2label))
        top_indices  = np.argsort(s2_probs)[::-1][:top_k_actual]

        return {
            "codes":         [ch_id2label.get(int(i), "UNKNOWN") for i in top_indices],
            "scores":        [float(s2_probs[i]) for i in top_indices],
            "chapter":       pred_chapter,
            "stage2_source": "resolver",
        }


# ---------------------------------------------------------------------------
# Convenience function — matches the README's documented API exactly
# ---------------------------------------------------------------------------

def predict(
    note: str,
    top_k: int = 5,
    experiment_name: str = "E-005a_Hierarchical_ICD10_Extended",
    stage1_experiment: str = "E-003_Hierarchical_ICD10",
) -> dict:
    """
    End-to-end ICD-10 prediction for a single clinical note.

    Instantiates a fresh ``HierarchicalPredictor`` on each call.
    For repeated inference, instantiate ``HierarchicalPredictor`` directly
    to avoid reloading all models on every call.

    Parameters
    ----------
    note : str
        Raw clinical note in SOAP format.
    top_k : int
        Number of top ICD-10 predictions to return.
    experiment_name : str
        Hierarchical experiment whose Stage-2 resolvers to use.
    stage1_experiment : str
        Experiment providing the Stage-1 router.

    Returns
    -------
    dict with keys: codes, scores, chapter, stage2_source

    Example
    -------
    >>> from src.inference import predict
    >>> result = predict(note, top_k=5)
    >>> print(f"Top prediction: {result['codes'][0]} ({result['scores'][0]:.1%})")
    Top prediction: E11.65 (84.2%)
    """
    predictor = HierarchicalPredictor(
        experiment_name=experiment_name,
        stage1_experiment=stage1_experiment,
    )
    return predictor.predict(note, top_k=top_k)


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
            "codes":  [self.model.config.id2label[int(i)] for i in top_idx],
            "scores": [float(probs[i]) for i in top_idx],
        }


def predict_icd3(
    text: str,
    top_k: int = 5,
    experiment_name: str = "E-001_Baseline_ICD3",
) -> dict:
    """Flat single-model prediction. Legacy API for notebooks 01–02."""
    return ClinicalPredictor(experiment_name=experiment_name).predict(text, top_k=top_k)