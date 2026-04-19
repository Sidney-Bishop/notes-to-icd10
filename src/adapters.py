"""
adapters.py — ModelAdapter interface and EncoderAdapter implementation.

This module is the foundation of the script layer's model abstraction.
It decouples the data pipeline, training strategy, and evaluation layer
from any specific model implementation — so the training and evaluation
scripts never need to know which model is underneath.

Architecture
------------
The design reflects Decision 3 from Prj_Overview.md:

    Encoder models (ClinicalBERT, MedBERT, PubMedBERT)
        Produce a [CLS] vector → linear classification head → logits.
        Fast, memory-efficient, well-understood. The entire current pipeline
        is built on this pattern.

    Generative models (MedGemma, BioMistral, GPT-4o)
        Autoregressive, constrained decoding or structured output parsing.
        Slower but potentially stronger for low-confidence cases.

These are not competing approaches — they are complementary layers.
EncoderAdapter is the first concrete implementation; GenerativeAdapter
is a stub that documents the interface contract for future use.

Interface contract
------------------
Any concrete adapter must implement:

    train(train_dataset, val_dataset, cfg, output_dir) → TrainingResult
        Fine-tune the model on the provided datasets.

    predict(note, top_k) → PredictionResult
        Single-note end-to-end prediction (preprocessing included).

    predict_batch(notes, top_k) → list[PredictionResult]
        Batch prediction — defaults to sequential predict() calls,
        override for efficiency.

    save(output_dir)
        Persist model weights and config to output_dir.

    load(model_dir) → cls
        Class method. Restore a saved adapter from disk.

Result dataclasses
------------------
PredictionResult and TrainingResult are plain dataclasses — no
framework dependencies. This keeps the evaluation layer framework-agnostic.

Usage
-----
    from src.adapters import EncoderAdapter, PredictionResult

    adapter = EncoderAdapter.from_config(cfg)
    result  = adapter.predict(note, top_k=5)
    print(result.codes[0], result.scores[0], result.confidence)
"""

from __future__ import annotations

import json
import time
import torch
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)

from src.preprocessing import prepare_inference_input


# ==============================================================================
# Result dataclasses
# ==============================================================================

@dataclass
class PredictionResult:
    """
    Structured output from any ModelAdapter.predict() call.

    Attributes
    ----------
    codes : list[str]
        ICD codes ranked by confidence, highest first.
    scores : list[float]
        Softmax probability for each code. Sums to ≤1.0 (top-k subset).
    confidence : float
        A scalar summary of model certainty for the top-1 prediction.
        For encoders: top-1 softmax probability.
        For generative adapters: may reflect log-prob or a calibrated score.
        Used by HybridRouter to decide whether to escalate to a stronger model.
    top1_code : str
        Convenience accessor: codes[0].
    top1_score : float
        Convenience accessor: scores[0].
    metadata : dict[str, Any]
        Adapter-specific diagnostic information (e.g. chapter, stage2_source,
        model_name, latency_ms). Not used by the training or evaluation layer —
        preserved for logging and debugging.
    """
    codes:      list[str]
    scores:     list[float]
    confidence: float
    metadata:   dict[str, Any] = field(default_factory=dict)

    @property
    def top1_code(self) -> str:
        return self.codes[0] if self.codes else "UNKNOWN"

    @property
    def top1_score(self) -> float:
        return self.scores[0] if self.scores else 0.0

    def to_dict(self) -> dict:
        return {
            "codes":      self.codes,
            "scores":     self.scores,
            "confidence": self.confidence,
            "top1_code":  self.top1_code,
            "top1_score": self.top1_score,
            "metadata":   self.metadata,
        }


@dataclass
class TrainingResult:
    """
    Structured output from any ModelAdapter.train() call.

    Attributes
    ----------
    best_val_accuracy : float
        Validation accuracy at the best checkpoint.
    best_val_f1 : float
        Validation Macro F1 at the best checkpoint.
    best_epoch : int
        Epoch at which the best checkpoint was saved.
    training_history : list[dict]
        Per-epoch log: {epoch, train_loss, val_loss, val_accuracy, val_f1}.
    output_dir : Path
        Where the best model was saved.
    experiment_name : str
        Name tag for MLflow / audit logging.
    elapsed_seconds : float
        Wall-clock training time.
    """
    best_val_accuracy:  float
    best_val_f1:        float
    best_epoch:         int
    training_history:   list[dict]
    output_dir:         Path
    experiment_name:    str
    elapsed_seconds:    float = 0.0

    def summary(self) -> str:
        return (
            f"[{self.experiment_name}] "
            f"best epoch {self.best_epoch} — "
            f"val_acc={self.best_val_accuracy:.3f}, "
            f"val_f1={self.best_val_f1:.3f} "
            f"({self.elapsed_seconds:.0f}s)"
        )


# ==============================================================================
# Abstract base interface
# ==============================================================================

class ModelAdapter(ABC):
    """
    Abstract base class for all model adapters.

    Defines the contract that training, evaluation, and routing code depend on.
    Concrete subclasses implement the four abstract methods; everything else
    (batch prediction, confidence routing) is provided as defaults.
    """

    # ── Core interface ──────────────────────────────────────────────────────

    @abstractmethod
    def predict(self, note: str, top_k: int = 5) -> PredictionResult:
        """
        Predict ICD codes for a single clinical note.

        Implementations must apply the standard preprocessing pipeline
        (APSO-Flip + ICD-10 redaction via prepare_inference_input) before
        tokenisation, so the input to the model exactly matches training data.

        Parameters
        ----------
        note : str
            Raw clinical note in SOAP format.
        top_k : int
            Number of top predictions to return.

        Returns
        -------
        PredictionResult
            Ranked codes, scores, and a scalar confidence value.
        """
        ...

    @abstractmethod
    def train(
        self,
        train_dataset,
        val_dataset,
        cfg: dict,
        output_dir: Path,
    ) -> TrainingResult:
        """
        Fine-tune the model on the provided datasets.

        Parameters
        ----------
        train_dataset, val_dataset :
            HuggingFace Dataset objects with 'input_ids', 'attention_mask',
            'labels' columns.
        cfg : dict
            Flat hyperparameter dict from artifacts.yaml / CLI overrides.
            Keys: num_epochs, learning_rate, batch_size, warmup_ratio,
            weight_decay, max_length, experiment_name.
        output_dir : Path
            Directory to save the best checkpoint.

        Returns
        -------
        TrainingResult
            Training metrics and output location.
        """
        ...

    @abstractmethod
    def save(self, output_dir: Path) -> None:
        """Persist model weights, tokenizer, and label maps to output_dir."""
        ...

    @classmethod
    @abstractmethod
    def load(cls, model_dir: Path, device: Optional[str] = None) -> "ModelAdapter":
        """
        Restore a saved adapter from disk.

        Parameters
        ----------
        model_dir : Path
            Directory containing weights and config (as saved by save()).
        device : str or None
            Torch device string. Auto-detects if None.
        """
        ...

    # ── Default implementations ─────────────────────────────────────────────

    def predict_batch(
        self,
        notes: list[str],
        top_k: int = 5,
    ) -> list[PredictionResult]:
        """
        Batch prediction. Default: sequential predict() calls.

        Subclasses should override this with efficient batched tokenisation
        and inference when throughput matters (e.g. bulk evaluation).

        Parameters
        ----------
        notes : list[str]
            Raw clinical notes in SOAP format.
        top_k : int
            Number of top predictions per note.

        Returns
        -------
        list[PredictionResult]
            One result per input note, in the same order.
        """
        return [self.predict(note, top_k=top_k) for note in notes]


# ==============================================================================
# EncoderAdapter — wraps any HuggingFace sequence classification model
# ==============================================================================

class EncoderAdapter(ModelAdapter):
    """
    Adapter for HuggingFace encoder models (ClinicalBERT, MedBERT, PubMedBERT, etc.).

    This is the primary adapter for all current experiments (E-001 through E-005a).
    The model is a drop-in: change the model_name_or_path config value and
    the full training + evaluation pipeline runs unchanged.

    Flat (single-stage) use
    -----------------------
    Used directly for E-001 (ICD-3) and E-002 (flat ICD-10 baseline):

        adapter = EncoderAdapter.from_config(cfg)
        result  = adapter.predict(note, top_k=5)

    Hierarchical (two-stage) use
    ----------------------------
    Stage-1 and Stage-2 resolvers in the hierarchical pipeline are each an
    EncoderAdapter. The HierarchicalPredictor in inference.py wires them
    together — the adapters themselves are unaware of the hierarchy.

    Parameters
    ----------
    model :
        HuggingFace AutoModelForSequenceClassification.
    tokenizer :
        HuggingFace tokenizer.
    label2id : dict[str, int]
        ICD code string → integer class index.
    id2label : dict[int, str]
        Integer class index → ICD code string.
    device : torch.device
        Inference device.
    model_name : str
        Human-readable model identifier for logging/metadata.
    """

    def __init__(
        self,
        model,
        tokenizer,
        label2id:   dict[str, int],
        id2label:   dict[int, str],
        device:     torch.device,
        model_name: str = "unknown",
    ) -> None:
        self.model      = model
        self.tokenizer  = tokenizer
        self.label2id   = label2id
        self.id2label   = id2label
        self.device     = device
        self.model_name = model_name
        self.model.eval()

    # ── Construction helpers ────────────────────────────────────────────────

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        label2id:           dict[str, int],
        id2label:           dict[int, str],
        device:             Optional[str] = None,
    ) -> "EncoderAdapter":
        """
        Initialise from a HuggingFace model name or local path.

        Parameters
        ----------
        model_name_or_path : str
            HuggingFace Hub name (e.g. 'emilyalsentzer/Bio_ClinicalBERT')
            or a local directory path.
        label2id, id2label : dict
            Label mappings — required for the classification head.
        device : str or None
            Auto-detects MPS → CUDA → CPU if None.
        """
        resolved_device = cls._resolve_device(device)

        num_labels = len(label2id)
        tokenizer  = AutoTokenizer.from_pretrained(model_name_or_path)
        model      = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path,
            num_labels=num_labels,
            id2label={str(k): v for k, v in id2label.items()},
            label2id=label2id,
            ignore_mismatched_sizes=True,   # allows re-using pre-trained weights
                                            # with a new classification head
        ).to(resolved_device)

        return cls(
            model=model,
            tokenizer=tokenizer,
            label2id=label2id,
            id2label=id2label,
            device=resolved_device,
            model_name=str(model_name_or_path),
        )

    @classmethod
    def from_config(cls, cfg: dict) -> "EncoderAdapter":
        """
        Construct from a config dict.

        Expects keys: model_name_or_path, label2id, id2label.
        Optional: device.

        This is the canonical construction path for scripts/train.py —
        the model is a config value, not a hard-coded import.
        """
        return cls.from_pretrained(
            model_name_or_path=cfg["model_name_or_path"],
            label2id=cfg["label2id"],
            id2label=cfg["id2label"],
            device=cfg.get("device"),
        )

    @classmethod
    def load(cls, model_dir: Path, device: Optional[str] = None) -> "EncoderAdapter":
        """
        Restore a saved EncoderAdapter from a directory produced by save().

        Expects:
            model_dir/model/          — HuggingFace model + tokenizer
            model_dir/label_map.json  — {label2id: {...}, id2label: {...}}
        """
        model_dir   = Path(model_dir)
        hf_dir      = model_dir / "model"
        lmap_path   = model_dir / "label_map.json"

        if not hf_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {hf_dir}")
        if not lmap_path.exists():
            raise FileNotFoundError(f"Label map not found: {lmap_path}")

        with open(lmap_path) as f:
            lmap = json.load(f)

        label2id = lmap["label2id"]
        id2label = {int(k): v for k, v in lmap["id2label"].items()}

        resolved_device = cls._resolve_device(device)
        tokenizer = AutoTokenizer.from_pretrained(str(hf_dir))
        model     = AutoModelForSequenceClassification.from_pretrained(
            str(hf_dir)
        ).to(resolved_device)

        return cls(
            model=model,
            tokenizer=tokenizer,
            label2id=label2id,
            id2label=id2label,
            device=resolved_device,
            model_name=str(hf_dir),
        )

    # ── ModelAdapter interface ───────────────────────────────────────────────

    def predict(self, note: str, top_k: int = 5) -> PredictionResult:
        """
        Single-note prediction with APSO-Flip + redaction preprocessing.

        Confidence = top-1 softmax probability. Used by HybridRouter to
        decide whether to escalate to a generative model or human review.
        """
        t0   = time.perf_counter()
        text = prepare_inference_input(note)

        inputs = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt",
        )
        # Drop token_type_ids for models that don't use them (RoBERTa variants)
        inputs = {
            k: v.to(self.device)
            for k, v in inputs.items()
            if k in ("input_ids", "attention_mask", "token_type_ids")
            and k in self.tokenizer.model_input_names
        }

        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs  = torch.softmax(logits, dim=-1).cpu().numpy()[0]

        top_k_actual = min(top_k, len(self.id2label))
        top_indices  = np.argsort(probs)[::-1][:top_k_actual]

        codes  = [self.id2label.get(int(i), "UNKNOWN") for i in top_indices]
        scores = [float(probs[i]) for i in top_indices]

        latency_ms = (time.perf_counter() - t0) * 1000

        return PredictionResult(
            codes=codes,
            scores=scores,
            confidence=scores[0] if scores else 0.0,
            metadata={
                "model_name":  self.model_name,
                "latency_ms":  round(latency_ms, 1),
                "num_labels":  len(self.id2label),
            },
        )

    def predict_batch(
        self,
        notes:  list[str],
        top_k:  int = 5,
        batch_size: int = 32,
    ) -> list[PredictionResult]:
        """
        Efficient batched prediction.

        Tokenises in batches of batch_size, runs a single forward pass per
        batch. Significantly faster than sequential predict() for evaluation.

        Parameters
        ----------
        notes : list[str]
            Raw clinical notes in SOAP format.
        top_k : int
            Number of top predictions per note.
        batch_size : int
            Number of notes per forward pass.
        """
        preprocessed = [prepare_inference_input(n) for n in notes]
        all_results: list[PredictionResult] = []

        for i in range(0, len(preprocessed), batch_size):
            batch_texts = preprocessed[i : i + batch_size]

            inputs = self.tokenizer(
                batch_texts,
                truncation=True,
                padding="max_length",
                max_length=512,
                return_tensors="pt",
            )
            inputs = {
                k: v.to(self.device)
                for k, v in inputs.items()
                if k in self.tokenizer.model_input_names
            }

            with torch.no_grad():
                logits     = self.model(**inputs).logits
                probs_batch = torch.softmax(logits, dim=-1).cpu().numpy()

            for probs in probs_batch:
                top_k_actual = min(top_k, len(self.id2label))
                top_indices  = np.argsort(probs)[::-1][:top_k_actual]
                codes  = [self.id2label.get(int(idx), "UNKNOWN") for idx in top_indices]
                scores = [float(probs[idx]) for idx in top_indices]
                all_results.append(
                    PredictionResult(
                        codes=codes,
                        scores=scores,
                        confidence=scores[0] if scores else 0.0,
                        metadata={"model_name": self.model_name},
                    )
                )

        return all_results

    def train(
        self,
        train_dataset,
        val_dataset,
        cfg:        dict,
        output_dir: Path,
    ) -> TrainingResult:
        """
        Fine-tune using HuggingFace Trainer.

        Mirrors the training loop used in notebooks 02–05:
          - AdamW, linear warmup schedule
          - Save best checkpoint by val_loss
          - Early stopping (patience = 3)
          - MLflow logging disabled here — handled externally in scripts/train.py

        Parameters
        ----------
        train_dataset, val_dataset :
            HuggingFace Dataset with input_ids, attention_mask, labels columns.
        cfg : dict
            Keys used: num_epochs, learning_rate, batch_size, warmup_ratio,
            weight_decay, experiment_name. Unrecognised keys are ignored.
        output_dir : Path
            Best checkpoint saved here.
        """
        from transformers import DataCollatorWithPadding
        from src.evaluation import hf_compute_metrics

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Set model to training mode (it's in eval() after construction)
        self.model.train()

        # Compute warmup_steps explicitly — mirrors notebook Phase 7 exactly.
        # warmup_ratio is deprecated in HuggingFace v5.2.
        num_epochs = cfg.get("num_epochs", 10)
        batch_size = cfg.get("batch_size", 8)
        total_steps = (len(train_dataset) // batch_size) * num_epochs
        warmup_steps = int(cfg.get("warmup_ratio", 0.1) * total_steps)

        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=cfg.get("learning_rate", 2e-5),
            warmup_steps=warmup_steps,        # replaces deprecated warmup_ratio
            weight_decay=cfg.get("weight_decay", 0.01),
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="macro_f1", # matches notebook: best by F1 not loss
            greater_is_better=True,
            save_total_limit=3,
            logging_steps=20,
            report_to=[],                     # MLflow handled externally in train.py
            fp16=False,                       # MPS doesn't support fp16
            dataloader_num_workers=0,         # avoid fork issues on macOS
            dataloader_pin_memory=False,      # MPS doesn't support pinned memory
            seed=cfg.get("seed", 42),
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            processing_class=self.tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer=self.tokenizer),
            compute_metrics=hf_compute_metrics,   # accuracy + macro_f1 + top5
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        )

        t0 = time.perf_counter()
        train_output = trainer.train()
        elapsed = time.perf_counter() - t0

        # Extract training history — now includes macro_f1 from compute_metrics
        history = []
        for log in trainer.state.log_history:
            if "eval_loss" in log:
                history.append({
                    "epoch":      log.get("epoch"),
                    "eval_loss":  log.get("eval_loss"),
                    "eval_macro_f1":   log.get("eval_macro_f1"),
                    "eval_accuracy":   log.get("eval_accuracy"),
                    "train_loss": log.get("loss"),
                })

        # Best epoch = highest eval_macro_f1 (matches load_best_model_at_end)
        best_f1 = max(
            (h["eval_macro_f1"] for h in history if h.get("eval_macro_f1") is not None),
            default=-1.0,
        )
        best_epoch_entry = next(
            (h for h in history if h.get("eval_macro_f1") == best_f1),
            {"epoch": -1},
        )
        best_epoch   = int(best_epoch_entry.get("epoch", -1))
        best_acc     = best_epoch_entry.get("eval_accuracy", -1.0) or -1.0

        result = TrainingResult(
            best_val_accuracy=float(best_acc),
            best_val_f1=float(best_f1),
            best_epoch=best_epoch,
            training_history=history,
            output_dir=output_dir,
            experiment_name=cfg.get("experiment_name", "unknown"),
            elapsed_seconds=elapsed,
        )

        # Return to eval mode
        self.model.eval()
        return result

    def save(self, output_dir: Path) -> None:
        """
        Save model weights, tokenizer, and label maps to output_dir.

        Layout produced:
            output_dir/model/          — HuggingFace save_pretrained artifacts
            output_dir/label_map.json  — {label2id, id2label}
        """
        output_dir = Path(output_dir)
        hf_dir     = output_dir / "model"
        hf_dir.mkdir(parents=True, exist_ok=True)

        self.model.save_pretrained(str(hf_dir))
        self.tokenizer.save_pretrained(str(hf_dir))

        label_map = {
            "label2id": self.label2id,
            "id2label": {str(k): v for k, v in self.id2label.items()},
        }
        with open(output_dir / "label_map.json", "w") as f:
            json.dump(label_map, f, indent=2)

        print(f"   ✅ EncoderAdapter saved: {hf_dir} ({len(self.label2id)} labels)")

    # ── Internal helpers ────────────────────────────────────────────────────

    @staticmethod
    def _resolve_device(device: Optional[str]) -> torch.device:
        """Auto-detect MPS → CUDA → CPU if device is None."""
        if device is not None:
            return torch.device(device)
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def __repr__(self) -> str:
        return (
            f"EncoderAdapter(model='{self.model_name}', "
            f"labels={len(self.label2id)}, "
            f"device={self.device})"
        )


# ==============================================================================
# HybridRouter — confidence-gated cascade between two adapters
# ==============================================================================

class HybridRouter:
    """
    Routes each prediction through a fast encoder first, escalates to a
    stronger (generative) model only when confidence is below threshold.

    This implements the Use Case B design from Prj_Overview.md:
        high confidence → auto-code, log for audit
        low confidence  → generative model → still uncertain → human review

    The routing threshold is a single float in [0, 1]. Records above the
    threshold are returned from the fast model; records below trigger
    escalation to the strong model.

    The HybridRouter itself satisfies the ModelAdapter interface (predict and
    predict_batch) so it can be used anywhere an adapter is expected — the
    caller doesn't need to know routing is happening.

    Parameters
    ----------
    fast : ModelAdapter
        The primary encoder adapter (fast, cheap, runs on every record).
    strong : ModelAdapter
        The escalation adapter (slower, runs on ~20-30% of records).
    threshold : float
        Confidence cutoff. fast.confidence >= threshold → use fast result.
        Default 0.7 — tune on held-out validation data.
    """

    def __init__(
        self,
        fast:       ModelAdapter,
        strong:     ModelAdapter,
        threshold:  float = 0.7,
    ) -> None:
        self.fast      = fast
        self.strong    = strong
        self.threshold = threshold

    def predict(self, note: str, top_k: int = 5) -> PredictionResult:
        """
        Predict with automatic model selection based on confidence.

        The 'routed_to' key in result.metadata records which model was used.
        """
        fast_result = self.fast.predict(note, top_k=top_k)

        if fast_result.confidence >= self.threshold:
            fast_result.metadata["routed_to"]  = "fast"
            fast_result.metadata["threshold"]  = self.threshold
            return fast_result

        strong_result = self.strong.predict(note, top_k=top_k)
        strong_result.metadata["routed_to"]   = "strong"
        strong_result.metadata["fast_confidence"] = fast_result.confidence
        strong_result.metadata["threshold"]   = self.threshold
        return strong_result

    def predict_batch(
        self,
        notes:  list[str],
        top_k:  int = 5,
    ) -> list[PredictionResult]:
        """
        Batch prediction with routing.

        Two-pass: fast model first (all records), then strong model for
        low-confidence records only. More efficient than sequential predict().
        """
        fast_results = self.fast.predict_batch(notes, top_k=top_k)

        low_conf_indices = [
            i for i, r in enumerate(fast_results)
            if r.confidence < self.threshold
        ]

        if low_conf_indices:
            low_conf_notes  = [notes[i] for i in low_conf_indices]
            strong_results  = self.strong.predict_batch(low_conf_notes, top_k=top_k)

            for rank, idx in enumerate(low_conf_indices):
                strong_results[rank].metadata["routed_to"]       = "strong"
                strong_results[rank].metadata["fast_confidence"] = fast_results[idx].confidence
                fast_results[idx] = strong_results[rank]

        for r in fast_results:
            if "routed_to" not in r.metadata:
                r.metadata["routed_to"] = "fast"

        return fast_results

    def escalation_rate(self, results: list[PredictionResult]) -> float:
        """Fraction of predictions escalated to the strong model."""
        n_strong = sum(1 for r in results if r.metadata.get("routed_to") == "strong")
        return n_strong / len(results) if results else 0.0


# ==============================================================================
# GenerativeAdapter — stub for future MedGemma / BioMistral integration
# ==============================================================================

class GenerativeAdapter(ModelAdapter):
    """
    Adapter stub for generative / instruction-tuned models.

    NOT IMPLEMENTED. This class exists to document the interface contract
    so that MedGemma, BioMistral, or GPT-4o can be integrated later without
    touching the training, evaluation, or routing code.

    When implementing:
      - predict() should use constrained decoding (if the model supports it)
        or parse the model output to extract a valid ICD code string.
      - train() is optional — generative models may be used zero-shot or
        prompt-tuned without a full gradient update loop.
      - The metadata dict in PredictionResult should include the clinical
        rationale generated by the model — this is the primary value-add
        over encoder-only models for the human review queue.
    """

    def predict(self, note: str, top_k: int = 5) -> PredictionResult:
        raise NotImplementedError(
            "GenerativeAdapter.predict() is not yet implemented.\n"
            "See src/adapters.py for the interface contract and implementation guidance."
        )

    def train(self, train_dataset, val_dataset, cfg: dict, output_dir: Path) -> TrainingResult:
        raise NotImplementedError(
            "GenerativeAdapter.train() is not yet implemented.\n"
            "For generative models, consider prompt-tuning or zero-shot evaluation "
            "via GenerativeAdapter.predict() without a training step."
        )

    def save(self, output_dir: Path) -> None:
        raise NotImplementedError("GenerativeAdapter.save() is not yet implemented.")

    @classmethod
    def load(cls, model_dir: Path, device: Optional[str] = None) -> "GenerativeAdapter":
        raise NotImplementedError("GenerativeAdapter.load() is not yet implemented.")