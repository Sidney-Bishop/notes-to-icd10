# src/inference.py
import torch
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.config import config


class ClinicalPredictor:
    """
    Inference engine for ClinicalBERT ICD classification.
    Loads a saved model from the experiment registry.
    """
    def __init__(self, experiment_name: str = "E-001_Baseline_ICD3"):
        self.device = torch.device(
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
        # Resolve model path from registry
        model_path = (
            config.resolve_path("outputs", "evaluations")
            / "registry"
            / experiment_name
            / "model"
        )
        if not model_path.exists():
            raise FileNotFoundError(
                f"No registry model found at {model_path}\n"
                f"Please run Phase 10 (model registry promotion) first."
            )
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        self.model     = AutoModelForSequenceClassification.from_pretrained(
            str(model_path)
        ).to(self.device)
        self.model.eval()
        print(f"✅ Model loaded from registry: {experiment_name}")
        print(f"   Device: {self.device}")
        print(f"   Labels: {self.model.config.num_labels}")

    def predict(self, text: str, top_k: int = 5) -> dict:
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding="max_length"
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]

        top_idx = np.argsort(probs)[::-1][:top_k]
        return {
            "codes":  [self.model.config.id2label[int(i)] for i in top_idx],
            "scores": [float(probs[i]) for i in top_idx]
        }


def predict_icd3(
    text: str,
    top_k: int = 5,
    experiment_name: str = "E-001_Baseline_ICD3"
) -> dict:
    """
    Simplified API for demos and downstream notebooks.

    Usage:
        from src.inference import predict_icd3
        result = predict_icd3("Patient presents with chest pain...", top_k=5)
        for code, score in zip(result['codes'], result['scores']):
            print(f"{code}: {score:.4f}")
    """
    predictor = ClinicalPredictor(experiment_name=experiment_name)
    return predictor.predict(text, top_k=top_k)