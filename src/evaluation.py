# src/evaluation.py
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from src.config import config

def compute_comprehensive_metrics(y_true, y_pred, logits=None, top_k=5):
    """
    Standard metrics + Top-K accuracy for clinical relevance.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
    }
    
    # Calculate Top-K Accuracy if logits are provided
    if logits is not None:
        # Convert to torch tensor if it's numpy for easy topk
        if isinstance(logits, np.ndarray):
            logits = torch.from_numpy(logits)
        
        _, top_indices = logits.topk(top_k, dim=-1)
        y_true_tensor = torch.tensor(y_true).unsqueeze(1)
        correct_in_top_k = (top_indices == y_true_tensor).any(dim=1).float().mean().item()
        metrics[f"top_{top_k}_accuracy"] = correct_in_top_k
        
    return metrics

def hf_compute_metrics(eval_pred):
    """
    Hugging Face Trainer hook with Top-5 support.
    """
    logits, labels = eval_pred
    y_pred = np.argmax(logits, axis=-1)
    
    metrics = compute_comprehensive_metrics(labels, y_pred, logits=logits, top_k=5)
    
    # Log to persistent audit trail for Phase 25 Traceability [cite: 2026-02-10]
    config.log_event(
        phase="Evaluation",
        action="compute_metrics",
        details=metrics
    )
    
    return metrics