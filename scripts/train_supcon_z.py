"""
train_supcon_z.py — Supervised Contrastive fine-tuning for the Z-chapter resolver.

Motivation
----------
The Z-chapter (administrative codes, 263 codes, 62.1% accuracy) is the primary
remaining performance gap in E-010. The root cause is feature similarity: clinical
notes for different Z-codes are nearly lexically identical. Standard cross-entropy
training cannot distinguish codes whose training examples have near-identical text.

Supervised Contrastive Learning (Khosla et al., 2020) addresses this directly:
it trains the encoder to pull embeddings for the same ICD-10 code together and
push embeddings for different codes apart — regardless of surface text similarity.

Strategy
--------
Rather than fine-tuning from scratch, we start from the E-010 Z resolver — a
model that already has 62.1% accuracy on Z-codes. We then apply SupCon fine-tuning
to sharpen the embedding geometry for the Z chapter specifically, without touching
any other chapter resolver.

Two-phase training:
  Phase 1 — Contrastive:  update encoder weights using SupCon loss on [CLS] embeddings
  Phase 2 — Head refit:   freeze encoder, retrain classification head with CE loss

The two-phase approach avoids the competing-gradients problem of joint SupCon+CE
training, and ensures the improved embedding geometry is captured before the head
is refit to it.

Batch construction
------------------
SupCon requires at least 2 examples of the same class per batch to form positive
pairs. With ~4 examples per Z-code across 263 codes, naive random batching
frequently produces batches with no positive pairs. A BalancedBatchSampler
guarantees each batch contains exactly n_per_class examples of n_classes_per_batch
randomly sampled classes — ensuring positive pairs exist in every batch.

Output
------
Drop-in replacement for the E-010 Z resolver at:
  outputs/evaluations/E-014_SupCon_Z/stage2/Z/

Calibrate and evaluate with:
  uv run python scripts/calibrate.py \\
      --experiment E-014_SupCon_Z \\
      --stage1-experiment E-003_Hierarchical_ICD10
  uv run python scripts/evaluate.py \\
      --experiment E-014_SupCon_Z \\
      --mode hierarchical \\
      --stage1-experiment E-003_Hierarchical_ICD10 \\
      --threshold 0.7

Usage
-----
  uv run python scripts/train_supcon_z.py [options]

  --source-experiment   Experiment to load the Z resolver from (default: E-010_40ep_E002Init)
  --experiment          Output experiment name (default: E-014_SupCon_Z)
  --contrastive-epochs  Epochs for SupCon encoder training (default: 10)
  --head-epochs         Epochs for frozen-encoder head retraining (default: 5)
  --batch-size          Batch size (default: 16)
  --n-per-class         Examples per class per batch (default: 2)
  --temperature         SupCon temperature τ (default: 0.07)
  --learning-rate       Learning rate (default: 2e-5)
  --head-lr             Learning rate for head retraining (default: 1e-3)
  --dry-run             Print config and exit without training
"""

import sys
sys.path.insert(0, '.')

import argparse
import json
import math
import random
import shutil
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler, Dataset
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import accuracy_score, f1_score

from src.paths import ExperimentPaths
from src.experiment_logger import ExperimentLogger

# ── Reproducibility ────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ==============================================================================
# Balanced Batch Sampler
# ==============================================================================

class BalancedBatchSampler(Sampler):
    """
    Yields batch indices that guarantee n_per_class examples of
    n_classes_per_batch randomly sampled classes per batch.

    This ensures every batch contains positive pairs for SupCon loss.

    Parameters
    ----------
    labels : list[int]
        Integer class label for each sample in the dataset.
    n_per_class : int
        How many examples of each class to include per batch.
    n_classes_per_batch : int
        How many distinct classes per batch. Batch size = n_per_class * n_classes_per_batch.
    """

    def __init__(
        self,
        labels: list[int],
        n_per_class: int = 2,
        n_classes_per_batch: int = 8,
    ) -> None:
        self.n_per_class = n_per_class
        self.n_classes_per_batch = n_classes_per_batch

        # Group indices by class — only include classes with >= n_per_class examples
        class_to_indices: dict[int, list[int]] = defaultdict(list)
        for idx, lbl in enumerate(labels):
            class_to_indices[lbl].append(idx)

        self.class_to_indices = {
            cls: idxs
            for cls, idxs in class_to_indices.items()
            if len(idxs) >= n_per_class
        }
        self.classes = list(self.class_to_indices.keys())

        n_usable = len(self.classes)
        batch_size = n_per_class * n_classes_per_batch
        # Number of batches ≈ total usable samples / batch_size
        total_samples = sum(len(v) for v in self.class_to_indices.values())
        self._n_batches = max(1, total_samples // batch_size)

        print(
            f" BalancedBatchSampler: {n_usable} classes with ≥{n_per_class} examples, "
            f"{self._n_batches} batches × {batch_size} = "
            f"~{self._n_batches * batch_size} samples/epoch"
        )

    def __len__(self) -> int:
        return self._n_batches

    def __iter__(self):
        for _ in range(self._n_batches):
            # Sample n_classes_per_batch distinct classes
            batch_classes = random.sample(
                self.classes,
                min(self.n_classes_per_batch, len(self.classes)),
            )
            batch_indices = []
            for cls in batch_classes:
                pool = self.class_to_indices[cls]
                chosen = random.choices(pool, k=self.n_per_class)
                batch_indices.extend(chosen)
            random.shuffle(batch_indices)
            yield batch_indices


# ==============================================================================
# Dataset
# ==============================================================================

class ZDataset(Dataset):
    """
    Tokenised Z-chapter dataset.

    Returns input_ids, attention_mask, and integer label_id.
    """

    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        tokenizer,
        max_length: int = 512,
    ) -> None:
        assert len(texts) == len(labels)
        self.labels = labels

        print(f" Tokenising {len(texts)} records...", flush=True)
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        print(f" Done.", flush=True)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        return {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "label":          torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ==============================================================================
# Supervised Contrastive Loss
# ==============================================================================

class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss (Khosla et al., 2020).

    For each anchor, treats all other examples with the same label as
    positive pairs and all examples with different labels as negatives.

    Parameters
    ----------
    temperature : float
        Scaling factor τ. Lower = sharper distribution. Default 0.07.
    """

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        embeddings : (N, D) — L2-normalised [CLS] embeddings
        labels     : (N,)   — integer class labels

        Returns
        -------
        Scalar loss value.
        """
        device = embeddings.device
        N = embeddings.shape[0]

        # L2-normalise (caller should do this too, but belt-and-braces)
        embeddings = F.normalize(embeddings, dim=1)

        # Similarity matrix: (N, N)
        sim = torch.matmul(embeddings, embeddings.T) / self.temperature

        # Mask out the diagonal (self-similarity)
        self_mask = torch.eye(N, dtype=torch.bool, device=device)
        sim = sim.masked_fill(self_mask, -1e4)  # -inf causes NaN on MPS

        # Positive pair mask: same label, not self
        labels_row = labels.unsqueeze(1)   # (N, 1)
        labels_col = labels.unsqueeze(0)   # (1, N)
        pos_mask = (labels_row == labels_col) & ~self_mask  # (N, N)

        # For numerical stability: subtract row-wise max before exp
        sim_max, _ = sim.masked_fill(self_mask, -1e4).max(dim=1, keepdim=True)
        sim = sim - sim_max.detach()

        # exp of all non-self pairs (denominator)
        exp_sim = torch.exp(sim) * (~self_mask).float()

        # Log-probability of each positive pair
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

        # Mean over positives — skip anchors with no positive in the batch
        n_positives = pos_mask.sum(dim=1).float()
        has_positive = n_positives > 0
        if not has_positive.any():
            return torch.tensor(0.0, device=device, requires_grad=True)

        mean_log_prob_pos = (log_prob * pos_mask.float()).sum(dim=1) / (n_positives + 1e-8)
        loss = -mean_log_prob_pos[has_positive].mean()
        return loss


# ==============================================================================
# Encoder wrapper (extracts [CLS] embedding)
# ==============================================================================

class CLSEncoder(nn.Module):
    """
    Wraps a HuggingFace encoder, returns L2-normalised [CLS] embeddings.
    Used during the contrastive phase — no classification head.
    """

    def __init__(self, encoder) -> None:
        super().__init__()
        self.encoder = encoder

    def forward(self, input_ids, attention_mask) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = outputs.last_hidden_state[:, 0, :]   # (B, hidden_size)
        return F.normalize(cls, dim=1)


# ==============================================================================
# Training phases
# ==============================================================================

def _resolve_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def run_contrastive_phase(
    encoder: CLSEncoder,
    train_loader: DataLoader,
    val_dataset: ZDataset,
    cfg: dict,
    device: torch.device,
) -> CLSEncoder:
    """
    Phase 1: Train the encoder with SupCon loss.

    Does NOT update any classification head — pure embedding geometry training.
    Saves the best encoder by val SupCon loss.

    Returns the best encoder.
    """
    print("\n── Phase 1: Contrastive encoder training ─────────────────────────────")

    loss_fn = SupConLoss(temperature=cfg["temperature"])
    optimizer = torch.optim.AdamW(
        encoder.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=0.01,
    )

    n_epochs = cfg["contrastive_epochs"]
    total_steps = len(train_loader) * n_epochs
    warmup_steps = max(1, int(0.1 * total_steps))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, n_epochs + 1):
        encoder.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            embeddings = encoder(input_ids, attention_mask)
            loss = loss_fn(embeddings, labels)

            if loss.requires_grad:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)

        # Validation: compute SupCon loss on val set
        val_loss = _eval_contrastive_loss(encoder, val_dataset, loss_fn, device, cfg["batch_size"])

        print(
            f" Epoch {epoch:2d}/{n_epochs} | "
            f"train_loss={avg_loss:.4f} | val_loss={val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in encoder.state_dict().items()}
            print(f"            ↑ best")

    if best_state is not None:
        encoder.load_state_dict(best_state)
    print(f"\n Phase 1 complete. Best val SupCon loss: {best_val_loss:.4f}")
    return encoder


def _eval_contrastive_loss(
    encoder: CLSEncoder,
    dataset: ZDataset,
    loss_fn: SupConLoss,
    device: torch.device,
    batch_size: int,
) -> float:
    """Compute mean SupCon loss on a dataset with random batching."""
    encoder.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    total_loss = 0.0
    n = 0
    with torch.no_grad():
        for batch in loader:
            emb = encoder(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
            )
            loss = loss_fn(emb, batch["label"].to(device))
            total_loss += loss.item()
            n += 1
    return total_loss / max(n, 1)


def run_head_retraining_phase(
    model: nn.Module,
    train_dataset: ZDataset,
    val_dataset: ZDataset,
    cfg: dict,
    device: torch.device,
    n_labels: int,
) -> tuple[nn.Module, dict]:
    """
    Phase 2: Freeze encoder, retrain classification head with CE loss.

    The encoder's improved embedding geometry is now fixed.
    We fit a linear head on top of it.

    Returns (best_model, best_metrics).
    """
    print("\n── Phase 2: Classification head retraining (frozen encoder) ──────────")

    # Freeze all encoder layers — only train classifier head
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad_(False)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f" Trainable parameters: {n_trainable:,} (head only)")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["head_lr"],
        weight_decay=0.01,
    )
    ce_loss = nn.CrossEntropyLoss()

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
    )

    best_val_f1 = -1.0
    best_state = None
    best_metrics: dict = {}

    for epoch in range(1, cfg["head_epochs"] + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            loss = ce_loss(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        val_metrics = _eval_classification(model, val_dataset, device, cfg["batch_size"])

        print(
            f" Epoch {epoch:2d}/{cfg['head_epochs']} | "
            f"train_loss={avg_loss:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f} | "
            f"val_f1={val_metrics['macro_f1']:.4f}"
        )

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_metrics = val_metrics
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"            ↑ best")

    if best_state is not None:
        model.load_state_dict(best_state)

    # Unfreeze all parameters (model saved in trainable state)
    for param in model.parameters():
        param.requires_grad_(True)

    print(f"\n Phase 2 complete. Best val F1: {best_val_f1:.4f}")
    return model, best_metrics


def _eval_classification(
    model: nn.Module,
    dataset: ZDataset,
    device: torch.device,
    batch_size: int,
) -> dict:
    """Compute accuracy and macro F1 on a dataset."""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            logits = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            ).logits
            preds = logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch["label"].numpy())

    return {
        "accuracy":  accuracy_score(all_labels, all_preds),
        "macro_f1":  f1_score(all_labels, all_preds, average="macro", zero_division=0),
    }


# ==============================================================================
# Main
# ==============================================================================

def main(cfg: dict) -> None:
    t0 = time.perf_counter()
    device = _resolve_device()
    print(f"🔧 Device: {device}")

    # ── Resolve source Z resolver ──────────────────────────────────────────────
    src_paths = ExperimentPaths(cfg["source_experiment"])
    src_model_dir = src_paths.stage2_model_dir("Z")
    if src_model_dir is None or not src_model_dir.exists():
        raise FileNotFoundError(
            f"Z resolver not found for {cfg['source_experiment']}. "
            f"Expected at: {src_paths.stage2_base / 'Z'}"
        )

    label_map_path = src_paths.stage2_label_map("Z")
    with open(label_map_path) as f:
        lmap = json.load(f)

    label2id: dict[str, int] = lmap["label2id"]
    id2label: dict[int, str] = {int(k): v for k, v in lmap["id2label"].items()}
    n_labels = len(label2id)
    print(f"📥 Loaded label map: {n_labels} Z-codes")

    # ── Load train/val splits (reuse E-010 Z splits) ───────────────────────────
    train_path = src_paths.stage2_train_split("Z")
    val_path   = src_paths.stage2_val_split("Z")

    train_df = pl.read_parquet(train_path)
    val_df   = pl.read_parquet(val_path)

    print(f"📊 Z train: {len(train_df)} records | val: {len(val_df)} records")
    print(f"   Unique codes — train: {train_df['standard_icd10'].n_unique()} | "
          f"val: {val_df['standard_icd10'].n_unique()}")

    # ── Build integer labels — filter out codes not in label2id ───────────────
    def encode_aligned(df: pl.DataFrame) -> tuple[list[str], list[int]]:
        texts, labels = [], []
        dropped = 0
        for text, code in zip(df["apso_note"].to_list(), df["standard_icd10"].to_list()):
            if code in label2id:
                texts.append(text)
                labels.append(label2id[code])
            else:
                dropped += 1
        if dropped:
            print(f" ⚠️  Dropped {dropped} records with codes not in label2id")
        return texts, labels

    train_texts, train_labels = encode_aligned(train_df)
    val_texts,   val_labels   = encode_aligned(val_df)

    # ── Tokenise ───────────────────────────────────────────────────────────────
    print(f"\n📥 Loading tokenizer from {src_model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(str(src_model_dir))

    train_dataset = ZDataset(train_texts, train_labels, tokenizer)
    val_dataset   = ZDataset(val_texts,   val_labels,   tokenizer)

    # ── Balanced sampler for contrastive phase ─────────────────────────────────
    sampler = BalancedBatchSampler(
        labels=train_labels,
        n_per_class=cfg["n_per_class"],
        n_classes_per_batch=cfg["batch_size"] // cfg["n_per_class"],
    )
    train_loader_contrastive = DataLoader(
        train_dataset,
        batch_sampler=sampler,
    )

    # ── Load base model for contrastive phase (encoder only) ──────────────────
    print(f"\n📥 Loading Z resolver from {src_model_dir}")
    base_encoder = AutoModel.from_pretrained(str(src_model_dir)).to(device)
    cls_encoder  = CLSEncoder(base_encoder)

    # ── Baseline: evaluate E-010 Z accuracy before SupCon ─────────────────────
    print("\n── Baseline evaluation (E-010 Z resolver) ────────────────────────────")
    baseline_model = AutoModelForSequenceClassification.from_pretrained(
        str(src_model_dir),
        num_labels=n_labels,
        ignore_mismatched_sizes=True,
    ).to(device)
    baseline_metrics = _eval_classification(baseline_model, val_dataset, device, cfg["batch_size"])
    print(
        f" Baseline val_acc={baseline_metrics['accuracy']:.4f} | "
        f"val_f1={baseline_metrics['macro_f1']:.4f}"
    )
    del baseline_model

    # ── Phase 1: Contrastive training ─────────────────────────────────────────
    cls_encoder = run_contrastive_phase(
        cls_encoder, train_loader_contrastive, val_dataset, cfg, device
    )

    # ── Phase 2: Head retraining ───────────────────────────────────────────────
    # Rebuild full classification model, initialise encoder from Phase 1 weights
    full_model = AutoModelForSequenceClassification.from_pretrained(
        str(src_model_dir),
        num_labels=n_labels,
        ignore_mismatched_sizes=True,
    ).to(device)

    # Transplant Phase 1 encoder weights into full model
    # The base encoder is the 'bert' or equivalent sub-module
    encoder_state = cls_encoder.encoder.state_dict()
    missing, unexpected = full_model.base_model.load_state_dict(encoder_state, strict=False)
    if missing:
        print(f" ⚠️  Missing keys when transplanting encoder: {len(missing)}")
    if unexpected:
        print(f" ⚠️  Unexpected keys when transplanting encoder: {len(unexpected)}")
    print(f" ✅ Phase 1 encoder weights transplanted into full model")

    full_model, best_metrics = run_head_retraining_phase(
        full_model, train_dataset, val_dataset, cfg, device, n_labels
    )

    # ── Save output ────────────────────────────────────────────────────────────
    out_paths = ExperimentPaths(cfg["experiment"])
    out_z_dir = out_paths.stage2_base / "Z"
    out_z_dir.mkdir(parents=True, exist_ok=True)

    # Save model + tokenizer
    full_model.save_pretrained(str(out_z_dir))
    tokenizer.save_pretrained(str(out_z_dir))

    # Save label map
    with open(out_z_dir / "label_map.json", "w") as f:
        json.dump(lmap, f, indent=2)

    # Copy test split from source (evaluate.py needs it)
    src_test = src_paths.stage2_test_split("Z")
    shutil.copy2(src_test, out_paths.stage2_base / "Z" / "test_split.parquet")

    # Copy stage2_results.json from source (calibrate.py needs skip_chapters)
    src_results = src_paths.stage2_results() if callable(src_paths.stage2_results) else src_paths.stage2_results
    if src_results.exists():
        shutil.copy2(src_results, out_paths.stage2_base / "stage2_results.json")

    # Save experiment config
    elapsed = time.perf_counter() - t0
    result_record = {
        "experiment":        cfg["experiment"],
        "source_experiment": cfg["source_experiment"],
        "baseline_val_acc":  baseline_metrics["accuracy"],
        "baseline_val_f1":   baseline_metrics["macro_f1"],
        "best_val_acc":      best_metrics["accuracy"],
        "best_val_f1":       best_metrics["macro_f1"],
        "delta_acc":         best_metrics["accuracy"] - baseline_metrics["accuracy"],
        "delta_f1":          best_metrics["macro_f1"] - baseline_metrics["macro_f1"],
        "cfg":               cfg,
        "elapsed_seconds":   elapsed,
    }
    with open(out_z_dir / "supcon_result.json", "w") as f:
        json.dump(result_record, f, indent=2)

    print(f"\n{'='*70}")
    print(f" ✅ E-014 SupCon Z training complete — {elapsed:.0f}s")
    print(f" Outputs: {out_z_dir}")
    print(f"\n Baseline → SupCon:")
    print(f"   val_acc:  {baseline_metrics['accuracy']:.4f} → {best_metrics['accuracy']:.4f}"
          f"  ({best_metrics['accuracy'] - baseline_metrics['accuracy']:+.4f})")
    print(f"   val_f1:   {baseline_metrics['macro_f1']:.4f} → {best_metrics['macro_f1']:.4f}"
          f"  ({best_metrics['macro_f1'] - baseline_metrics['macro_f1']:+.4f})")
    print(f"\n Next steps:")
    print(f"   uv run python scripts/calibrate.py \\")
    print(f"       --experiment {cfg['experiment']} \\")
    print(f"       --stage1-experiment E-003_Hierarchical_ICD10")
    print(f"   uv run python scripts/evaluate.py \\")
    print(f"       --experiment {cfg['experiment']} \\")
    print(f"       --mode hierarchical \\")
    print(f"       --stage1-experiment E-003_Hierarchical_ICD10 \\")
    print(f"       --threshold 0.7")
    print(f"{'='*70}")


# ==============================================================================
# CLI
# ==============================================================================

def parse_args() -> dict:
    parser = argparse.ArgumentParser(
        description="SupCon fine-tuning for Z-chapter resolver"
    )
    parser.add_argument(
        "--source-experiment",
        default="E-010_40ep_E002Init",
        help="Experiment to load Z resolver from (default: E-010_40ep_E002Init)",
    )
    parser.add_argument(
        "--experiment",
        default="E-014_SupCon_Z",
        help="Output experiment name (default: E-014_SupCon_Z)",
    )
    parser.add_argument("--contrastive-epochs", type=int, default=10)
    parser.add_argument("--head-epochs",        type=int, default=5)
    parser.add_argument("--batch-size",         type=int, default=16)
    parser.add_argument("--n-per-class",        type=int, default=2,
                        help="Examples per class per batch (default: 2)")
    parser.add_argument("--temperature",        type=float, default=0.07,
                        help="SupCon temperature τ (default: 0.07)")
    parser.add_argument("--learning-rate",      type=float, default=2e-5)
    parser.add_argument("--head-lr",            type=float, default=1e-3)
    parser.add_argument("--dry-run",            action="store_true")
    args = parser.parse_args()

    cfg = {
        "source_experiment":  args.source_experiment,
        "experiment":         args.experiment,
        "contrastive_epochs": args.contrastive_epochs,
        "head_epochs":        args.head_epochs,
        "batch_size":         args.batch_size,
        "n_per_class":        args.n_per_class,
        "temperature":        args.temperature,
        "learning_rate":      args.learning_rate,
        "head_lr":            args.head_lr,
    }
    return cfg, args.dry_run


if __name__ == "__main__":
    cfg, dry_run = parse_args()

    print("=" * 70)
    print(f"  train_supcon_z.py — Supervised Contrastive Z-Chapter Fine-Tuning")
    print(f"  Source:      {cfg['source_experiment']}")
    print(f"  Experiment:  {cfg['experiment']}")
    print(f"  SupCon epochs: {cfg['contrastive_epochs']} | Head epochs: {cfg['head_epochs']}")
    print(f"  Batch: {cfg['batch_size']} ({cfg['n_per_class']} per class) | τ={cfg['temperature']}")
    print(f"  LR: {cfg['learning_rate']} (encoder) | {cfg['head_lr']} (head)")
    print("=" * 70)

    if dry_run:
        print("\n[DRY RUN] Config printed. Exiting.")
        sys.exit(0)

    main(cfg)
