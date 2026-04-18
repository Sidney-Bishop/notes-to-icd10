"""
dataset.py — PyTorch Dataset wrapper for ClinicalBERT fine-tuning.

Provides ``MedSynthDataset``, a ``torch.utils.data.Dataset`` that wraps
a Polars Gold layer DataFrame for use with a PyTorch DataLoader.

Usage context
-------------
The model training notebooks (02–05) use the HuggingFace ``DatasetDict``
+ ``Trainer`` pattern rather than this class — labels are encoded as
integers before dataset construction, and the Trainer handles batching
and collation internally. ``MedSynthDataset`` is NOT used in any current
training notebook.

This class is retained as a standalone PyTorch DataLoader alternative,
useful if you want to step outside the HuggingFace Trainer API — for
example, to write a custom training loop, integrate with a non-HuggingFace
framework, or run targeted per-batch debugging.

If you use this class, note that it expects:
- A ``apso_note`` column (not ``text``) — the APSO-flipped, redacted
  model input produced by notebook 01 Phase 3.
- A ``label_id`` column containing pre-encoded integer class indices —
  produced by the label encoding step in notebook 02/03 Phase 2.
- A ``label_to_id`` dict mapping ICD code strings to integer indices.

These conventions match the Gold layer schema exactly. The original class
used a ``text`` column and a ``label`` column (string), which matched
neither the Gold layer nor the HuggingFace DatasetDict schema.
"""

import torch
import polars as pl
from torch.utils.data import Dataset


class MedSynthDataset(Dataset):
    """
    PyTorch Dataset wrapper for the MedSynth Gold layer.

    Wraps a Polars DataFrame for use with ``torch.utils.data.DataLoader``.
    Expects labels to be pre-encoded as integers in a ``label_id`` column,
    matching the encoding produced by the Phase 2 label encoding step in
    the training notebooks.

    Parameters
    ----------
    dataframe : pl.DataFrame
        Gold layer DataFrame containing at minimum:
            - ``apso_note``  (str)  — APSO-flipped, redacted model input
            - ``label_id``   (int)  — pre-encoded integer class index
    tokenizer :
        A HuggingFace tokenizer (e.g. AutoTokenizer) compatible with
        Bio_ClinicalBERT.
    label_to_id : dict[str, int]
        Mapping from ICD code string to integer class index. Used for
        validation only — labels are read from the ``label_id`` column,
        not re-encoded here.
    max_length : int
        Maximum token sequence length. Default 512 (Bio_ClinicalBERT limit).

    Raises
    ------
    ValueError
        At construction time, if required columns are missing from the DataFrame.
    KeyError
        At ``__getitem__`` time, if a ``label_id`` value is missing (null).
        This indicates a data preparation problem upstream and should not
        be silently swallowed — it means the Phase 2 label encoding step
        produced an incomplete result.
    """

    _REQUIRED_COLUMNS = {"apso_note", "label_id"}

    def __init__(
        self,
        dataframe: pl.DataFrame,
        tokenizer,
        label_to_id: dict,
        max_length: int = 512,
    ) -> None:
        missing = self._REQUIRED_COLUMNS - set(dataframe.columns)
        if missing:
            raise ValueError(
                f"MedSynthDataset requires columns {self._REQUIRED_COLUMNS}. "
                f"Missing: {missing}. "
                f"Ensure the DataFrame comes from the Gold layer after Phase 2 "
                f"label encoding (notebook 02 or 03)."
            )

        self.df          = dataframe
        self.tokenizer   = tokenizer
        self.label_to_id = label_to_id
        self.max_length  = max_length

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.row(idx, named=True)

        encoding = self.tokenizer(
            str(row["apso_note"]),
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        label_id = row["label_id"]
        if label_id is None:
            raise ValueError(
                f"Record at index {idx} has a null label_id. "
                f"Re-run Phase 2 label encoding to ensure all records have "
                f"a valid integer class index before constructing this dataset."
            )

        return {
            "input_ids":      encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels":         torch.tensor(int(label_id), dtype=torch.long),
        }