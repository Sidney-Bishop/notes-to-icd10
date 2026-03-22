# src/dataset.py
import torch
import polars as pl
from torch.utils.data import Dataset

class MedSynthDataset(Dataset):
    """
    High-performance PyTorch Dataset for ClinicalBERT.
    Consumes Polars DataFrames from the Gold (Surgical) layer.
    """
    def __init__(self, dataframe: pl.DataFrame, tokenizer, label_to_id, max_length=512):
        # Polars dataframes are already efficiently indexed; no need for reset_index
        self.df = dataframe
        self.tokenizer = tokenizer
        self.label_to_id = label_to_id
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Polars row access (returns a tuple or dict depending on method)
        # Using row() for speed, or select() for safety.
        row = self.df.row(idx, named=True)
        
        # The 'text' column already has A-P-S-O prioritization [cite: 2026-01-27]
        encoding = self.tokenizer(
            str(row['text']),
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        label_str = str(row['label'])
        
        # Zero-Trust Label Lookup
        try:
            label_id = self.label_to_id[label_str]
        except KeyError:
            # This catches any 'Hidden Surprises' that leaked through
            # In a demo, this shows you handle out-of-vocabulary labels gracefully.
            print(f"⚠️ Warning: Label {label_str} not in mapping. Defaulting to 0.")
            label_id = 0
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label_id, dtype=torch.long)
        }