import argparse
import os
import re

import pandas as pd
import torch
from transformers import AutoTokenizer

MODEL_NAME = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
LABEL_MAP = {"DEF": 0, "SX": 1, "TX": 2, "DX": 3, "CLS": 4}
MAX_LEN = 128


def normalize_question(text: str) -> str:
    """Used identically in training, evaluation and the app."""
    text = re.sub(r"\[.*?\]|\(.*?\)", "", str(text))
    return " ".join(text.split()).strip().lower()


class MedicalPreprocessor:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def preprocess_split(self, df, split_name):
        print(f"\nProcessing {split_name}...")
        assert set(df["q_type"].unique()).issubset(LABEL_MAP.keys())

        labels = torch.tensor(df["q_type"].map(LABEL_MAP).values)
        texts = df["question"].map(normalize_question).tolist()  # question ONLY

        enc = dict(
            self.tokenizer(
                texts,
                max_length=MAX_LEN,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
        )
        enc["labels"] = labels

        print("Class distribution:")
        for cls, idx in LABEL_MAP.items():
            print(f"  {cls}: {(labels == idx).sum().item()}")
        return enc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", default="data/splits")
    parser.add_argument("--out", default="data/tokenized")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    pre = MedicalPreprocessor()
    for split in ["train", "val", "test"]:
        df = pd.read_csv(f"{args.splits}/{split}.csv")
        torch.save(pre.preprocess_split(df, split), f"{args.out}/{split}_encodings.pt")
        print(f"Saved {split} ({len(df)} samples)")