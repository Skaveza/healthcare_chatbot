import argparse
import os
import re

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

# Order matters: specific intents first, generic definition last
RULES = [
    ("SX", re.compile(r"\b(symptoms?|signs?)\b")),
    ("TX", re.compile(r"\b(treat\w*|therap\w*)\b")),
    ("DX", re.compile(r"\b(diagnos\w*|tests? for)\b")),
    ("DEF", re.compile(r"\b(what is|define)\b")),
]


def get_question_type(q: str) -> str:
    q = q.lower()
    for label, pattern in RULES:
        if pattern.search(q):
            return label
    return "CLS"


def _group_split(df, test_size, seed):
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    a_idx, b_idx = next(gss.split(df, groups=df["focus_area"]))
    return df.iloc[a_idx], df.iloc[b_idx]


def medical_split(df, seed=42):
    df = df.dropna(subset=["question", "answer", "focus_area"])
    df = df[df["focus_area"] != "UNKNOWN"].copy()
    df["q_type"] = df["question"].apply(get_question_type)

    # Split by focus_area so the same condition never appears in two splits
    train, temp = _group_split(df, test_size=0.3, seed=seed)
    val, test = _group_split(temp, test_size=0.5, seed=seed)

    for name, split in [("TRAIN", train), ("VAL", val), ("TEST", test)]:
        print(f"\n{name}: {len(split)} samples")
        print(split["q_type"].value_counts())

    # Sanity check: no condition shared between splits
    assert not set(train.focus_area) & set(test.focus_area)
    assert not set(train.focus_area) & set(val.focus_area)
    assert not set(val.focus_area) & set(test.focus_area)
    return train, val, test


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/medquad.csv")
    parser.add_argument("--out", default="data/splits")
    args = parser.parse_args()

    train_df, val_df, test_df = medical_split(pd.read_csv(args.input))
    os.makedirs(args.out, exist_ok=True)
    train_df.to_csv(f"{args.out}/train.csv", index=False)
    val_df.to_csv(f"{args.out}/val.csv", index=False)
    test_df.to_csv(f"{args.out}/test.csv", index=False)
    print("\nSplits saved!")