
import pandas as pd
import torch
from transformers import T5Tokenizer
import os

# Use FLAN tokenizer
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")

# Sequence lengths (adjusted for short QA)
max_input_length = 48
max_target_length = 48

def clean_and_filter(df):
    df = df.dropna(subset=['question', 'answer'])
    df['question'] = df['question'].astype(str)
    df['answer'] = df['answer'].astype(str)
    df['q_word_len'] = df['question'].apply(lambda x: len(x.split()))
    df['a_word_len'] = df['answer'].apply(lambda x: len(x.split()))
    df = df[(df['q_word_len'] >= 3) & (df['q_word_len'] <= 30)]
    df = df[(df['a_word_len'] >= 3) & (df['a_word_len'] <= 50)]
    df = df.head(2000)  # adjust based on memory
    return df

def preprocess_data(df_subset):
    # FLAN uses instruction-style prompts
    inputs = ["Answer the medical question: " + q for q in df_subset['question']]
    targets = [str(a) for a in df_subset['answer']]

    model_inputs = tokenizer(
        inputs,
        max_length=max_input_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets,
            max_length=max_target_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

    model_inputs["labels"] = labels["input_ids"]
    model_inputs["labels"][model_inputs["labels"] == tokenizer.pad_token_id] = -100
    return model_inputs

# Main Execution
if __name__ == "__main__":
    os.makedirs("/kaggle/working/tokenized", exist_ok=True)

    for split in ["train", "val", "test"]:
        try:
            print(f"Processing {split}.csv...")
            df = pd.read_csv(f"/kaggle/working/{split}.csv")
            df = clean_and_filter(df)
            encodings = preprocess_data(df)
            torch.save(encodings, f"/kaggle/working/tokenized/{split}_encodings.pt")
            print(f"Saved {split}_encodings.pt")
        except FileNotFoundError:
            print(f" {split}.csv not found. Skipping.")

