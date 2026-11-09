import os

import numpy as np
import torch
from sklearn.metrics import f1_score
from transformers import (
    AutoTokenizer,
    BertForSequenceClassification,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

os.environ["WANDB_DISABLED"] = "true"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

MODEL_NAME = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
TOKENIZED_DIR = "/kaggle/working/tokenized"
OUTPUT_DIR = "/kaggle/working/output"
FINAL_MODEL_DIR = "/kaggle/working/final_model"


class MedQADataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.encodings["labels"][idx],
        }


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": float(np.mean(predictions == labels)),
        "f1_macro": f1_score(labels, predictions, average="macro"),
        "f1_weighted": f1_score(labels, predictions, average="weighted"),
    }


def load_encodings(path):
    return torch.load(path)  # plain dict of tensors


def train_medquad():
    print("\n=== Loading Data ===")
    train_dataset = MedQADataset(load_encodings(f"{TOKENIZED_DIR}/train_encodings.pt"))
    val_dataset = MedQADataset(load_encodings(f"{TOKENIZED_DIR}/val_encodings.pt"))
    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} samples")

    print("\n=== Model Setup ===")
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=5)

    # ~715 optimizer steps/epoch x 5 epochs ≈ 3575 total; warm up over ~10%
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=2,
        num_train_epochs=2,
        learning_rate=2e-5,
        warmup_steps=350,
        weight_decay=0.01,
        max_grad_norm=1.0,
        fp16=torch.cuda.is_available(),
        eval_strategy="steps",
        eval_steps=400,
        save_strategy="steps",
        save_steps=400,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        logging_steps=50,
        report_to="none",
        dataloader_pin_memory=False,
        dataloader_num_workers=2,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print("\n=== Training ===")
    trainer.train()

    print("\n=== Saving ===")
    trainer.save_model(FINAL_MODEL_DIR)  # best checkpoint (load_best_model_at_end=True)
    AutoTokenizer.from_pretrained(MODEL_NAME).save_pretrained(FINAL_MODEL_DIR)
    print(f"Training complete! Model + tokenizer saved to {FINAL_MODEL_DIR}")


if __name__ == "__main__":
    train_medquad()