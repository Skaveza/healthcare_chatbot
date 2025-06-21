
import torch
from torch.utils.data import Dataset
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
import os

# Disable W&B to avoid logging errors on Kaggle
os.environ["WANDB_MODE"] = "disabled"

# Load FLAN-T5 tokenizer and model
model_name = "google/flan-t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Load tokenized data
train_encodings = torch.load("/kaggle/working/tokenized/train_encodings.pt")
val_encodings = torch.load("/kaggle/working/tokenized/val_encodings.pt")

# Dataset class
class MedQADataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    def __len__(self):
        return self.encodings["input_ids"].size(0)
    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()}

train_dataset = MedQADataset(train_encodings)
val_dataset = MedQADataset(val_encodings)

# Training config
training_args = TrainingArguments(
    output_dir="/kaggle/working/final_model",
    num_train_epochs=5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
    load_best_model_at_end=True,
    logging_dir="/kaggle/working/logs",
    fp16=True,
    learning_rate=3e-4,
    evaluation_strategy="epoch",
    lr_scheduler_type="linear",
    logging_steps=20,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

# Train
print("Training FLAN-T5-small model...")
trainer.train()

# Save model
model.save_pretrained("/kaggle/working/healthcare_model")
tokenizer.save_pretrained("/kaggle/working/healthcare_model")
print("✅ Model saved to /kaggle/working/healthcare_model")

