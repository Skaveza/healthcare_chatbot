import torch
from transformers import (
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from transformers.tokenization_utils_base import BatchEncoding
import numpy as np
import os
from sklearn.metrics import f1_score

# Register BatchEncoding for torch.load (PyTorch 2.6+)
torch.serialization.add_safe_globals([BatchEncoding])

# Environment setup
os.environ['WANDB_DISABLED'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use only 1 GPU to save memory

class MedQADataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings['input_ids'])

    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'labels': self.encodings['labels'][idx]
        }

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        'accuracy': np.mean(predictions == labels),
        'f1_macro': f1_score(labels, predictions, average='macro')
    }

def load_encodings(path):
    try:
        return torch.load(path, weights_only=False)
    except:
        return torch.load(path)

def train_medquad():
    print('\n=== Loading Data ===')
    train_data = load_encodings('/kaggle/working/tokenized/train_encodings.pt')
    val_data = load_encodings('/kaggle/working/tokenized/val_encodings.pt')
    
    train_dataset = MedQADataset(train_data)
    val_dataset = MedQADataset(val_data)
    print(f'Train: {len(train_dataset)} | Val: {len(val_dataset)} samples')

    print('\n=== Model Setup ===')
    model = BertForSequenceClassification.from_pretrained(
        'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract',
        num_labels=5,
        ignore_mismatched_sizes=True
    )
    
    # Backward-compatible TrainingArguments setup
    try:
        args = TrainingArguments(
            output_dir='/kaggle/working/output',
            per_device_train_batch_size=8,
            per_device_eval_batch_size=16,
            num_train_epochs=5,
            learning_rate=2e-5,
            warmup_ratio=0.1,
            weight_decay=0.01,
            max_grad_norm=1.0,
            fp16=True,
            eval_strategy='steps',  # older versions
            eval_steps=200,
            save_steps=200,
            load_best_model_at_end=True,
            metric_for_best_model='f1_macro',
            logging_steps=50,
            report_to='none',
            gradient_accumulation_steps=2,
            dataloader_pin_memory=False,
            dataloader_num_workers=2
        )
        print('Using eval_strategy="steps"')
    except TypeError:
        try:
            args = TrainingArguments(
                output_dir='/kaggle/working/output',
                per_device_train_batch_size=8,
                per_device_eval_batch_size=16,
                num_train_epochs=5,
                learning_rate=2e-5,
                warmup_ratio=0.1,
                weight_decay=0.01,
                max_grad_norm=1.0,
                fp16=True,
                evaluation_strategy='steps',  # newer versions
                eval_steps=200,
                save_steps=200,
                load_best_model_at_end=True,
                metric_for_best_model='f1_macro',
                logging_steps=50,
                report_to='none',
                gradient_accumulation_steps=2,
                dataloader_pin_memory=False,
                dataloader_num_workers=2
            )
            print('Using evaluation_strategy="steps"')
        except Exception as e:
            print(f'Error creating TrainingArguments: {e}')
            args = TrainingArguments(
                output_dir='/kaggle/working/output',
                per_device_train_batch_size=8,
                per_device_eval_batch_size=16,
                num_train_epochs=5,
                learning_rate=2e-5,
                warmup_ratio=0.1,
                weight_decay=0.01,
                max_grad_norm=1.0,
                fp16=True,
                gradient_accumulation_steps=2,
                dataloader_pin_memory=False,
                dataloader_num_workers=2
            )
            print('Using basic TrainingArguments without eval strategy')

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    print('\n=== Training ===')
    trainer.train()
    
    # Save model & tokenizer
    model.save_pretrained('/kaggle/working/final_model', save_tokenizer=True)
    print('Training complete!')

if __name__ == '__main__':
    train_medquad()
