import pandas as pd
import torch
from transformers import BertTokenizer
from transformers.tokenization_utils_base import BatchEncoding
import os
import re
from sklearn.utils import resample
import numpy as np

torch.serialization._import_dotted_name = lambda name: eval(name)
torch.serialization.add_safe_globals([BatchEncoding])

class MedicalPreprocessor:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained(
            "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
            add_prefix_space=True
        )
        self.label_map = {"DEF": 0, "SX": 1, "TX": 2, "DX": 3, "CLS": 4}
    
    def clean_text(self, text):
        text = re.sub(r'\[.*?\]|\(.*?\)', '', text)
        text = re.sub(r'\b(e\.g|i\.e|etc)\.?\b', '', text, flags=re.IGNORECASE)
        return ' '.join(text.split()).strip()
    
    def preprocess_split(self, df, split_name):
        print(f"\nProcessing {split_name}...")
        assert set(df['q_type'].unique()).issubset(self.label_map.keys())
        
        df['label'] = df['q_type'].map(self.label_map)
        texts = (df['q_type'] + " " + df['question'].str.lower()).tolist()
        
        encodings = self.tokenizer(
            texts,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        encodings['labels'] = torch.tensor(df['label'].values)
        
        print("\nClass Distribution:")
        for cls, name in self.label_map.items():
            count = (encodings['labels'] == name).sum().item()
            print(f"{cls}: {count} samples")
        
        return encodings

if __name__ == "__main__":
    preprocessor = MedicalPreprocessor()
    os.makedirs("/kaggle/working/tokenized", exist_ok=True)
    
    for split in ["train", "val", "test"]:
        try:
            df = pd.read_csv(f"/kaggle/working/splits/{split}.csv")
            encodings = preprocessor.preprocess_split(df, split)
            torch.save(
                encodings,
                f"/kaggle/working/tokenized/{split}_encodings.pt",
                _use_new_zipfile_serialization=True
            )
            print(f"Saved {split} ({len(df)} samples)")
        except Exception as e:
            print(f"Error: {str(e)}")
