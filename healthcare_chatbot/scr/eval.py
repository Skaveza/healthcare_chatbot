import torch
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
)
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    f1_score,
    confusion_matrix
)
import numpy as np
import pandas as pd
from tqdm import tqdm
import os

class MedQuadEvaluator:
    def __init__(self, model_path="/kaggle/working/final_model"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load tokenizer - fallback to original model if not found
        try:
            self.tokenizer = BertTokenizer.from_pretrained(model_path)
        except OSError:
            print(f"Tokenizer not found in {model_path}, using original model's tokenizer")
            self.tokenizer = BertTokenizer.from_pretrained(
                "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
            )
        
        # Load model
        self.model = BertForSequenceClassification.from_pretrained(model_path).to(self.device)
        self.model.eval()
        
        self.label_map = {
            "DEF": 0,  # Definitions
            "SX": 1,   # Symptoms
            "TX": 2,   # Treatments
            "DX": 3,   # Diagnosis
            "CLS": 4   # General
        }
        self.inverse_map = {v: k for k, v in self.label_map.items()}
    
    def load_test_data(self, test_path="/kaggle/working/splits/test.csv"):
        """Load and clean test data"""
        if not os.path.exists(test_path):
            raise FileNotFoundError(f"Test file not found at {test_path}")
            
        df = pd.read_csv(test_path)
        df = df.dropna(subset=['question', 'answer', 'q_type'])
        return df
    
    def batch_predict(self, texts, batch_size=32):
        """Run batched predictions with progress bar"""
        predictions = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Predicting"):
            batch = texts[i:i+batch_size]
            inputs = self.tokenizer(
                batch,
                max_length=128,
                padding='max_length',
                truncation=True,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            predictions.extend(torch.argmax(outputs.logits, dim=-1).cpu().numpy())
        return predictions
    
    def evaluate(self):
        """Run full evaluation pipeline"""
        try:
            test_df = self.load_test_data()
            
            # Prepare data - maintain same format as training
            texts = (test_df['q_type'] + " " + test_df['question'].str.lower()).tolist()
            y_true = test_df['q_type'].map(self.label_map).values
            
            # Predict
            y_pred = self.batch_predict(texts)
            
            # Calculate metrics
            accuracy = accuracy_score(y_true, y_pred)
            f1_macro = f1_score(y_true, y_pred, average='macro')
            f1_weighted = f1_score(y_true, y_pred, average='weighted')
            
            print("\n=== Evaluation Results ===")
            print(f"Accuracy: {accuracy:.2%}")
            print(f"Macro F1: {f1_macro:.2%}")
            print(f"Weighted F1: {f1_weighted:.2%}")
            
            # Detailed reports
            target_names = list(self.label_map.keys())
            print("\nClassification Report:")
            print(classification_report(
                y_true, y_pred,
                target_names=target_names,
                digits=4
            ))
            
            print("\nConfusion Matrix:")
            print(confusion_matrix(y_true, y_pred))
            
            return {
                'accuracy': accuracy,
                'f1_macro': f1_macro,
                'f1_weighted': f1_weighted,
                'y_true': y_true,
                'y_pred': y_pred
            }
            
        except Exception as e:
            print(f"\nError during evaluation: {str(e)}")
            return None

if __name__ == "__main__":
    print("=== Starting Evaluation ===")
    evaluator = MedQuadEvaluator()
    results = evaluator.evaluate()
    
    if results:
        print("\nEvaluation completed successfully!")
    else:
        print("\nEvaluation failed")
