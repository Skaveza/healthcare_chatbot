import pandas as pd
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import evaluate
import numpy as np

# Load tokenizer and model
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
model = T5ForConditionalGeneration.from_pretrained("/kaggle/working/healthcare_model")
model.eval()

# Load and preprocess test data
df = pd.read_csv("/kaggle/working/test.csv")
df = df.dropna(subset=["question", "answer"])
df["question"] = df["question"].astype(str)
df["answer"] = df["answer"].astype(str)

# Generate predictions
predictions = []
for question in df["question"]:
    input_text = "question: " + question
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True).to(model.device)

    with torch.no_grad():
        output = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=64,
            num_beams=4,
            early_stopping=True
        )
    
    pred = tokenizer.decode(output[0], skip_special_tokens=True)
    predictions.append(pred)

# Save predictions
df["generated_answer"] = predictions
df.to_csv("/kaggle/working/test_with_predictions.csv", index=False)
print("Predictions saved to test_with_predictions.csv")

# Evaluation Section

references = df["answer"].tolist()

# BLEU
bleu = load_metric("bleu")
bleu_score = bleu.compute(predictions=[[p.split()] for p in predictions],
                          references=[[r.split()] for r in references])["bleu"]
print(f"\nBLEU: {bleu_score:.4f}")

# ROUGE
rouge = load_metric("rouge")
rouge_result = rouge.compute(predictions=predictions, references=references)
print("ROUGE:")
for key in ["rouge1", "rouge2", "rougeL", "rougeLsum"]:
    print(f"  {key}: {rouge_result[key].mid.fmeasure:.4f}")

# F1 (simple token-overlap-based)
def simple_f1(pred, ref):
    pred_tokens = set(pred.split())
    ref_tokens = set(ref.split())
    common = pred_tokens & ref_tokens
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)

f1_scores = [simple_f1(p, r) for p, r in zip(predictions, references)]
print(f"Average F1 Score: {np.mean(f1_scores):.4f}")
