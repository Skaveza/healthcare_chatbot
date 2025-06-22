import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

@st.cache_resource
def load_model():
    model_repo = "sifakaveza/healthcare-chatbot"
    tokenizer_repo = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"

    try:
        # Load model from your Hugging Face repo
        model = AutoModelForSequenceClassification.from_pretrained(model_repo)

        # Load tokenizer from the base model
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_repo)
    except Exception as e:
        raise RuntimeError(f"Failed to load model/tokenizer: {str(e)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, tokenizer, device

def classify_question(question, model, tokenizer, device):
    inputs = tokenizer(
        question,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    pred_class = torch.argmax(outputs.logits).item()

    class_names = {
        0: "Definition (DEF)",
        1: "Symptoms (SX)",
        2: "Treatment (TX)",
        3: "Diagnosis (DX)",
        4: "General (CLS)"
    }

    return class_names.get(pred_class, "General (CLS)")

def main():
    st.title("Healthcare Question Classifier")

    model, tokenizer, device = load_model()

    question = st.text_input("Type your medical question here:")
    if question:
        label = classify_question(question, model, tokenizer, device)
        st.write(f"**Predicted Question Type:** {label}")

if __name__ == "__main__":
    main()
