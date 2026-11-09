import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from scr.preprocessing import normalize_question

MODEL_REPO = "sifakaveza/healthcare-chatbot"
TOKENIZER_REPO = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
CLASS_NAMES = ["Definition (DEF)", "Symptoms (SX)", "Treatment (TX)",
               "Diagnosis (DX)", "General (CLS)"]
CONFIDENCE_THRESHOLD = 0.60


@st.cache_resource
def load_model():
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_REPO)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_REPO)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    return model, tokenizer, device


def classify_question(question, model, tokenizer, device):
    inputs = tokenizer(
        normalize_question(question),
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    return torch.softmax(logits, dim=-1)[0].cpu()


def main():
    st.title("Healthcare Question Classifier")
    st.caption("Informational only. This is not medical advice.")

    model, tokenizer, device = load_model()
    question = st.text_input("Type your medical question here:")

    if question:
        probs = classify_question(question, model, tokenizer, device)
        top = int(probs.argmax())
        st.write(f"**Predicted Question Type:** {CLASS_NAMES[top]} ({probs[top]:.0%})")
        if probs[top] < CONFIDENCE_THRESHOLD:
            st.warning("Low confidence. This question may need human review.")
        st.bar_chart({name: float(p) for name, p in zip(CLASS_NAMES, probs)})


if __name__ == "__main__":
    main()