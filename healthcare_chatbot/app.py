import streamlit as st
from transformers import BertForSequenceClassification, BertTokenizer
import torch
import numpy as np
import os

@st.cache_resource
def load_model():
    model_path = "final_model"

    # Check if the model directory exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model directory '{model_path}' not found. "
            "Please upload it manually if running locally, "
            "or deploy using Hugging Face Hub or a cloud bucket if deploying."
        )

    try:
        tokenizer = BertTokenizer.from_pretrained(model_path)
        model = BertForSequenceClassification.from_pretrained(model_path)
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
    st.set_page_config(page_title="Healthcare Classifier", page_icon="💊")
    st.title("💬 Healthcare Question Classifier")
    st.write("Ask a medical question and the model will classify its type.")

    try:
        model, tokenizer, device = load_model()
    except Exception as e:
        st.error(f"⚠️ Model loading failed: {str(e)}")
        return

    # Maintain chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Type your medical question here..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Classifying..."):
                try:
                    q_type = classify_question(prompt, model, tokenizer, device)
                    label = q_type.split()[0]  # Extract "DEF", "SX", etc.
                    explanations = {
                        "DEF": "📖 This appears to be a definition question about medical terms or conditions.",
                        "SX": "🤒 This question seems to be about symptoms of a medical condition.",
                        "TX": "💊 This looks like a treatment or therapy-related question.",
                        "DX": "🧪 This is likely about diagnostic tests or procedures.",
                        "CLS": "🩺 This is a general medical question."
                    }
                    response = f"**Question Type:** {q_type}\n\n{explanations.get(label, '')}"

                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    error_msg = f"❌ Error: {str(e)}"
                    st.markdown(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

if __name__ == "__main__":
    main()
