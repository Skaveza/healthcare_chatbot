import streamlit as st
from transformers import (
    BertForSequenceClassification,
    BertTokenizer
)
import torch
import numpy as np

@st.cache_resource
def load_model():
    model_path = "./final_model"
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)
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
    st.write("Ask any medical question to get it classified")
    
    try:
        model, tokenizer, device = load_model()
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return
    
    # Chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # User input
    if prompt := st.chat_input("Type your medical question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.write(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                try:
                    # Classify question
                    q_type = classify_question(prompt, model, tokenizer, device)
                    response = f"Question Type: **{q_type}**\n\n"
                    
                    # Add explanation
                    explanations = {
                        "DEF": "This appears to be a definition question about medical terms or conditions.",
                        "SX": "This question seems to be about symptoms of a medical condition.",
                        "TX": "This looks like a treatment/therapy-related question.",
                        "DX": "This is likely about diagnostic tests or procedures.",
                        "CLS": "This is a general medical question."
                    }
                    response += explanations.get(q_type.split()[0], "")
                    
                    st.write(response)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response
                    })
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.write(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })

if __name__ == "__main__":
    main()