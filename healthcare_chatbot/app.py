import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

@st.cache_resource
def load_model():
    model_repo = "sifakaveza/healthcare-chatbot"

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_repo)
        model = AutoModelForSequenceClassification.from_pretrained(model_repo)
    except Exception as e:
        raise RuntimeError(f"Failed to load model/tokenizer from Hugging Face Hub: {str(e)}")

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
    st.set_page_config(page_title="Healthcare Classifier")
    st.title("Healthcare Question Classifier")
    st.write("Ask a medical question and the model will classify its type.")

    try:
        model, tokenizer, device = load_model()
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Type your medical question here..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Classifying..."):
                try:
                    q_type = classify_question(prompt, model, tokenizer, device)
                    label = q_type.split()[0]
                    explanations = {
                        "DEF": "This appears to be a definition question about medical terms or conditions.",
                        "SX": "This question seems to be about symptoms of a medical condition.",
                        "TX": "This looks like a treatment or therapy-related question.",
                        "DX": "This is likely about diagnostic tests or procedures.",
                        "CLS": "This is a general medical question."
                    }
                    response = f"**Question Type:** {q_type}\n\n{explanations.get(label, '')}"
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.markdown(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

if __name__ == "__main__":
    main()
