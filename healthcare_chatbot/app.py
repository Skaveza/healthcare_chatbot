import streamlit as st
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

@st.cache_resource(show_spinner=False)
def load_model():
    model = T5ForConditionalGeneration.from_pretrained("./model/final_model")
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model.eval()
    return model, tokenizer

def generate_answer(question, model, tokenizer):
    input_text = "question: " + question
    inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    with torch.no_grad():
        outputs = model.generate(inputs, max_length=150, num_beams=4, early_stopping=True)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

def main():
    st.title("Healthcare Chatbot")
    st.write("Ask me any healthcare-related question!")

    # Load model once
    try:
        model, tokenizer = load_model()
        st.success("Model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # User input
    if user_question := st.chat_input("Type your healthcare question here..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.write(user_question)

        # Generate assistant response
        with st.chat_message("assistant"):
            with st.spinner("Generating answer..."):
                try:
                    answer = generate_answer(user_question, model, tokenizer)
                    st.write(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    error_msg = f"Sorry, I encountered an error: {e}"
                    st.write(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

if __name__ == "__main__":
    main()
