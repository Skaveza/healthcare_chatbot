# Medical Question Classifier Chatbot

This project is a BERT-based medical chatbot that classifies user questions into predefined clinical categories and provides informative responses. It leverages domain-specific pretrained models like PubMedBERT to improve performance on biomedical language.

---

## Overview

The chatbot supports classification of medical inquiries into five categories:
- DEF: Definitions
- SX: Symptoms
- TX: Treatments
- DX: Diagnosis
- CLS: General/Other

It processes natural language input from users, predicts the relevant category, and returns concise medical explanations tailored to that category.

---

## Highlights

- Custom preprocessing pipeline for biomedical text
- Balanced, stratified dataset creation from MedQuAD
- Fine-tuning of PubMedBERT using weighted macro F1 as the key metric
- Clean evaluation reports and confusion matrix visualization
- Streamlit-based frontend for real-time interaction

---

## Evaluation Summary

Final model performance on the test set:
- Accuracy: 85%
- Macro F1 Score: 81%
- Weighted F1 Score: 83%

This configuration achieved a strong balance across all five question types, and was selected after tuning learning rate, batch size, and dropout values.

---

## Deployment with Streamlit

You can run the chatbot interface locally using Streamlit.

### How to Launch

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## Features

- Interactive text input for user questions
- Real-time classification and response generation
- Visual feedback on predicted category
- Responsive layout optimized for desktop/tablet

---

## Future Improvements

- Integrate retrieval-based or generative answering
- Add multilingual support
- Improve handling of ambiguous or rare questions
