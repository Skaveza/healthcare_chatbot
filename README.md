# Healthcare Chatbot

A healthcare question-answering chatbot built using T5 transformer model and deployed with Streamlit. This project fine-tunes a T5-small model on medical Q&A data to provide intelligent responses to healthcare-related questions.

## Features

- **AI-Powered Responses**: Uses fine-tuned T5 transformer model for generating healthcare answers
- **Interactive Web Interface**: Streamlit-based chat interface for easy interaction
- **Memory Optimized**: Designed to run efficiently on Google Colab with limited resources
- **Real-time Chat**: Chat-based interface with conversation history
- **Medical Domain Focused**: Trained specifically on healthcare Q&A dataset

## Dataset

The project uses the MedQuAD (Medical Question Answering Dataset) which contains:
- Medical questions and their corresponding answers
- Questions covering various healthcare topics
- Preprocessed for optimal model training

## Model Architecture

- **Base Model**: T5-small (Text-to-Text Transfer Transformer)
- **Task**: Conditional text generation for Q&A
- **Fine-tuning**: Domain-specific training on healthcare data
- **Optimization**: Memory-efficient training with reduced parameters

## Requirements

```
transformers
torch
datasets
rouge-score
nltk
streamlit
pyngrok
accelerate
pandas
numpy
matplotlib
scikit-learn
```

## Installation & Setup

### For Google Colab (Recommended)

1. **Mount Google Drive** and upload your `medquad.csv` dataset to `/content/drive/MyDrive/Colab Notebooks/`

2. **Run the installation cell**:
```python
!pip install transformers torch datasets rouge-score nltk streamlit pyngrok
!pip install accelerate -U
```

3. **Execute each cell step by step** as provided in the notebook

### For Local Environment

1. **Clone the repository**:
```bash
git clone <repository-url>
cd healthcare-chatbot
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Download NLTK data**:
```python
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
```

## Usage

### Training the Model

The training process is divided into manageable steps:

1. **Data Preprocessing**: 
   - Loads and cleans the MedQuAD dataset
   - Filters questions and answers by length
   - Uses subset of data (2000 samples) for memory efficiency

2. **Model Training**:
   - Fine-tunes T5-small model
   - Configurable hyperparameters
   - Evaluation metrics include BLEU score and accuracy

3. **Model Saving**:
   - Saves trained model and tokenizer for deployment

### Running the Chatbot

#### In Google Colab:
```python
# The notebook automatically sets up ngrok tunnel
# Follow the generated public URL to access your chatbot
```

#### Locally:
```bash
streamlit run healthcare_chatbot_app.py
```

### Sample Questions

Try asking questions like:
- "What are the symptoms of diabetes?"
- "How can I prevent heart disease?"
- "What should I do if I have a fever?"
- "What are the side effects of aspirin?"

## Configuration

### Training Parameters

```python
# Adjustable parameters
max_input_length = 64      # Maximum input sequence length
max_target_length = 64     # Maximum output sequence length
learning_rate = 3e-4       # Learning rate for training
batch_size = 4             # Training batch size
num_epochs = 1             # Number of training epochs
```

### Generation Parameters

```python
# Text generation settings
max_length = 128           # Maximum response length
num_beams = 4             # Number of beams for beam search
temperature = 0.7         # Sampling temperature
top_p = 0.9              # Nucleus sampling parameter
```

## Model Performance

The model is evaluated using:
- **BLEU Score**: Measures text generation quality
- **Accuracy**: Exact match accuracy
- **ROUGE Score**: Evaluates text summarization quality

## Memory Optimization Features

- **Reduced Dataset Size**: Uses subset of data for training
- **Batch Processing**: Processes data in smaller batches
- **Efficient Tokenization**: Optimized sequence lengths
- **Model Checkpointing**: Saves best model during training

## File Structure

```
healthcare-chatbot/
├── healthcare_chatbot_app.py    # Streamlit web application
├── healthcare_chatbot_model/    # Trained model directory
│   ├── pytorch_model.bin
│   ├── config.json
│   └── tokenizer files
├── medquad.csv                  # Training dataset
├── logs/                        # Training logs
└── README.md                    # This file
```

## Limitations

- **Dataset Size**: Limited to 2000 samples for memory efficiency
- **Model Size**: Uses T5-small for resource constraints
- **Sequence Length**: Limited input/output lengths (64 tokens)
- **Domain Scope**: Focused on general healthcare questions

## Deployment Options

### Google Colab + ngrok
- Quick setup and testing
- Free GPU access
- Temporary public URLs

### Local Deployment
- Full control over environment
- Persistent deployment
- Custom domain options

### Cloud Deployment
- Scalable solutions
- Production-ready
- Services like Heroku, AWS, or Google Cloud

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Disclaimer

 **Important**: This chatbot is for educational and informational purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare providers for medical concerns.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Hugging Face Transformers library
- Google T5 model
- MedQuAD dataset creators
- Streamlit framework
- Google Colab platform

## Support

For questions and support:
- Create an issue in the repository
- Check the documentation
- Review the code comments for implementation details
