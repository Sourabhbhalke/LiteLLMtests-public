from transformers import AutoModel, AutoTokenizer, logging
import os

# Suppress unnecessary warnings
logging.set_verbosity_error()

# List of model names to check
models_to_check = [
    "bert-base-uncased",
    "facebook/bart-large-cnn",
    "t5-base",
    "distilbert-base-uncased",
    "google/pegasus-xsum",
    "facebook/mbart-large-50-many-to-many-mmt",
    "Helsinki-NLP/opus-mt-en-ro",
    "xlnet-base-cased",
    "allenai/longformer-base-4096",
    "google/reformer-crime-and-punishment",
    "tiiuae/falcon-7b",
    "facebook/llama",  # This one may require zuckerberg authentication for download
]

# Function to check if the model files exist and can be loaded
def check_model_downloaded(model_name):
    try:
        # Load the tokenizer and model to see if the model is accessible
        model = AutoModel.from_pretrained(model_name, local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        
        print(f"Model {model_name} is successfully downloaded and accessible.")
    except Exception as e:
        print(f"Model {model_name} failed to load. Error: {str(e)}")

# Check all models
for model_name in models_to_check:
    print(f"Testing model: {model_name}")
    check_model_downloaded(model_name)
    print("-" * 50)
