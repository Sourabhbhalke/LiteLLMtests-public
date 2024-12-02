from transformers import AutoModel, AutoTokenizer, logging
import os
import json

# Suppress unnecessary warnings
logging.set_verbosity_error()

# Path to cache the model download status
CACHE_FILE = 'model_download_cache.json'

# List of model names to check (commented out facebook/llama for now)
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
    # "facebook/llama",  # This one may require zuckerberg authentication for download
]

# Load cached errors if the cache file exists
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, 'r') as cache_file:
        model_errors = json.load(cache_file)
else:
    model_errors = {}

# Function to check if the model files exist and can be loaded
def check_model_downloaded(model_name):
    # Skip models with previously recorded errors
    if model_name in model_errors:
        print(f"Skipping model {model_name} due to previous error: {model_errors[model_name]}")
        return
    
    try:
        # Attempt to load the model and tokenizer
        model = AutoModel.from_pretrained(model_name, local_files_only=False)
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=False)
        
        print(f"Model {model_name} is successfully downloaded and accessible.")
        model_errors[model_name] = 'success'  # Cache success

    except Exception as e:
        # Capture and store the error if the model can't be loaded
        error_message = str(e)
        print(f"Model {model_name} failed to load. Error: {error_message}")
        model_errors[model_name] = error_message  # Cache error message

    # Save updated cache after each model check
    with open(CACHE_FILE, 'w') as cache_file:
        json.dump(model_errors, cache_file, indent=4)

# Check all models
for model_name in models_to_check:
    print(f"Testing model: {model_name}")
    check_model_downloaded(model_name)
    print("-" * 50)
