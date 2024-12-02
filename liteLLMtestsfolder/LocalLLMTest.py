from transformers import pipeline

# List of models to test for summarization
models = {
    "BERT (bert-base-uncased)": "bert-base-uncased",
    "BART (Facebook's BART Transformer)": "facebook/bart-large-cnn",
    "T5 (Google’s T5 Transformer)": "t5-base",
    "DistilBERT (Distilled BERT)": "distilbert-base-uncased",
    "Pegasus (Google's Pegasus)": "google/pegasus-xsum",
    "MBART (Facebook's MBART)": "facebook/mbart-large-50-many-to-many-mmt",
    "Helsinki-NLP Opus MT (English-Romanian)": "Helsinki-NLP/opus-mt-en-ro",
    "XLNet (XLNet Transformer)": "xlnet-base-cased",
    "Longformer (AllenAI's Longformer)": "allenai/longformer-base-4096",
    "Reformer (Google's Reformer)": "google/reformer-crime-and-punishment",
    "Falcon-7B (tiiuae Falcon-7B)": "tiiuae/falcon-7b",
    # "Llama (Facebook Llama)": "facebook/llama",  # Uncomment if accessible
}

# Sample text for summarization
sample_text = """
In artificial intelligence (AI), a language model (LM) is a statistical model that calculates 
the probability distribution of words, phrases, and sentences. It uses this information to 
predict or generate words or sentences. Language models are foundational to many natural 
language processing tasks such as translation, summarization, and text generation.
"""

def test_model(model_name, model_identifier):
    print(f"\nTesting {model_name}...")

    try:
        # Load the model pipeline
        model = pipeline("summarization", model=model_identifier)
        
        # Test with the sample text
        result = model(sample_text)
        print(f"Result from {model_name}:")
        print(result[0]['summary_text']) # i can't believe the jupyter notebook extension allows for line by line interactive window, big win for extension guy
    except Exception as e:
        print(f"Error loading {model_name}: {e}")

def main():
    for model_name, model_identifier in models.items():
        test_model(model_name, model_identifier)

if __name__ == "__main__":
    main()
