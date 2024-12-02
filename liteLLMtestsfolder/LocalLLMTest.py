from transformers import pipeline

# List of models to test for summarization
models = {
    "BART (Facebook's BART Transformer)": "facebook/bart-large-cnn",
    "T5 (Google’s T5 Transformer)": "t5-base",
    # Only including models known to work for summarization tasks
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
        print(result[0]['summary_text'])
    except Exception as e:
        print(f"Error loading {model_name}: {e}")

def main():
    for model_name, model_identifier in models.items():
        test_model(model_name, model_identifier)

if __name__ == "__main__":
    main()
