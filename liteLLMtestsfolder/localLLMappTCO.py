import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import time
import pandas as pd

# Streamlit App Setup
st.title("LLM Token Cost Optimization")
st.sidebar.header("Settings")

# Model Options
MODEL_CHOICES = [
    "t5-base",
    "facebook/bart-large-cnn",
    "google/pegasus-xsum",
    "google/bigbird-roberta-base",
    "allenai/longformer-base-4096"
]
OPTIMIZATION_METHODS = ["Summarization", "Text Compression (BPE)"]

# User Inputs
selected_model = st.sidebar.selectbox("Select Model", MODEL_CHOICES)
optimization_method = st.sidebar.radio("Select Optimization Method", OPTIMIZATION_METHODS)
user_input = st.text_area("Enter your text:", height=200)

# Helper Functions
def load_model_and_tokenizer(model_name):
    """Attempt to load the model and tokenizer."""
    try:
        model = pipeline("summarization", model=model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return model, tokenizer
    except Exception as e:
        st.warning(f"Model '{model_name}' could not be loaded. Error: {str(e)}")
        return None, None

def generic_call(input_text, model_name):
    """Generic call for LLM summarization with error handling."""
    model, _ = load_model_and_tokenizer(model_name)
    if model is None:
        return "Unavailable", 0
    start_time = time.time()
    try:
        output = model(input_text, max_length=100, num_return_sequences=1)[0]['summary_text']
    except Exception as e:
        st.error(f"Error during inference for '{model_name}': {str(e)}")
        return "Unavailable", 0
    duration = time.time() - start_time
    return output, duration

def bpe_compression(input_text, model_name):
    """Apply Byte Pair Encoding (BPE) compression."""
    _, tokenizer = load_model_and_tokenizer(model_name)
    if tokenizer is None:
        return "Unavailable"
    tokens = tokenizer.tokenize(input_text)
    return " ".join(tokens)

def generate_comparison_table(results):
    """Display results in a comparison table."""
    st.subheader("Model Performance Comparison")
    df = pd.DataFrame(results)
    st.write(df)
    if not df[df["Output"] != "Unavailable"].empty:
        best_token_reduction = df.loc[df["Token Reduction"].idxmax(), "Model"]
        best_inference_time = df.loc[df["Inference Time (s)"].idxmin(), "Model"]
        st.subheader("Summary")
        st.write(pd.DataFrame({
            "Metric": ["Best Model for Token Reduction", "Best Model for Inference Time"],
            "Model": [best_token_reduction, best_inference_time]
        }))

# Model Evaluation
if st.button("Evaluate Models"):
    if not user_input.strip():
        st.error("Please enter valid text.")
    else:
        results = []
        for model in MODEL_CHOICES:
            output, inference_time = generic_call(user_input, model)
            token_reduction = len(user_input.split()) - len(output.split()) if output != "Unavailable" else 0
            results.append({
                "Model": model,
                "Original Token Count": len(user_input.split()),
                "Optimized Token Count": len(output.split()) if output != "Unavailable" else "Unavailable",
                "Token Reduction": token_reduction,
                "Inference Time (s)": inference_time,
                "Output": output
            })
        generate_comparison_table(results)

# Text Compression
if st.button("Optimize (BPE)"):
    if not user_input.strip():
        st.error("Please enter valid text.")
    else:
        compressed_results = [{"Model": model, "Compressed Output": bpe_compression(user_input, model)}
                              for model in MODEL_CHOICES]
        st.subheader("BPE Compression Results")
        st.write(pd.DataFrame(compressed_results))

# Info Section
st.write("""
### Models and Methods Overview:
- **T5, BART, Pegasus**: Effective for summarization.
- **BigBird, Longformer**: Handle longer text sequences.
- **Text Compression (BPE)**: Token-based reduction.
""")
