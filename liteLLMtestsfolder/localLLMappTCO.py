import streamlit as st
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
from sentence_transformers import SentenceTransformer, util
import time
import pandas as pd  # Import pandas for structured data

# Streamlit UI Setup
st.title("Token Cost Optimization with Local Models")
st.sidebar.header("Optimization Settings")

# Model Choices (only local models)
model_choices = [
    "t5-base", "bart-large",  # HuggingFace models for Summarization
    "google/pegasus-xsum",    # Pretrained summarizer model
    "google/bigbird-roberta-base",  # BigBird for long text
    "allenai/longformer-base-4096"  # Longformer for long document processing
]

selected_model = st.sidebar.selectbox("Select Model:", model_choices)

# User Input
user_input = st.text_area("Enter your text:", height=200)

# Optimization Algorithm Selection (Now always available)
method = st.sidebar.radio("Choose Optimization Algorithm:", ["Summarization", "Text Compression (BPE)", "Semantic Compression", "Hybrid Method"])

# Function to perform generic model call (without optimization)
def generic_model_call(model, input_text):
    model_call_start = time.time()
    
    if model in ["t5-base", "bart-large", "google/pegasus-xsum"]:
        generator = pipeline("summarization", model=model)
        output = generator(input_text, max_length=100, num_return_sequences=1)
        output_text = output[0]['summary_text']
    
    else:  # Longformer and BigBird (for long texts)
        generator = pipeline("text-generation", model=model)
        output = generator(input_text, max_length=100, num_return_sequences=1)
        output_text = output[0]['generated_text']
    
    model_call_end = time.time()
    inference_time = model_call_end - model_call_start
    return output_text, inference_time

# Optimization and Text Processing
def process_with_model(method, user_input):
    optimized_text = ""  # Initialize optimized_text to avoid UnboundLocalError
    
    if method == "Summarization":
        summarizer = pipeline("summarization", model=selected_model)
        summarized = summarizer(user_input, max_length=100, min_length=25, do_sample=False)
        optimized_text = summarized[0]['summary_text']
        st.subheader("Summarized Output")
        st.write(optimized_text)

    elif method == "Text Compression (BPE)":
        tokenizer = AutoTokenizer.from_pretrained(selected_model)
        compressed_text = tokenizer.tokenize(user_input)
        optimized_text = " ".join(compressed_text)
        st.subheader("Compressed Output (BPE)")
        st.write(optimized_text)

    elif method == "Semantic Compression":
        sentence_model = SentenceTransformer(selected_model)
        embeddings = sentence_model.encode(user_input, convert_to_tensor=True)
        closest_sentence = util.semantic_search(embeddings, embeddings, top_k=1)
        optimized_text = user_input[closest_sentence[0][0]['corpus_id']]
        st.subheader("Semantically Compressed Output")
        st.write(optimized_text)

    elif method == "Hybrid Method":
        st.subheader("Hybrid Method Placeholder")
        st.write("This is a placeholder for hybrid approaches (RAG, REALM, Haystack).")
    
    return optimized_text

# Trigger Optimization
if st.button("Optimize"):
    if not user_input.strip():
        st.error("Please enter valid text.")
    else:
        optimized_text = process_with_model(method, user_input)

# Function to display token count and compare metrics
def display_token_comparison(original_text, optimized_text, inference_time_generic, inference_time_optimized):
    original_token_count = len(original_text.split())  # Tokenize based on space (simple tokenization)
    optimized_token_count = len(optimized_text.split())
    
    # Prepare data for table
    data = {
        "Metric": ["Original Token Count", "Optimized Token Count", "Token Reduction", 
                   "Generic Inference Time", "Optimized Inference Time"],
        "Value": [original_token_count, optimized_token_count, original_token_count - optimized_token_count, 
                  f"{inference_time_generic:.4f} seconds", f"{inference_time_optimized:.4f} seconds"]
    }
    
    # Create a DataFrame from the data
    df = pd.DataFrame(data)
    
    # Display the table
    st.subheader("Comparison Metrics")
    st.write(df)

# If optimization was done, compare token counts and inference times
if user_input.strip() and st.button("Show Comparison Metrics"):
    # Perform generic model call
    generic_output, generic_inference_time = generic_model_call(selected_model, user_input)
    
    # Perform optimized processing
    optimized_text = process_with_model(method, user_input)
    
    # Calculate optimized inference time
    optimized_inference_time = time.time() - generic_inference_time
    
    # Display comparison between generic call and optimized output
    display_token_comparison(user_input, optimized_text, generic_inference_time, optimized_inference_time)

# Explanation of Methods and Models
st.write("""
### Available Models and Methods:
- **Summarization**: Generate concise summaries for large texts.
- **Text Compression (BPE)**: Use Byte Pair Encoding to reduce text size.
- **Semantic Compression**: Compress text while preserving semantics using Sentence Transformers.
- **Hybrid Method**: A placeholder for combining semantic search and text generation (e.g., RAG, Haystack).
### Models Used:
- **T5, BART, Pegasus** (Summarization models from Hugging Face)
- **BigBird, Longformer** (Efficient transformers for long sequences)
- **Sentence Transformers** (For Semantic Compression)
""")
# Debug: Output currently selected model and method
st.write(f"Currently selected model: {selected_model}")
