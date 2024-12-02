# LiteLLMtests - Token Cost Optimization

This repository contains a Streamlit application designed to optimize token usage for text processing with local machine learning models. The app allows users to choose from several pre-trained models for tasks like summarization, text compression, and semantic compression, providing a way to reduce token usage while maintaining the quality of the text.

## Features

- **Model Selection**: Users can select from various local models, including T5, BART, Pegasus, Longformer, and BigBird, all available through Hugging Face and Sentence Transformers.
  
- **Optimization Algorithms**:
  - **Summarization**: Condenses long text into shorter summaries.
  - **Text Compression (BPE)**: Reduces text size using Byte Pair Encoding.
  - **Semantic Compression**: Uses Sentence Transformers to compress text while preserving its semantic meaning.
  - **Hybrid Method**: A placeholder for combining different compression methods, such as RAG and Haystack.

- **Token Comparison**: After processing, the app compares the original and optimized text based on token count and inference time, helping users understand the efficiency of each optimization method.

## Usage

1. **Model Choice**: Select the model from the sidebar.
2. **Text Input**: Enter the text you want to optimize in the provided text area.
3. **Select Optimization Method**: Choose the optimization algorithm to apply.
4. **Results**: After processing, the app will display the optimized text and comparison metrics (e.g., token count reduction and inference times).

## Available Models and Methods

- **Summarization**: Generate concise summaries for long texts.
- **Text Compression (BPE)**: Reduce text size using Byte Pair Encoding.
- **Semantic Compression**: Compress text using Sentence Transformers while preserving meaning.
- **Hybrid Method**: Placeholder for combining multiple techniques.

### Models Used:
- **T5, BART, Pegasus**: Pre-trained summarization models from Hugging Face.
- **BigBird, Longformer**: Efficient transformers for processing long texts.
- **Sentence Transformers**: For semantic compression.
