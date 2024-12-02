from aisuite import AISuite
from transformers import pipeline

# Initialize AISuite for local model switching
ai_suite = AISuite()

# Model Choices (local models)
model_choices = ["t5-base", "bart-large", "google/pegasus-xsum", "google/bigbird-roberta-base", "allenai/longformer-base-4096"]
selected_model = ai_suite.switch_model(model_choices)

# Function to perform generic model call (without optimization)
def generic_model_call(input_text):
    model_call_start = time.time()
    
    # Use AISuite to switch and use a local model
    generator = ai_suite.get_pipeline("summarization", model=selected_model)
    output = generator(input_text, max_length=100, num_return_sequences=1)
    output_text = output[0]['summary_text']
    
    model_call_end = time.time()
    inference_time = model_call_end - model_call_start
    return output_text, inference_time
