import os 
import requests
import streamlit as st
import sys
import yaml
from dotenv import load_dotenv, find_dotenv
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import json

# Add custom module paths (hope this doesn't break i did it 6 times today!)
sys.path.append("../../../aisuite")
from aisuite.client import Client

# Set up Streamlit UI layout
st.set_page_config(layout="wide", menu_items={})
st.markdown(
    "<div style='padding-top: 1rem;'><h2 style='text-align: center; color: #ffffff;'>Chat & Compare LLM Responses</h2></div>",
    unsafe_allow_html=True,
)
st.markdown(
    """
    <style>
        /* Global font size and custom button styles */
        html, body, [class*="css"] {
            font-size: 14px !important;
        }
        
        button[data-testid="stButton"][aria-label="Reset Chat"]:focus {
            border-color: red !important;
            box-shadow: 0 0 0 2px red !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Load configuration and initialize aisuite client (meh ...)
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)
configured_llms = config["llms"]
load_dotenv(find_dotenv())  # Do you feel loaded? 
client = Client()

# Local model cache (aka the cache of dreams)
CACHE_FILE = 'model_download_cache.json'

# Load cached errors if the cache file exists (to keep the nightmares in check)
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, 'r') as cache_file:
        model_errors = json.load(cache_file)
else:
    model_errors = {}

# Check if local models are downloaded (i just don't trust API calls bruh)
def check_model_downloaded(model_name):
    if model_name in model_errors:
        return model_errors[model_name]

    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, local_files_only=False)
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=False)
        model_errors[model_name] = 'success'  # Everything is fine here :)
        return 'success'
    except Exception as e:
        model_errors[model_name] = str(e)  # help chatGPT!
        return str(e)

# Function to display chat history (yea insert armchair historian joke here)
def display_chat_history(chat_history, model_name):
    for message in chat_history:
        role_display = "User" if message["role"] == "user" else model_name
        role = "user" if message["role"] == "user" else "assistant"
        if role == "user":
            with st.chat_message(role, avatar="👤"):
                st.write(message["content"])
        else:
            with st.chat_message(role, avatar="🤖"):
                st.write(message["content"])

# Function to query LLM (do i understand this part?)
def query_llm(model_config, chat_history):
    print(f"Querying {model_config['name']} with {chat_history}")
    try:
        if model_config["provider"] == "api":
            model = model_config["provider"] + ":" + model_config["model"]
            response = client.chat.completions.create(model=model, messages=chat_history)
            return response.choices[0].message.content
        else:
            model_name = model_config["model"]
            model_check = check_model_downloaded(model_name)
            
            if model_check != 'success':
                raise Exception(f"Model download issue: {model_check}")

            local_model = pipeline("text-generation", model=model_name, tokenizer=model_name)
            response = local_model(chat_history[-1]["content"])
            return response[0]['generated_text']
    except Exception as e:
        st.error(f"Error querying {model_config['name']}: {e}")
        return "Error with LLM response."  # More like "Error with human sanity."

# Initialize session states (even if it's not always stable, looking at you --> huggingface cache :( yk )
if "chat_history_1" not in st.session_state:
    st.session_state.chat_history_1 = []
if "chat_history_2" not in st.session_state:
    st.session_state.chat_history_2 = []
if "is_processing" not in st.session_state:
    st.session_state.is_processing = False
if "use_comparison_mode" not in st.session_state:
    st.session_state.use_comparison_mode = False

# Layout for controls and model selection
col1, col2 = st.columns([1, 2])
with col1:
    st.session_state.use_comparison_mode = st.checkbox("Comparison Mode", value=True)

llm_col1, llm_col2 = st.columns(2)
with llm_col1:
    selected_model_1 = st.selectbox(
        "Choose LLM Model 1",
        [llm["name"] for llm in configured_llms],
        key="model_1",
        index=0 if configured_llms else 0,
    )
with llm_col2:
    if st.session_state.use_comparison_mode:
        selected_model_2 = st.selectbox(
            "Choose LLM Model 2",
            [llm["name"] for llm in configured_llms],
            key="model_2",
            index=1 if len(configured_llms) > 1 else 0,
        )

# Display chat history (We store history, unlike what VPNs claim, i should stop watching youtube ig)
if st.session_state.use_comparison_mode:
    col1, col2 = st.columns(2)
    with col1:
        chat_container = st.container(height=500)
        with chat_container:
            display_chat_history(st.session_state.chat_history_1, selected_model_1)
    with col2:
        chat_container = st.container(height=500)
        with chat_container:
            display_chat_history(st.session_state.chat_history_2, selected_model_2)
else:
    chat_container = st.container(height=500)
    with chat_container:
        display_chat_history(st.session_state.chat_history_1, selected_model_1)

# User input section ( i am attracting bugs here, not a frontend dev -_-)
col1, col2, col3 = st.columns([6, 1, 1])
with col1:
    user_query = st.text_area(
        label="Enter your query",
        label_visibility="collapsed",
        placeholder="Enter your query...",
        key="query_input",
        height=70,
    )

with col2:
    send_button = False
    if st.session_state.is_processing:
        st.markdown("<div class='button-container'>Processing... ⏳</div>", unsafe_allow_html=True)
    else:
        send_button = st.button("Send Query", use_container_width=True)

with col3:
    if st.button("Reset Chat", use_container_width=True):
        st.session_state.chat_history_1 = []
        st.session_state.chat_history_2 = []
        st.rerun()

# Handle the send button click (throw error? check for clearing cache and ...idk)
if send_button and user_query and not st.session_state.is_processing:
    st.session_state.is_processing = True
    st.session_state.chat_history_1.append({"role": "user", "content": user_query})
    if st.session_state.use_comparison_mode:
        st.session_state.chat_history_2.append({"role": "user", "content": user_query})
    st.rerun()

# Process the query with selected models (tbh my favorite part here)
if st.session_state.is_processing and user_query:
    model_config_1 = next(
        llm for llm in configured_llms if llm["name"] == selected_model_1
    )
    response_1 = query_llm(model_config_1, st.session_state.chat_history_1)
    st.session_state.chat_history_1.append({"role": "assistant", "content": response_1})

    if st.session_state.use_comparison_mode:
        model_config_2 = next(
            llm for llm in configured_llms if llm["name"] == selected_model_2
        )
        response_2 = query_llm(model_config_2, st.session_state.chat_history_2)
        st.session_state.chat_history_2.append(
            {"role": "assistant", "content": response_2}
        )

    st.session_state.is_processing = False
    st.rerun()
