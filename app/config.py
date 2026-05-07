# config.py
# Handles API key loading for both local and cloud environments
# Locally we use .env file
# On Streamlit Cloud we use their secrets manager

import os
from dotenv import load_dotenv
import streamlit as st

def get_nvidia_key():
    """
    Gets NVIDIA API key from either:
    - Streamlit secrets (when deployed on cloud)
    - .env file (when running locally)
    """
    try:
        # st.secrets works on Streamlit Cloud
        return st.secrets["NVIDIA_API_KEY"]
    except:
        # fallback to .env for local development
        load_dotenv()
        return os.getenv('NVIDIA_API_KEY')