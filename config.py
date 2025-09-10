# config.py
import os
import streamlit as st
from dotenv import load_dotenv

# Load .env if running locally
load_dotenv()

def get_secret(key: str) -> str:
    """Get a secret value from Streamlit Cloud or local .env"""
    if key in st.secrets:        # Streamlit Cloud
        return st.secrets[key]
    return os.getenv(key)        # Local fallback
