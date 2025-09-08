# app/config.py
import os
from dotenv import load_dotenv

load_dotenv() # Load variables from a .env file

REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not all([REPLICATE_API_TOKEN, HUGGINGFACE_API_TOKEN, OPENAI_API_KEY, GROQ_API_KEY]):
    raise ValueError("One or more required environment variables are not set.")