# app/config.py
import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN") 
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Application Settings
DEFAULT_MODEL = "flux"  # Most cost-effective default
SUPPORTED_MODELS = ["flux", "stable-diffusion", "openai", "flux-replicate"]
SUPPORTED_AGE_GROUPS = ["0-6", "7-12", "13-16"]

# Display Settings
OLED_WIDTH = 128
OLED_HEIGHT = 64

# Validation - Only require free API keys by default
required_free_keys = [HUGGINGFACE_API_TOKEN, GROQ_API_KEY]
if not all(required_free_keys):
    raise ValueError("Required free API keys not set. Please set HUGGINGFACE_API_TOKEN and GROQ_API_KEY")

# Optional paid API validation
def validate_paid_apis():
    """Validate paid API keys when needed"""
    missing_keys = []
    if not REPLICATE_API_TOKEN:
        missing_keys.append("REPLICATE_API_TOKEN")
    if not OPENAI_API_KEY:
        missing_keys.append("OPENAI_API_KEY")
    
    if missing_keys:
        print(f"Warning: Paid API keys not set: {', '.join(missing_keys)}")
        print("Only free models (FLUX via HuggingFace, Stable Diffusion) will be available.")
    
    return len(missing_keys) == 0

# Cost information
MODEL_COSTS = {
    "flux": "FREE (HuggingFace Serverless)",
    "stable-diffusion": "FREE (HuggingFace Serverless)", 
    "flux-replicate": "~$0.003 per image (Replicate)",
    "openai": "~$0.04 per image (OpenAI DALL-E 3)"
}

MODEL_DESCRIPTIONS = {
    "flux": {
        "name": "FLUX.1-schnell",
        "provider": "HuggingFace Serverless API",
        "speed": "Fast (4-8 seconds)",
        "quality": "Excellent",
        "cost": "FREE",
        "recommended": True
    },
    "stable-diffusion": {
        "name": "Stable Diffusion XL", 
        "provider": "HuggingFace Serverless API",
        "speed": "Medium (10-15 seconds)",
        "quality": "Very Good", 
        "cost": "FREE",
        "recommended": False
    },
    "flux-replicate": {
        "name": "FLUX.1-schnell",
        "provider": "Replicate API",
        "speed": "Very Fast (2-4 seconds)",
        "quality": "Excellent",
        "cost": "Low ($0.003/image)",
        "recommended": False
    },
    "openai": {
        "name": "DALL-E 3",
        "provider": "OpenAI API", 
        "speed": "Medium (10-30 seconds)",
        "quality": "Excellent",
        "cost": "High ($0.04/image)",
        "recommended": False
    }
}