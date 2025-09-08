# app/main.py - Enhanced Cost-Effective Educational Poster Generator

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.responses import JSONResponse, Response, HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import requests
from groq import Groq
import os
import base64
import io
from PIL import Image, ImageEnhance, ImageOps
import numpy as np
from typing import Optional, Dict, Any
import asyncio
import logging
from datetime import datetime
import json
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Educational Poster Generator - Cost Effective", version="2.0.0")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

load_dotenv()

# Environment variables
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize clients
groq_client = Groq(api_key=GROQ_API_KEY)

# Pydantic models
class PosterRequest(BaseModel):
    topic: str
    age_group: str  # "0-6", "7-12", "13-16"
    model: str = "flux-schnell"  # Default to most cost-effective
    complexity: Optional[str] = "simple"
    style: Optional[str] = "infographic"

class PosterResponse(BaseModel):
    status: str
    image_id: str
    model_used: str
    generation_time: float
    cost_estimate: str
    message: Optional[str] = None

# In-memory cache
poster_cache: Dict[str, bytes] = {}

def create_enhanced_educational_prompt(topic: str, age_group: str, style: str = "infographic") -> str:
    """
    Enhanced Groq prompt creation specifically for educational infographics
    """
    try:
        age_specifications = {
            "0-6": {
                "complexity": "very simple, single concept focus",
                "elements": "basic shapes, primary colors, minimal text, large icons",
                "text": "single words, maximum 3-4 labels",
                "visual_style": "cartoon-like, friendly, rounded shapes"
            },
            "7-12": {
                "complexity": "moderate detail, step-by-step processes",
                "elements": "clear diagrams, process arrows, explanatory icons",
                "text": "short phrases, clear labeling, instructional text",
                "visual_style": "clean infographic, educational diagram style"
            },
            "13-16": {
                "complexity": "detailed scientific accuracy, technical concepts",
                "elements": "complex diagrams, molecular structures, detailed processes",
                "text": "technical terminology, detailed explanations, scientific accuracy",
                "visual_style": "scientific poster, technical illustration"
            }
        }
        
        age_spec = age_specifications.get(age_group, age_specifications["7-12"])
        
        # Enhanced system prompt for better instruction following
        system_prompt = f"""You are an expert educational content creator and prompt engineer. Create a highly detailed image generation prompt for an educational poster about '{topic}' designed for {age_group} year olds.

CRITICAL REQUIREMENTS for ESP32 OLED Display (128x64 monochrome):
- HIGH CONTRAST: Pure black text/icons on white background
- BOLD TYPOGRAPHY: Sans-serif, minimum 12pt equivalent, thick strokes
- SIMPLE GEOMETRY: No fine details, minimum 3px line weights
- CLEAR HIERARCHY: Title at top, main concept center, details around edges
- MINIMAL ELEMENTS: Maximum 6 visual components total
- INFOGRAPHIC STYLE: Clean, educational, diagram-like

Age-specific requirements:
- Complexity: {age_spec['complexity']}
- Visual elements: {age_spec['elements']}
- Text style: {age_spec['text']}
- Visual approach: {age_spec['visual_style']}

Your response should be a single, detailed prompt starting with:
"Educational infographic poster about {topic}, {age_spec['visual_style']}, high contrast black and white design, bold typography, simplified vector illustrations, {age_spec['elements']}, clean white background, minimalist layout, classroom wall poster, {age_spec['text']}, OLED display optimized"

Include specific visual elements that would help students understand {topic} at the {age_group} level.
End with: "Negative: photorealistic, cluttered, small text, complex gradients, dark backgrounds, fine details, low contrast"
"""

        response = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Create the perfect image generation prompt for teaching {topic} to {age_group} year olds, optimized for monochrome display"}
            ],
            model="llama-3.1-8b-instant",  # Better instruction following
            temperature=0.3,  # Lower for more consistent results
            max_tokens=300
        )
        
        enhanced_prompt = response.choices[0].message.content.strip()
        logger.info(f"Enhanced prompt generated: {enhanced_prompt}")
        return enhanced_prompt
        
    except Exception as e:
        logger.error(f"Groq API error: {e}")
        # Improved fallback based on your reference image style
        return f"Educational infographic poster about {topic}, high contrast black and white design, bold typography, simplified vector illustrations, clean diagrams with labels, arrows showing process flow, child-friendly icons, white background, classroom poster style, OLED display optimized. Negative: photorealistic, cluttered, small text, dark backgrounds"

def convert_to_oled_bitmap_enhanced(image_data: bytes) -> bytes:
    """
    Enhanced conversion optimized for educational content
    """
    try:
        # Open and process image
        image = Image.open(io.BytesIO(image_data))
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize maintaining aspect ratio
        image.thumbnail((128, 64), Image.Resampling.LANCZOS)
        
        # Create new image with white background
        oled_image = Image.new('RGB', (128, 64), 'white')
        
        # Center the image
        x_offset = (128 - image.width) // 2
        y_offset = (64 - image.height) // 2
        oled_image.paste(image, (x_offset, y_offset))
        
        # Enhanced processing for educational content
        # 1. Increase contrast significantly
        enhancer = ImageEnhance.Contrast(oled_image)
        oled_image = enhancer.enhance(2.5)
        
        # 2. Increase sharpness for better text readability
        enhancer = ImageEnhance.Sharpness(oled_image)
        oled_image = enhancer.enhance(2.0)
        
        # 3. Convert to grayscale with better algorithm
        oled_image = oled_image.convert('L')
        
        # 4. Apply threshold for clean black/white conversion
        threshold = 128
        oled_image = oled_image.point(lambda x: 255 if x > threshold else 0, mode='1')
        
        # Convert to XBM format for ESP32
        bitmap_data = []
        pixels = np.array(oled_image)
        
        # Optimized bitmap generation
        for y in range(64):
            for x in range(0, 128, 8):
                byte = 0
                for bit in range(8):
                    if x + bit < 128:
                        if pixels[y, x + bit] == 0:  # Black pixel
                            byte |= (1 << bit)
                bitmap_data.append(byte)
        
        return bytes(bitmap_data)
        
    except Exception as e:
        logger.error(f"Enhanced image conversion error: {e}")
        raise HTTPException(status_code=500, detail=f"Image conversion failed: {str(e)}")

# =============================================================================
# COST-EFFECTIVE IMAGE GENERATION FUNCTIONS
# =============================================================================

def generate_with_flux_schnell_free(prompt: str):
    """
    Generates an image using the FLUX.1-schnell model on Hugging Face.

    Args:
        prompt (str): The text prompt for image generation.

    Returns:
        A PIL.Image object if successful, None otherwise.
    """
    print("Attempting to generate image with FLUX.1-schnell...")
    try:
        # Check for the required environment variable
        hf_token = os.getenv("HUGGINGFACE_API_TOKEN")
        if not hf_token:
            print("ERROR: HF_TOKEN environment variable not found. Please set it.")
            return None

        # Initialize the InferenceClient with the correct provider and model
        # The 'together' provider is often used for FLUX.1-schnell.
        client = InferenceClient(
            provider="together",
            api_key=hf_token,
        )

        # Make the text-to-image API call
        # The 'model' argument is specified here, but can also be passed in the client constructor.
        image = client.text_to_image(
            prompt,
            model="black-forest-labs/FLUX.1-schnell",
            # Add any additional parameters required by the model, e.g., negative_prompt, width, height
        )

        print("Successfully generated image with FLUX.1-schnell!")
        return image

    except Exception as e:
        print(f"ERROR: FLUX generation failed: {e}")
        return None

def generate_with_stable_diffusion_free(prompt: str):
    """
    Generates an image using the Stable Diffusion XL model on Hugging Face.

    Args:
        prompt (str): The text prompt for image generation.

    Returns:
        A PIL.Image object if successful, None otherwise.
    """
    print("Attempting to generate image with Stable Diffusion XL...")
    try:
        # Check for the required environment variable
        hf_token = os.getenv("HUGGINGFACE_API_TOKEN")
        
        if not hf_token:
            print("ERROR: HF_TOKEN environment variable not found. Please set it.")
            return None

        # Initialize the InferenceClient. The default provider works for many models.
        client = InferenceClient(
            api_key=hf_token,
        )

        # Make the text-to-image API call with the SDXL model
        image = client.text_to_image(
            prompt,
            model="stabilityai/stable-diffusion-xl-base-1.0",
            # SDXL works best with a specific image size, so it's good practice to specify it.
            width=1024,
            height=1024
        )
        print("Successfully generated image with Stable Diffusion XL!")
        return image

    except Exception as e:
        print(f"ERROR: Stable Diffusion XL generation failed: {e}")
        return None

if __name__ == '__main__':
    # This is an example of how you would use the functions.
    # Make sure to set your HF_TOKEN environment variable before running.
    # For example, in your terminal: export HF_TOKEN="your_hugging_face_token_here"
    # or if you are on Windows: set HF_TOKEN="your_hugging_face_token_here"

    test_prompt = "A high-contrast black and white poster about the water cycle, vector art style, clean and simple design"

    # Test FLUX.1-schnell generation
    flux_image = generate_with_flux_schnell_free(test_prompt)
    if flux_image:
        # You can save the image or process it further here
        flux_image.save("water_cycle_flux.png")
        print("Saved image as water_cycle_flux.png")

    print("-" * 20)

    # Test Stable Diffusion XL generation
    sdxl_image = generate_with_stable_diffusion_free(test_prompt)
    if sdxl_image:
        sdxl_image.save("water_cycle_sdxl.png")
        print("Saved image as water_cycle_sdxl.png")



# =============================================================================
# ENHANCED API ENDPOINTS
# =============================================================================

@app.post("/generate-poster", response_model=PosterResponse)
async def generate_educational_poster(request: PosterRequest, background_tasks: BackgroundTasks):
    """
    Generate cost-effective educational poster with enhanced instruction following
    """
    start_time = datetime.now()
    
    try:
        # Create enhanced prompt using Groq
        enhanced_prompt = create_enhanced_educational_prompt(
            request.topic, 
            request.age_group, 
            request.style
        )
        
        logger.info(f"Topic: {request.topic} | Age: {request.age_group}")
        logger.info(f"Enhanced prompt: {enhanced_prompt[:200]}...")
        
        # Generate image with cost-effective model
        if request.model.lower() == "flux-schnell":
            image = generate_with_flux_schnell_free(enhanced_prompt)
            cost_estimate = "FREE (HF Serverless)"
        elif request.model.lower() == "stable-diffusion":
            image = generate_with_stable_diffusion_free(enhanced_prompt)
            cost_estimate = "FREE (HF Serverless)"
        else:
            image = generate_with_flux_schnell_free(enhanced_prompt)
            cost_estimate = "FREE (HF Serverless)"

        if image is None:
            raise HTTPException(status_code=500, detail="Image generation failed.")

        # Convert PIL Image to bytes (PNG format)
        img_buffer = io.BytesIO()
        image.save(img_buffer, format="PNG")
        img_buffer.seek(0)
        image_bytes = img_buffer.getvalue()

        # Convert to optimized OLED bitmap
        oled_bitmap = convert_to_oled_bitmap_enhanced(image_bytes)
        
        # Generate unique ID and cache
        image_id = f"{request.topic.replace(' ', '_').replace('/', '_')}_{request.age_group}_{request.model}_{int(datetime.now().timestamp())}"
        poster_cache[image_id] = oled_bitmap
        
        generation_time = (datetime.now() - start_time).total_seconds()
        
        return PosterResponse(
            status="success",
            image_id=image_id,
            model_used=request.model,
            generation_time=generation_time,
            cost_estimate=cost_estimate,
            message=f"Generated educational poster for '{request.topic}' (Age: {request.age_group}) using {request.model}"
        )
        
    except Exception as e:
        logger.error(f"Poster generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/poster/{image_id}")
async def get_poster_bitmap(image_id: str):
    """
    Retrieve generated poster bitmap for ESP32 OLED display
    """
    if image_id not in poster_cache:
        raise HTTPException(status_code=404, detail="Poster not found")
    
    bitmap_data = poster_cache[image_id]
    
    return Response(
        content=bitmap_data,
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": f"attachment; filename={image_id}.xbm",
            "Content-Length": str(len(bitmap_data))
        }
    )

@app.get("/poster/{image_id}/preview")
async def get_poster_preview(image_id: str):
    """
    Get poster as PNG preview with enhanced visualization
    """
    if image_id not in poster_cache:
        raise HTTPException(status_code=404, detail="Poster not found")
    
    try:
        bitmap_data = poster_cache[image_id]
        
        # Create preview with better visualization
        pixels = []
        for byte in bitmap_data:
            for bit in range(8):
                pixel = 0 if (byte & (1 << bit)) else 255
                pixels.append(pixel)
        
        # Reshape and create preview
        image_array = np.array(pixels[:128*64]).reshape((64, 128))
        preview_image = Image.fromarray(image_array.astype(np.uint8), mode='L')
        
        # Scale up for better visibility (4x)
        preview_image = preview_image.resize((512, 256), Image.NEAREST)
        
        img_buffer = io.BytesIO()
        preview_image.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        
        return Response(
            content=img_buffer.getvalue(),
            media_type="image/png"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Preview generation failed: {str(e)}")

@app.get("/models/cost-effective")
async def list_cost_effective_models():
    """
    List cost-effective models optimized for educational content
    """
    return {
        "recommended_models": {
            "flux-schnell": {
                "name": "FLUX.1-schnell",
                "provider": "Hugging Face Serverless API",
                "cost": "FREE",
                "speed": "Fast (4-8 seconds)",
                "quality": "Excellent",
                "instruction_following": "Excellent",
                "best_for": "Educational infographics, high instruction following",
                "recommended": True
            },
            "stable-diffusion": {
                "name": "Stable Diffusion XL",
                "provider": "Hugging Face Serverless API",
                "cost": "FREE",
                "speed": "Medium (10-15 seconds)",
                "quality": "Very Good",
                "instruction_following": "Good",
                "best_for": "General educational content, fallback option",
                "recommended": False
            }
        },
        "daily_limits": {
            "huggingface_serverless": "Generous free tier, rate limited but sufficient for classroom use",
            "estimated_daily_posters": "50-100 posters per day (free tier)"
        }
    }

# Enhanced test endpoint for your workflow
@app.post("/test-workflow")
async def test_complete_workflow():
    """
    Test the complete workflow: Teacher input → Groq → Image Generation → ESP32 bitmap
    """
    test_cases = [
        {"topic": "Photosynthesis", "age_group": "7-12"},
        {"topic": "Solar System", "age_group": "0-6"},
        {"topic": "Water Cycle", "age_group": "13-16"}
    ]
    
    results = []
    
    for test_case in test_cases:
        try:
            request = PosterRequest(
                topic=test_case["topic"],
                age_group=test_case["age_group"],
                model="flux-schnell",
                style="infographic"
            )
            
            result = await generate_educational_poster(request, BackgroundTasks())
            
            results.append({
                "topic": test_case["topic"],
                "age_group": test_case["age_group"],
                "status": "success",
                "image_id": result.image_id,
                "generation_time": result.generation_time,
                "cost": result.cost_estimate,
                "preview_url": f"/poster/{result.image_id}/preview",
                "esp32_bitmap_url": f"/poster/{result.image_id}"
            })
            
        except Exception as e:
            results.append({
                "topic": test_case["topic"],
                "age_group": test_case["age_group"],
                "status": "failed",
                "error": str(e)
            })
    
    return {
        "workflow_test": "complete",
        "results": results,
        "summary": {
            "successful": len([r for r in results if r.get("status") == "success"]),
            "failed": len([r for r in results if r.get("status") == "failed"]),
            "total_cost": "FREE (All generated using HF Serverless API)"
        }
    }

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)