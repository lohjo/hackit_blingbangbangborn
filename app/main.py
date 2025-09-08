# app/main.py - Enhanced Cost-Effective Educational Poster Generator

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, Response
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Educational Poster Generator - Cost Effective", version="2.0.0")

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
            model="llama-3.1-70b-versatile",  # Better instruction following
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

async def generate_with_flux_schnell_free(prompt: str) -> bytes:
    """
    Generate image using FLUX.1-schnell via Hugging Face Serverless API (FREE)
    Best cost-effective option with excellent instruction following
    """
    try:
        logger.info("Generating image with FLUX.1-schnell (FREE via HF)...")
        
        api_url = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-schnell"
        headers = {"Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}"}
        
        payload = {
            "inputs": prompt,
            "parameters": {
                "num_inference_steps": 4,  # Fast generation
                "guidance_scale": 3.5,     # Good instruction following
                "width": 1024,
                "height": 1024,
                "scheduler": "FlowMatchEulerDiscreteScheduler"
            }
        }
        
        # Make API request with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.post(api_url, headers=headers, json=payload, timeout=120)
                response.raise_for_status()
                
                if response.headers.get("content-type", "").startswith("image"):
                    return response.content
                else:
                    # Handle model loading time
                    error_data = response.json()
                    if "loading" in str(error_data).lower():
                        logger.info(f"Model loading, attempt {attempt + 1}/{max_retries}")
                        await asyncio.sleep(20)  # Wait for model to load
                        continue
                    else:
                        raise Exception(f"API Error: {error_data}")
                        
            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    logger.info(f"Request timeout, retrying... ({attempt + 1}/{max_retries})")
                    await asyncio.sleep(10)
                    continue
                else:
                    raise Exception("Request timed out after all retries")
                    
        raise Exception("Failed to generate image after all retries")
        
    except Exception as e:
        logger.error(f"FLUX generation error: {e}")
        raise HTTPException(status_code=500, detail=f"FLUX generation failed: {str(e)}")

async def generate_with_stable_diffusion_free(prompt: str) -> bytes:
    """
    Fallback: Stable Diffusion via HF Serverless API (FREE)
    """
    try:
        logger.info("Generating image with Stable Diffusion XL (FREE)...")
        
        api_url = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
        headers = {"Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}"}
        
        payload = {
            "inputs": prompt,
            "parameters": {
                "negative_prompt": "blurry, low quality, dark, cluttered, small text, photorealistic, complex gradients, fine details, adult complexity, low contrast, busy background, overlapping elements",
                "num_inference_steps": 20,
                "guidance_scale": 8.0,
                "width": 1024,
                "height": 1024
            }
        }
        
        response = requests.post(api_url, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        
        if response.headers.get("content-type", "").startswith("image"):
            return response.content
        else:
            error_data = response.json()
            raise Exception(f"API Error: {error_data}")
            
    except Exception as e:
        logger.error(f"Stable Diffusion generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Stable Diffusion generation failed: {str(e)}")

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
            image_data = await generate_with_flux_schnell_free(enhanced_prompt)
            cost_estimate = "FREE (HF Serverless)"
        elif request.model.lower() == "stable-diffusion":
            image_data = await generate_with_stable_diffusion_free(enhanced_prompt)
            cost_estimate = "FREE (HF Serverless)"
        else:
            # Default to most cost-effective
            image_data = await generate_with_flux_schnell_free(enhanced_prompt)
            cost_estimate = "FREE (HF Serverless)"
        
        # Convert to optimized OLED bitmap
        oled_bitmap = convert_to_oled_bitmap_enhanced(image_data)
        
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)