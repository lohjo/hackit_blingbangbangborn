# app/utils.py
import requests
import replicate
import openai
from groq import Groq
from fastapi import HTTPException
import logging
import io
import numpy as np
import base64
from PIL import Image, ImageEnhance, ImageOps
import asyncio


logger = logging.getLogger(__name__)


def create_enhanced_educational_prompt(groq_client: Groq, topic: str, age_group: str, style: str = "infographic") -> str:
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
            model="llama-3.1-70b-versatile",
            temperature=0.3,
            max_tokens=300
        )
       
        enhanced_prompt = response.choices[0].message.content.strip()
        logger.info(f"Enhanced prompt generated: {enhanced_prompt}")
        return enhanced_prompt
       
    except Exception as e:
        logger.error(f"Groq API error: {e}")
        return f"Educational infographic poster about {topic}, high contrast black and white design, bold typography, simplified vector illustrations, clean diagrams with labels, arrows showing process flow, child-friendly icons, white background, classroom poster style, OLED display optimized. Negative: photorealistic, cluttered, small text, dark backgrounds"


def convert_to_oled_bitmap_enhanced(image_data: bytes) -> bytes:
    """
    Enhanced conversion optimized for educational content
    """
    try:
        image = Image.open(io.BytesIO(image_data))
       
        if image.mode != 'RGB':
            image = image.convert('RGB')
       
        image.thumbnail((128, 64), Image.Resampling.LANCZOS)
       
        oled_image = Image.new('RGB', (128, 64), 'white')
       
        x_offset = (128 - image.width) // 2
        y_offset = (64 - image.height) // 2
        oled_image.paste(image, (x_offset, y_offset))
       
        # Enhanced processing for educational content
        enhancer = ImageEnhance.Contrast(oled_image)
        oled_image = enhancer.enhance(2.5)
       
        enhancer = ImageEnhance.Sharpness(oled_image)
        oled_image = enhancer.enhance(2.0)
       
        oled_image = oled_image.convert('L')
       
        threshold = 128
        oled_image = oled_image.point(lambda x: 255 if x > threshold else 0, mode='1')
       
        bitmap_data = []
        pixels = np.array(oled_image)
       
        for y in range(64):
            for x in range(0, 128, 8):
                byte = 0
                for bit in range(8):
                    if x + bit < 128:
                        if pixels[y, x + bit] == 0:
                            byte |= (1 << bit)
                bitmap_data.append(byte)
       
        return bytes(bitmap_data)
       
    except Exception as e:
        logger.error(f"Enhanced image conversion error: {e}")
        raise HTTPException(status_code=500, detail=f"Image conversion failed: {str(e)}")


async def generate_with_flux_schnell_free(prompt: str, hf_token: str) -> bytes:
    """
    Generate image using FLUX.1-schnell via Hugging Face Serverless API (FREE)
    """
    try:
        logger.info("Generating image with FLUX.1-schnell (FREE via HF)...")
       
        api_url = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-schnell"
        headers = {"Authorization": f"Bearer {hf_token}"}
       
        payload = {
            "inputs": prompt,
            "parameters": {
                "num_inference_steps": 4,
                "guidance_scale": 3.5,
                "width": 1024,
                "height": 1024,
                "scheduler": "FlowMatchEulerDiscreteScheduler"
            }
        }
       
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.post(api_url, headers=headers, json=payload, timeout=120)
                response.raise_for_status()
               
                if response.headers.get("content-type", "").startswith("image"):
                    return response.content
                else:
                    error_data = response.json()
                    if "loading" in str(error_data).lower():
                        logger.info(f"Model loading, attempt {attempt + 1}/{max_retries}")
                        await asyncio.sleep(20)
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


async def generate_with_stable_diffusion_free(prompt: str, hf_token: str) -> bytes:
    """
    Fallback: Stable Diffusion via HF Serverless API (FREE)
    """
    try:
        logger.info("Generating image with Stable Diffusion XL (FREE)...")
       
        api_url = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
        headers = {"Authorization": f"Bearer {hf_token}"}
       
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


async def generate_with_flux_schnell_replicate(prompt: str) -> bytes:
    """
    Generate image using FLUX.1-schnell via Replicate (PAID but fast)
    """
    try:
        logger.info("Generating image with FLUX.1-schnell via Replicate...")
       
        output = replicate.run(
            "black-forest-labs/flux-schnell",
            input={
                "prompt": prompt,
                "go_fast": True,
                "megapixels": "1",
                "num_outputs": 1,
                "aspect_ratio": "1:1",
                "output_format": "png",
                "output_quality": 80,
                "num_inference_steps": 4
            }
        )
       
        if output and len(output) > 0:
            image_url = output[0]
            response = requests.get(image_url)
            response.raise_for_status()
            return response.content
        else:
            raise Exception("No output generated from FLUX")
           
    except Exception as e:
        logger.error(f"FLUX Replicate generation error: {e}")
        raise HTTPException(status_code=500, detail=f"FLUX Replicate generation failed: {str(e)}")


async def generate_with_openai_dalle3(openai_client: openai.OpenAI, prompt: str) -> bytes:
    """
    Generate image using OpenAI DALL-E 3 (PAID but highest quality)
    """
    try:
        logger.info("Generating image with DALL-E 3...")
       
        response = openai_client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024",
            quality="standard",
            n=1,
            response_format="b64_json"
        )
       
        if response.data and len(response.data) > 0:
            image_b64 = response.data[0].b64_json
            image_data = base64.b64decode(image_b64)
            return image_data
        else:
            raise Exception("No output generated from DALL-E 3")
           
    except Exception as e:
        logger.error(f"DALL-E 3 generation error: {e}")
        raise HTTPException(status_code=500, detail=f"DALL-E 3 generation failed: {str(e)}")