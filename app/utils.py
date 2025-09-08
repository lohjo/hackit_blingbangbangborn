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
from PIL import Image, ImageEnhance
import asyncio

logger = logging.getLogger(__name__)

def create_educational_prompt(groq_client: Groq, topic: str, age_group: str, complexity: str = "simple") -> str:
    # ... (paste the create_educational_prompt function from CODE_1)
    try:
        age_descriptions = {
            "0-6": "kindergarten level, very simple concepts, basic shapes, minimal text",
            "7-12": "elementary school level, moderate detail, step-by-step processes",
            "13-16": "high school level, detailed diagrams, technical terminology"
        }
        
        system_prompt = f"""You are an educational content creator specializing in visual learning materials. 
        Create a detailed image generation prompt for a classroom poster about '{topic}' suitable for {age_descriptions.get(age_group, 'general audience')}.
        
        The poster must be:
        - Educational and informative
        - High contrast for monochrome OLED display (128x64 pixels)
        - Clean, minimalist design with bold typography
        - Child-friendly illustrations with simple geometric shapes
        - Infographic layout with clear visual hierarchy
        
        Format: Educational poster, [TOPIC], [specific visual elements], bold sans-serif typography, simplified vector illustrations, high contrast black and white compatible, child-friendly icons, classroom wall poster aesthetic, minimalist background.
        """
        
        response = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Create an image generation prompt for: {topic}"}
            ],
            model="llama3-8b-8192",
            temperature=0.7,
            max_tokens=200
        )
        
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        logger.error(f"Groq API error: {e}")
        return f"Educational poster about {topic}, clean infographic style, bold typography, simplified illustrations, high contrast, child-friendly, classroom appropriate"

def convert_to_oled_bitmap(image_data: bytes) -> bytes:
    # ... (paste the convert_to_oled_bitmap function from CODE_1)
    try:
        image = Image.open(io.BytesIO(image_data))
        
        image.thumbnail((128, 64), Image.Resampling.LANCZOS)
        
        oled_image = Image.new('RGB', (128, 64), 'white')
        
        x_offset = (128 - image.width) // 2
        y_offset = (64 - image.height) // 2
        oled_image.paste(image, (x_offset, y_offset))
        
        enhancer = ImageEnhance.Contrast(oled_image)
        oled_image = enhancer.enhance(2.0)
        
        oled_image = oled_image.convert('L')
        oled_image = oled_image.convert('1')
        
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
        logger.error(f"Image conversion error: {e}")
        raise HTTPException(status_code=500, detail=f"Image conversion failed: {str(e)}")

async def generate_with_flux_schnell(prompt: str) -> bytes:
    # ... (paste the generate_with_flux_schnell function from CODE_1)
    try:
        logger.info("Generating image with FLUX.1-schnell...")
        
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
        logger.error(f"FLUX generation error: {e}")
        raise HTTPException(status_code=500, detail=f"FLUX generation failed: {str(e)}")

async def generate_with_stable_diffusion(prompt: str, hf_token: str) -> bytes:
    # ... (paste the generate_with_stable_diffusion function from CODE_1)
    try:
        logger.info("Generating image with Stable Diffusion XL...")
        
        api_url = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
        headers = {"Authorization": f"Bearer {hf_token}"}
        
        payload = {
            "inputs": prompt,
            "parameters": {
                "negative_prompt": "blurry, low quality, dark, cluttered, small text, photorealistic, complex gradients, fine details, adult complexity, low contrast",
                "num_inference_steps": 25,
                "guidance_scale": 7.5,
                "width": 1024,
                "height": 1024
            }
        }
        
        response = requests.post(api_url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        
        if response.headers.get("content-type", "").startswith("image"):
            return response.content
        else:
            error_data = response.json()
            raise Exception(f"API Error: {error_data}")
            
    except Exception as e:
        logger.error(f"Stable Diffusion generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Stable Diffusion generation failed: {str(e)}")

async def generate_with_openai_dalle3(openai_client: openai.OpenAI, prompt: str) -> bytes:
    # ... (paste the generate_with_openai_dalle3 function from CODE_1)
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