from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from typing import List
import os
from ..services.image_processor import process_image

router = APIRouter()

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output'

@router.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    if not file.filename.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
        raise HTTPException(status_code=400, detail="Invalid file type. Only images are allowed.")
    
    file_location = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(file_location, "wb") as f:
        f.write(await file.read())
    
    processed_image_path = process_image(file_location)
    
    if processed_image_path:
        return JSONResponse(content={"message": "Image uploaded and processed successfully", "image_path": processed_image_path})
    else:
        raise HTTPException(status_code=500, detail="Image processing failed.")

@router.get("/latest")
async def get_latest_image():
    try:
        files = os.listdir(OUTPUT_FOLDER)
        if not files:
            raise HTTPException(status_code=404, detail="No processed images available.")
        
        latest_file = max(files, key=lambda f: os.path.getctime(os.path.join(OUTPUT_FOLDER, f)))
        return JSONResponse(content={"filename": latest_file})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))