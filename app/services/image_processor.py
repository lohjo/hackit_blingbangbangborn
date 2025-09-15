from PIL import Image, ImageOps
import os

OUTPUT_FOLDER = 'output'
UPLOAD_FOLDER = 'uploads'

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

def process_image(image_path):
    """Process the uploaded image for OLED display."""
    try:
        with Image.open(image_path) as img:
            # Convert to grayscale
            img = img.convert('L')
            
            # Resize to fit OLED display
            img = img.resize((128, 64), Image.Resampling.LANCZOS)
            
            # Apply dithering and convert to 1-bit
            img = img.convert('1', dither=Image.Dither.FLOYDSTEINBERG)
            
            # Save processed image
            output_path = os.path.join(OUTPUT_FOLDER, os.path.basename(image_path))
            img.save(output_path)
            
            return output_path
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

def get_processed_image(filename):
    """Retrieve the processed image for the OLED display."""
    processed_image_path = os.path.join(OUTPUT_FOLDER, filename)
    if os.path.exists(processed_image_path):
        return processed_image_path
    return None