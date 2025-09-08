import os
from huggingface_hub import InferenceClient

client = InferenceClient(
    api_key=os.getenv("HUGGINGFACE_API_TOKEN"),
)

# output is a PIL.Image object
image = client.text_to_image(
    "Astronaut riding a horse",
    model="black-forest-labs/FLUX.1-schnell",
)