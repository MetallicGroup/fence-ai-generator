import uuid
import os
from PIL import Image

def run_fence_replacement(image_file, model):
    try:
        # Asigurăm că folderul static există
        os.makedirs("static", exist_ok=True)

        temp_path = f"static/{uuid.uuid4()}.png"
        img = Image.open(image_file.stream).convert("RGB")
        img.save(temp_path)

        # Aici pui in viitor SAM + Inpainting
        return "https://i.ibb.co/example/generated-image.png"

    except Exception as e:
        print("ERROR in AI logic:", str(e))
        return "https://i.ibb.co/example/error.png"
