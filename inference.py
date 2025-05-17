import uuid
import os
import requests
from PIL import Image

# Config
HF_API_URL = "https://api-inference.huggingface.co/models/Sanster/anything-3.0-inpainting"
HF_API_TOKEN = os.getenv("HF_API_TOKEN")

def run_fence_replacement(image_file, model_name):
    try:
        os.makedirs("static", exist_ok=True)
        image_id = str(uuid.uuid4())
        input_path = f"static/{image_id}.png"
        print("Saving input image...")

        # Save image locally
        image = Image.open(image_file.stream).convert("RGB")
        image.save(input_path)

        # Prepare prompt
        prompt = f"A modern horizontal metal fence, model {model_name}, matte black, minimalistic, in front of a house"

        # Call Hugging Face API with only image and prompt
        print("Sending request to Hugging Face...")
        headers = {
            "Authorization": f"Bearer {HF_API_TOKEN}"
        }
        with open(input_path, "rb") as image_data:
            response = requests.post(
                HF_API_URL,
                headers=headers,
                files={"image": image_data},
                data={"prompt": prompt}
            )
        print("Response status:", response.status_code)

        # Try to extract image_url if available
        if response.ok:
            try:
                json_data = response.json()
                print("HF response:", json_data)
                return json_data.get("image_url", "https://i.ibb.co/error.jpg")
            except Exception as json_err:
                print("Error decoding JSON:", json_err)
                print("Raw response:", response.text)
                return "https://i.ibb.co/error.jpg"
        else:
            print("API error:", response.text)
            return "https://i.ibb.co/error.jpg"

    except Exception as e:
        print("Unexpected ERROR:", str(e))
        return "https://i.ibb.co/error.jpg"
