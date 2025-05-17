import uuid
import os
import requests
from PIL import Image
from io import BytesIO

HF_API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2-inpainting"
HF_API_TOKEN = os.getenv("HF_API_TOKEN")

def run_fence_replacement(image_file, model_name):
    try:
        os.makedirs("static", exist_ok=True)
        image_id = str(uuid.uuid4())
        input_path = f"static/{image_id}.png"
        print("Saving input image...")

        # Convert image from upload
        image_bytes = image_file.read()
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        image.save(input_path)

        prompt = f"A modern horizontal metal fence, model {model_name}, matte black, minimalistic, in front of a house"
        print("Prompt:", prompt)

        headers = {
            "Authorization": f"Bearer {HF_API_TOKEN}"
        }

        print("Sending request to Hugging Face...")
        files = {
            "image": ("input.png", BytesIO(image_bytes), "image/png"),
            "mask_image": ("mask.png", BytesIO(image_bytes), "image/png")
        }
        data = {
            "prompt": prompt
        }

        response = requests.post(HF_API_URL, headers=headers, files=files, data=data)

        print("HF Status:", response.status_code)
        print("HF Headers:", response.headers)
        print("HF Raw Response:", response.text)

        if response.ok:
            try:
                data = response.json()
                return data.get("image_url", "https://i.ibb.co/error.jpg")
            except Exception as json_err:
                print("JSON error:", json_err)
                return "https://i.ibb.co/error.jpg"
        else:
            return "https://i.ibb.co/error.jpg"

    except Exception as e:
        print("Unexpected ERROR:", str(e))
        return "https://i.ibb.co/error.jpg"
