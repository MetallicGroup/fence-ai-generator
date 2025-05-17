import uuid
import os
import requests
from PIL import Image
import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor

# Config
HF_API_URL = "https://api-inference.huggingface.co/models/Sanster/anything-3.0-inpainting"
HF_API_TOKEN = os.getenv("HF_API_TOKEN")

def run_fence_replacement(image_file, model_name):
    try:
        os.makedirs("static", exist_ok=True)
        image_id = str(uuid.uuid4())
        input_path = f"static/{image_id}.png"
        mask_path = f"static/{image_id}_mask.png"
        print("Saving input image...")

        # 1. Save image
        image = Image.open(image_file.stream).convert("RGB")
        image.save(input_path)

        # 2. Run SAM
        print("Loading SAM model...")
        sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
        sam.to("cpu")
        predictor = SamPredictor(sam)

        print("Predicting mask...")
        image_np = np.array(image)
        predictor.set_image(image_np)
        masks, _, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            multimask_output=False
        )
        mask = masks[0]
        mask_img = Image.fromarray((mask * 255).astype(np.uint8))
        mask_img.save(mask_path)
        print("Mask saved to:", mask_path)

        # 3. Call Hugging Face inpainting
        print("Calling Hugging Face Inpainting API...")
        prompt = f"A modern horizontal metal fence, model {model_name}, matte black, minimalistic, in front of a house"
        headers = {
            "Authorization": f"Bearer {HF_API_TOKEN}"
        }
        with open(input_path, "rb") as image_data, open(mask_path, "rb") as mask_data:
            response = requests.post(
                HF_API_URL,
                headers=headers,
                files={"image": image_data, "mask": mask_data},
                data={"prompt": prompt}
            )
        print("Response status:", response.status_code)
        print("Response headers:", response.headers)

        if response.ok:
            try:
                output_url = response.json().get("image_url", "")
                if output_url:
                    print("Image URL:", output_url)
                    return output_url
                else:
                    print("No image_url found in response JSON")
                    return "https://i.ibb.co/error.jpg"
            except Exception as json_err:
                print("JSON decode error:", json_err)
                print("Raw response:", response.text)
                return "https://i.ibb.co/error.jpg"
        else:
            print("API returned error:", response.text)
            return "https://i.ibb.co/error.jpg"

    except Exception as e:
        print("Unexpected ERROR:", str(e))
        return "https://i.ibb.co/error.jpg"
