import uuid
import os
import requests
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as T
from segment_anything import sam_model_registry, SamPredictor

# Config
HF_API_URL = "https://api-inference.huggingface.co/models/Sanster/anything-3.0-inpainting"
HF_API_TOKEN = os.getenv("HF_API_TOKEN")  # trebuie setat în Render

def run_fence_replacement(image_file, model_name):
    os.makedirs("static", exist_ok=True)
    image_id = str(uuid.uuid4())
    input_path = f"static/{image_id}.png"
    mask_path = f"static/{image_id}_mask.png"
    output_url = ""

    # 1. Salvează imaginea
    image = Image.open(image_file.stream).convert("RGB")
    image.save(input_path)

    # 2. Segmentare cu SAM
    try:
        image_np = np.array(image)
        sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
        sam.to("cpu")
        predictor = SamPredictor(sam)
        predictor.set_image(image_np)
        masks, _, _ = predictor.predict(point_coords=None, point_labels=None, multimask_output=False)
        mask = masks[0]

        # Salvează masca
        mask_img = Image.fromarray((mask * 255).astype(np.uint8))
        mask_img.save(mask_path)
    except Exception as e:
        print("SAM ERROR:", e)
        return "https://i.ibb.co/error.jpg"

    # 3. Inpainting cu Hugging Face
    try:
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
        if response.ok:
            output_url = response.json()["image_url"]
        else:
            print("Inpainting ERROR:", response.text)
            output_url = "https://i.ibb.co/error.jpg"
    except Exception as e:
        print("HF ERROR:", e)
        output_url = "https://i.ibb.co/error.jpg"

    return output_url
