import replicate
import base64

REPLICATE_API_TOKEN = "r8_ORN2qhbR6uqRqmOVoH5wEDPDpORHoY42x9rSu"

def run_inpainting(image_bytes, mask_bytes, prompt):
    replicate.Client(api_token=REPLICATE_API_TOKEN)

    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    mask_b64 = base64.b64encode(mask_bytes).decode("utf-8")

    model = replicate.models.get("stability-ai/stable-diffusion-inpainting")
    version = model.versions.get("f9863fca2146b317...")  # actualizează cu ultima versiune stabilă

    output = version.predict(
        input={
            "image": f"data:image/png;base64,{image_b64}",
            "mask": f"data:image/png;base64,{mask_b64}",
            "prompt": prompt
        }
    )

    return output
