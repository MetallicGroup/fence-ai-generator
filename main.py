from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
from utils.replicate_api import run_inpainting
from utils.imgbb import upload_to_imgbb

app = FastAPI()

@app.post("/generate")
async def generate_image(image: UploadFile, mask: UploadFile, prompt: str = Form(...)):
    image_bytes = await image.read()
    mask_bytes = await mask.read()

    try:
        output_image = run_inpainting(image_bytes, mask_bytes, prompt)
        image_url = upload_to_imgbb(output_image)
        return {"status": "success", "url": image_url}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
