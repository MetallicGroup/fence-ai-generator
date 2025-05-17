def run_fence_replacement(image_file, model):
    import uuid
    from PIL import Image
    temp_path = f"static/{uuid.uuid4()}.png"
    Image.open(image_file).save(temp_path)

    # TODO: aici integrezi SAM + Inpainting
    # Pentru demo, returnăm o imagine statică
    return "https://i.ibb.co/example/generated-image.png"
