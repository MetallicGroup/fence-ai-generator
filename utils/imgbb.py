import requests

IMGBB_API_KEY = "f81bb722e0833f163bffd4ae7e9bffa8"

def upload_to_imgbb(image_url):
    response = requests.post(
        "https://api.imgbb.com/1/upload",
        params={"key": IMGBB_API_KEY, "image": image_url}
    )
    response.raise_for_status()
    return response.json()["data"]["url"]
