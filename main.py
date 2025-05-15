import os
import replicate
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route("/", methods=["GET"])
def home():
    return "🟢 Server funcționează!"

@app.route("/generate", methods=["POST"])
def generate():
    try:
        input_file = request.files["input"]
        mask_file = request.files["mask"]
        prompt = request.form["prompt"]

        input_path = "/tmp/input.png"
        mask_path = "/tmp/mask.png"
        input_file.save(input_path)
        mask_file.save(mask_path)

        # Trimite cererea la Replicate (model de inpainting)
        output = replicate.run(
            "stability-ai/stable-diffusion-inpainting:6cf94c5dbd41d2b8e3fd3709fcabff96049b145c109f4f5a77d1045c22f1b7cf",
            input={
                "image": open(input_path, "rb"),
                "mask": open(mask_path, "rb"),
                "prompt": prompt
            }
        )

        return jsonify({"output_url": output}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
