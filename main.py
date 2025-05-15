import os
import replicate
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ✅ Asigură-te că cheia este setată corect în Render ENV (Settings > Environment > Add Secret)
REPLICATE_API_TOKEN = os.environ.get("REPLICATE_API_TOKEN")
if not REPLICATE_API_TOKEN:
    raise ValueError("⚠️ REPLICATE_API_TOKEN nu este setat în environment!")
os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_TOKEN

@app.route("/", methods=["GET"])
def home():
    return "🟢 Fence AI Generator este activ!"

@app.route("/generate", methods=["POST"])
def generate():
    try:
        # ✅ Verifică cererea
        if "input" not in request.files or "mask" not in request.files or "prompt" not in request.form:
            return jsonify({"error": "Lipsesc câmpuri: 'input', 'mask' sau 'prompt'"}), 400

        input_file = request.files["input"]
        mask_file = request.files["mask"]
        prompt = request.form["prompt"]

        # ✅ Salvează temporar fișierele
        input_path = "/tmp/input.png"
        mask_path = "/tmp/mask.png"
        input_file.save(input_path)
        mask_file.save(mask_path)

        print(f"📥 Prompt: {prompt}")
        print(f"📂 Fișiere salvate: {input_path}, {mask_path}")

        # ✅ Trimite cererea la Replicate
        output = replicate.run(
            "stability-ai/stable-diffusion-inpainting:6cf94c5dbd41d2b8e3fd3709fcabff96049b145c109f4f5a77d1045c22f1b7cf",
            input={
                "image": open(input_path, "rb"),
                "mask": open(mask_path, "rb"),
                "prompt": prompt
            }
        )

        print(f"✅ Imagine generată: {output}")
        return jsonify({"output_url": output}), 200

    except Exception as e:
        print(f"❌ Eroare la generare: {str(e)}")
        return jsonify({"error": str(e)}), 500

# ✅ Rulează pe portul necesar pentru Render
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
