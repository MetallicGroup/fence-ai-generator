from flask import Flask, request, jsonify
from inference import run_fence_replacement

app = Flask(__name__)

@app.route('/generate', methods=['POST'])
def generate():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    image = request.files['image']
    model = request.args.get('model', 'MX25')  # fixed default
    result_url = run_fence_replacement(image, model)

    return jsonify({'image_url': result_url})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
