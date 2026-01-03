from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

app = Flask(__name__)

quality_model = tf.lite.Interpreter(model_path="handwriting-quality.tflite")
feedback_model = tf.lite.Interpreter(model_path="handwriting-feedback-ai.tflite")

quality_model.allocate_tensors()
feedback_model.allocate_tensors()

quality_input = quality_model.get_input_details()[0]
quality_output = quality_model.get_output_details()[0]

feedback_input = feedback_model.get_input_details()[0]
feedback_output = feedback_model.get_output_details()[0]

@app.route("/")
def home():
    return "AI server running"

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image = Image.open(request.files["image"]).convert("L")
    image = image.resize((224, 224))
    image = np.array(image, dtype=np.float32) / 255.0
    image = image.reshape(1, 224, 224, 1)

    quality_model.set_tensor(quality_input["index"], image)
    quality_model.invoke()
    quality = quality_model.get_tensor(quality_output["index"])

    feedback_model.set_tensor(feedback_input["index"], image)
    feedback_model.invoke()
    feedback = feedback_model.get_tensor(feedback_output["index"])

    return jsonify({
        "quality": float(np.max(quality)),
        "feedback": float(np.max(feedback))
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
