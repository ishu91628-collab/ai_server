from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)

# Load Model 1
model1 = tf.lite.Interpreter(model_path="model1.tflite")
model1.allocate_tensors()
input1 = model1.get_input_details()
output1 = model1.get_output_details()

# Load Model 2
model2 = tf.lite.Interpreter(model_path="model2.tflite")
model2.allocate_tensors()
input2 = model2.get_input_details()
output2 = model2.get_output_details()

def preprocess(img):
    img = img.resize((224, 224))
    img = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(img, axis=0)

@app.route("/")
def home():
    return "AI Server Running"

@app.route("/predict1", methods=["POST"])
def predict1():
    img = Image.open(request.files["image"])
    model1.set_tensor(input1[0]["index"], preprocess(img))
    model1.invoke()
    result = model1.get_tensor(output1[0]["index"])
    return jsonify({"result": int(np.argmax(result))})

@app.route("/predict2", methods=["POST"])
def predict2():
    img = Image.open(request.files["image"])
    model2.set_tensor(input2[0]["index"], preprocess(img))
    model2.invoke()
    result = model2.get_tensor(output2[0]["index"])
    return jsonify({"result": int(np.argmax(result))})

app.run(host="0.0.0.0", port=8000)