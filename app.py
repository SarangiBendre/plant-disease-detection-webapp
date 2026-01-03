from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import json

app = Flask(__name__)

# ===============================
# Load trained model
# ===============================
model = tf.keras.models.load_model("plant_disease_model.keras")

# ===============================
# Load class names (same as training)
# ===============================
with open("class_names.json", "r") as f:
    class_names = json.load(f)

# ===============================
# Upload folder (must be inside static)
# ===============================
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ===============================
# Home route
# ===============================
@app.route("/", methods=["GET", "POST"])
def home():
    image_name = None
    prediction = None
    confidence = None

    if request.method == "POST":
        file = request.files.get("image")

        if file and file.filename != "":
            # Save image
            image_name = file.filename
            image_path = os.path.join(UPLOAD_FOLDER, image_name)
            file.save(image_path)

            # Preprocess image
            img = Image.open(image_path).convert("RGB")
            img = img.resize((180, 180))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Predict
            preds = model.predict(img_array)
            probs = tf.nn.softmax(preds[0]).numpy()

            idx = int(np.argmax(probs))
            prediction = class_names[idx]
            confidence = round(float(probs[idx] * 100), 2)

    return render_template(
        "index.html",
        image_name=image_name,
        prediction=prediction,
        confidence=confidence
    )

# ===============================
# Run app (Render compatible)
# ===============================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
