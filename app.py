from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import json
import cv2

app = Flask(__name__)

# ===============================
# Configurations
# ===============================
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
CONFIDENCE_THRESHOLD = 65  # %

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ===============================
# Helper functions
# ===============================
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def is_leaf_image(image_path):
    img = cv2.imread(image_path)

    if img is None:
        return False

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Green color range for leaf detection
    lower_green = np.array([25, 40, 40])
    upper_green = np.array([85, 255, 255])

    mask = cv2.inRange(hsv, lower_green, upper_green)
    green_ratio = np.sum(mask > 0) / mask.size

    return green_ratio > 0.15


# ===============================
# Load model and class names
# ===============================
model = tf.keras.models.load_model("plant_disease_model.keras")

with open("class_names.json", "r") as f:
    class_names = json.load(f)

# ===============================
# Routes
# ===============================
@app.route("/", methods=["GET", "POST"])
def home():
    image_name = None
    prediction = None
    confidence = None
    error = None

    if request.method == "POST":
        file = request.files.get("image")

        # ❌ No file selected
        if not file or file.filename == "":
            error = "No file selected. Please upload an image."

        # ❌ Invalid file type
        elif not allowed_file(file.filename):
            error = "Only JPG, JPEG, and PNG image files are allowed."

        else:
            # ✅ Save image
            image_name = file.filename
            image_path = os.path.join(UPLOAD_FOLDER, image_name)
            file.save(image_path)

            # ❌ Non-leaf image
            if not is_leaf_image(image_path):
                return render_template(
                    "index.html",
                    error="Invalid image. Please upload a clear plant leaf image only."
                )

            # ===============================
            # Preprocess image (same as training)
            # ===============================
            img = Image.open(image_path).convert("RGB")
            img = img.resize((180, 180))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # ===============================
            # Predict
            # ===============================
            preds = model.predict(img_array)
            probs = tf.nn.softmax(preds[0]).numpy()

            idx = int(np.argmax(probs))
            confidence = float(probs[idx] * 100)

            # ❌ Low confidence prediction
            if confidence < CONFIDENCE_THRESHOLD:
                return render_template(
                    "index.html",
                    error="Low confidence prediction. Please upload a clearer leaf image."
                )

            # ✅ Valid prediction
            prediction = class_names[idx]
            confidence = round(confidence, 2)

    return render_template(
        "index.html",
        image_name=image_name,
        prediction=prediction,
        confidence=confidence,
        error=error
    )

# ===============================
# Run app
# ===============================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
