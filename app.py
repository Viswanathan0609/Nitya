import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os

# ================= STREAMLIT CONFIG =================
st.set_page_config(
    page_title="Plant Disease Detection",
    layout="centered"
)

st.title("🌿 Plant Disease Detection System")
st.write("Upload a plant leaf image to detect disease and remedies")

# ================= CONSTANTS =================
IMG_SIZE = 128
MODEL_PATH = "plant_disease_model.h5"

# ================= CLASS LABELS =================
class_names = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Healthy"
]

# ================= DISEASE DETAILS =================
disease_data = {
    "Apple___Apple_scab": {
        "name": "Apple Scab",
        "remedies": [
            "Spray fungicides like Mancozeb or Captan",
            "Remove infected leaves",
            "Avoid overhead irrigation",
            "Improve air circulation"
        ]
    },
    "Apple___Black_rot": {
        "name": "Apple Black Rot",
        "remedies": [
            "Prune infected branches",
            "Apply copper fungicide",
            "Destroy infected fruits",
            "Maintain orchard cleanliness"
        ]
    },
    "Apple___Healthy": {
        "name": "Healthy Leaf",
        "remedies": [
            "No disease detected",
            "Water regularly",
            "Apply balanced fertilizer",
            "Monitor crop condition"
        ]
    }
}

# ================= LOAD MODEL (STREAMLIT SAFE) =================
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

if model is None:
    st.error("❌ Model file not found!")
    st.info("Place **plant_disease_model.h5** in the same folder as app.py")
    st.stop()

# ================= IMAGE UPLOAD =================
uploaded_file = st.file_uploader(
    "Upload Leaf Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Read image
    image_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

    if image is None:
        st.error("Invalid image file")
        st.stop()

    # Preprocess
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)

    st.image(uploaded_file, caption="Uploaded Leaf Image", use_column_width=True)

    # ================= PREDICTION =================
    prediction = model.predict(image)
    class_index = int(np.argmax(prediction))
    confidence = float(np.max(prediction)) * 100

    predicted_class = class_names[class_index]
    disease = disease_data[predicted_class]

    # ================= OUTPUT =================
    st.subheader("🦠 Disease Detected")
    st.success(disease["name"])

    st.subheader("📊 Confidence")
    st.write(f"{confidence:.2f}%")

    st.subheader("💊 Remedies")
    for remedy in disease["remedies"]:
        st.write("✔️", remedy)
