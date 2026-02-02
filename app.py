import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
import os

# ================= CONFIG =================
IMG_SIZE = 128
MODEL_PATH = "plant_disease_model.h5"

st.set_page_config(
    page_title="Plant Disease Detection",
    layout="centered"
)

st.title("🌿 Plant Disease Detection System")
st.write("Upload a plant leaf image to detect disease and get remedies")

# ================= CLASS NAMES =================
class_names = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Healthy"
]

# ================= DISEASE INFO =================
disease_info = {
    "Apple___Apple_scab": {
        "name": "Apple Scab",
        "remedies": [
            "Apply fungicides like Captan or Mancozeb",
            "Remove infected leaves",
            "Avoid overhead watering",
            "Ensure good air circulation"
        ]
    },
    "Apple___Black_rot": {
        "name": "Apple Black Rot",
        "remedies": [
            "Prune infected branches",
            "Apply copper-based fungicide",
            "Remove infected fruits",
            "Maintain field sanitation"
        ]
    },
    "Apple___Healthy": {
        "name": "Healthy Plant",
        "remedies": [
            "No disease detected",
            "Maintain proper irrigation",
            "Apply balanced fertilizers",
            "Regular crop monitoring"
        ]
    }
}

# ================= LOAD MODEL SAFELY =================
@st.cache_resource
def load_model_safe():
    if not os.path.exists(MODEL_PATH):
        return None
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model_safe()

if model is None:
    st.warning(
        "⚠️ Model file not found!\n\n"
        "Please place **plant_disease_model.h5** in the same folder as app.py"
    )
    st.stop()

# ================= IMAGE UPLOAD =================
uploaded_file = st.file_uploader(
    "Upload Leaf Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Read image safely
    image_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

    if image is None:
        st.error("❌ Unable to read the image. Please upload a valid image.")
        st.stop()

    # Preprocess
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)

    st.image(uploaded_file, caption="Uploaded Leaf Image", use_column_width=True)

    # ================= PREDICTION =================
    prediction = model.predict(image)

    if prediction.shape[1] != len(class_names):
        st.error("❌ Model output does not match class labels")
        st.stop()

    class_index = int(np.argmax(prediction))
    confidence = float(np.max(prediction)) * 100

    predicted_class = class_names[class_index]
    result = disease_info[predicted_class]

    # ================= OUTPUT =================
    st.subheader("🦠 Disease Identified")
    st.success(result["name"])

    st.subheader("📊 Prediction Confidence")
    st.write(f"{confidence:.2f}%")

    st.subheader("💊 Remedies & Prevention")
    for remedy in result["remedies"]:
        st.write("✔️", remedy)
