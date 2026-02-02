import streamlit as st
import tensorflow as tf
import numpy as np
import cv2

# ================== CONFIG ==================
IMG_SIZE = 128
MODEL_PATH = "plant_disease_model.h5"

# ================== LOAD MODEL ==================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# ================== CLASS NAMES ==================
# MUST match the order used while training
class_names = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Healthy"
]

# ================== DISEASE & REMEDIES ==================
disease_info = {
    "Apple___Apple_scab": {
        "name": "Apple Scab",
        "remedies": [
            "Use fungicides like Captan or Mancozeb",
            "Remove and destroy infected leaves",
            "Avoid overhead irrigation",
            "Ensure proper air circulation"
        ]
    },
    "Apple___Black_rot": {
        "name": "Apple Black Rot",
        "remedies": [
            "Prune infected branches immediately",
            "Apply copper-based fungicide",
            "Remove mummified fruits",
            "Maintain orchard sanitation"
        ]
    },
    "Apple___Healthy": {
        "name": "Healthy Plant",
        "remedies": [
            "No disease detected",
            "Maintain proper watering",
            "Apply balanced fertilizer",
            "Regular monitoring recommended"
        ]
    }
}

# ================== STREAMLIT UI ==================
st.set_page_config(page_title="Plant Disease Detection", layout="centered")

st.title("🌿 Plant Disease Detection System")
st.write("Upload a leaf image to detect disease and get remedies")

uploaded_file = st.file_uploader(
    "Upload Leaf Image",
    type=["jpg", "jpeg", "png"]
)

# ================== PREDICTION ==================
if uploaded_file is not None:
    # Read image
    image_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

    if image is None:
        st.error("Invalid image file")
    else:
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        image = image / 255.0
        image = np.expand_dims(image, axis=0)

        st.image(uploaded_file, caption="Uploaded Leaf Image", use_column_width=True)

        # Predict
        predictions = model.predict(image)
        class_index = np.argmax(predictions)
        confidence = float(np.max(predictions)) * 100

        predicted_class = class_names[class_index]
        result = disease_info[predicted_class]

        # ================== OUTPUT ==================
        st.subheader("🦠 Disease Identified")
        st.success(result["name"])

        st.subheader("📊 Prediction Confidence")
        st.write(f"{confidence:.2f}%")

        st.subheader("💊 Recommended Remedies")
        for remedy in result["remedies"]:
            st.write("✔️", remedy)
