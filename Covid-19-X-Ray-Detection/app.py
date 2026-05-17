# -*- coding: utf-8 -*-
import base64
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
import gdown
import zipfile
import os

# =========================
# PATHS & CONSTANTS
# =========================
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "model_2_vgg16.keras"
BACKGROUND_IMAGE = BASE_DIR / "blur_HB.jpg"
CLASS_NAMES = ["Covid", "Normal", "Viral Pneumonia"]

# =========================
# UI: BACKGROUND STYLE
# =========================
def set_background_image(image_path: Path):
    if not image_path.is_file():
        st.warning(f"Background image not found: {image_path}")
        return

    encoded = base64.b64encode(image_path.read_bytes()).decode()

    css = f"""
    <style>
    .stApp {{
        background-image: linear-gradient(
            rgba(0, 12, 30, 0.68),
            rgba(0, 12, 30, 0.68)
        ), url("data:image/jpeg;base64,{encoded}");
        
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
        color: #f1f9ff;
    }}

    .stButton>button {{
        background: linear-gradient(135deg, #00a4ff, #00ffcc);
        color: #051923;
        font-weight: 700;
        border-radius: 10px;
    }}

    .card {{
        border-radius: 20px;
        padding: 20px;
        background: rgba(3, 18, 35, 0.82);
        margin-bottom: 20px;
    }}
    </style>
    """

    st.markdown(css, unsafe_allow_html=True)

# =========================
# DOWNLOAD MODEL
# =========================
def download_and_extract_model():
    file_id = "17OucXrgs5Ovv1Q0W9xHg_Y4kYwuRLQ9E"
    zip_path = BASE_DIR / "models.zip"
    extract_path = BASE_DIR

    if MODEL_PATH.exists():
        return

    st.info("📥 Downloading model from Google Drive...")

    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, str(zip_path), quiet=False)

    st.info("📦 Extracting model...")

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

    os.remove(zip_path)
    st.success("✅ Model ready!")

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model(path: Path):
    if not path.is_file():
        return None
    try:
        return tf.keras.models.load_model(str(path))
    except Exception as e:
        st.error(f"Model loading error: {e}")
        return None

# =========================
# IMAGE PREPROCESSING
# =========================
def preprocess_image(image: Image.Image, target_size=(224, 224)) -> np.ndarray:
    image = image.convert("RGB").resize(target_size)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return np.expand_dims(array, axis=0)

# =========================
# MAIN APP
# =========================
def main():
    st.set_page_config(
        page_title="COVID-19 X-Ray Detector",
        page_icon="🩺",
        layout="wide",
    )

    # Step 1: Download model if needed
    download_and_extract_model()

    # Step 2: Apply background
    set_background_image(BACKGROUND_IMAGE)

    # Step 3: UI
    st.title("🩺 COVID-19 X-Ray Detector")
    st.write("Upload a chest X-ray image to classify COVID-19, viral pneumonia, or normal lungs.")

    # Step 4: Load model
    model = load_model(MODEL_PATH)

    if model is None:
        st.error("❌ Model not found. Please check: models/model_2_vgg16.keras")
        return

    # Step 5: Upload
    uploaded_file = st.file_uploader(
        "Upload X-ray image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        with st.spinner("Analyzing..."):
            inputs = preprocess_image(image)
            predictions = model.predict(inputs, verbose=0)[0]

            prediction_index = int(np.argmax(predictions))
            prediction_label = CLASS_NAMES[prediction_index]
            confidence = float(predictions[prediction_index] * 100)

        st.success(f"Prediction: {prediction_label}")
        st.info(f"Confidence: {confidence:.2f}%")

        st.subheader("Class Probabilities")
        for class_name, prob in zip(CLASS_NAMES, predictions):
            st.write(f"{class_name}: {prob*100:.2f}%")

# =========================
# RUN
# =========================
if __name__ == "__main__":
    main()