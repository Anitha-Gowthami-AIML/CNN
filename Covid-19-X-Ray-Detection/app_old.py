# -*- coding: utf-8 -*-
import base64
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

import streamlit as st
import base64
import os

# 👇 PLACE YOUR FUNCTION HERE
def set_background_image(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

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
    }}
    </style>
    """

      st.markdown(css, unsafe_allow_html=True);
      background-attachment: fixed;
      color: #f1f9ff;
      }
      .stApp::before {
      content: '';
      position: fixed;
      inset: 0;
      background: rgba(0, 11, 22, 0.35);
      z-index: 0;
      }
      .main > div.block-container {
      position: relative;
      z-index: 1;
      }
      .stButton>button {
      background: linear-gradient(

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR/"models"/"model_2_vgg16.keras"
BACKGROUND_IMAGE = "blur_HB.jpg"
CLASS_NAMES = ["Covid", "Normal", "Viral Pneumonia"]


def set_background_image(image_path: Path) -> None:
    if not image_path.is_file():
        return
    encoded = base64.b64encode(image_path.read_bytes()).decode()
    page_bg = f"""
    <style>
    .stApp {
        background-image: linear-gradient(rgba(0, 12, 30, 0.68), rgba(0, 12, 30, 0.68)), url('data:image/jpeg;base64,{encoded}');
        background-size: cover;
        background-position: center135deg, #00a4ff, #00ffcc);
        color: #051923;
        font-weight: 700;
        border: none;
    }
    .stMarkdown, .stText, .stDivider {
        color: #e9f7ff;
    }
    .card {
        border: 1px solid rgba(0, 196, 255, 0.18);
        border-radius: 20px;
        padding: 24px;
        background: rgba(3, 18, 35, 0.82);
        box-shadow: 0 0 40px rgba(0, 138, 255, 0.12);
    }
    .small-muted {
        color: rgba(190, 220, 255, 0.7);
        font-size: 0.95rem;
    }
    .label-text {
        color: #9ee8ff;
        font-size: 0.95rem;
        font-weight: 600;
    }
    .prob-row {
        margin-bottom: 14px;
    }
    .prob-track {
        position: relative;
        height: 12px;
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.08);
        overflow: hidden;
    }
    .prob-fill {
        height: 100%;
        border-radius: 999px;
    }
    </style>
    """
    st.markdown(page_bg, unsafe_allow_html=True)


@st.cache_resource
def load_model(path: Path):
    if not path.is_file():
        return None
    try:
        return tf.keras.models.load_model(str(path))
    except Exception:
        return None


def preprocess_image(image: Image.Image, target_size=(224, 224)) -> np.ndarray:
    image = image.convert("RGB").resize(target_size)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return np.expand_dims(array, axis=0)


def format_probability_row(label: str, value: float) -> str:
    color = "#ff5c5c" if label == "Covid" else ("#ffb74d" if label == "Viral Pneumonia" else "#4bd37f")
    width = max(1, min(100, int(value * 100)))
    return f"""
    <div class='prob-row'>
      <div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;'>
        <span style='font-weight:700; color:#e8f9ff;'>{label}</span>
        <span style='color:{color}; font-weight:700;'>{value * 100:.1f}%</span>
      </div>
      <div class='prob-track'>
        <div class='prob-fill' style='width:{width}%; background: linear-gradient(90deg, {color} 0%, {color}cc 100%);'></div>
      </div>
    </div>
    """


def main() -> None:
    BACKGROUND_IMAGE = "blur_HB.jpg"
    st.set_page_config(
        page_title="COVID-19 X-Ray Detector",
        page_icon="🩺",
        layout="wide",
    )

    set_background_image(BACKGROUND_IMAGE)

    st.markdown(
        """
        <div class='card'>
          <h1 style='margin-bottom:0.25rem;'>COVID-19 X-Ray Detector</h1>
          <p class='small-muted'>Upload a chest X-ray image to classify COVID-19, viral pneumonia, or normal lungs using a trained CNN model.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    model = load_model(MODEL_PATH)

    if model is None:
        st.error(
            "The model failed to load. Make sure the file `models/model_2_vgg16.keras` is present and the app has permission to read it."
        )
        if not MODEL_PATH.is_file():
            st.info(f"Expected model path: `{MODEL_PATH}`")
        return

    col1, col2 = st.columns([1.2, 1], gap="large")

    with col1:
        st.markdown(
            """
            <div class='card'>
              <div style='display:flex; justify-content:space-between; align-items:center;'>
                <div>
                  <p class='label-text'>Step 1</p>
                  <h2 style='margin:0;'>Upload your X-ray image</h2>
                </div>
              </div>
            """,
            unsafe_allow_html=True,
        )

        uploaded_file = st.file_uploader(
            "Choose an X-ray image (JPG, JPEG, PNG)",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded chest X-ray", use_column_width=True)

            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown(
                """
                <div class='card'>
                  <div style='display:flex; justify-content:space-between; align-items:center;'>
                    <div>
                      <p class='label-text'>Step 2</p>
                      <h2 style='margin:0;'>Analyze the image</h2>
                    </div>
                  </div>
                """,
                unsafe_allow_html=True,
            )

            with st.spinner("Analyzing image..."):
                inputs = preprocess_image(image)
                predictions = model.predict(inputs, verbose=0)[0]
                prediction_index = int(np.argmax(predictions))
                prediction_label = CLASS_NAMES[prediction_index]
                confidence = float(predictions[prediction_index] * 100)

            icon = "🔴" if prediction_label == "Covid" else ("🟡" if prediction_label == "Viral Pneumonia" else "🟢")

            st.markdown(
                f"""
                <div class='card'>
                  <h3 style='margin-bottom:0.5rem;'>Result</h3>
                  <p style='font-size:1.1rem; margin:0.15rem 0; color:#d6faff;'>Prediction: <strong>{prediction_label}</strong> {icon}</p>
                  <p style='margin:0; color:#c7eaff;'>Confidence: <strong>{confidence:.1f}%</strong></p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                "<div class='card' style='padding-top:1rem;'>",
                unsafe_allow_html=True,
            )
            for class_name, probability in zip(CLASS_NAMES, predictions):
                st.markdown(format_probability_row(class_name, float(probability)), unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            if prediction_label == "Covid":
                st.error("⚠️ COVID-19 detected. Please consult a medical professional immediately.")
            elif prediction_label == "Viral Pneumonia":
                st.warning("⚠️ Viral pneumonia detected. A medical evaluation is strongly recommended.")
            else:
                st.success("✅ Normal lung appearance detected. Follow routine healthcare guidance.")

        else:
            st.markdown("</div>", unsafe_allow_html=True)
            st.info("Upload a chest X-ray image to see a prediction.")

    with col2:
        st.markdown(
            """
            <div class='card'>
              <h3>How this app works</h3>
              <ul style='padding-left:1.15rem; color:#d5eeff;'>
                <li>Load a trained CNN model from `models/model_2_vgg16.keras`.</li>
                <li>Resize and normalize the uploaded X-ray to 224×224 pixels.</li>
                <li>Predict among Covid, Normal, and Viral Pneumonia.</li>
                <li>Show class probabilities and a clear recommendation.</li>
              </ul>
            </div>
            <div class='card' style='margin-top:20px;'>
              <h3>Model status</h3>
              <p class='small-muted'>Model file loaded successfully from:</p>
              <p style='word-break:break-all; color:#b8ecff;'>{MODEL_PATH}</p>
            </div>
            """.format(MODEL_PATH=MODEL_PATH),
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
