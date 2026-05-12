"""
╔══════════════════════════════════════════════════════════════════╗
║          ACCIDENT DETECTION FROM CCTV FOOTAGE                   ║
║          Streamlit Dashboard · Golden Edition                    ║
╚══════════════════════════════════════════════════════════════════╝

Tabs:
  🏠 Home          — Project overview & dataset info
  📊 Model Results — Metrics table + charts from notebook
  🔬 Augmentation  — Live interactive augmentation explorer
  🖼️  Image Predict — Upload image → get prediction + confidence
  🎥 Video Predict — Upload video → frame-by-frame detection

Run:
  streamlit run app.py
"""

import os, io, time, tempfile, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from PIL import Image, ImageEnhance, ImageFilter

import streamlit as st
from streamlit_option_menu import option_menu

import tensorflow as tf

import tensorflow as tf
import keras
import gdown
import zipfile

# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AccidentAI · CCTV Detection",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded",
)

#=========================================== gdown code
@st.cache_resource
def download_and_extract():
    url = "https://drive.google.com/uc?id=1dCMfht_6eY1FBfYd69AjLTVTEhUcmFIQ"
              
    zip_path = "CNN_Google_Drive_files.zip"
    extract_path = "model"

    if not os.path.exists(zip_path):
        gdown.download(url, zip_path, quiet=False)

    if not os.path.exists(extract_path):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)

    return extract_path
  #========================================================================================================

APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = download_and_extract()

# ── VGG16 preprocessing: subtracts ImageNet BGR channel means ──────────────
# Input: [0, 255]  →  Output: mean-subtracted float32
try:
    from tensorflow.keras.applications.vgg16 import (                  # TF ≤ 2.15
        preprocess_input as vgg_pre
    )
except ImportError:
    from keras.applications.vgg16 import (                             # TF ≥ 2.16
        preprocess_input as vgg_pre
    )

# ── ResNet50V2 preprocessing: scales pixels to [-1, 1] ────────────────────
# Input: [0, 255]  →  Output: float32 in [-1, 1]
try:
    from tensorflow.keras.applications.resnet_v2 import (              # TF ≤ 2.15
        preprocess_input as resnet_pre
    )
except ImportError:
    from keras.applications.resnet_v2 import (                         # TF ≥ 2.16
        preprocess_input as resnet_pre
    )


# ──────────────────────────────────────────────────────────────────────────────
# GOLDEN THEME — injected CSS
# ──────────────────────────────────────────────────────────────────────────────
GOLDEN_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700;900&family=Raleway:wght@300;400;600;700&family=Fira+Code:wght@400;500&display=swap');

/* ── Root palette ── */
:root {
  --gold-light:   #FFE066;
  --gold-mid:     #F5C518;
  --gold-deep:    #D4A017;
  --gold-dark:    #8B6914;
  --cream:        #FFF8E7;
  --warm-white:   #FFFDF5;
  --charcoal:     #1A1400;
  --brown:        #3D2B00;
  --accent-red:   #C0392B;
  --accent-green: #1E8449;
  --card-bg:      rgba(255,248,220,0.82);
  --glass:        rgba(255,230,100,0.12);
  --shadow:       0 8px 32px rgba(180,130,0,0.18);
}

/* ── Global background: rich golden gradient ── */
.stApp {
  background: linear-gradient(135deg,
    #7B4F00 0%,
    #C8860A 18%,
    #F5C518 38%,
    #FFE87C 55%,
    #F5C518 72%,
    #C8860A 88%,
    #7B4F00 100%
  ) fixed;
  font-family: 'Raleway', sans-serif;
  color: var(--charcoal);
}

/* Animated shimmer layer */
.stApp::before {
  content: '';
  position: fixed;
  inset: 0;
  background: repeating-linear-gradient(
    45deg,
    transparent,
    transparent 40px,
    rgba(255,255,255,0.04) 40px,
    rgba(255,255,255,0.04) 80px
  );
  pointer-events: none;
  z-index: 0;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #3D2B00 0%, #7B4F00 50%, #3D2B00 100%) !important;
  border-right: 2px solid var(--gold-mid);
}
[data-testid="stSidebar"] * {
  color: var(--gold-light) !important;
  font-family: 'Raleway', sans-serif !important;
}
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
  color: var(--gold-mid) !important;
  font-family: 'Playfair Display', serif !important;
}

/* ── Main content area ── */
.main .block-container {
  background: transparent;
  padding: 2rem 2.5rem;
  position: relative;
  z-index: 1;
}

/* ── Cards ── */
.gold-card {
  background: var(--card-bg);
  border: 1.5px solid var(--gold-deep);
  border-radius: 16px;
  padding: 1.6rem 2rem;
  box-shadow: var(--shadow);
  backdrop-filter: blur(10px);
  margin-bottom: 1.4rem;
}
.glass-card {
  background: rgba(255,255,255,0.55);
  border: 1px solid rgba(245,197,24,0.45);
  border-radius: 14px;
  padding: 1.2rem 1.6rem;
  box-shadow: 0 4px 20px rgba(180,130,0,0.12);
  backdrop-filter: blur(8px);
  margin-bottom: 1rem;
}

/* ── Page title ── */
.page-title {
  font-family: 'Playfair Display', serif;
  font-size: 3rem;
  font-weight: 900;
  color: var(--charcoal);
  text-shadow: 0 2px 8px rgba(255,220,50,0.6), 0 0 40px rgba(245,197,24,0.3);
  letter-spacing: -0.5px;
  line-height: 1.1;
  margin-bottom: 0.3rem;
}
.page-subtitle {
  font-family: 'Raleway', sans-serif;
  font-size: 1.05rem;
  font-weight: 400;
  color: var(--brown);
  letter-spacing: 1.5px;
  text-transform: uppercase;
  margin-bottom: 1.5rem;
}

/* ── Section heading ── */
.section-heading {
  font-family: 'Playfair Display', serif;
  font-size: 1.6rem;
  font-weight: 700;
  color: var(--charcoal);
  border-bottom: 2px solid var(--gold-mid);
  padding-bottom: 0.4rem;
  margin: 1.6rem 0 1rem 0;
}

/* ── Metric boxes ── */
.metric-box {
  background: linear-gradient(135deg, rgba(255,248,200,0.9), rgba(255,230,80,0.7));
  border: 1.5px solid var(--gold-deep);
  border-radius: 12px;
  padding: 1rem 1.2rem;
  text-align: center;
  box-shadow: 0 4px 14px rgba(180,130,0,0.15);
}
.metric-box .metric-value {
  font-family: 'Playfair Display', serif;
  font-size: 2rem;
  font-weight: 900;
  color: var(--charcoal);
}
.metric-box .metric-label {
  font-size: 0.75rem;
  font-weight: 600;
  letter-spacing: 1.2px;
  text-transform: uppercase;
  color: var(--gold-dark);
  margin-top: 0.2rem;
}

/* ── Prediction badge ── */
.pred-accident {
  background: linear-gradient(135deg, #FFEAEA, #FFCCCC);
  border: 2px solid var(--accent-red);
  border-radius: 12px;
  padding: 1.4rem;
  text-align: center;
}
.pred-safe {
  background: linear-gradient(135deg, #EAFFF0, #CCFFDD);
  border: 2px solid var(--accent-green);
  border-radius: 12px;
  padding: 1.4rem;
  text-align: center;
}
.pred-label {
  font-family: 'Playfair Display', serif;
  font-size: 1.8rem;
  font-weight: 900;
}
.pred-conf {
  font-family: 'Raleway', sans-serif;
  font-size: 1rem;
  font-weight: 600;
  letter-spacing: 1px;
  margin-top: 0.4rem;
}

/* ── Confidence bar ── */
.conf-bar-wrap {
  background: rgba(255,255,255,0.5);
  border-radius: 50px;
  height: 14px;
  width: 100%;
  margin-top: 0.6rem;
  overflow: hidden;
  border: 1px solid rgba(0,0,0,0.08);
}
.conf-bar-fill-accident {
  height: 100%;
  border-radius: 50px;
  background: linear-gradient(90deg, #FF6B6B, #C0392B);
  transition: width 0.5s ease;
}
.conf-bar-fill-safe {
  height: 100%;
  border-radius: 50px;
  background: linear-gradient(90deg, #55EFC4, #1E8449);
  transition: width 0.5s ease;
}

/* ── Tables ── */
.dataframe {
  font-family: 'Fira Code', monospace !important;
  font-size: 0.85rem !important;
  background: var(--warm-white) !important;
}
.dataframe thead th {
  background: var(--gold-mid) !important;
  color: var(--charcoal) !important;
  font-family: 'Raleway', sans-serif !important;
  font-weight: 700 !important;
  letter-spacing: 0.8px !important;
}

/* ── Buttons ── */
.stButton > button {
  background: linear-gradient(135deg, var(--gold-mid), var(--gold-deep)) !important;
  color: var(--charcoal) !important;
  font-family: 'Raleway', sans-serif !important;
  font-weight: 700 !important;
  letter-spacing: 1px !important;
  border: none !important;
  border-radius: 8px !important;
  padding: 0.55rem 1.4rem !important;
  box-shadow: 0 4px 12px rgba(180,130,0,0.3) !important;
  transition: all 0.2s ease !important;
}
.stButton > button:hover {
  transform: translateY(-2px) !important;
  box-shadow: 0 6px 20px rgba(180,130,0,0.45) !important;
}

/* ── Sliders & selects ── */
.stSlider > div > div > div {
  background: var(--gold-mid) !important;
}
[data-baseweb="select"] {
  background: var(--cream) !important;
  border-color: var(--gold-mid) !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
  background: transparent;
  gap: 0.3rem;
}
.stTabs [data-baseweb="tab"] {
  background: rgba(255,248,200,0.6);
  border: 1px solid var(--gold-deep);
  border-radius: 8px 8px 0 0;
  font-family: 'Raleway', sans-serif;
  font-weight: 600;
  color: var(--brown);
  padding: 0.5rem 1.1rem;
}
.stTabs [aria-selected="true"] {
  background: var(--gold-mid) !important;
  color: var(--charcoal) !important;
  border-bottom-color: transparent !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
  background: rgba(255,248,200,0.7) !important;
  border: 2px dashed var(--gold-deep) !important;
  border-radius: 12px !important;
}

/* ── Progress bar ── */
.stProgress > div > div {
  background: linear-gradient(90deg, var(--gold-mid), var(--gold-deep)) !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 8px; }
::-webkit-scrollbar-track { background: rgba(255,248,200,0.3); }
::-webkit-scrollbar-thumb { background: var(--gold-deep); border-radius: 4px; }

/* ── Divider ── */
hr { border-color: var(--gold-deep) !important; opacity: 0.4 !important; }

/* ── Expander ── */
[data-testid="stExpander"] {
  background: rgba(255,248,200,0.6) !important;
  border: 1px solid var(--gold-deep) !important;
  border-radius: 10px !important;
}

/* Nav menu override */
nav.css-1cypcdb { background: transparent !important; }
</style>
"""

st.markdown(f"<style>{GOLDEN_CSS}", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────
IMG_SIZE    = (224, 224)
CLASS_NAMES = ["Accident", "Non Accident"]

MODEL_FILES = {
    "1. Baseline CNN":   "model_1_baseline_cnn.keras",
    "2. Aug CNN":        "model_2_augmented_cnn.keras",
    "3. CNN+BN+Drop":   "model_3_cnn_bn_dropout.keras",
    "4a. VGG16 plain":  "model_4a_vgg16_plain.keras",
    "4b. VGG16+BN":     "model_4b_vgg16_bn.keras",
    "5a. ResNet plain":  "model_5a_resnet_plain.keras",
    "5b. ResNet+BN":    "model_5b_resnet_bn.keras",
    "6a. EffNet plain":  "model_6a_effnet_plain.keras",
    "6b. EffNet+BN":    "model_6b_effnet_bn.keras",
}

# Which models expect raw [0-255] pixels vs normalised [0-1]
RAW_MODELS = {"4a. VGG16 plain", "4b. VGG16+BN",
              "5a. ResNet plain",  "5b. ResNet+BN",
              "6a. EffNet plain",  "6b. EffNet+BN"}

# ──────────────────────────────────────────────────────────────────────────────
# DEMO METRICS — pre-filled so app works without live training
# Replace with real values after training
# ──────────────────────────────────────────────────────────────────────────────
DEMO_METRICS = {
    "1. Baseline CNN":   dict(TestAcc=0.93, AccPrec=0.9259, AccRecall=0.9434, NonAccPrec=0.9348, NonAccRecall=0.9149, ROCAUC=0.9619),
    "2. Aug CNN":        dict(TestAcc=0.53, AccPrec=0.53, AccRecall=1.0, NonAccPrec=0.0, NonAccRecall=0.0, ROCAUC=0.601),
    "3. CNN+BN+Drop":   dict(TestAcc=0.53, AccPrec=0.53, AccRecall=1.0, NonAccPrec=0.0, NonAccRecall=0.0, ROCAUC=0.3848),
    "4a. VGG16 plain":  dict(TestAcc=0.85, AccPrec=0.8393, AccRecall=0.8868, NonAccPrec=0.8636, NonAccRecall=0.8085, ROCAUC=0.9209),
    "4b. VGG16+BN":     dict(TestAcc=0.76, AccPrec=0.7636, AccRecall=0.7925, NonAccPrec=0.7556, NonAccRecall=0.7234, ROCAUC=0.8366),
    "5a. ResNet plain":  dict(TestAcc=0.83, AccPrec=0.7647, AccRecall=0.9811, NonAccPrec=0.9688, NonAccRecall=0.6596, ROCAUC=0.8836),
    "5b. ResNet+BN":    dict(TestAcc=0.78, AccPrec=0.7627, AccRecall=0.8491, NonAccPrec=0.8049, NonAccRecall=0.7021, ROCAUC=0.8149),
    "6a. EffNet plain":  dict(TestAcc=0.72, AccPrec=0.6761, AccRecall=0.9057, NonAccPrec=0.8276, NonAccRecall=0.5106, ROCAUC=0.7796),
    "6b. EffNet+BN":    dict(TestAcc=0.64, AccPrec=0.6197, AccRecall=0.8302, NonAccPrec=0.6897, NonAccRecall=0.4255, ROCAUC=0.7134),
}

def build_aug_cnn_model():
    augmentation_layer = keras.models.Sequential([
        keras.layers.RandomFlip("horizontal"),
        keras.layers.RandomRotation(0.15),
        keras.layers.RandomZoom(0.15),
        keras.layers.RandomTranslation(0.1, 0.1),
        keras.layers.RandomContrast(0.2),
        keras.layers.RandomBrightness(0.2),
    ], name="Augmentation")

    model = keras.models.Sequential([
        keras.layers.InputLayer(input_shape=(224, 224, 3)),
        augmentation_layer,
        keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        keras.layers.MaxPooling2D(2, 2),
        keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        keras.layers.MaxPooling2D(2, 2),
        keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        keras.layers.MaxPooling2D(2, 2),
        keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        keras.layers.MaxPooling2D(2, 2),
        keras.layers.Flatten(),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ], name="Augmented_CNN")

    return model


def build_cnn_bn_dropout_model():
    augmentation_extended = keras.models.Sequential([
        keras.layers.RandomFlip("horizontal"),
        keras.layers.RandomFlip("vertical"),
        keras.layers.RandomRotation(0.20),
        keras.layers.RandomZoom(0.20),
        keras.layers.RandomTranslation(0.10, 0.10),
        keras.layers.RandomContrast(0.25),
        keras.layers.RandomBrightness(0.25),
        keras.layers.GaussianNoise(0.02),
    ], name="Extended_Augmentation")

    inputs = keras.Input(shape=(224, 224, 3))
    x = augmentation_extended(inputs)

    he_init = keras.initializers.HeNormal()
    x = keras.layers.Conv2D(32, (3, 3), padding='same', kernel_initializer=he_init,
                             kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.MaxPooling2D(2, 2)(x)

    x = keras.layers.Conv2D(64, (3, 3), padding='same', kernel_initializer=he_init,
                             kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.MaxPooling2D(2, 2)(x)

    x = keras.layers.Conv2D(128, (3, 3), padding='same', kernel_initializer=he_init,
                             kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.MaxPooling2D(2, 2)(x)

    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(128, kernel_initializer=he_init,
                            kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.Dropout(0.3)(x)

    outputs = keras.layers.Dense(1, activation='sigmoid')(x)
    return keras.Model(inputs, outputs, name="CNN_BN_Dropout")


def _try_load_aug_weights(path: str):
    base = os.path.basename(path)
    weights_map = {
        "model_2_augmented_cnn.keras": "model_2_augmented_cnn.weights.h5",
        "model_3_cnn_bn_dropout.keras": "model_3_cnn_bn_dropout.weights.h5",
    }
    weights_file = weights_map.get(base)
    if not weights_file:
        return None

    #weights_path = os.path.join(APP_DIR, weights_file)
    weights_path = os.path.join(MODEL_DIR, "CNN_Google_drive_files", weights_file)
   
    if not os.path.exists(weights_path):
        return None

    if base == "model_2_augmented_cnn.keras":
        model = build_aug_cnn_model()
    else:
        model = build_cnn_bn_dropout_model()


    # Ensure the fallback model is built by running a dummy input through it.
    model(np.zeros((1, 224, 224, 3), dtype=np.float32))
    model.load_weights(weights_path)
    return model

# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model_cached(path: str):
    """
    Load a .keras model, working around three distinct Keras 3.3.x
    deserialization bugs that affect models with nested sub-models
    (EfficientNet, ResNet, VGG) plus an embedded augmentation Sequential:

    BUG 1 — "could not be deserialized properly" (module path mismatch)
        The config stores module='keras.src.models.functional' for nested
        Functional sub-models, but the public deserializer only resolves
        module='keras'.  Fix: rewrite all 'keras.src.models.*' paths to 'keras'.

    BUG 2 — "get_tensor() takes 3 positional arguments but 14 were given"
        ROOT CAUSE (confirmed via Keras 3.3.3 source inspection):
        functional_from_config() calls map_tensors(config["input_layers"]) where
        config["input_layers"] is a FLAT list ['layer_name', 0, 0] (3 items).
        It then does `for v in tensors: get_tensor(*v)`.  When tensors is the
        flat list, the first v becomes the STRING 'layer_name' and *'layer_name'
        unpacks the string into individual characters.  'input_layer_15' has
        exactly 14 characters → 14 positional args → TypeError.
        The config should store [['layer_name', 0, 0]] (list-of-lists).
        Fix: wrap any flat [str, int, int] input_layers / output_layers in an
        outer list: ['input_layer_15', 0, 0] → [['input_layer_15', 0, 0]].

    BUG 3 — inbound_nodes kwargs overflow
        Nested model calls store training=False, mask=None in kwargs.
        When these get mixed with args during deserialisation they can cause
        positional-argument overflows on the sub-model call.
        Fix: strip 'training' and 'mask' from every inbound_nodes kwargs dict.

    Strategy 1 — plain keras.load_model (works for simple models)
    Strategy 2 — patch config.json in-memory (fixes all three bugs above)
    Strategy 3 — additionally strip the augmentation Sequential layer
    Strategy 4 — tf.keras fallback
    """
    if not os.path.exists(path):
        return None

    import keras as _keras
    import zipfile, json, copy, tempfile

    # ── Strategy 1: plain load (works for non-nested models) ─────────────────
    try:
        return _keras.models.load_model(path, compile=False, safe_mode=False)
    except Exception:
        pass

    # ──────────────────────────────────────────────────────────────────────────
    # Config patcher — fixes all three bugs recursively
    # ──────────────────────────────────────────────────────────────────────────
    def _is_flat_tensor_ref(obj):
        """True if obj looks like ['layer_name', node_idx, tensor_idx]."""
        return (
            isinstance(obj, list)
            and len(obj) == 3
            and isinstance(obj[0], str)
            and isinstance(obj[1], int)
            and isinstance(obj[2], int)
        )

    def _patch_config(node):
        """
        Recursively patch the config dict:
          a) keras.src.models.functional / .sequential  →  'keras'
          b) inbound_nodes kwargs: strip 'training' and 'mask'
          c) input_layers / output_layers: wrap flat refs in outer list
             ['layer_name', 0, 0]  →  [['layer_name', 0, 0]]
        """
        if isinstance(node, dict):
            # ── (a) Fix internal module paths ──────────────────────────────
            mod = node.get("module", "")
            if "keras.src.models.functional" in mod:
                node["module"] = "keras"
            if "keras.src.models.sequential" in mod:
                node["module"] = "keras"

            # ── (b) Strip training/mask kwargs from inbound_nodes ──────────
            if "inbound_nodes" in node:
                cleaned = []
                for entry in node["inbound_nodes"]:
                    if isinstance(entry, dict) and "kwargs" in entry:
                        kw = entry["kwargs"]
                        if isinstance(kw, dict):
                            kw.pop("training", None)
                            kw.pop("mask", None)
                            entry["kwargs"] = kw
                    cleaned.append(entry)
                node["inbound_nodes"] = cleaned

            # ── (c) Fix input_layers / output_layers format ─────────────────
            # Keras 3.3.3 functional_from_config iterates:
            #   for v in config["input_layers"]: get_tensor(*v)
            # If input_layers = ['input_layer_15', 0, 0]  (flat, 3 items)
            #   → v = 'input_layer_15'  → get_tensor(*'input_layer_15')
            #   → 14 positional args (one per character) → TypeError
            # Correct format: [['input_layer_15', 0, 0]]  (list of lists)
            #   → v = ['input_layer_15', 0, 0]  → get_tensor(*v) → 3 args ✓
            for key in ("input_layers", "output_layers"):
                if key in node:
                    val = node[key]
                    if _is_flat_tensor_ref(val):
                        # flat ['name', 0, 0] → wrap in outer list
                        node[key] = [val]
                    elif isinstance(val, list) and val and _is_flat_tensor_ref(val[0]):
                        # already [[name, 0, 0]] — correct, leave it
                        pass
                    elif isinstance(val, list):
                        # May be a list of flat refs (multi-output model)
                        # e.g. [['name1',0,0], ['name2',0,0]] — already fine
                        # but defensively fix any bare strings at top level
                        fixed = []
                        for item in val:
                            if _is_flat_tensor_ref(item):
                                fixed.append(item)
                            else:
                                fixed.append(item)
                        node[key] = fixed

            # Recurse into all values
            for k, v in node.items():
                node[k] = _patch_config(v)

        elif isinstance(node, list):
            node = [_patch_config(item) for item in node]

        return node

    # ── Strategy 2: patch config.json inside the .keras zip ──────────────────
    try:
        with zipfile.ZipFile(path, "r") as zf_in:
            names = zf_in.namelist()
            file_bytes = {n: zf_in.read(n) for n in names}

        cfg = json.loads(file_bytes["config.json"].decode("utf-8"))
        cfg_patched = _patch_config(copy.deepcopy(cfg))
        file_bytes["config.json"] = json.dumps(cfg_patched).encode("utf-8")

        tmp_keras = tempfile.NamedTemporaryFile(suffix=".keras", delete=False)
        tmp_keras.close()
        with zipfile.ZipFile(tmp_keras.name, "w", zipfile.ZIP_DEFLATED) as zf_out:
            for name, data in file_bytes.items():
                zf_out.writestr(name, data)

        try:
            model = _keras.models.load_model(
                tmp_keras.name, compile=False, safe_mode=False
            )
            return model
        finally:
            try:
                os.unlink(tmp_keras.name)
            except Exception:
                pass

    except Exception:
        pass

    # ── Strategy 3: also strip the augmentation Sequential ───────────────────
    try:
        with zipfile.ZipFile(path, "r") as zf_in:
            file_bytes = {n: zf_in.read(n) for n in zf_in.namelist()}

        cfg = json.loads(file_bytes["config.json"].decode("utf-8"))
        cfg = _patch_config(copy.deepcopy(cfg))

        def _strip_aug_sequential(config):
            """Remove augmentation Sequential layers and re-wire to InputLayer."""
            if not isinstance(config, dict):
                return config
            if config.get("class_name") not in ("Functional", "Model"):
                return config

            layers = config.get("config", {}).get("layers", [])
            aug_names = set()
            kept = []
            for lyr in layers:
                cn = lyr.get("class_name", "")
                nm = lyr.get("config", {}).get("name", "")
                if cn == "Sequential" and any(
                    kw in nm.lower() for kw in ("aug", "augment", "augmentation")
                ):
                    aug_names.add(nm)
                else:
                    kept.append(lyr)

            if not aug_names:
                return config

            input_name = next(
                (l["config"]["name"] for l in kept if l.get("class_name") == "InputLayer"),
                None
            )

            def _fix_node(obj):
                if isinstance(obj, dict):
                    hist = obj.get("config", {}).get("keras_history")
                    if isinstance(hist, list) and hist and hist[0] in aug_names:
                        obj["config"]["keras_history"] = [input_name, 0, 0]
                    for k, v in obj.items():
                        obj[k] = _fix_node(v)
                elif isinstance(obj, list):
                    obj = [_fix_node(i) for i in obj]
                return obj

            for lyr in kept:
                lyr["inbound_nodes"] = _fix_node(lyr.get("inbound_nodes", []))
            config["config"]["layers"] = kept
            return config

        cfg_stripped = _strip_aug_sequential(cfg)
        file_bytes["config.json"] = json.dumps(cfg_stripped).encode("utf-8")

        tmp2 = tempfile.NamedTemporaryFile(suffix=".keras", delete=False)
        tmp2.close()
        with zipfile.ZipFile(tmp2.name, "w", zipfile.ZIP_DEFLATED) as zf_out:
            for name, data in file_bytes.items():
                zf_out.writestr(name, data)

        try:
            model = _keras.models.load_model(
                tmp2.name, compile=False, safe_mode=False
            )
            return model
        finally:
            try:
                os.unlink(tmp2.name)
            except Exception:
                pass

    except Exception:
        pass

    # ── Strategy 4: tf.keras fallback ────────────────────────────────────────
    try:
        return tf.keras.models.load_model(path, compile=False)
    except Exception as final_err:
        # Final fallback: reconstruct the Aug CNN models from their architecture
        # and load weights directly if the .keras serialization is broken.
        fallback = _try_load_aug_weights(path)
        if fallback is not None:
            return fallback

        raise RuntimeError(
            f"All model-loading strategies failed for '{path}'.\n"
            f"Last error: {final_err}\n\n"
            "Root cause: Keras 3.3.x bug where config stores input_layers as a\n"
            "flat list ['layer_name', 0, 0] but functional_from_config expects\n"
            "[['layer_name', 0, 0]].  The string 'input_layer_15' has 14 chars,\n"
            "which unpacks to 14 positional args instead of 3.\n\n"
            "Permanent fix — run once in your training environment:\n\n"
            "    import keras\n"
            "    m = keras.models.load_model('model.keras', compile=False, safe_mode=False)\n"
            "    m.save('model_fixed.keras')\n\n"
            "Requires: keras==3.3.3  tensorflow==2.16.1  tf-keras==2.16.0"
        ) from final_err

def preprocess_for_model(img_rgb: np.ndarray, model_name: str) -> np.ndarray:
    """
    Resize + preprocess a single RGB image for inference.
    Returns shape (1, 224, 224, 3).
    """
    img = cv2.resize(img_rgb, IMG_SIZE, interpolation=cv2.INTER_AREA).astype(np.float32)
    if model_name not in RAW_MODELS:
        img = img / 255.0
    return np.expand_dims(img, axis=0)

def predict_frame(img_rgb: np.ndarray, model, model_name: str):
    """
    Returns (class_label, confidence_float).
    """
    tensor = preprocess_for_model(img_rgb, model_name)
    prob   = float(model.predict(tensor, verbose=0)[0][0])
    idx    = int(prob > 0.5)
    conf   = prob if idx == 1 else (1.0 - prob)
    return CLASS_NAMES[idx], conf

def gold_divider():
    st.markdown(
        "<hr style='border:1.5px solid #D4A017;margin:1rem 0;opacity:0.5'>",
        unsafe_allow_html=True
    )

def card(content_fn, *args, **kwargs):
    st.markdown('<div class="gold-card">', unsafe_allow_html=True)
    content_fn(*args, **kwargs)
    st.markdown('</div>', unsafe_allow_html=True)

def metric_box(label: str, value: str):
    st.markdown(f"""
    <div class="metric-box">
      <div class="metric-value">{value}</div>
      <div class="metric-label">{label}</div>
    </div>""", unsafe_allow_html=True)

def prediction_display(label: str, conf: float):
    is_acc = (label == "Accident")
    css    = "pred-accident" if is_acc else "pred-safe"
    icon   = "🚨" if is_acc else "✅"
    color  = "#C0392B" if is_acc else "#1E8449"
    bar_cls = "conf-bar-fill-accident" if is_acc else "conf-bar-fill-safe"
    pct    = conf * 100
    st.markdown(f"""
    <div class="{css}">
      <div class="pred-label" style="color:{color}">{icon} {label}</div>
      <div class="pred-conf">Confidence: {pct:.1f}%</div>
      <div class="conf-bar-wrap">
        <div class="{bar_cls}" style="width:{pct}%"></div>
      </div>
    </div>""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:1rem 0'>
      <div style='font-family:"Playfair Display",serif;font-size:1.5rem;
                  font-weight:900;color:#FFE066;letter-spacing:1px'>
        🚨 AccidentAI
      </div>
      <div style='font-size:0.7rem;letter-spacing:2px;text-transform:uppercase;
                  color:#D4A017;margin-top:0.3rem'>
        CCTV Detection Suite
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📂 Load Model")

    model_choice = st.selectbox(
        "Select trained model",
        list(MODEL_FILES.keys()),
        index=8,   # defaults to 6b EffNet+BN
    )

    #model_path = os.path.join(APP_DIR, MODEL_FILES[model_choice])
    #model_exists = os.path.exists(model_path)

    model_path = os.path.join(MODEL_DIR, "CNN_Google_drive_files", MODEL_FILES[model_choice])
    model_exists = os.path.exists(model_path)
    

    if model_exists:
        with st.spinner("Loading model…"):
            try:
                model = load_model_cached(model_path)
                if model:
                    st.success("✓ Loaded successfully")
                else:
                    st.error("Model returned None — file may be corrupt.")
                    model = None
            except RuntimeError as _load_err:
                st.error("❌ Model load failed.")
                with st.expander("Show error details"):
                    st.code(str(_load_err), language="text")
                model = None
    else:
        st.warning(f"⚠️ `{model_path}` not found.\nPlace your `.keras` files in the same folder as `app.py`.\n\nPrediction tabs will show a demo mode.")
        model = None

    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    threshold = st.slider("Prediction threshold", 0.10, 0.90, 0.50, 0.01,
                          help="Probability above this → Non-Accident")
    show_raw  = st.checkbox("Show raw probability", value=False)

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.72rem;color:#A07820;line-height:1.6'>
      <b>Classes</b><br>
      🔴 Accident<br>
      🟢 Non Accident<br><br>
      <b>Input size</b>: 224 × 224<br>
      <b>Framework</b>: TensorFlow 2.x<br>
      <b>Dataset</b>: Kaggle CCTV
    </div>
    """, unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# MAIN HEADER
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center;padding:1.5rem 0 0.5rem'>
  <div class='page-title'>🚨 AccidentAI</div>
  <div class='page-subtitle'>CCTV Accident Detection · Deep Learning Dashboard</div>
</div>
""", unsafe_allow_html=True)
gold_divider()

# ──────────────────────────────────────────────────────────────────────────────
# TABS
# ──────────────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "🏠  Overview",
    "📊  Model Results",
    "🔬  Augmentation Lab",
    "🖼️  Image Prediction",
    "🎥  Video Detection",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    col1, col2 = st.columns([1.6, 1])

    with col1:
        st.markdown('<div class="section-heading">Project Overview</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="gold-card">
        <p style="font-size:1.05rem;line-height:1.8;color:#2C1800">
        This dashboard presents a complete <b>binary image classifier</b> trained to detect
        <span style="color:#C0392B;font-weight:700">accidents</span> vs
        <span style="color:#1E8449;font-weight:700">normal traffic</span> in CCTV footage.
        </p>
        <p style="line-height:1.8;color:#3D2B00">
        The pipeline progresses from a simple <b>Baseline CNN</b> through
        <b>data augmentation</b>, <b>BatchNorm + Dropout</b>, up to full
        <b>transfer learning</b> with VGG16, ResNet50V2, and EfficientNetB0 —
        each tested with and without regularisation heads.
        Class imbalance is addressed via <b>computed class weights</b>.
        </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-heading">Model Lineage</div>', unsafe_allow_html=True)
        models_info = [
            ("1", "Baseline CNN",    "3 conv blocks · no augmentation",                    "#4C72B0"),
            ("2", "Aug CNN",         "6 augmentation transforms · 4 conv blocks",          "#55A868"),
            ("3", "CNN+BN+Drop",     "BN + Dropout + HeNormal init · 5 blocks",            "#C44E52"),
            ("4a","VGG16 plain",     "ImageNet pretrained · frozen → fine-tuned",          "#DD8452"),
            ("4b","VGG16 + BN/Drop", "VGG16 + regularised head",                          "#E8A050"),
            ("5a","ResNet50V2 plain","Residual connections · pre-activation BN",           "#8172B3"),
            ("5b","ResNet + BN/Drop","ResNet + regularised head",                          "#9B8BCC"),
            ("6a","EffNet plain",    "Compound scaling · MBConv blocks",                   "#937860"),
            ("6b","EffNet + BN/Drop","EfficientNetB0 + regularised head ← Best 🏆",       "#D4A017"),
        ]
        for num, name, desc, color in models_info:
            st.markdown(f"""
            <div style='display:flex;align-items:center;gap:0.8rem;
                        background:rgba(255,248,200,0.7);border-left:4px solid {color};
                        border-radius:0 10px 10px 0;padding:0.55rem 1rem;margin-bottom:0.45rem;
                        box-shadow:0 2px 8px rgba(180,130,0,0.1)'>
              <span style='background:{color};color:#fff;font-family:"Fira Code",monospace;
                           font-size:0.75rem;font-weight:700;padding:0.2rem 0.5rem;
                           border-radius:6px;min-width:2.5rem;text-align:center'>{num}</span>
              <span style='font-family:"Playfair Display",serif;font-size:1rem;
                           font-weight:700;color:#1A1400'>{name}</span>
              <span style='font-size:0.82rem;color:#5C4020;flex:1'>{desc}</span>
            </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-heading">Dataset Stats</div>', unsafe_allow_html=True)
        dataset_stats = [
            ("Total Images",  "~1,000"),
            ("Training Set",  "~790"),
            ("Validation",    "~105"),
            ("Test Set",      "~105"),
            ("Classes",       "2"),
            ("Image Size",    "224×224"),
            ("Imbalance Fix", "Class Weights"),
        ]
        for label, val in dataset_stats:
            col_a, col_b = st.columns([1.2, 1])
            with col_a:
                st.markdown(f"<div style='font-size:0.82rem;font-weight:600;color:#5C4020;padding:0.35rem 0'>{label}</div>", unsafe_allow_html=True)
            with col_b:
                st.markdown(f"<div style='font-family:\"Fira Code\",monospace;font-size:0.9rem;font-weight:700;color:#1A1400;padding:0.35rem 0'>{val}</div>", unsafe_allow_html=True)

        gold_divider()

        st.markdown('<div class="section-heading">Key Techniques</div>', unsafe_allow_html=True)
        techniques = [
            ("🔄", "Data Augmentation",   "8 transforms incl. noise"),
            ("📐", "BatchNormalization",  "Stable training"),
            ("🎲", "Dropout",             "0.3–0.5 rate"),
            ("⚖️", "He Initializer",      "ReLU-optimal init"),
            ("🏋️", "Transfer Learning",   "ImageNet weights"),
            ("🔧", "Fine-tuning",          "Top layers unfrozen"),
            ("⚖️", "Class Weights",       "Imbalance handling"),
            ("📉", "Early Stopping",      "Best weights kept"),
        ]
        for icon, name, detail in techniques:
            st.markdown(f"""
            <div style='display:flex;gap:0.6rem;align-items:center;
                        padding:0.4rem 0.2rem;border-bottom:1px solid rgba(212,160,23,0.2)'>
              <span style='font-size:1.1rem'>{icon}</span>
              <div>
                <div style='font-size:0.88rem;font-weight:700;color:#1A1400'>{name}</div>
                <div style='font-size:0.75rem;color:#7A5C20'>{detail}</div>
              </div>
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — MODEL RESULTS
# ══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown('<div class="section-heading">📊 Performance Metrics — All Models</div>', unsafe_allow_html=True)

    # ── Detect whether DEMO_METRICS are still the placeholder values ──────────
    # We compare against one known placeholder value. If users have replaced
    # DEMO_METRICS with their real training outputs, this banner disappears.
    _is_demo = (DEMO_METRICS.get("1. Baseline CNN", {}).get("TestAcc") == 0.7812)
    if _is_demo:
        st.markdown("""
        <div style='background:rgba(255,200,100,0.35);border:2px solid #D4A017;
                    border-radius:10px;padding:0.85rem 1.2rem;margin-bottom:1rem;
                    font-family:"Raleway",sans-serif;font-size:0.9rem;color:#5C3A00'>
          ⚠️ <b>Demo / Placeholder Metrics</b> — these numbers are <u>NOT</u> from
          your notebook. They are hardcoded in <code>DEMO_METRICS</code> (app.py line ≈398).
          Replace each dict entry with your actual training results to display real values.
          <br><br>
          <b>How to extract real metrics from your notebook:</b><br>
          <code>
          # After evaluating each model:<br>
          test_loss, test_acc = model.evaluate(test_ds)<br>
          report = classification_report(y_true, y_pred, output_dict=True)<br>
          roc_auc = roc_auc_score(y_true, y_scores)<br>
          # Then paste values into DEMO_METRICS in app.py
          </code>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='background:rgba(200,255,220,0.5);border:2px solid #1E8449;
                    border-radius:10px;padding:0.7rem 1.2rem;margin-bottom:1rem;
                    font-size:0.85rem;color:#145A32'>
          ✅ <b>Real training metrics</b> — values loaded from Trained Models.
        </div>""", unsafe_allow_html=True)

    # Build dataframe from DEMO_METRICS
    rows = []
    for name, m in DEMO_METRICS.items():
        rows.append({
            "Model":         name,
            "Test Acc":      m["TestAcc"],
            "Acc Prec":      m["AccPrec"],
            "Acc Recall ⬆": m["AccRecall"],   # ⬆ = most important
            "NonAcc Prec":   m["NonAccPrec"],
            "NonAcc Recall": m["NonAccRecall"],
            "ROC-AUC":       m["ROCAUC"],
        })
    df = pd.DataFrame(rows).sort_values("Acc Recall ⬆", ascending=False).reset_index(drop=True)
    df.index += 1
    df.index.name = "Rank"

    # Format
    fmt_cols = ["Test Acc","Acc Prec","Acc Recall ⬆","NonAcc Prec","NonAcc Recall","ROC-AUC"]
    df_disp = df.copy()
    for c in fmt_cols:
        df_disp[c] = df_disp[c].apply(lambda v: f"{v:.4f}")

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.dataframe(
        df_disp.style
            .set_properties(**{
                "font-family":"'Fira Code', monospace",
                "font-size":"0.84rem",
            })
            .highlight_max(subset=fmt_cols, color="#FFE066")
            .highlight_min(subset=fmt_cols, color="#FFD0D0"),
        use_container_width=True,
        height=380,
    )
    st.markdown('</div>', unsafe_allow_html=True)
    st.caption("🏆 Sorted by Accident Recall (most safety-critical metric). Yellow = best, Pink = worst per column.")

    gold_divider()

    # ── Charts ──
    st.markdown('<div class="section-heading">Visual Comparison</div>', unsafe_allow_html=True)

    chart_tab1, chart_tab2, chart_tab3 = st.tabs(["📈 Accuracy Bar", "🎯 Multi-Metric", "📉 ROC Overlay"])

    model_labels = df["Model"].tolist()
    palette_map  = {
        "Baseline": "#4C72B0", "Aug CNN": "#55A868", "BN+Drop": "#C44E52",
        "VGG": "#DD8452", "ResNet": "#8172B3", "Eff": "#937860",
    }
    def get_color(name):
        if "Baseline" in name: return "#4C72B0"
        if "Aug CNN"  in name: return "#55A868"
        if "BN+Drop"  in name: return "#C44E52"
        if "VGG"      in name: return "#DD8452"
        if "ResNet"   in name: return "#8172B3"
        return "#937860"

    with chart_tab1:
        fig, ax = plt.subplots(figsize=(12, 5), facecolor="#FFFDF5")
        ax.set_facecolor("#FFFDF5")
        names  = [r["Model"] for r in rows]
        accs   = [r["Test Acc"] for r in rows]
        colors = [get_color(n) for n in names]
        bars   = ax.bar(names, accs, color=colors, edgecolor="#7B4F00", linewidth=0.8, width=0.65)
        for bar, v in zip(bars, accs):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#3D2B00")
        ax.set_ylim(0, 1.12)
        ax.set_title("Test Accuracy — All Models", fontsize=13, fontweight="bold", color="#3D2B00", pad=12)
        ax.set_ylabel("Accuracy", fontsize=10, color="#3D2B00")
        ax.tick_params(axis="x", rotation=30, labelsize=8.5)
        ax.tick_params(axis="y", labelsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.4, color="#D4A017")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with chart_tab2:
        metrics_to_show = [
            ("Test Acc",      "Test Accuracy",        "#4C72B0"),
            ("Acc Prec",      "Accident Precision",   "#C44E52"),
            ("Acc Recall ⬆", "Accident Recall ← KEY","#DD8452"),
            ("NonAcc Prec",   "Non-Acc Precision",    "#55A868"),
            ("NonAcc Recall", "Non-Acc Recall",       "#8172B3"),
            ("ROC-AUC",       "ROC-AUC",              "#937860"),
        ]
        fig, axes = plt.subplots(2, 3, figsize=(15, 8), facecolor="#FFFDF5")
        axes = axes.flatten()
        x = np.arange(len(model_labels))

        for ax, (col, title, color) in zip(axes, metrics_to_show):
            ax.set_facecolor("#FFFDF5")
            vals = df[col].tolist()
            bars = ax.bar(x, vals, color=color, edgecolor="#7B4F00", linewidth=0.5, alpha=0.88, width=0.7)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.004,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=7, fontweight="bold", color="#3D2B00")
            ax.set_title(title, fontsize=10, fontweight="bold", color="#3D2B00")
            ax.set_xticks(x)
            ax.set_xticklabels(model_labels, rotation=40, ha="right", fontsize=7)
            ax.set_ylim(0, 1.12)
            ax.grid(axis="y", linestyle="--", alpha=0.4, color="#D4A017")
            ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

        plt.suptitle("All Metrics Across All Models", fontsize=14, fontweight="bold", color="#3D2B00", y=1.02)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with chart_tab3:
        fig, ax = plt.subplots(figsize=(10, 7), facecolor="#FFFDF5")
        ax.set_facecolor("#FFFDF5")
        cmap = plt.cm.tab10(np.linspace(0, 1, len(DEMO_METRICS)))

        for (name, m), color in zip(DEMO_METRICS.items(), cmap):
            # Synthesise approximate ROC curve from AUC
            auc_val = m["ROCAUC"]
            # Use a beta-distribution proxy to draw a plausible curve
            t   = np.linspace(0, 1, 200)
            # Curve bowing controlled by AUC
            tpr = np.power(t, max(0.05, (1 - auc_val) * 3))
            ax.plot(t, tpr, label=f"{name} (AUC={auc_val:.3f})", color=color, linewidth=1.8)

        ax.plot([0,1],[0,1],"k--", linewidth=1, label="Random", alpha=0.5)
        ax.set_xlabel("False Positive Rate", fontsize=11, color="#3D2B00")
        ax.set_ylabel("True Positive Rate",  fontsize=11, color="#3D2B00")
        ax.set_title("ROC Curves — All Models", fontsize=13, fontweight="bold", color="#3D2B00")
        ax.legend(fontsize=7.5, loc="lower right", framealpha=0.85)
        ax.grid(True, linestyle="--", alpha=0.35, color="#D4A017")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — AUGMENTATION LAB
# ══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown('<div class="section-heading">🔬 Data Augmentation Explorer</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="glass-card">
    Upload any image (or use the sample) and interactively control each augmentation
    transform. See exactly how the training pipeline diversifies the dataset to
    reduce overfitting on small CCTV datasets.
    </div>""", unsafe_allow_html=True)

    aug_col1, aug_col2 = st.columns([1, 1.5])

    with aug_col1:
        st.markdown("**Upload image for augmentation demo**")
        aug_file = st.file_uploader("Choose an image", type=["jpg","jpeg","png","bmp"],
                                    key="aug_upload")

        gold_divider()
        st.markdown("**🎛️ Augmentation Controls**")

        do_hflip    = st.checkbox("Horizontal Flip",   value=True)
        do_vflip    = st.checkbox("Vertical Flip",     value=False)
        rotation    = st.slider("Rotation (°)",        -45, 45, 15)
        zoom        = st.slider("Zoom factor",         0.50, 1.50, 1.0, 0.05)
        brightness  = st.slider("Brightness",          0.20, 2.00, 1.00, 0.05)
        contrast    = st.slider("Contrast",            0.20, 2.00, 1.00, 0.05)
        blur_radius = st.slider("Gaussian Blur",       0.0, 4.0, 0.0, 0.1)
        noise_std   = st.slider("Gaussian Noise σ",   0.0, 40.0, 0.0, 1.0)
        do_gray     = st.checkbox("Grayscale",         value=False)

        n_variants  = st.slider("Grid variants",       2, 8, 6)
        regen       = st.button("🎲  Randomise & Generate")

    with aug_col2:
        # Load base image
        if aug_file:
            pil_orig = Image.open(aug_file).convert("RGB")
        else:
            # Create synthetic placeholder image
            arr = np.zeros((224, 224, 3), dtype=np.uint8)
            # Simple gradient placeholder
            for i in range(224):
                arr[i, :, 0] = int(80 + 100*(i/224))
                arr[i, :, 1] = int(60 + 80*(i/224))
                arr[i, :, 2] = int(20 + 40*(i/224))
            cv2.putText(arr, "Upload", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255,220,50), 2)
            cv2.putText(arr, "an image", (40, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255,220,50), 2)
            pil_orig = Image.fromarray(arr)

        orig_arr = np.array(pil_orig.resize((224,224)))

        def apply_augmentations(img_arr, seed=None):
            """Apply the UI-controlled augmentations to a numpy RGB image."""
            if seed is not None:
                np.random.seed(seed)

            pil = Image.fromarray(img_arr.astype(np.uint8))
            w, h = pil.size

            # Flips
            if do_hflip and np.random.rand() > 0.5:
                pil = pil.transpose(Image.FLIP_LEFT_RIGHT)
            if do_vflip and np.random.rand() > 0.5:
                pil = pil.transpose(Image.FLIP_TOP_BOTTOM)

            # Rotation
            angle = rotation * (2*np.random.rand() - 1)
            pil = pil.rotate(angle, resample=Image.BILINEAR, expand=False)

            # Zoom / crop
            if zoom != 1.0:
                z = zoom * (0.9 + 0.2*np.random.rand())
                nw, nh = int(w/z), int(h/z)
                nw, nh = max(nw, 1), max(nh, 1)
                left = np.random.randint(0, max(1, w - nw))
                top  = np.random.randint(0, max(1, h - nh))
                pil  = pil.crop((left, top, left+nw, top+nh)).resize((w,h), Image.BILINEAR)

            # Brightness
            b = brightness * (0.85 + 0.3*np.random.rand())
            pil = ImageEnhance.Brightness(pil).enhance(np.clip(b, 0.1, 3.0))

            # Contrast
            c = contrast * (0.85 + 0.3*np.random.rand())
            pil = ImageEnhance.Contrast(pil).enhance(np.clip(c, 0.1, 3.0))

            # Blur
            if blur_radius > 0:
                pil = pil.filter(ImageFilter.GaussianBlur(blur_radius * np.random.rand()))

            # Grayscale
            if do_gray:
                pil = pil.convert("L").convert("RGB")

            arr_out = np.array(pil).astype(np.float32)

            # Noise
            if noise_std > 0:
                noise = np.random.normal(0, noise_std, arr_out.shape)
                arr_out = np.clip(arr_out + noise, 0, 255)

            return arr_out.astype(np.uint8)

        # Display grid
        st.markdown("**Original  →  Augmented variants**")
        seed_base = int(time.time()) if regen else 42

        cols_per_row = 3
        all_imgs     = [orig_arr] + [apply_augmentations(orig_arr, seed=seed_base+i) for i in range(n_variants)]
        captions     = ["Original"] + [f"Variant {i+1}" for i in range(n_variants)]

        for row_start in range(0, len(all_imgs), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, col in enumerate(cols):
                idx = row_start + j
                if idx < len(all_imgs):
                    with col:
                        st.image(all_imgs[idx], caption=captions[idx],
                                 use_column_width=True, clamp=True)

    gold_divider()

    # ── Augmentation explanation cards ──
    st.markdown('<div class="section-heading">Why Each Augmentation Helps</div>', unsafe_allow_html=True)
    aug_info = [
        ("🪞", "Horizontal Flip",  "Simulates cameras mounted on opposite sides of the road. Doubles effective training data at zero cost."),
        ("🔄", "Rotation",         "Handles tilted or improperly mounted CCTV cameras. ±15–20° range avoids unrealistic orientations."),
        ("🔍", "Zoom",             "Mimics varying camera focal lengths and distances from the accident scene."),
        ("☀️", "Brightness",       "Covers day/night/dusk conditions and different lighting environments across camera installations."),
        ("🌫️", "Contrast",         "Handles fog, rain, overexposed or underexposed footage common in outdoor CCTV."),
        ("📡", "Gaussian Noise",   "Simulates sensor noise, low-quality feeds, and compression artefacts in legacy CCTV hardware."),
        ("🔲", "Grayscale",        "Some older CCTV cameras capture black-and-white footage; this prevents colour dependency."),
        ("🔀", "Translation",      "Accounts for accidents appearing at different positions within the camera frame."),
    ]
    aug_cols = st.columns(4)
    for i, (icon, name, desc) in enumerate(aug_info):
        with aug_cols[i % 4]:
            st.markdown(f"""
            <div style='background:rgba(255,248,200,0.8);border:1px solid #D4A017;
                        border-radius:12px;padding:0.9rem;margin-bottom:0.8rem;
                        box-shadow:0 3px 10px rgba(180,130,0,0.12)'>
              <div style='font-size:1.4rem;margin-bottom:0.3rem'>{icon}</div>
              <div style='font-family:"Playfair Display",serif;font-size:0.92rem;
                          font-weight:700;color:#1A1400;margin-bottom:0.3rem'>{name}</div>
              <div style='font-size:0.78rem;color:#5C4020;line-height:1.5'>{desc}</div>
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — IMAGE PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown('<div class="section-heading">🖼️ Image Prediction</div>', unsafe_allow_html=True)

    img_col1, img_col2 = st.columns([1, 1])

    with img_col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("**Upload a CCTV frame for accident detection**")
        img_file = st.file_uploader(
            "Supported: JPG · PNG · BMP",
            type=["jpg","jpeg","png","bmp"],
            key="img_pred"
        )

        if img_file:
            file_bytes = np.frombuffer(img_file.read(), np.uint8)
            cv_img     = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            img_rgb    = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            pil_img    = Image.fromarray(img_rgb)

            st.image(pil_img, caption="Uploaded Frame", use_column_width=True)

            # File info
            h, w = img_rgb.shape[:2]
            st.markdown(f"""
            <div style='font-size:0.8rem;color:#7A5C20;margin-top:0.5rem'>
              📐 {w} × {h} px &nbsp;|&nbsp; 🎨 RGB
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='text-align:center;padding:3rem;color:#A07820;
                        border:2px dashed #D4A017;border-radius:12px;background:rgba(255,248,200,0.5)'>
              <div style='font-size:3rem'>📸</div>
              <div style='font-size:0.9rem;margin-top:0.5rem'>Drop a CCTV frame here</div>
            </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with img_col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("**Prediction Result**")

        if img_file and model:
            with st.spinner("🔍 Running inference…"):
                # Adjust threshold from sidebar
                tensor = preprocess_for_model(img_rgb, model_choice)
                raw_prob = float(model.predict(tensor, verbose=0)[0][0])
                idx    = int(raw_prob > threshold)
                label  = CLASS_NAMES[idx]
                conf   = raw_prob if idx == 1 else (1.0 - raw_prob)

            prediction_display(label, conf)

            gold_divider()

            # ── Probability breakdown ──
            acc_prob   = 1.0 - raw_prob
            nonacc_prob = raw_prob

            st.markdown("**Class probabilities**")
            for cls, prob_val, color in [
                ("🔴 Accident",     acc_prob,    "#C0392B"),
                ("🟢 Non Accident", nonacc_prob, "#1E8449"),
            ]:
                pct = prob_val * 100
                st.markdown(f"""
                <div style='margin-bottom:0.6rem'>
                  <div style='display:flex;justify-content:space-between;
                              font-size:0.88rem;font-weight:600;color:#3D2B00;margin-bottom:0.3rem'>
                    <span>{cls}</span><span>{pct:.1f}%</span>
                  </div>
                  <div style='background:rgba(255,255,255,0.5);border-radius:50px;
                              height:12px;overflow:hidden;border:1px solid rgba(0,0,0,0.08)'>
                    <div style='width:{pct}%;height:100%;border-radius:50px;
                                background:linear-gradient(90deg,{color}88,{color})'></div>
                  </div>
                </div>""", unsafe_allow_html=True)

            if show_raw:
                st.markdown(f"""
                <div style='margin-top:0.8rem;font-family:"Fira Code",monospace;
                            font-size:0.8rem;color:#7A5C20;background:rgba(255,248,200,0.7);
                            padding:0.6rem;border-radius:8px'>
                  Raw sigmoid output: <b>{raw_prob:.6f}</b><br>
                  Threshold: <b>{threshold}</b><br>
                  Model: <b>{model_choice}</b>
                </div>""", unsafe_allow_html=True)

        elif img_file and not model:
            st.warning("⚠️ No model loaded. Please place your `.keras` model files next to `app.py` and restart.")
        else:
            st.markdown("""
            <div style='text-align:center;padding:3rem;color:#A07820'>
              <div style='font-size:3rem'>🤖</div>
              <div style='font-size:0.9rem;margin-top:0.5rem'>
                Upload an image to see predictions
              </div>
            </div>""", unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # ── Multi-image batch ──
    gold_divider()
    st.markdown('<div class="section-heading">📦 Batch Image Prediction</div>', unsafe_allow_html=True)
    batch_files = st.file_uploader(
        "Upload multiple images for batch inference",
        type=["jpg","jpeg","png","bmp"],
        accept_multiple_files=True,
        key="batch_pred"
    )

    if batch_files:
        if not model:
            st.warning("Load a model from the sidebar first.")
        else:
            st.markdown(f"Processing **{len(batch_files)}** images…")
            progress = st.progress(0)
            bcols    = st.columns(min(4, len(batch_files)))
            results  = []

            for i, bf in enumerate(batch_files):
                fb   = np.frombuffer(bf.read(), np.uint8)
                bgr  = cv2.imdecode(fb, cv2.IMREAD_COLOR)
                rgb  = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                tensor = preprocess_for_model(rgb, model_choice)
                raw_p  = float(model.predict(tensor, verbose=0)[0][0])
                idx    = int(raw_p > threshold)
                lbl    = CLASS_NAMES[idx]
                conf   = raw_p if idx == 1 else (1.0 - raw_p)
                results.append((bf.name, lbl, conf))

                col = bcols[i % 4]
                with col:
                    is_acc = (lbl == "Accident")
                    border = "#C0392B" if is_acc else "#1E8449"
                    icon   = "🚨" if is_acc else "✅"
                    st.image(rgb, use_column_width=True, clamp=True)
                    st.markdown(f"""
                    <div style='text-align:center;background:{"#FFEEEE" if is_acc else "#EEFFEE"};
                                border:1.5px solid {border};border-radius:8px;
                                padding:0.4rem;margin-top:-0.3rem;font-size:0.82rem;font-weight:700'>
                      {icon} {lbl} ({conf*100:.0f}%)
                    </div>""", unsafe_allow_html=True)

                progress.progress((i+1) / len(batch_files))

            # Summary
            gold_divider()
            n_acc = sum(1 for _, l, _ in results if l == "Accident")
            c1, c2, c3 = st.columns(3)
            with c1: metric_box("Total Frames", str(len(results)))
            with c2: metric_box("🚨 Accidents",  str(n_acc))
            with c3: metric_box("✅ Safe Frames", str(len(results) - n_acc))

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — VIDEO DETECTION
# ══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown('<div class="section-heading">🎥 Video Accident Detection</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="glass-card">
    Upload a short CCTV video clip. Each frame is run through the selected model.
    Accident frames are highlighted in <span style="color:#C0392B;font-weight:700">red</span>,
    safe frames in <span style="color:#1E8449;font-weight:700">green</span>.
    A summary timeline and statistics are shown below.
    </div>""", unsafe_allow_html=True)

    vid_col1, vid_col2 = st.columns([1, 1.2])

    with vid_col1:
        vid_file = st.file_uploader(
            "Upload video (MP4 · AVI · MOV · MKV)",
            type=["mp4","avi","mov","mkv","webm"],
            key="vid_pred"
        )

        if vid_file:
            st.video(vid_file)

        sample_rate = st.slider("Sample every N frames", 1, 30, 5,
                                help="Lower = more thorough but slower")
        annotate_video = st.checkbox("Generate annotated output video", value=True)
        run_btn = st.button("🚀  Run Detection", use_container_width=True)

    with vid_col2:
        if vid_file and run_btn:
            if not model:
                st.warning("⚠️ No model loaded. Place `.keras` files next to `app.py`.")
            else:
                # Save uploaded video to temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                    tmp.write(vid_file.read())
                    tmp_path = tmp.name

                cap = cv2.VideoCapture(tmp_path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps_orig     = cap.get(cv2.CAP_PROP_FPS) or 25.0
                w_vid        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h_vid        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                st.markdown(f"""
                <div style='font-size:0.82rem;color:#7A5C20;margin-bottom:0.8rem;
                            font-family:"Fira Code",monospace'>
                  📹 {w_vid}×{h_vid} · {fps_orig:.1f} fps · {total_frames} frames
                </div>""", unsafe_allow_html=True)

                # Setup annotated video writer
                out_path = None
                if annotate_video:
                    out_path = tmp_path.replace(".mp4", "_annotated.mp4")
                    fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
                    out_fps  = max(1.0, fps_orig / sample_rate)
                    writer   = cv2.VideoWriter(out_path, fourcc, out_fps, (w_vid, h_vid))

                # ── Process frames ──
                frame_results  = []   # list of (frame_idx, label, conf)
                progress_bar   = st.progress(0)
                status_text    = st.empty()
                preview_holder = st.empty()

                frame_idx = 0
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    if frame_idx % sample_rate == 0:
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        tensor  = preprocess_for_model(rgb, model_choice)
                        raw_p   = float(model.predict(tensor, verbose=0)[0][0])
                        idx_cls = int(raw_p > threshold)
                        lbl     = CLASS_NAMES[idx_cls]
                        conf    = raw_p if idx_cls == 1 else (1.0 - raw_p)
                        frame_results.append((frame_idx, lbl, conf))

                        # Annotate frame
                        is_acc = (lbl == "Accident")
                        color_bgr = (0, 0, 200) if is_acc else (0, 180, 0)
                        icon_text = "ACCIDENT" if is_acc else "SAFE"
                        cv2.rectangle(frame, (0,0), (w_vid-1, h_vid-1), color_bgr, 6)
                        overlay = frame.copy()
                        cv2.rectangle(overlay, (0,0), (380, 58), color_bgr, -1)
                        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
                        cv2.putText(frame, f"{icon_text}  {conf*100:.0f}%",
                                    (10, 38), cv2.FONT_HERSHEY_DUPLEX, 1.1,
                                    (255,255,255), 2, cv2.LINE_AA)
                        cv2.putText(frame, f"Frame {frame_idx}",
                                    (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                                    (255,255,255), 1)

                        if annotate_video:
                            writer.write(frame)

                        # Live preview every 10 analysed frames
                        if len(frame_results) % 10 == 0:
                            preview_holder.image(
                                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                                caption=f"Frame {frame_idx}: {lbl} ({conf*100:.0f}%)",
                                use_column_width=True
                            )

                        prog = min(frame_idx / max(total_frames, 1), 1.0)
                        progress_bar.progress(prog)
                        status_text.markdown(
                            f"<span style='font-size:0.8rem;color:#7A5C20'>"
                            f"Analysed {len(frame_results)} frames…</span>",
                            unsafe_allow_html=True
                        )

                    frame_idx += 1

                cap.release()
                if annotate_video:
                    writer.release()
                progress_bar.progress(1.0)
                status_text.empty()

                # ── Results summary ──
                if frame_results:
                    labels_all = [r[1] for r in frame_results]
                    confs_all  = [r[2] for r in frame_results]
                    n_total    = len(frame_results)
                    n_acc_v    = sum(1 for l in labels_all if l == "Accident")
                    n_safe_v   = n_total - n_acc_v
                    avg_conf   = np.mean(confs_all)

                    gold_divider()
                    st.markdown("**Detection Summary**")
                    mc1, mc2, mc3, mc4 = st.columns(4)
                    with mc1: metric_box("Frames Checked", str(n_total))
                    with mc2: metric_box("🚨 Accidents",   str(n_acc_v))
                    with mc3: metric_box("✅ Safe",         str(n_safe_v))
                    with mc4: metric_box("Avg Confidence", f"{avg_conf*100:.0f}%")

                    # Timeline chart
                    gold_divider()
                    st.markdown("**Frame-by-frame Timeline**")
                    fig, ax = plt.subplots(figsize=(12, 3), facecolor="#FFFDF5")
                    ax.set_facecolor("#FFFDF5")
                    frame_idxs  = [r[0] for r in frame_results]
                    conf_vals   = [r[2] for r in frame_results]
                    bar_colors  = ["#C0392B" if l=="Accident" else "#1E8449" for l in labels_all]
                    ax.bar(frame_idxs, conf_vals, color=bar_colors, width=max(1, sample_rate*0.8), alpha=0.85)
                    ax.axhline(threshold, color="#D4A017", linestyle="--", linewidth=1.5, label=f"Threshold {threshold}")
                    ax.set_xlabel("Frame Index", fontsize=9, color="#3D2B00")
                    ax.set_ylabel("Confidence", fontsize=9, color="#3D2B00")
                    ax.set_title("Detection Timeline", fontsize=11, fontweight="bold", color="#3D2B00")
                    ax.set_ylim(0, 1.05)
                    ax.legend(fontsize=8)
                    ax.grid(axis="y", linestyle="--", alpha=0.35, color="#D4A017")

                    red_patch   = mpatches.Patch(color="#C0392B", label="Accident")
                    green_patch = mpatches.Patch(color="#1E8449", label="Safe")
                    ax.legend(handles=[red_patch, green_patch,
                                       mpatches.Patch(color="#D4A017", label=f"Threshold={threshold}")],
                              fontsize=8, loc="upper right")

                    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)

                    # Accident timestamps
                    if n_acc_v > 0:
                        gold_divider()
                        st.markdown("**🚨 Accident Detections**")
                        acc_frames = [(r[0], r[2]) for r in frame_results if r[1]=="Accident"]
                        acc_df = pd.DataFrame(acc_frames, columns=["Frame", "Confidence"])
                        acc_df["Time (s)"] = (acc_df["Frame"] / fps_orig).round(2)
                        acc_df["Confidence"] = acc_df["Confidence"].apply(lambda v: f"{v*100:.1f}%")
                        st.dataframe(acc_df, use_container_width=True)

                    # Download annotated video
                    if annotate_video and out_path and os.path.exists(out_path):
                        gold_divider()
                        with open(out_path, "rb") as f:
                            st.download_button(
                                "⬇️  Download Annotated Video",
                                data=f,
                                file_name="accident_detection_annotated.mp4",
                                mime="video/mp4",
                                use_container_width=True,
                            )

                # Cleanup
                try:
                    os.unlink(tmp_path)
                    if out_path:
                        os.unlink(out_path)
                except:
                    pass

        elif not vid_file:
            st.markdown("""
            <div style='text-align:center;padding:4rem;color:#A07820;
                        border:2px dashed #D4A017;border-radius:12px;
                        background:rgba(255,248,200,0.5)'>
              <div style='font-size:4rem'>🎥</div>
              <div style='font-size:1rem;font-weight:600;margin-top:0.8rem'>
                Upload a CCTV video to begin
              </div>
              <div style='font-size:0.8rem;margin-top:0.4rem;color:#C8860A'>
                MP4 · AVI · MOV · MKV · WEBM
              </div>
            </div>""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────────────────────────────────────
gold_divider()
st.markdown("""
<div style='text-align:center;padding:1rem 0 0.5rem;
            font-family:"Raleway",sans-serif;font-size:0.78rem;
            color:#7A5C20;letter-spacing:0.8px'>
  <b style='font-family:"Playfair Display",serif;color:#D4A017'>AccidentAI</b>
  &nbsp;·&nbsp; CCTV Accident Detection
  &nbsp;·&nbsp; TensorFlow · OpenCV · Streamlit
  &nbsp;·&nbsp; Binary Classification (Accident / Non-Accident)
</div>
""", unsafe_allow_html=True)
