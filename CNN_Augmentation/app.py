import streamlit as st
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageOps, ImageDraw
import io
import random
import math

# ─── PAGE CONFIG ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AugmentLab",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── CUSTOM CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400&family=Syne:wght@400;700;800&family=DM+Sans:wght@300;400;500&display=swap');

/* ── ROOT VARIABLES ── */
:root {
  --bg-primary:    #080c10;
  --bg-secondary:  #0d1117;
  --bg-card:       #111820;
  --bg-card2:      #141c24;
  --accent-lime:   #b8ff57;
  --accent-cyan:   #00e5ff;
  --accent-pink:   #ff3d9a;
  --accent-amber:  #ffb830;
  --border:        rgba(255,255,255,0.07);
  --text-primary:  #e8edf2;
  --text-muted:    #5a6a7a;
  --text-dim:      #3a4a5a;
  --font-head:     'Syne', sans-serif;
  --font-mono:     'Space Mono', monospace;
  --font-body:     'DM Sans', sans-serif;
}

/* ── GLOBAL RESET ── */
html, body, [class*="css"] {
  font-family: var(--font-body);
  background-color: var(--bg-primary);
  color: var(--text-primary);
}

.stApp {
  background: var(--bg-primary);
  background-image:
    radial-gradient(ellipse 80% 50% at 20% -10%, rgba(0,229,255,0.06) 0%, transparent 60%),
    radial-gradient(ellipse 60% 40% at 80% 110%, rgba(184,255,87,0.05) 0%, transparent 55%);
}

/* ── HIDE STREAMLIT CHROME ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 0 !important; max-width: 100% !important; }

/* ── TOP HEADER BAR ── */
.lab-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 18px 40px 16px;
  border-bottom: 1px solid var(--border);
  background: rgba(8,12,16,0.85);
  backdrop-filter: blur(12px);
  position: sticky;
  top: 0;
  z-index: 100;
  margin-bottom: 0;
}
.lab-logo {
  display: flex;
  align-items: center;
  gap: 12px;
}
.lab-logo-hex {
  font-size: 26px;
  color: var(--accent-lime);
  line-height: 1;
}
.lab-logo-text {
  font-family: var(--font-head);
  font-size: 22px;
  font-weight: 800;
  letter-spacing: -0.5px;
  color: var(--text-primary);
}
.lab-logo-text span { color: var(--accent-lime); }
.lab-tagline {
  font-family: var(--font-mono);
  font-size: 11px;
  color: var(--text-muted);
  letter-spacing: 2px;
  text-transform: uppercase;
}
.lab-status {
  display: flex;
  align-items: center;
  gap: 8px;
  font-family: var(--font-mono);
  font-size: 11px;
  color: var(--text-muted);
}
.status-dot {
  width: 7px; height: 7px;
  border-radius: 50%;
  background: var(--accent-lime);
  box-shadow: 0 0 8px var(--accent-lime);
  animation: pulse 2s infinite;
}
@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.4; }
}

/* ── UPLOAD ZONE ── */
.upload-section {
  padding: 36px 40px 24px;
}
.section-label {
  font-family: var(--font-mono);
  font-size: 10px;
  letter-spacing: 3px;
  text-transform: uppercase;
  color: var(--accent-cyan);
  margin-bottom: 14px;
}

[data-testid="stFileUploader"] {
  background: var(--bg-card) !important;
  border: 1px dashed rgba(0,229,255,0.25) !important;
  border-radius: 12px !important;
  padding: 8px !important;
  transition: border-color 0.3s ease;
}
[data-testid="stFileUploader"]:hover {
  border-color: rgba(0,229,255,0.5) !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] {
  color: var(--text-muted) !important;
  font-family: var(--font-body) !important;
}

/* ── IMAGE PREVIEW PANEL ── */
.preview-label {
  font-family: var(--font-mono);
  font-size: 10px;
  letter-spacing: 3px;
  text-transform: uppercase;
  color: var(--text-muted);
  margin-bottom: 10px;
}
.img-panel {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 12px;
  overflow: hidden;
  position: relative;
}
.img-panel-tag {
  position: absolute;
  top: 12px; left: 12px;
  background: rgba(0,0,0,0.6);
  backdrop-filter: blur(8px);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 3px 10px;
  font-family: var(--font-mono);
  font-size: 10px;
  color: var(--text-muted);
  z-index: 10;
}

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"] {
  background: transparent !important;
  border-bottom: 1px solid var(--border) !important;
  gap: 0 !important;
  padding: 0 40px !important;
}
.stTabs [data-baseweb="tab"] {
  background: transparent !important;
  border: none !important;
  border-bottom: 2px solid transparent !important;
  color: var(--text-muted) !important;
  font-family: var(--font-mono) !important;
  font-size: 11px !important;
  letter-spacing: 1.5px !important;
  text-transform: uppercase !important;
  padding: 14px 20px !important;
  border-radius: 0 !important;
  transition: all 0.25s ease !important;
}
.stTabs [data-baseweb="tab"]:hover {
  color: var(--text-primary) !important;
}
.stTabs [aria-selected="true"] {
  color: var(--accent-lime) !important;
  border-bottom-color: var(--accent-lime) !important;
  background: transparent !important;
}
.stTabs [data-baseweb="tab-panel"] {
  background: transparent !important;
  padding: 0 !important;
}

/* ── AUGMENTATION GRID ── */
.aug-grid-header {
  padding: 32px 40px 16px;
}
.aug-grid-title {
  font-family: var(--font-head);
  font-size: 28px;
  font-weight: 800;
  color: var(--text-primary);
  margin-bottom: 4px;
}
.aug-grid-title span { color: var(--accent-lime); }
.aug-grid-sub {
  font-size: 13px;
  color: var(--text-muted);
}

.aug-content {
  padding: 16px 40px 40px;
}

/* ── BUTTONS ── */
.stButton > button {
  font-family: var(--font-mono) !important;
  font-size: 11px !important;
  letter-spacing: 1.5px !important;
  text-transform: uppercase !important;
  background: var(--bg-card2) !important;
  color: var(--text-primary) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
  padding: 10px 18px !important;
  width: 100% !important;
  transition: all 0.25s ease !important;
  position: relative !important;
  overflow: hidden !important;
}
.stButton > button:hover {
  background: rgba(184,255,87,0.08) !important;
  border-color: rgba(184,255,87,0.4) !important;
  color: var(--accent-lime) !important;
  transform: translateY(-1px) !important;
  box-shadow: 0 4px 20px rgba(184,255,87,0.1) !important;
}
.stButton > button:active {
  transform: translateY(0px) !important;
}

/* ── SLIDERS ── */
.stSlider [data-baseweb="slider"] {
  padding: 0 !important;
}
.stSlider [data-baseweb="thumb"] {
  background: var(--accent-lime) !important;
  border: none !important;
  width: 14px !important; height: 14px !important;
}
.stSlider [data-baseweb="track-fill"] {
  background: var(--accent-lime) !important;
}
.stSlider label {
  font-family: var(--font-mono) !important;
  font-size: 11px !important;
  letter-spacing: 1px !important;
  color: var(--text-muted) !important;
  text-transform: uppercase !important;
}

/* ── METRIC CARDS ── */
.metric-row {
  display: flex;
  gap: 12px;
  margin-bottom: 28px;
}
.metric-card {
  flex: 1;
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 16px 20px;
}
.metric-val {
  font-family: var(--font-head);
  font-size: 22px;
  font-weight: 700;
  color: var(--text-primary);
}
.metric-val.lime { color: var(--accent-lime); }
.metric-val.cyan { color: var(--accent-cyan); }
.metric-val.pink { color: var(--accent-pink); }
.metric-lbl {
  font-family: var(--font-mono);
  font-size: 10px;
  letter-spacing: 2px;
  color: var(--text-dim);
  text-transform: uppercase;
  margin-top: 2px;
}

/* ── RESULT CARD ── */
.result-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 12px;
  overflow: hidden;
  margin-bottom: 20px;
}
.result-card-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 14px 18px;
  border-bottom: 1px solid var(--border);
  background: var(--bg-card2);
}
.result-card-name {
  font-family: var(--font-head);
  font-size: 14px;
  font-weight: 700;
  color: var(--text-primary);
}
.result-card-badge {
  font-family: var(--font-mono);
  font-size: 10px;
  letter-spacing: 1px;
  color: var(--accent-cyan);
  background: rgba(0,229,255,0.08);
  border: 1px solid rgba(0,229,255,0.2);
  border-radius: 20px;
  padding: 3px 10px;
}
.result-card-body { padding: 18px; }
.result-card-desc {
  font-size: 12px;
  color: var(--text-muted);
  margin-bottom: 14px;
  line-height: 1.6;
}

/* ── SELECTBOX / DROPDOWN ── */
.stSelectbox [data-baseweb="select"] > div {
  background: var(--bg-card) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
  color: var(--text-primary) !important;
  font-family: var(--font-mono) !important;
  font-size: 12px !important;
}
.stSelectbox label {
  font-family: var(--font-mono) !important;
  font-size: 10px !important;
  letter-spacing: 2px !important;
  color: var(--text-muted) !important;
  text-transform: uppercase !important;
}

/* ── CHECKBOX ── */
.stCheckbox label {
  font-family: var(--font-body) !important;
  font-size: 13px !important;
  color: var(--text-muted) !important;
}

/* ── NUMBER INPUT ── */
.stNumberInput label {
  font-family: var(--font-mono) !important;
  font-size: 10px !important;
  letter-spacing: 2px !important;
  color: var(--text-muted) !important;
  text-transform: uppercase !important;
}
.stNumberInput input {
  background: var(--bg-card) !important;
  border: 1px solid var(--border) !important;
  color: var(--text-primary) !important;
  font-family: var(--font-mono) !important;
  border-radius: 8px !important;
}

/* ── DIVIDERS ── */
hr { border-color: var(--border) !important; }

/* ── COLOR PICKER ── */
.stColorPicker label {
  font-family: var(--font-mono) !important;
  font-size: 10px !important;
  letter-spacing: 2px !important;
  color: var(--text-muted) !important;
  text-transform: uppercase !important;
}

/* ── COMPARISON LABEL ── */
.compare-wrap {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 16px;
}

/* ── EMPTY STATE ── */
.empty-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 80px 40px;
  text-align: center;
}
.empty-hex {
  font-size: 56px;
  margin-bottom: 20px;
  opacity: 0.3;
}
.empty-title {
  font-family: var(--font-head);
  font-size: 20px;
  font-weight: 700;
  color: var(--text-muted);
  margin-bottom: 8px;
}
.empty-sub {
  font-size: 13px;
  color: var(--text-dim);
}

/* ── TOOLTIP-STYLE INFO ── */
.info-pill {
  display: inline-block;
  background: rgba(255,184,48,0.08);
  border: 1px solid rgba(255,184,48,0.2);
  border-radius: 20px;
  padding: 4px 12px;
  font-family: var(--font-mono);
  font-size: 10px;
  letter-spacing: 1px;
  color: var(--accent-amber);
  margin-bottom: 16px;
}

/* ── CATEGORY BADGE ── */
.cat-badge {
  display: inline-block;
  border-radius: 4px;
  padding: 2px 8px;
  font-family: var(--font-mono);
  font-size: 9px;
  letter-spacing: 1.5px;
  text-transform: uppercase;
  margin-bottom: 12px;
}
.cat-geometric { background: rgba(184,255,87,0.1); color: var(--accent-lime); }
.cat-photometric { background: rgba(0,229,255,0.1); color: var(--accent-cyan); }
.cat-noise { background: rgba(255,61,154,0.1); color: var(--accent-pink); }
.cat-filter { background: rgba(255,184,48,0.1); color: var(--accent-amber); }
.cat-artistic { background: rgba(160,100,255,0.1); color: #a064ff; }
.cat-advanced { background: rgba(255,80,80,0.1); color: #ff5050; }

/* ── SCROLLBAR ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--text-dim); }

/* stImage */
[data-testid="stImage"] img {
  border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)


# ─── HEADER ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="lab-header">
  <div class="lab-logo">
    <div class="lab-logo-hex">⬡</div>
    <div>
      <div class="lab-logo-text">Augment<span>Lab</span></div>
      <div class="lab-tagline">Image Augmentation Studio</div>
    </div>
  </div>
  <div class="lab-status">
    <div class="status-dot"></div>
    SYSTEM ACTIVE &nbsp;·&nbsp; v2.0
  </div>
</div>
""", unsafe_allow_html=True)


# ─── AUGMENTATION FUNCTIONS ─────────────────────────────────────────────────
def pil_to_bytes(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def apply_horizontal_flip(img): return img.transpose(Image.FLIP_LEFT_RIGHT)
def apply_vertical_flip(img): return img.transpose(Image.FLIP_TOP_BOTTOM)

def apply_rotation(img, angle):
    return img.rotate(angle, expand=True, fillcolor=(20, 28, 36))

def apply_crop(img, pct):
    w, h = img.size
    m = int(min(w, h) * pct / 200)
    return img.crop((m, m, w - m, h - m)).resize((w, h), Image.LANCZOS)

def apply_zoom(img, factor):
    w, h = img.size
    nw, nh = int(w / factor), int(h / factor)
    left, top = (w - nw) // 2, (h - nh) // 2
    return img.crop((left, top, left + nw, top + nh)).resize((w, h), Image.LANCZOS)

def apply_shear(img, shear_factor):
    w, h = img.size
    xshift = abs(shear_factor) * w
    new_w = w + int(round(xshift))
    img_arr = np.array(img.convert("RGBA"))
    result = Image.new("RGBA", (new_w, h), (20, 28, 36, 255))
    for y in range(h):
        shift = int(shear_factor * y)
        for x in range(w):
            nx = x + shift
            if 0 <= nx < new_w:
                result.putpixel((nx, y), tuple(img_arr[y, x]))
    try:
        return result.resize((w, h), Image.Resampling.LANCZOS).convert(img.mode)
    except AttributeError:
        return result.resize((w, h), Image.LANCZOS).convert(img.mode)

def apply_brightness(img, factor): return ImageEnhance.Brightness(img).enhance(factor)
def apply_contrast(img, factor): return ImageEnhance.Contrast(img).enhance(factor)
def apply_saturation(img, factor): return ImageEnhance.Color(img).enhance(factor)
def apply_sharpness(img, factor): return ImageEnhance.Sharpness(img).enhance(factor)

def apply_gamma(img, gamma):
    arr = np.array(img).astype(np.float32) / 255.0
    arr = np.clip(arr ** (1.0 / gamma), 0, 1)
    return Image.fromarray((arr * 255).astype(np.uint8))

def apply_grayscale(img): return ImageOps.grayscale(img).convert("RGB")
def apply_sepia(img):
    arr = np.array(img.convert("RGB"), dtype=np.float32)
    r = np.clip(arr[:,:,0]*0.393 + arr[:,:,1]*0.769 + arr[:,:,2]*0.189, 0, 255)
    g = np.clip(arr[:,:,0]*0.349 + arr[:,:,1]*0.686 + arr[:,:,2]*0.168, 0, 255)
    b = np.clip(arr[:,:,0]*0.272 + arr[:,:,1]*0.534 + arr[:,:,2]*0.131, 0, 255)
    return Image.fromarray(np.stack([r, g, b], axis=2).astype(np.uint8))

def apply_invert(img): return ImageOps.invert(img.convert("RGB"))
def apply_equalize(img): return ImageOps.equalize(img.convert("RGB"))

def apply_hue_shift(img, shift):
    import colorsys
    arr = np.array(img.convert("RGB")).astype(np.float32) / 255.0
    out = np.zeros_like(arr)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            r, g, b = arr[i, j]
            h, s, v = colorsys.rgb_to_hsv(r, g, b)
            h = (h + shift / 360.0) % 1.0
            rgb = colorsys.hsv_to_rgb(h, s, v)
            out[i, j] = rgb
    result = Image.fromarray((out * 255).astype(np.uint8))
    return result.convert(img.mode) if img.mode != 'RGB' else result

def apply_gaussian_noise(img, std):
    arr = np.array(img).astype(np.float32)
    noise = np.random.normal(0, std, arr.shape)
    return Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))

def apply_salt_pepper(img, amount):
    arr = np.array(img).copy()
    n = int(amount * arr.size // arr.shape[2] if len(arr.shape) == 3 else amount * arr.size)
    coords = [np.random.randint(0, i, n) for i in arr.shape[:2]]
    arr[coords[0], coords[1]] = 255
    coords = [np.random.randint(0, i, n) for i in arr.shape[:2]]
    arr[coords[0], coords[1]] = 0
    return Image.fromarray(arr)

def apply_speckle(img, intensity):
    arr = np.array(img).astype(np.float32)
    noise = np.random.randn(*arr.shape) * intensity
    return Image.fromarray(np.clip(arr + arr * noise / 100, 0, 255).astype(np.uint8))

def apply_blur(img, radius): return img.filter(ImageFilter.GaussianBlur(radius))
def apply_motion_blur(img, size):
    kernel = np.zeros((size, size))
    kernel[size // 2, :] = 1.0 / size
    from PIL import ImageFilter as IF
    return img.filter(ImageFilter.Kernel(size=(size, size), kernel=kernel.flatten().tolist(),
                                         scale=1, offset=0)) if size in [3, 5, 7] else img.filter(IF.GaussianBlur(size // 3))

def apply_sharpen(img): return img.filter(ImageFilter.SHARPEN)
def apply_edge_enhance(img): return img.filter(ImageFilter.EDGE_ENHANCE_MORE)
def apply_emboss(img): return img.filter(ImageFilter.EMBOSS)
def apply_median_filter(img, size): return img.filter(ImageFilter.MedianFilter(size=size))

def apply_pixelate(img, block):
    w, h = img.size
    small = img.resize((max(1, w // block), max(1, h // block)), Image.NEAREST)
    return small.resize((w, h), Image.NEAREST)

def apply_vignette(img, strength):
    arr = np.array(img.convert("RGB")).astype(np.float32)
    h, w = arr.shape[:2]
    Y, X = np.ogrid[:h, :w]
    cx, cy = w / 2, h / 2
    dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
    max_d = np.sqrt(cx**2 + cy**2)
    mask = 1 - strength * (dist / max_d) ** 2
    mask = np.clip(mask, 0, 1)
    arr *= mask[:, :, np.newaxis]
    return Image.fromarray(arr.astype(np.uint8))

def apply_posterize(img, bits): return ImageOps.posterize(img.convert("RGB"), bits)
def apply_solarize(img, thresh): return ImageOps.solarize(img.convert("RGB"), thresh)

def apply_color_jitter(img, b, c, s):
    img = ImageEnhance.Brightness(img).enhance(b)
    img = ImageEnhance.Contrast(img).enhance(c)
    img = ImageEnhance.Color(img).enhance(s)
    return img

def apply_cutout(img, n, size):
    arr = np.array(img).copy()
    h, w = arr.shape[:2]
    for _ in range(n):
        x = random.randint(0, w - size)
        y = random.randint(0, h - size)
        arr[y:y+size, x:x+size] = 0
    return Image.fromarray(arr)

def apply_grid_distortion(img, steps, magnitude):
    w, h = img.size
    arr = np.array(img.convert("RGB")).astype(np.uint8)
    out = np.zeros_like(arr)
    sw, sh = w // steps, h // steps
    for i in range(h):
        for j in range(w):
            ox = int(magnitude * math.sin(2 * math.pi * j / (sw + 1)))
            oy = int(magnitude * math.cos(2 * math.pi * i / (sh + 1)))
            src_x = min(max(j + ox, 0), w - 1)
            src_y = min(max(i + oy, 0), h - 1)
            out[i, j] = arr[src_y, src_x]
    return Image.fromarray(out).convert(img.mode)

def apply_channel_shuffle(img):
    arr = np.array(img.convert("RGB"))
    channels = [arr[:,:,i] for i in range(3)]
    random.shuffle(channels)
    return Image.fromarray(np.stack(channels, axis=2))

def apply_clahe(img):
    try:
        import cv2
        gray = np.array(img.convert("L"))
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        result = clahe.apply(gray)
        return Image.fromarray(result).convert("RGB")
    except ImportError:
        return apply_equalize(img)

def apply_color_overlay(img, color_hex, alpha):
    overlay = Image.new("RGBA", img.size, color_hex + hex(int(alpha * 255))[2:].zfill(2))
    base = img.convert("RGBA")
    return Image.alpha_composite(base, overlay).convert("RGB")


# ─── UPLOAD ──────────────────────────────────────────────────────────────────
st.markdown('<div class="upload-section">', unsafe_allow_html=True)
st.markdown('<div class="section-label">↑ Input Source</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "Drop your image here, or click to browse",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
    label_visibility="collapsed"
)
st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file is None:
    st.markdown("""
    <div class="empty-state">
      <div class="empty-hex">⬡</div>
      <div class="empty-title">No Image Loaded</div>
      <div class="empty-sub">Upload an image above to start exploring augmentations</div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# Load image
raw_img = Image.open(uploaded_file).convert("RGB")
w, h = raw_img.size

# ─── METRIC STRIP ────────────────────────────────────────────────────────────
col_m1, col_m2, col_m3, col_m4 = st.columns(4)
with col_m1:
    st.markdown(f"""<div class="metric-card">
      <div class="metric-val lime">{w}×{h}</div>
      <div class="metric-lbl">Resolution</div>
    </div>""", unsafe_allow_html=True)
with col_m2:
    st.markdown(f"""<div class="metric-card">
      <div class="metric-val cyan">{uploaded_file.size // 1024} KB</div>
      <div class="metric-lbl">File Size</div>
    </div>""", unsafe_allow_html=True)
with col_m3:
    st.markdown(f"""<div class="metric-card">
      <div class="metric-val pink">{uploaded_file.type.split("/")[1].upper()}</div>
      <div class="metric-lbl">Format</div>
    </div>""", unsafe_allow_html=True)
with col_m4:
    st.markdown(f"""<div class="metric-card">
      <div class="metric-val" style="color:var(--accent-amber)">{w/h:.2f}</div>
      <div class="metric-lbl">Aspect Ratio</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─── TABS ────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "⬡ Geometric",
    "◈ Photometric",
    "∿ Noise",
    "⟁ Filters",
    "✦ Artistic",
    "⊞ Advanced",
    "⊕ Compose All"
])

# ─── CODE LIBRARY FOR ALL TRANSFORMATIONS ──────────────────────────────────────
TRANSFORMATION_CODE = {
    "Horizontal Flip": """def apply_horizontal_flip(img):
    return ImageOps.mirror(img)""",
    
    "Vertical Flip": """def apply_vertical_flip(img):
    return ImageOps.flip(img)""",
    
    "Rotation": """def apply_rotation(img, angle):
    return img.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC, fillcolor=(20, 28, 36))""",
    
    "Crop": """def apply_crop(img, percentage):
    w, h = img.size
    c = int(percentage * min(w, h) / 100)
    return img.crop((c, c, w-c, h-c)).resize((w, h), Image.Resampling.LANCZOS)""",
    
    "Shear": """def apply_shear(img, shear_factor):
    w, h = img.size
    xshift = abs(shear_factor) * w
    new_w = w + int(round(xshift))
    img_arr = np.array(img.convert("RGBA"))
    result = Image.new("RGBA", (new_w, h), (20, 28, 36, 255))
    for y in range(h):
        shift = int(shear_factor * y)
        for x in range(w):
            nx = x + shift
            if 0 <= nx < new_w:
                result.putpixel((nx, y), tuple(img_arr[y, x]))
    return result.resize((w, h), Image.Resampling.LANCZOS).convert(img.mode)""",
    
    "Brightness": """def apply_brightness(img, factor):
    return ImageEnhance.Brightness(img).enhance(factor)""",
    
    "Contrast": """def apply_contrast(img, factor):
    return ImageEnhance.Contrast(img).enhance(factor)""",
    
    "Saturation": """def apply_saturation(img, factor):
    return ImageEnhance.Color(img).enhance(factor)""",
    
    "Sharpness": """def apply_sharpness(img, factor):
    return ImageEnhance.Sharpness(img).enhance(factor)""",
    
    "Gamma": """def apply_gamma(img, gamma):
    arr = np.array(img).astype(np.float32) / 255.0
    arr = np.clip(arr ** (1.0 / gamma), 0, 1)
    return Image.fromarray((arr * 255).astype(np.uint8))""",
    
    "Hue Shift": """def apply_hue_shift(img, shift):
    import colorsys
    arr = np.array(img.convert("RGB")).astype(np.float32) / 255.0
    out = np.zeros_like(arr)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            r, g, b = arr[i, j]
            h, s, v = colorsys.rgb_to_hsv(r, g, b)
            h = (h + shift / 360.0) % 1.0
            rgb = colorsys.hsv_to_rgb(h, s, v)
            out[i, j] = rgb
    result = Image.fromarray((out * 255).astype(np.uint8))
    return result.convert(img.mode) if img.mode != 'RGB' else result""",
    
    "Grayscale": """def apply_grayscale(img):
    return ImageOps.grayscale(img).convert("RGB")""",
    
    "Sepia": """def apply_sepia(img):
    arr = np.array(img.convert("RGB"), dtype=np.float32)
    r = np.clip(arr[:,:,0]*0.393 + arr[:,:,1]*0.769 + arr[:,:,2]*0.189, 0, 255)
    g = np.clip(arr[:,:,0]*0.349 + arr[:,:,1]*0.686 + arr[:,:,2]*0.168, 0, 255)
    b = np.clip(arr[:,:,0]*0.272 + arr[:,:,1]*0.534 + arr[:,:,2]*0.131, 0, 255)
    return Image.fromarray(np.stack([r, g, b], axis=2).astype(np.uint8))""",
    
    "Gaussian Noise": """def apply_gaussian_noise(img, std):
    arr = np.array(img).astype(np.float32)
    noise = np.random.normal(0, std, arr.shape)
    return Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))""",
    
    "Salt & Pepper": """def apply_salt_pepper(img, amount):
    arr = np.array(img).copy()
    n = int(amount * arr.size // arr.shape[2] if len(arr.shape) == 3 else amount * arr.size)
    coords = [np.random.randint(0, i, n) for i in arr.shape[:2]]
    arr[coords[0], coords[1]] = 255
    coords = [np.random.randint(0, i, n) for i in arr.shape[:2]]
    arr[coords[0], coords[1]] = 0
    return Image.fromarray(arr)""",
    
    "Gaussian Blur": """def apply_blur(img, radius):
    return img.filter(ImageFilter.GaussianBlur(radius))""",
    
    "Sharpen": """def apply_sharpen(img):
    return img.filter(ImageFilter.SHARPEN)""",
    
    "Grid Distortion": """def apply_grid_distortion(img, steps, magnitude):
    w, h = img.size
    arr = np.array(img.convert("RGB")).astype(np.uint8)
    out = np.zeros_like(arr)
    sw, sh = w // steps, h // steps
    for i in range(h):
        for j in range(w):
            ox = int(magnitude * math.sin(2 * math.pi * j / (sw + 1)))
            oy = int(magnitude * math.cos(2 * math.pi * i / (sh + 1)))
            src_x = min(max(j + ox, 0), w - 1)
            src_y = min(max(i + oy, 0), h - 1)
            out[i, j] = arr[src_y, src_x]
    return Image.fromarray(out).convert(img.mode)""",
    
    "Vignette": """def apply_vignette(img, strength):
    arr = np.array(img.convert("RGB")).astype(np.float32)
    h, w = arr.shape[:2]
    Y, X = np.ogrid[:h, :w]
    cx, cy = w / 2, h / 2
    dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
    max_d = np.sqrt(cx**2 + cy**2)
    mask = 1 - strength * (dist / max_d) ** 2
    mask = np.clip(mask, 0, 1)
    arr *= mask[:, :, np.newaxis]
    return Image.fromarray(arr.astype(np.uint8))""",
    
    "Invert": """def apply_invert(img):
    return ImageOps.invert(img.convert("RGB"))""",
    
    "Equalize": """def apply_equalize(img):
    return ImageOps.equalize(img.convert("RGB"))""",
    
    "Posterize": """def apply_posterize(img, bits):
    return ImageOps.posterize(img.convert("RGB"), bits)""",
    
    "Solarize": """def apply_solarize(img, thresh):
    return ImageOps.solarize(img.convert("RGB"), thresh)""",
    
    "Speckle": """def apply_speckle(img, intensity):
    arr = np.array(img).astype(np.float32)
    noise = np.random.randn(*arr.shape) * intensity
    return Image.fromarray(np.clip(arr + arr * noise / 100, 0, 255).astype(np.uint8))""",
    
    "Motion Blur": """def apply_motion_blur(img, size):
    kernel = np.zeros((size, size))
    kernel[size // 2, :] = 1.0 / size
    return img.filter(ImageFilter.GaussianBlur(size // 3))""",
    
    "Median Filter": """def apply_median_filter(img, size):
    return img.filter(ImageFilter.MedianFilter(size=size))""",
    
    "Edge Enhance": """def apply_edge_enhance(img):
    return img.filter(ImageFilter.EDGE_ENHANCE_MORE)""",
    
    "Emboss": """def apply_emboss(img):
    return img.filter(ImageFilter.EMBOSS)""",
    
    "Pixelate": """def apply_pixelate(img, block):
    w, h = img.size
    small = img.resize((max(1, w // block), max(1, h // block)), Image.NEAREST)
    return small.resize((w, h), Image.NEAREST)""",
    
    "Cutout": """def apply_cutout(img, n, size):
    arr = np.array(img).copy()
    h, w = arr.shape[:2]
    for _ in range(n):
        x = random.randint(0, w - size)
        y = random.randint(0, h - size)
        arr[y:y+size, x:x+size] = 0
    return Image.fromarray(arr)""",
    
    "Color Jitter": """def apply_color_jitter(img, b, c, s):
    img = ImageEnhance.Brightness(img).enhance(b)
    img = ImageEnhance.Contrast(img).enhance(c)
    img = ImageEnhance.Color(img).enhance(s)
    return img""",
    
    "Channel Shuffle": """def apply_channel_shuffle(img):
    arr = np.array(img.convert("RGB"))
    channels = [arr[:,:,i] for i in range(3)]
    random.shuffle(channels)
    return Image.fromarray(np.stack(channels, axis=2))""",
    
    "Both Flips": """def apply_both_flips(img):
    return ImageOps.flip(ImageOps.mirror(img))""",
    
    "Center Crop": """def apply_crop(img, percentage):
    w, h = img.size
    c = int(percentage * min(w, h) / 100)
    return img.crop((c, c, w-c, h-c)).resize((w, h), Image.Resampling.LANCZOS)""",
    
    "Zoom": """def apply_zoom(img, factor):
    w, h = img.size
    new_w, new_h = int(w * factor), int(h * factor)
    zoomed = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    x = (new_w - w) // 2
    y = (new_h - new_h) // 2
    return zoomed.crop((x, y, x + w, y + h)) if factor > 1 else zoomed""",
    
    "Rotation": """def apply_rotation(img, angle):
    return img.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC, fillcolor=(20, 28, 36))""",
    
    "Vertical Flip": """def apply_vertical_flip(img):
    return ImageOps.flip(img)""",
    
    "Equalized": """def apply_equalize(img):
    return ImageOps.equalize(img.convert("RGB"))""",
    
    "Speckle Noise": """def apply_speckle(img, intensity):
    arr = np.array(img).astype(np.float32)
    noise = np.random.randn(*arr.shape) * intensity
    return Image.fromarray(np.clip(arr + arr * noise / 100, 0, 255).astype(np.uint8))""",
    
    "Sepia": """def apply_sepia(img):
    arr = np.array(img.convert("RGB"), dtype=np.float32)
    r = np.clip(arr[:,:,0]*0.393 + arr[:,:,1]*0.769 + arr[:,:,2]*0.189, 0, 255)
    g = np.clip(arr[:,:,0]*0.349 + arr[:,:,1]*0.686 + arr[:,:,2]*0.168, 0, 255)
    b = np.clip(arr[:,:,0]*0.272 + arr[:,:,1]*0.534 + arr[:,:,2]*0.131, 0, 255)
    return Image.fromarray(np.stack([r, g, b], axis=2).astype(np.uint8))""",
    
    "Color Overlay": """def apply_color_overlay(img, color_hex, alpha):
    overlay = Image.new("RGBA", img.size, color_hex + hex(int(alpha * 255))[2:].zfill(2))
    base = img.convert("RGBA")
    return Image.alpha_composite(base, overlay).convert("RGB")""",
    
    "CLAHE": """def apply_clahe(img):
    try:
        import cv2
        gray = np.array(img.convert("L"))
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        result = clahe.apply(gray)
        return Image.fromarray(result).convert("RGB")
    except ImportError:
        return apply_equalize(img)""",
    
    "Color Jitter": """def apply_color_jitter(img, b, c, s):
    img = ImageEnhance.Brightness(img).enhance(b)
    img = ImageEnhance.Contrast(img).enhance(c)
    img = ImageEnhance.Color(img).enhance(s)
    return img""",
}

def show_code_modal(title):
    """Display transformation code in a modal - only if code exists"""
    if title not in TRANSFORMATION_CODE:
        return
    
    code = TRANSFORMATION_CODE[title]
    st.markdown(f"""
    <div style="background: var(--bg-card); border: 1px solid var(--border); border-radius: 8px; 
                padding: 16px; margin-top: 12px;">
        <div style="font-family: var(--font-mono); font-size: 10px; color: var(--accent-cyan); 
                    letter-spacing: 2px; text-transform: uppercase; margin-bottom: 10px;">📋 Backend Code</div>
        <pre style="background: var(--bg-primary); padding: 12px; border-radius: 6px; 
                   border-left: 3px solid var(--accent-lime); overflow-x: auto;
                   color: var(--text-primary); font-family: var(--font-mono); font-size: 11px;
                   line-height: 1.5;">{code}</pre>
    </div>
    """, unsafe_allow_html=True)


def show_result(col_orig, col_aug, orig, aug, title, badge, desc):
    with col_orig:
        st.markdown('<div class="preview-label">Original</div>', unsafe_allow_html=True)
        st.image(orig, use_column_width=True)
    with col_aug:
        st.markdown(f'<div class="preview-label">{title}</div>', unsafe_allow_html=True)
        st.image(aug, use_column_width=True)
        buf = io.BytesIO()
        aug.save(buf, format="PNG")
        st.download_button("↓ Download", buf.getvalue(), f"{title.lower().replace(' ','_')}.png",
                           "image/png")
        show_code_modal(title)


# ══════════════════════════════════════════════════════
# TAB 1: GEOMETRIC
# ══════════════════════════════════════════════════════
with tabs[0]:
    st.markdown("""
    <div class="aug-grid-header">
      <div class="aug-grid-title">Geometric <span>Transforms</span></div>
      <div class="aug-grid-sub">Spatial transformations that alter the shape, orientation, and layout of the image</div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="aug-content">', unsafe_allow_html=True)

    # ── Flip ──
    st.markdown('<span class="cat-badge cat-geometric">Flip</span>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        if st.button("⟵⟶ Horizontal Flip"):
            aug = apply_horizontal_flip(raw_img)
            cols = st.columns(2)
            show_result(cols[0], cols[1], raw_img, aug, "Horizontal Flip", "GEO", "")
    with c2:
        if st.button("↑↓ Vertical Flip"):
            aug = apply_vertical_flip(raw_img)
            cols = st.columns(2)
            show_result(cols[0], cols[1], raw_img, aug, "Vertical Flip", "GEO", "")
    with c3:
        if st.button("↻ Both Flips"):
            aug = apply_horizontal_flip(apply_vertical_flip(raw_img))
            cols = st.columns(2)
            show_result(cols[0], cols[1], raw_img, aug, "Both Flips", "GEO", "")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Rotation ──
    st.markdown('<span class="cat-badge cat-geometric">Rotation</span>', unsafe_allow_html=True)
    angle = st.slider("Rotation Angle (°)", -180, 180, 45, key="rot_ang")
    col_b, _ = st.columns([1, 3])
    with col_b:
        if st.button("↻ Apply Rotation"):
            aug = apply_rotation(raw_img, angle)
            cols = st.columns(2)
            show_result(cols[0], cols[1], raw_img, aug, f"Rotation {angle}°", "GEO", "")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Crop ──
    st.markdown('<span class="cat-badge cat-geometric">Crop & Zoom</span>', unsafe_allow_html=True)
    cc1, cc2 = st.columns(2)
    with cc1:
        crop_pct = st.slider("Crop Margin %", 5, 40, 15, key="crop_pct")
        if st.button("✂ Random Crop"):
            aug = apply_crop(raw_img, crop_pct)
            cols = st.columns(2)
            show_result(cols[0], cols[1], raw_img, aug, "Center Crop", "GEO", "")
    with cc2:
        zoom_f = st.slider("Zoom Factor", 1.1, 3.0, 1.5, 0.1, key="zoom_f")
        if st.button("⊕ Zoom In"):
            aug = apply_zoom(raw_img, zoom_f)
            cols = st.columns(2)
            show_result(cols[0], cols[1], raw_img, aug, "Zoom", "GEO", "")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Shear ──
    st.markdown('<span class="cat-badge cat-geometric">Shear</span>', unsafe_allow_html=True)
    shear_v = st.slider("Shear Factor", -0.5, 0.5, 0.2, 0.05, key="shear_v")
    col_sh, _ = st.columns([1, 3])
    with col_sh:
        if st.button("⟋ Apply Shear"):
            aug = apply_shear(raw_img, shear_v)
            cols = st.columns(2)
            show_result(cols[0], cols[1], raw_img, aug, "Shear", "GEO", "")

    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
# TAB 2: PHOTOMETRIC
# ══════════════════════════════════════════════════════
with tabs[1]:
    st.markdown("""
    <div class="aug-grid-header">
      <div class="aug-grid-title">Photometric <span>Adjustments</span></div>
      <div class="aug-grid-sub">Modify pixel intensity, color balance, and tonal characteristics</div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="aug-content">', unsafe_allow_html=True)

    st.markdown('<span class="cat-badge cat-photometric">Intensity Controls</span>', unsafe_allow_html=True)
    pa1, pa2 = st.columns(2)
    with pa1:
        bright_f = st.slider("Brightness", 0.1, 3.0, 1.5, 0.1, key="br")
        if st.button("☀ Adjust Brightness"):
            aug = apply_brightness(raw_img, bright_f)
            cols = st.columns(2)
            show_result(cols[0], cols[1], raw_img, aug, "Brightness", "PHOTO", "")

        sat_f = st.slider("Saturation", 0.0, 4.0, 2.0, 0.1, key="sat")
        if st.button("⬤ Adjust Saturation"):
            aug = apply_saturation(raw_img, sat_f)
            cols = st.columns(2)
            show_result(cols[0], cols[1], raw_img, aug, "Saturation", "PHOTO", "")

    with pa2:
        cont_f = st.slider("Contrast", 0.1, 4.0, 2.0, 0.1, key="ct")
        if st.button("◑ Adjust Contrast"):
            aug = apply_contrast(raw_img, cont_f)
            cols = st.columns(2)
            show_result(cols[0], cols[1], raw_img, aug, "Contrast", "PHOTO", "")

        sharp_f = st.slider("Sharpness", 0.0, 5.0, 2.5, 0.1, key="sh")
        if st.button("◈ Adjust Sharpness"):
            aug = apply_sharpness(raw_img, sharp_f)
            cols = st.columns(2)
            show_result(cols[0], cols[1], raw_img, aug, "Sharpness", "PHOTO", "")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<span class="cat-badge cat-photometric">Gamma & Color</span>', unsafe_allow_html=True)

    pb1, pb2 = st.columns(2)
    with pb1:
        gamma_v = st.slider("Gamma Value", 0.2, 3.0, 1.5, 0.1, key="gm")
        if st.button("γ Gamma Correction"):
            aug = apply_gamma(raw_img, gamma_v)
            cols = st.columns(2)
            show_result(cols[0], cols[1], raw_img, aug, "Gamma", "PHOTO", "")

    with pb2:
        hue_v = st.slider("Hue Shift (°)", -180, 180, 60, key="hue")
        if st.button("◌ Hue Shift"):
            aug = apply_hue_shift(raw_img, hue_v)
            cols = st.columns(2)
            show_result(cols[0], cols[1], raw_img, aug, "Hue Shift", "PHOTO", "")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<span class="cat-badge cat-photometric">Tone Presets</span>', unsafe_allow_html=True)

    pc1, pc2, pc3 = st.columns(3)
    with pc1:
        if st.button("▣ Grayscale"):
            aug = apply_grayscale(raw_img)
            cols = st.columns(2)
            show_result(cols[0], cols[1], raw_img, aug, "Grayscale", "PHOTO", "")
    with pc2:
        if st.button("◫ Invert"):
            aug = apply_invert(raw_img)
            cols = st.columns(2)
            show_result(cols[0], cols[1], raw_img, aug, "Invert", "PHOTO", "")
    with pc3:
        if st.button("⬡ Equalize"):
            aug = apply_equalize(raw_img)
            cols = st.columns(2)
            show_result(cols[0], cols[1], raw_img, aug, "Equalized", "PHOTO", "")

    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
# TAB 3: NOISE
# ══════════════════════════════════════════════════════
with tabs[2]:
    st.markdown("""
    <div class="aug-grid-header">
      <div class="aug-grid-title">Noise <span>Injection</span></div>
      <div class="aug-grid-sub">Add stochastic perturbations to improve model robustness and generalization</div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="aug-content">', unsafe_allow_html=True)
    st.markdown('<span class="cat-badge cat-noise">Statistical Noise</span>', unsafe_allow_html=True)

    na1, na2 = st.columns(2)
    with na1:
        g_std = st.slider("Gaussian Std Dev", 5, 80, 25, key="g_std")
        if st.button("∿ Gaussian Noise"):
            aug = apply_gaussian_noise(raw_img, g_std)
            cols = st.columns(2)
            show_result(cols[0], cols[1], raw_img, aug, "Gaussian Noise", "NOISE", "")

    with na2:
        sp_amt = st.slider("S&P Amount", 0.01, 0.2, 0.05, 0.01, key="sp")
        if st.button("∷ Salt & Pepper"):
            aug = apply_salt_pepper(raw_img, sp_amt)
            cols = st.columns(2)
            show_result(cols[0], cols[1], raw_img, aug, "Salt & Pepper", "NOISE", "")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<span class="cat-badge cat-noise">Multiplicative Noise</span>', unsafe_allow_html=True)

    nb1, _ = st.columns([1, 3])
    with nb1:
        sp_int = st.slider("Speckle Intensity", 10, 100, 40, key="sp_int")
        if st.button("⁕ Speckle Noise"):
            aug = apply_speckle(raw_img, sp_int)
            cols = st.columns(2)
            show_result(cols[0], cols[1], raw_img, aug, "Speckle Noise", "NOISE", "")

    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
# TAB 4: FILTERS
# ══════════════════════════════════════════════════════
with tabs[3]:
    st.markdown("""
    <div class="aug-grid-header">
      <div class="aug-grid-title">Convolutional <span>Filters</span></div>
      <div class="aug-grid-sub">Kernel-based spatial filters that alter frequency and texture characteristics</div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="aug-content">', unsafe_allow_html=True)
    st.markdown('<span class="cat-badge cat-filter">Blur Filters</span>', unsafe_allow_html=True)

    fa1, fa2 = st.columns(2)
    with fa1:
        blur_r = st.slider("Gaussian Blur Radius", 1, 20, 5, key="blur_r")
        if st.button("⊙ Gaussian Blur"):
            aug = apply_blur(raw_img, blur_r)
            cols = st.columns(2)
            show_result(cols[0], cols[1], raw_img, aug, "Gaussian Blur", "FILTER", "")

    with fa2:
        med_s = st.selectbox("Median Filter Size", [3, 5, 7, 9], key="med_s")
        if st.button("⊞ Median Filter"):
            aug = apply_median_filter(raw_img, med_s)
            cols = st.columns(2)
            show_result(cols[0], cols[1], raw_img, aug, "Median Filter", "FILTER", "")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<span class="cat-badge cat-filter">Enhancement Filters</span>', unsafe_allow_html=True)

    fb1, fb2, fb3 = st.columns(3)
    with fb1:
        if st.button("◈ Sharpen"):
            aug = apply_sharpen(raw_img)
            cols = st.columns(2)
            show_result(cols[0], cols[1], raw_img, aug, "Sharpen", "FILTER", "")
    with fb2:
        if st.button("⟁ Edge Enhance"):
            aug = apply_edge_enhance(raw_img)
            cols = st.columns(2)
            show_result(cols[0], cols[1], raw_img, aug, "Edge Enhance", "FILTER", "")
    with fb3:
        if st.button("⬡ Emboss"):
            aug = apply_emboss(raw_img)
            cols = st.columns(2)
            show_result(cols[0], cols[1], raw_img, aug, "Emboss", "FILTER", "")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<span class="cat-badge cat-filter">Pixelation</span>', unsafe_allow_html=True)

    pix_b = st.slider("Pixel Block Size", 2, 30, 8, key="pix_b")
    col_px, _ = st.columns([1, 3])
    with col_px:
        if st.button("⊞ Pixelate"):
            aug = apply_pixelate(raw_img, pix_b)
            cols = st.columns(2)
            show_result(cols[0], cols[1], raw_img, aug, "Pixelate", "FILTER", "")

    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
# TAB 5: ARTISTIC
# ══════════════════════════════════════════════════════
with tabs[4]:
    st.markdown("""
    <div class="aug-grid-header">
      <div class="aug-grid-title">Artistic <span>Effects</span></div>
      <div class="aug-grid-sub">Stylistic transformations inspired by photography, printing, and visual art</div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="aug-content">', unsafe_allow_html=True)
    st.markdown('<span class="cat-badge cat-artistic">Classic Film Looks</span>', unsafe_allow_html=True)

    art1, art2 = st.columns(2)
    with art1:
        if st.button("☕ Sepia Tone"):
            aug = apply_sepia(raw_img)
            cols = st.columns(2)
            show_result(cols[0], cols[1], raw_img, aug, "Sepia", "ART", "")

    with art2:
        vig_s = st.slider("Vignette Strength", 0.1, 1.5, 0.7, 0.1, key="vig_s")
        if st.button("◎ Vignette"):
            aug = apply_vignette(raw_img, vig_s)
            cols = st.columns(2)
            show_result(cols[0], cols[1], raw_img, aug, "Vignette", "ART", "")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<span class="cat-badge cat-artistic">Stylization</span>', unsafe_allow_html=True)

    art3, art4 = st.columns(2)
    with art3:
        post_bits = st.slider("Posterize Bits", 1, 7, 3, key="post_b")
        if st.button("⬤ Posterize"):
            aug = apply_posterize(raw_img, post_bits)
            cols = st.columns(2)
            show_result(cols[0], cols[1], raw_img, aug, "Posterize", "ART", "")

    with art4:
        sol_thresh = st.slider("Solarize Threshold", 50, 200, 128, key="sol_t")
        if st.button("☀ Solarize"):
            aug = apply_solarize(raw_img, sol_thresh)
            cols = st.columns(2)
            show_result(cols[0], cols[1], raw_img, aug, "Solarize", "ART", "")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<span class="cat-badge cat-artistic">Color Overlay</span>', unsafe_allow_html=True)

    ov1, ov2, ov3 = st.columns([1, 1, 2])
    with ov1:
        ov_color = st.color_picker("Overlay Color", "#00e5ff", key="ov_c")
    with ov2:
        ov_alpha = st.slider("Overlay Alpha", 0.05, 0.7, 0.25, 0.05, key="ov_a")
    with ov3:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("◈ Apply Color Overlay"):
            aug = apply_color_overlay(raw_img, ov_color, ov_alpha)
            cols = st.columns(2)
            show_result(cols[0], cols[1], raw_img, aug, "Color Overlay", "ART", "")

    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
# TAB 6: ADVANCED
# ══════════════════════════════════════════════════════
with tabs[5]:
    st.markdown("""
    <div class="aug-grid-header">
      <div class="aug-grid-title">Advanced <span>Techniques</span></div>
      <div class="aug-grid-sub">Complex augmentations used in state-of-the-art training pipelines</div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="aug-content">', unsafe_allow_html=True)
    st.markdown('<span class="cat-badge cat-advanced">Occlusion & Dropout</span>', unsafe_allow_html=True)

    ad1, ad2 = st.columns(2)
    with ad1:
        cut_n = st.slider("Cutout Patches (n)", 1, 10, 3, key="cut_n")
        cut_s = st.slider("Patch Size (px)", 10, 80, 30, key="cut_s")
        if st.button("⊞ Cutout (Erasing)"):
            aug = apply_cutout(raw_img, cut_n, cut_s)
            cols = st.columns(2)
            show_result(cols[0], cols[1], raw_img, aug, "Cutout", "ADV", "")

    with ad2:
        gd_steps = st.slider("Grid Steps", 3, 10, 5, key="gd_s")
        gd_mag = st.slider("Distortion Magnitude", 2, 20, 8, key="gd_m")
        if st.button("⟁ Grid Distortion"):
            aug = apply_grid_distortion(raw_img, gd_steps, gd_mag)
            cols = st.columns(2)
            show_result(cols[0], cols[1], raw_img, aug, "Grid Distortion", "ADV", "")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<span class="cat-badge cat-advanced">Channel Operations</span>', unsafe_allow_html=True)

    ad3, ad4 = st.columns(2)
    with ad3:
        if st.button("⬡ Channel Shuffle"):
            aug = apply_channel_shuffle(raw_img)
            cols = st.columns(2)
            show_result(cols[0], cols[1], raw_img, aug, "Channel Shuffle", "ADV", "")

    with ad4:
        if st.button("◈ CLAHE (Adaptive Hist Eq)"):
            aug = apply_clahe(raw_img)
            cols = st.columns(2)
            show_result(cols[0], cols[1], raw_img, aug, "CLAHE", "ADV", "")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<span class="cat-badge cat-advanced">Color Jitter (Combined)</span>', unsafe_allow_html=True)

    cj1, cj2, cj3 = st.columns(3)
    with cj1: cj_b = st.slider("Brightness", 0.5, 2.5, 1.3, 0.1, key="cj_b")
    with cj2: cj_c = st.slider("Contrast", 0.5, 2.5, 1.3, 0.1, key="cj_c")
    with cj3: cj_s = st.slider("Saturation", 0.0, 3.0, 1.5, 0.1, key="cj_s")
    col_cj, _ = st.columns([1, 3])
    with col_cj:
        if st.button("✦ Apply Color Jitter"):
            aug = apply_color_jitter(raw_img, cj_b, cj_c, cj_s)
            cols = st.columns(2)
            show_result(cols[0], cols[1], raw_img, aug, "Color Jitter", "ADV", "")

    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
# TAB 7: COMPOSE ALL
# ══════════════════════════════════════════════════════
with tabs[6]:
    st.markdown("""
    <div class="aug-grid-header">
      <div class="aug-grid-title">Compose <span>Pipeline</span></div>
      <div class="aug-grid-sub">Chain multiple augmentations into a single transformation pipeline</div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="aug-content">', unsafe_allow_html=True)
    st.markdown('<div class="info-pill">⊕ Select augmentations · Set parameters · Fire pipeline</div>', unsafe_allow_html=True)

    comp_col1, comp_col2 = st.columns([1, 2])

    with comp_col1:
        st.markdown('<span class="cat-badge cat-geometric">Geometric</span>', unsafe_allow_html=True)
        do_flip = st.checkbox("Horizontal Flip", value=True, key="c_flip")
        do_rot = st.checkbox("Rotation", key="c_rot")
        rot_a = st.slider("Angle", -90, 90, 20, key="c_rot_a") if do_rot else 20
        do_crop = st.checkbox("Center Crop", key="c_crop")
        crop_p = st.slider("Margin %", 5, 30, 10, key="c_crop_p") if do_crop else 10

        st.markdown('<span class="cat-badge cat-photometric">Photometric</span>', unsafe_allow_html=True)
        do_bright = st.checkbox("Brightness", key="c_bright")
        b_val = st.slider("Factor", 0.5, 2.0, 1.3, 0.1, key="c_bright_v") if do_bright else 1.3
        do_cont = st.checkbox("Contrast", key="c_cont")
        ct_val = st.slider("Factor  ", 0.5, 2.5, 1.3, 0.1, key="c_cont_v") if do_cont else 1.3
        do_gray = st.checkbox("Grayscale", key="c_gray")

        st.markdown('<span class="cat-badge cat-noise">Noise</span>', unsafe_allow_html=True)
        do_gnoise = st.checkbox("Gaussian Noise", key="c_gn")
        gn_std = st.slider("Std Dev  ", 5, 60, 20, key="c_gn_s") if do_gnoise else 20
        do_sp = st.checkbox("Salt & Pepper", key="c_sp")
        sp_a = st.slider("Amount  ", 0.01, 0.15, 0.03, 0.01, key="c_sp_a") if do_sp else 0.03

        st.markdown('<span class="cat-badge cat-filter">Filters</span>', unsafe_allow_html=True)
        do_blur = st.checkbox("Gaussian Blur", key="c_blur")
        bl_r = st.slider("Radius  ", 1, 15, 3, key="c_blur_r") if do_blur else 3
        do_sharp = st.checkbox("Sharpen", key="c_sharp")
        do_vig = st.checkbox("Vignette", key="c_vig")
        vig_str = st.slider("Strength  ", 0.1, 1.5, 0.6, 0.1, key="c_vig_s") if do_vig else 0.6

    with comp_col2:
        if st.button("⟁ RUN PIPELINE"):
            aug = raw_img.copy()
            pipeline_steps = []

            if do_flip:
                aug = apply_horizontal_flip(aug)
                pipeline_steps.append("H-Flip")
            if do_rot:
                aug = apply_rotation(aug, rot_a)
                pipeline_steps.append(f"Rotate {rot_a}°")
            if do_crop:
                aug = apply_crop(aug, crop_p)
                pipeline_steps.append(f"Crop {crop_p}%")
            if do_bright:
                aug = apply_brightness(aug, b_val)
                pipeline_steps.append(f"Bright ×{b_val}")
            if do_cont:
                aug = apply_contrast(aug, ct_val)
                pipeline_steps.append(f"Contrast ×{ct_val}")
            if do_gray:
                aug = apply_grayscale(aug)
                pipeline_steps.append("Grayscale")
            if do_gnoise:
                aug = apply_gaussian_noise(aug, gn_std)
                pipeline_steps.append(f"G-Noise σ={gn_std}")
            if do_sp:
                aug = apply_salt_pepper(aug, sp_a)
                pipeline_steps.append("S&P Noise")
            if do_blur:
                aug = apply_blur(aug, bl_r)
                pipeline_steps.append(f"Blur r={bl_r}")
            if do_sharp:
                aug = apply_sharpen(aug)
                pipeline_steps.append("Sharpen")
            if do_vig:
                aug = apply_vignette(aug, vig_str)
                pipeline_steps.append(f"Vignette {vig_str}")

            st.markdown(f"""
            <div style="background:var(--bg-card);border:1px solid var(--border);border-radius:8px;
                        padding:12px 16px;margin-bottom:16px;font-family:var(--font-mono);font-size:11px;
                        color:var(--text-muted);">
              Pipeline: {'  →  '.join(pipeline_steps) if pipeline_steps else 'No transforms selected'}
            </div>
            """, unsafe_allow_html=True)

            rc1, rc2 = st.columns(2)
            with rc1:
                st.markdown('<div class="preview-label">Original</div>', unsafe_allow_html=True)
                st.image(raw_img, use_column_width=True)
            with rc2:
                st.markdown('<div class="preview-label">Pipeline Output</div>', unsafe_allow_html=True)
                st.image(aug, use_column_width=True)
                buf = io.BytesIO()
                aug.save(buf, format="PNG")
                st.download_button("↓ Download Pipeline Result", buf.getvalue(),
                                   "augmented_pipeline.png", "image/png")
        else:
            st.markdown("""
            <div style="display:flex;align-items:center;justify-content:center;
                        height:320px;background:var(--bg-card);border:1px dashed var(--border);
                        border-radius:12px;flex-direction:column;gap:12px;">
              <div style="font-size:40px;opacity:0.2">⟁</div>
              <div style="font-family:var(--font-mono);font-size:11px;color:var(--text-dim);letter-spacing:2px;">
                SELECT TRANSFORMS · CLICK RUN
              </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
