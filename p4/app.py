import time
import io

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from core.preprocess import preprocess_pipeline
from core.detector import ObjectDetector, CLASSES, CLASS_COLORS, MANDATORY_MIN_CONFIDENCE

MODEL_PROTOTXT = "model/MobileNetSSD_deploy.prototxt"
MODEL_WEIGHTS = "model/MobileNetSSD_deploy.caffemodel"

st.set_page_config(page_title="VISION // Recognition Pipeline", page_icon="◎", layout="wide")


# ---------------------------------------------------------------- styling
# Inject custom styling into the Streamlit app
def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background: #0B0E11; }

    h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; letter-spacing: -0.01em; }

    .mono { font-family: 'JetBrains Mono', monospace; }

    /* eyebrow / header */
    .eyebrow {
        font-family: 'JetBrains Mono', monospace;
        color: #39FF88;
        font-size: 0.75rem;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        margin-bottom: 0.25rem;
    }
    .hero-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.4rem;
        font-weight: 700;
        color: #E8ECEF;
        margin: 0 0 0.3rem 0;
        line-height: 1.1;
    }
    .hero-sub {
        color: #8B96A5;
        font-size: 0.95rem;
        max-width: 640px;
    }

    /* pipeline stepper */
    .stage-row { display: flex; gap: 0; margin: 1.4rem 0 1.6rem 0; flex-wrap: wrap; }
    .stage {
        flex: 1;
        min-width: 130px;
        border-top: 2px solid #262B32;
        padding: 10px 12px 8px 0;
        position: relative;
    }
    .stage.done { border-top: 2px solid #39FF88; }
    .stage-num {
        font-family: 'JetBrains Mono', monospace;
        color: #8B96A5;
        font-size: 0.7rem;
    }
    .stage.done .stage-num { color: #39FF88; }
    .stage-label {
        font-family: 'Space Grotesk', sans-serif;
        color: #E8ECEF;
        font-size: 0.85rem;
        font-weight: 600;
        margin-top: 2px;
    }
    .stage-time {
        font-family: 'JetBrains Mono', monospace;
        color: #5A6472;
        font-size: 0.72rem;
        margin-top: 3px;
    }
    .stage.done .stage-time { color: #39FF88; }

    /* image frame with HUD corner brackets */
    .scan-frame {
        position: relative;
        border: 1px solid #262B32;
        background: #0F1317;
        padding: 6px;
    }
    .scan-frame::before, .scan-frame::after,
    .scan-frame .br, .scan-frame .bl {
        content: "";
        position: absolute;
        width: 18px; height: 18px;
        border-color: #39FF88;
        z-index: 2;
    }
    .scan-frame::before { top: -1px; left: -1px; border-top: 2px solid #39FF88; border-left: 2px solid #39FF88; }
    .scan-frame::after { top: -1px; right: -1px; border-top: 2px solid #39FF88; border-right: 2px solid #39FF88; }
    .frame-tag {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        color: #5A6472;
        margin-bottom: 6px;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }

    /* detection cards */
    .det-card {
        border: 1px solid #262B32;
        background: #12161B;
        border-radius: 4px;
        padding: 10px 14px;
        margin-bottom: 8px;
    }
    .det-top { display: flex; justify-content: space-between; align-items: baseline; }
    .det-label { font-family: 'Space Grotesk', sans-serif; font-weight: 600; font-size: 0.95rem; }
    .det-conf { font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; color: #39FF88; }
    .det-bar-bg { background: #1C2126; height: 5px; border-radius: 3px; margin-top: 6px; overflow: hidden; }
    .det-bar-fill { height: 5px; border-radius: 3px; }
    .det-coords { font-family: 'JetBrains Mono', monospace; color: #5A6472; font-size: 0.7rem; margin-top: 5px; }

    /* telemetry strip */
    .tele-strip { display: flex; gap: 0; border-top: 1px solid #262B32; border-bottom: 1px solid #262B32; margin: 1.2rem 0; }
    .tele-cell { flex: 1; padding: 10px 14px; border-right: 1px solid #262B32; }
    .tele-cell:last-child { border-right: none; }
    .tele-val { font-family: 'JetBrains Mono', monospace; font-size: 1.3rem; color: #E8ECEF; }
    .tele-key { font-family: 'JetBrains Mono', monospace; font-size: 0.68rem; color: #5A6472; text-transform: uppercase; letter-spacing: 0.08em; }

    .rejected-note {
        font-family: 'JetBrains Mono', monospace;
        color: #FFB020;
        font-size: 0.78rem;
        margin-top: 8px;
    }

    section[data-testid="stSidebar"] { background: #0F1317; border-right: 1px solid #262B32; }
    </style>
    """, unsafe_allow_html=True)


# Cache and load the object detector model
@st.cache_resource(show_spinner=False)
def load_detector():
    return ObjectDetector(MODEL_PROTOTXT, MODEL_WEIGHTS)


# Convert BGR color tuple to hex string for CSS
def to_hex(bgr_tuple):
    b, g, r = bgr_tuple
    return f"#{r:02X}{g:02X}{b:02X}"


# Render the stepper bar showing stage execution times
def render_stage_row(active_stage: int, timings: dict):
    stages = [
        ("01", "Input Acquisition", None),
        ("02", "Preprocessing", timings.get("preprocess_ms") if timings else None),
        ("03", "Blob Construction", timings.get("blob_construction_ms") if timings else None),
        ("04", "Neural Inference", timings.get("inference_ms") if timings else None),
        ("05", "Confidence Filter", timings.get("filtering_ms") if timings else None),
        ("06", "Render Output", timings.get("render_ms") if timings else None),
    ]
    html = '<div class="stage-row">'
    for i, (num, label, t) in enumerate(stages):
        done = "done" if i < active_stage else ""
        time_html = f'<div class="stage-time">{t:.1f} ms</div>' if t is not None else '<div class="stage-time">&mdash;</div>'
        html += f'<div class="stage {done}"><div class="stage-num">{num}</div><div class="stage-label">{label}</div>{time_html}</div>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


# ---------------------------------------------------------------- app
inject_css()

st.markdown('<div class="eyebrow">Industrial Training Kit · Project 4 · DecodeLabs</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-title">VISION // Object Recognition Pipeline</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">A pre-trained MobileNet-SSD detector reading an image end to end — '
    'noise reduction, adaptive contrast, blob normalization, neural inference, and confidence '
    'filtering — with every stage timed and every prediction shown at the number the model actually output.</div>',
    unsafe_allow_html=True,
)
st.write("")

with st.sidebar:
    st.markdown("### Instrument Settings")
    st.markdown(
        '<div class="mono" style="color:#8B96A5; font-size:0.8rem;">MODEL &nbsp;MobileNet-SSD<br>'
        'DATASET &nbsp;VOC0712 (20 classes)<br>mAP &nbsp;0.727<br>INPUT &nbsp;300×300 blob</div>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    st.markdown("**Confidence threshold**")
    st.caption(f"Mandatory floor: {MANDATORY_MIN_CONFIDENCE:.0%}. Nothing below this is ever accepted.")
    confidence_threshold = st.slider(
        "Confidence threshold", min_value=0.80, max_value=0.99,
        value=0.80, step=0.01, label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("**Preprocessing stage**")
    denoise = st.checkbox("Noise reduction (Gaussian blur)", value=True)
    enhance = st.checkbox("Adaptive contrast (CLAHE)", value=True)

    st.markdown("---")
    st.markdown("**Detectable classes**")
    legend_html = ""
    for cls in CLASSES:
        if cls == "background":
            continue
        legend_html += f'<span class="mono" style="font-size:0.68rem; color:#8B96A5;">&#9632; {cls}</span><br>'
    st.markdown(f'<div style="max-height:180px; overflow-y:auto; line-height:1.6;">{legend_html}</div>', unsafe_allow_html=True)

detector = load_detector()

st.markdown("#### Input")
source = st.radio("Image source", ["Upload an image", "Use a sample image"], horizontal=True, label_visibility="collapsed")

image_bgr = None
if source == "Upload an image":
    uploaded = st.file_uploader("Upload", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    if uploaded is not None:
        pil_img = Image.open(uploaded).convert("RGB")
        image_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
else:
    sample_choice = st.selectbox("Sample", ["demo_street.jpg (bicycle / car / dog)", "demo_person.jpg (person)"], label_visibility="collapsed")
    fname = sample_choice.split(" ")[0]
    image_bgr = cv2.imread(f"sample_images/{fname}")

if image_bgr is None:
    st.info("Upload an image, or select a sample, to run the pipeline.")
    st.stop()

# ---------------------------------------------------------------- run pipeline
t_pre0 = time.perf_counter()
preprocessed = preprocess_pipeline(image_bgr, denoise=denoise, enhance_contrast=enhance)
preprocess_ms = (time.perf_counter() - t_pre0) * 1000

result = detector.infer(preprocessed, confidence_threshold=confidence_threshold)
result.stage_timings_ms["preprocess_ms"] = preprocess_ms
total_ms = sum(result.stage_timings_ms.values())

render_stage_row(active_stage=6, timings=result.stage_timings_ms)

# ---------------------------------------------------------------- image comparison
col1, col2 = st.columns(2)
with col1:
    st.markdown('<div class="frame-tag">Source / Preprocessed</div>', unsafe_allow_html=True)
    st.markdown('<div class="scan-frame">', unsafe_allow_html=True)
    st.image(cv2.cvtColor(preprocessed, cv2.COLOR_BGR2RGB), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
with col2:
    st.markdown(f'<div class="frame-tag">Detection Output · {len(result.detections)} object(s) ≥ {confidence_threshold:.0%}</div>', unsafe_allow_html=True)
    st.markdown('<div class="scan-frame">', unsafe_allow_html=True)
    st.image(cv2.cvtColor(result.annotated_image, cv2.COLOR_BGR2RGB), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------- telemetry
avg_conf = np.mean([d.confidence for d in result.detections]) if result.detections else 0.0
tele_html = f"""
<div class="tele-strip">
  <div class="tele-cell"><div class="tele-val">{total_ms:.1f} ms</div><div class="tele-key">Total pipeline time</div></div>
  <div class="tele-cell"><div class="tele-val">{len(result.detections)}</div><div class="tele-key">Objects accepted</div></div>
  <div class="tele-cell"><div class="tele-val">{result.rejected_count}</div><div class="tele-key">Objects rejected (&lt; {confidence_threshold:.0%})</div></div>
  <div class="tele-cell"><div class="tele-val">{avg_conf*100:.1f}%</div><div class="tele-key">Average confidence</div></div>
</div>
"""
st.markdown(tele_html, unsafe_allow_html=True)

# ---------------------------------------------------------------- results panel
st.markdown("#### Recognition Results")
if not result.detections:
    st.warning(f"No object cleared the {confidence_threshold:.0%} confidence floor. Try a clearer image or lower the threshold.")
else:
    for det in result.detections:
        color_hex = to_hex(CLASS_COLORS[det.label])
        pct = det.confidence * 100
        x1, y1, x2, y2 = det.box
        card = f"""
        <div class="det-card">
            <div class="det-top">
                <span class="det-label" style="color:{color_hex};">{det.label}</span>
                <span class="det-conf">{pct:.1f}%</span>
            </div>
            <div class="det-bar-bg"><div class="det-bar-fill" style="width:{pct}%; background:{color_hex};"></div></div>
            <div class="det-coords">x1={x1}  y1={y1}  x2={x2}  y2={y2}  &nbsp;·&nbsp; box {x2-x1}×{y2-y1}px</div>
        </div>
        """
        st.markdown(card, unsafe_allow_html=True)

    if result.rejected_count:
        st.markdown(
            f'<div class="rejected-note">&#9650; {result.rejected_count} additional candidate(s) were detected '
            f'below {confidence_threshold:.0%} confidence and discarded by the filtering stage.</div>',
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------------- download
success, buf = cv2.imencode(".jpg", result.annotated_image)
if success:
    st.download_button(
        "Download annotated image",
        data=io.BytesIO(buf.tobytes()),
        file_name="vision_detection_output.jpg",
        mime="image/jpeg",
    )

st.markdown("---")
st.caption(
    "MobileNet-SSD · pre-trained on VOC0712 · depthwise separable convolutions for edge-viable inference · "
    f"mandatory confidence floor {MANDATORY_MIN_CONFIDENCE:.0%} enforced regardless of the slider above."
)
