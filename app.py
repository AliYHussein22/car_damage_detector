"""
Car Damage Detector
-------------------
YOLOv8-powered car damage detection with severity scoring.
Run with: streamlit run app.py
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
from src.detector import load_model, run_detection
from src.severity import compute_severity

# page config — wide layout works better for the two-column result view
st.set_page_config(
    page_title="CarDamage AI",
    page_icon=":material/car_crash:",
    layout="wide",
)

# inject a minimal dark theme so it doesn't look like default Streamlit
st.markdown("""
<style>
.stApp { background-color: #0e0e0e; }
[data-testid="stSidebar"] { background-color: #1a1a1a; border-right: 1px solid #2a2a2a; }
html, body, [class*="css"] { color: #e0e0e0; }
h1, h2, h3 { color: #ffffff !important; }
.stButton > button {
    background-color: #2a2a2a; color: #e0e0e0;
    border: 1px solid #3a3a3a; border-radius: 6px;
}
.stButton > button:hover { background-color: #3a3a3a; border-color: #555; }
.stAlert { background-color: #1e1e1e !important; border-color: #333 !important; }
[data-testid="stFileUploader"] {
    background-color: #1a1a1a; border: 1px dashed #3a3a3a; border-radius: 8px;
}
.stSlider [data-baseweb="slider"] { background-color: #2a2a2a; }
hr { border-color: #2a2a2a; }
.stRadio label { color: #ccc !important; }
</style>
""", unsafe_allow_html=True)

# header — using inline SVG instead of an emoji for the car icon
st.markdown("""
<div style="display:flex;align-items:center;gap:12px;margin-bottom:4px">
  <svg xmlns="http://www.w3.org/2000/svg" width="36" height="36" viewBox="0 0 24 24"
       fill="none" stroke="#aaa" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
    <path d="M5 17H3a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v5"/>
    <circle cx="16" cy="17" r="2"/><circle cx="7" cy="17" r="2"/>
    <path d="M14 17H9m-2 0H3m18 0h-2"/>
  </svg>
  <span style="font-size:2rem;font-weight:700;color:#fff;letter-spacing:-0.5px">CarDamage AI</span>
</div>
<p style="color:#666;margin-top:0;margin-bottom:24px;font-size:0.9rem">
  YOLOv8 car damage detection &mdash; upload a photo to get an instant damage report
</p>
""", unsafe_allow_html=True)

# sidebar settings
with st.sidebar:
    # settings header with a wrench icon instead of the gear emoji
    st.markdown("""
    <div style="display:flex;align-items:center;gap:8px;font-weight:600;font-size:1rem;color:#ddd;margin-bottom:16px">
      <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24"
           fill="none" stroke="#aaa" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
        <circle cx="12" cy="12" r="3"/>
        <path d="M19.07 4.93a10 10 0 0 1 0 14.14M4.93 4.93a10 10 0 0 0 0 14.14
                 M12 2v2m0 16v2M2 12h2m16 0h2"/>
      </svg>
      Settings
    </div>
    """, unsafe_allow_html=True)

    confidence = st.slider("Detection confidence", 0.1, 1.0, 0.25, 0.05)

    st.markdown("---")

    # model info block — plain text, no emojis
    st.markdown("""
    <div style="color:#555;font-size:0.8rem;line-height:1.6">
      <b style="color:#777">Model:</b> YOLOv8n fine-tuned on car damage dataset<br>
      <b style="color:#777">Classes:</b> scratch, dent, crack, broken part, flat tyre, shattered glass<br>
      <b style="color:#777">Dataset:</b> Roboflow Car Damage Detection
    </div>
    """, unsafe_allow_html=True)


# cache the model so it only loads once per session
@st.cache_resource
def get_model():
    return load_model()


with st.spinner("Loading model..."):
    model = get_model()

# two-column layout: left for upload, right for results
col_input, col_output = st.columns([1, 1])

with col_input:
    st.markdown('<div style="color:#888;font-size:0.82rem;margin-bottom:8px">INPUT</div>',
                unsafe_allow_html=True)
    source = st.radio("Source", ["Photo", "Video"], horizontal=True)

    if source == "Photo":
        uploaded = st.file_uploader("Upload car photo", type=["jpg", "jpeg", "png", "webp"])
    else:
        uploaded = st.file_uploader("Upload video clip", type=["mp4", "mov", "avi"])

with col_output:
    st.markdown('<div style="color:#888;font-size:0.82rem;margin-bottom:8px">DETECTION RESULT</div>',
                unsafe_allow_html=True)
    result_placeholder = st.empty()
    report_placeholder = st.empty()


# severity level -> accent color
SEVERITY_COLORS = {"Low": "#4ade80", "Medium": "#fb923c", "High": "#f87171"}


def render_report(detections: list, severity: dict):
    """Build and display the damage report card below the annotated image."""

    if not detections:
        report_placeholder.markdown(
            '<div style="color:#555;font-size:0.85rem;margin-top:12px">No damage detected.</div>',
            unsafe_allow_html=True,
        )
        return

    level  = severity["level"]
    score  = severity["score"]
    accent = SEVERITY_COLORS.get(level, "#aaa")

    # severity banner with left accent border
    banner = (
        f'<div style="background:#1e1e1e;border:1px solid #2e2e2e;border-left:3px solid {accent};'
        f'border-radius:8px;padding:14px 18px;margin-bottom:12px">'
        f'<div style="color:#888;font-size:0.78rem;margin-bottom:2px">Overall Severity</div>'
        f'<div style="color:{accent};font-size:1.5rem;font-weight:700">{level}</div>'
        f'<div style="color:#555;font-size:0.78rem">Damage score: {score:.1f} / 10</div>'
        f'</div>'
    )

    # count detections per label and sort by frequency
    seen = {}
    for d in detections:
        label = d["label"]
        seen[label] = seen.get(label, 0) + 1

    rows = []
    for label, count in sorted(seen.items(), key=lambda x: -x[1]):
        rows.append(
            f'<div style="display:flex;align-items:center;gap:10px;padding:7px 0;border-bottom:1px solid #1e1e1e">'
            f'<span style="width:8px;height:8px;border-radius:50%;background:{accent};display:inline-block"></span>'
            f'<span style="flex:1;color:#ddd;font-size:0.88rem">{label.replace("-", " ").title()}</span>'
            f'<span style="color:#555;font-size:0.8rem">&times;{count}</span>'
            f'</div>'
        )

    html = (
        banner
        + '<div style="color:#666;font-size:0.75rem;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:6px">Detected damage</div>'
        + "".join(rows)
    )
    report_placeholder.markdown(html, unsafe_allow_html=True)


if uploaded is not None:
    if source == "Photo":
        img    = Image.open(uploaded).convert("RGB")
        img_np = np.array(img)

        with st.spinner("Running detection..."):
            annotated, detections = run_detection(model, img_np, confidence)

        result_placeholder.image(annotated, channels="RGB", use_container_width=True)
        severity = compute_severity(detections)
        render_report(detections, severity)

    else:
        # for video we process every 3rd frame for speed, stream annotated frames live
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded.read())
        tfile.close()

        cap = cv2.VideoCapture(tfile.name)
        stframe       = result_placeholder.empty()
        all_detections = []
        frame_count    = 0

        with st.spinner("Processing video..."):
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame_count += 1
                if frame_count % 3 != 0:
                    continue

                frame_rgb             = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                annotated, detections = run_detection(model, frame_rgb, confidence)
                all_detections.extend(detections)
                stframe.image(annotated, channels="RGB", use_container_width=True)

        cap.release()
        os.unlink(tfile.name)

        severity = compute_severity(all_detections)
        render_report(all_detections, severity)
