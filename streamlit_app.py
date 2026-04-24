import io
import json
import time
import traceback
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_PATH = Path(__file__).parent / "best.pt"
MAX_FILE_MB = 10

# st.set_page_config must be the very first Streamlit call
st.set_page_config(
    page_title="Met Museum Artwork Detector",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Model loading ──────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading YOLOv8 model…")
def load_model() -> YOLO:
    return YOLO(str(MODEL_PATH))

# ── Inference ──────────────────────────────────────────────────────────────────

def run_inference(
    model: YOLO,
    img: np.ndarray,
    conf: float,
    iou: float,
) -> tuple:
    t0 = time.perf_counter()
    results = model(img, conf=conf, iou=iou)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return results, elapsed_ms


def build_detections(boxes, class_names: dict) -> list[dict]:
    detections = []
    for box in boxes:
        x1, y1, x2, y2 = [int(v) for v in box.xyxy.tolist()[0]]
        detections.append(
            {
                "class": class_names[int(box.cls.item())],
                "confidence": round(float(box.conf.item()), 4),
                "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            }
        )
    return sorted(detections, key=lambda d: d["confidence"], reverse=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────

def render_sidebar(model: YOLO) -> tuple[float, float]:
    st.sidebar.header("Detection Settings")

    conf = st.sidebar.slider(
        "Confidence Threshold", 0.0, 1.0, 0.25, 0.05,
        help="Minimum score for a detection to be shown.",
    )
    iou = st.sidebar.slider(
        "IoU Threshold (NMS)", 0.0, 1.0, 0.70, 0.05,
        help="Higher values keep more overlapping boxes.",
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Model Info")
    classes = ", ".join(v.title() for v in model.names.values())
    st.sidebar.markdown(
        f"**Architecture:** YOLOv8n  \n"
        f"**Classes:** {classes}  \n"
        f"**Training:** 50 epochs · Google Colab T4  \n"
        f"**Dataset:** Met Museum Open Access"
    )

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "📖 [GitHub](https://github.com/TejaReddy1402/met-artwork-detector)  ·  "
        "🌐 [Live Demo](https://met-artwork-detector.onrender.com)"
    )

    return conf, iou

# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    model = load_model()
    conf, iou = render_sidebar(model)

    st.title("🖼️ Met Museum Artwork Detector")
    st.markdown(
        "Upload a museum image to detect and locate artworks using a "
        "custom-trained **YOLOv8** model fine-tuned on the Met Museum Open Access collection."
    )

    uploaded_file = st.file_uploader(
        "Choose a JPG or PNG image",
        type=["jpg", "jpeg", "png"],
    )

    if uploaded_file is None:
        st.info("Upload an image above to get started.")
        return

    if uploaded_file.size / (1024 * 1024) > MAX_FILE_MB:
        st.error(
            f"File is too large ({uploaded_file.size / 1024 / 1024:.1f} MB). "
            f"Limit is {MAX_FILE_MB} MB."
        )
        return

    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)

    with st.spinner("Running inference…"):
        try:
            results, elapsed_ms = run_inference(model, img_np, conf, iou)
        except Exception:
            st.error("Inference failed. See details below.")
            st.code(traceback.format_exc())
            return

    first_result = results[0] if results else None
    boxes = getattr(first_result, "boxes", None)
    n_detected = len(boxes) if boxes is not None else 0

    # ── Metrics row ────────────────────────────────────────────────────────────
    m1, m2, m3 = st.columns(3)
    m1.metric("Artworks Detected", n_detected)
    m2.metric("Inference Time", f"{elapsed_ms:.0f} ms")
    m3.metric("Image Resolution", f"{image.width} × {image.height} px")

    # ── Side-by-side images ────────────────────────────────────────────────────
    col_orig, col_det = st.columns(2)
    with col_orig:
        st.subheader("Original")
        st.image(image, use_column_width=True)
    with col_det:
        st.subheader("Detections")
        annotated = first_result.plot() if first_result is not None else img_np
        st.image(annotated, use_column_width=True)

    # ── Detection details & export ─────────────────────────────────────────────
    if n_detected == 0:
        st.info("No artworks detected. Try lowering the confidence threshold.")
        return

    detections = build_detections(boxes, model.names)

    with st.expander(f"Detection Details ({n_detected} found)", expanded=True):
        for i, d in enumerate(detections, 1):
            b = d["bbox"]
            st.markdown(
                f"**{i}. {d['class'].title()}** — "
                f"confidence `{d['confidence']:.1%}` — "
                f"bbox `({b['x1']}, {b['y1']}) → ({b['x2']}, {b['y2']})`"
            )

    payload = json.dumps(
        {
            "filename": uploaded_file.name,
            "image_size": {"width": image.width, "height": image.height},
            "inference_time_ms": round(elapsed_ms, 2),
            "detections": detections,
        },
        indent=2,
    )
    st.download_button(
        label="⬇ Download results (JSON)",
        data=payload,
        file_name="detections.json",
        mime="application/json",
    )

    st.markdown("---")
    st.caption("Built by Teja Reddy · YOLOv8 · Streamlit · Render.com")


if __name__ == "__main__":
    main()
