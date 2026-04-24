# Met Museum Artwork Detector

[![Python 3.9](https://img.shields.io/badge/Python-3.9-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.46-red.svg)](https://streamlit.io/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-brightgreen.svg)](https://ultralytics.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Live Demo](https://img.shields.io/badge/Demo-Render.com-4D97EA)](https://met-artwork-detector.onrender.com)

An end-to-end computer vision application that uses a **custom-trained YOLOv8** model to detect artworks in museum images. Upload any image and the model returns annotated bounding boxes, confidence scores, inference timing, and a downloadable JSON result — all in a live web app.

**→ [Try the live demo](https://met-artwork-detector.onrender.com)**

---

## Features

| Feature | Detail |
|---|---|
| Custom object detection | YOLOv8n fine-tuned on Met Museum Open Access images |
| Interactive controls | Adjust confidence and IoU thresholds in real time |
| Side-by-side view | Original and annotated images rendered together |
| Performance metrics | Inference time, detection count, and image resolution |
| Export | Download structured detection results as JSON |
| Cloud deployment | Auto-deploys to Render.com on every push to `main` |

---

## How It Works

```
Upload image
     │
     ▼
PIL decode + RGB normalise
     │
     ▼
YOLOv8n forward pass  ←── confidence & IoU thresholds
     │
     ▼
Bounding-box overlay + confidence scores
     │
     ▼
Metrics display + JSON export
```

The model predicts axis-aligned bounding boxes over the `artwork` class. Non-maximum suppression (NMS) filters duplicate boxes using the configurable IoU threshold.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Computer vision | YOLOv8 (Ultralytics 8.0.196), Pillow 10.3 |
| Deep learning | PyTorch 2.3.1, torchvision 0.18.1 |
| Web UI | Streamlit 1.46.1 |
| Training environment | Google Colab (NVIDIA T4 GPU) |
| Hosting | Render.com (web service, auto-deploy) |
| Version control | Git / GitHub |

---

## Project Structure

```
met-artwork-detector/
├── streamlit_app.py      # Streamlit UI and inference pipeline
├── best.pt               # Fine-tuned YOLOv8n weights (6 MB)
├── requirements.txt      # Pinned runtime dependencies
└── README.md
```

---

## Local Setup

```bash
git clone https://github.com/TejaReddy1402/met-artwork-detector.git
cd met-artwork-detector

# (Optional) create a virtual environment
conda create --name artwork-detector python=3.9
conda activate artwork-detector

pip install -r requirements.txt
streamlit run streamlit_app.py
```

App opens at `http://localhost:8501`.

---

## Model Training

Training was done in Google Colab using free GPU acceleration.

| Setting | Value |
|---|---|
| Base model | `yolov8n.pt` — pretrained on COCO |
| Dataset | Met Museum Open Access — artwork images |
| Epochs | 50 |
| Train / Val split | Standard YOLO layout (`train/` + `val/`) |
| Label format | YOLO `.txt` annotations with `artwork_dataset.yaml` config |
| Hardware | Google Colab · NVIDIA T4 GPU |
| Output | `best.pt` — best validation checkpoint |

The fine-tuning pipeline:

1. Mount Google Drive for persistent storage
2. Prepare `train/images`, `train/labels`, `val/images`, `val/labels`
3. Run `model.train(data="artwork_dataset.yaml", epochs=50)`
4. Pull `runs/detect/train/weights/best.pt`

---

## Deployment (Render.com)

| Config | Value |
|---|---|
| Build command | `pip install -r requirements.txt` |
| Start command | `streamlit run streamlit_app.py --server.port $PORT --server.address 0.0.0.0` |
| Python version | `3.9.19` (set as `PYTHON_VERSION` env var) |
| Trigger | Auto-deploy on push to `main` |

---

## License

MIT — see [LICENSE](LICENSE).
