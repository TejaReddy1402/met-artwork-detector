# AI Artwork Detector & Museum Guide 🖼️

This project is an end-to-end computer vision application that uses a custom-trained YOLOv8 model to detect and identify artworks in images. The model was trained on a curated dataset of museum pieces and is deployed as an interactive web application using Streamlit and Render.com.

## 🚀 Live Demo

**You can access the live application here:** https://met-artwork-detector.onrender.com

## ✨ Key Features

* **Custom Object Detection:** A fine-tuned YOLOv8 model trained to recognize specific artworks.
* **Interactive Web Interface:** A user-friendly app built with Streamlit that allows for easy image uploads and real-time detection.
* **Adjustable Thresholds:** Users can dynamically change the confidence and IoU thresholds to see how they affect the model's predictions.
* **Cloud-Native Deployment:** The entire application is containerized and deployed as a web service on Render.com.
* **GPU-Accelerated Training:** The model was trained in Google Colab to leverage free GPU resources for efficient fine-tuning.

## 🛠️ Tech Stack

* **Computer Vision:** YOLOv8, Ultralytics, Pillow
* **ML & Data Science:** PyTorch, torchvision
* **Web Framework:** Streamlit
* **Cloud Platform:** Render.com (Deployment), Google Colab (Training)
* **Environment:** Git, Conda

## 📂 Project Structure

A typical structure for deploying this application would be:

```

/met-artwork-detector
|
|-- best.pt                 \# The trained YOLOv8 model weights
|-- streamlit\_app.py       \# The main Streamlit web application script
|-- requirements.txt        \# Project dependencies for deployment
|-- README.md               \# This file

````

## ⚙️ Setup and Local Installation

To run this project on your local machine, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/TejaReddy1402/met-artwork-detector.git](https://github.com/TejaReddy1402/met-artwork-detector.git)
    cd met-artwork-detector
    ```

2.  **Set up the Python environment:**
    ```bash
    # (Optional but recommended) Create a Conda environment
    conda create --name museum_app python=3.9
    conda activate museum_app
    ```

3.  **Install dependencies:**
    ```bash
    python -m pip install -r requirements.txt
    ```

4.  **Launch the App:**
    ```bash
    streamlit run streamlit_app.py
    ```

## Journey & Development Process

This project was built in three main phases: data preparation, model training, and cloud deployment.

### 1. Data Preparation & Organization

The foundation of the project was a well-organized dataset. Images and their corresponding YOLO-formatted labels were structured into `train` and `validation` sets. A `artwork_dataset.yaml` file was created to define the dataset paths and class names, serving as the master configuration for training.

### 2. Model Training (YOLOv8 in Google Colab)

To accelerate the training process, Google Colab was used for its free GPU access.

* **Environment Setup:** Google Drive was mounted to the Colab notebook to provide persistent storage for the dataset and trained model weights.
* **Fine-Tuning:** The Ultralytics library was used to fine-tune a pre-trained `yolov8n.pt` (nano) model on the custom artwork dataset for 50 epochs.
* **Model Artifact:** Upon completion, the best-performing model weights (`best.pt`) were automatically saved to Google Drive for later use in the deployment phase.

### 3. Deployment to the Cloud (Render.com)

The final phase involved deploying the trained model as a public web service using Streamlit and Render.com.

* **Local Preparation:** The `best.pt` model file, a `streamlit_app.py` script for the UI, and a `requirements.txt` file were prepared locally.
* **GitHub Push:** The project files were pushed to a public GitHub repository.
* **Render Service Configuration:** A new Web Service was created on Render.com and linked to the GitHub repository. Key configuration steps included:
    * **Build Command:** `pip install -r requirements.txt`
    * **Start Command:** `streamlit run streamlit_app.py --server.port $PORT --server.address 0.0.0.0`
    * **Environment:** A specific Python version (`3.9.19`) was set as an environment variable to ensure dependency compatibility.
* **Go-Live:** Render automatically pulled the code, built the environment, and deployed the application, making it accessible via a public URL.
