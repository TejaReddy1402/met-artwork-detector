## ⚙️ My Setup and Training Process

### 1. Google Drive Preparation

My journey began by organizing my data on Google Drive. I created a `SmartMuseumGuide` folder, and within it, a `data` folder to house my dataset. I made sure my images and their corresponding YOLO-formatted `.txt` labels were correctly placed in `images/train`, `images/val`, `labels/train`, and `labels/val` respectively. Crucially, I placed my `artwork_dataset.yaml` file directly in the `data` folder, defining my dataset paths, number of classes, and class names.

### 2. Model Training in Google Colab

Next, I moved to Google Colab for model training, leveraging its GPU capabilities.

1.  **Opening a new Google Colab Notebook:** I started a fresh notebook.
2.  **Mounting Google Drive:** I connected my Google Drive to access my data:
    ```python
    from google.colab import drive
    from pathlib import Path
    import os

    drive.mount('/content/drive')
    GOOGLE_DRIVE_PROJECT_ROOT = Path('/content/drive/My Drive/SmartMuseumGuide')
    DATASET_ROOT = GOOGLE_DRIVE_PROJECT_ROOT / 'data'
    YOLO_RUNS_DIR = GOOGLE_DRIVE_PROJECT_ROOT / 'runs'
    YOLO_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_YAML_FILE = DATASET_ROOT / 'artwork_dataset.yaml'

    print(f"Dataset root: {DATASET_ROOT}")
    print(f"Data YAML file: {DATA_YAML_FILE}")
    ```
    *After running this cell, I authorized Drive access.*

3.  **Installing & Upgrading Ultralytics:** I ensured I had the latest version of Ultralytics:
    ```python
    %pip install ultralytics --upgrade --quiet
    ```
    *I ran this cell to get the necessary libraries.*

4.  **Training my YOLOv8 Model:** This was the core training step. I used a pre-trained `yolov8n.pt` as my base:
    ```python
    from ultralytics import YOLO

    model = YOLO('yolov8n.pt') # Loads pre-trained nano model
    results = model.train(
        data=DATA_YAML_FILE.as_posix(),
        epochs=50,
        imgsz=640,
        batch=-1,
        name='artwork_detection_hf_run1', # I named this specific training run
        project=YOLO_RUNS_DIR.as_posix()
    )
    ```
    *I monitored the training progress, and once completed, my `best.pt` model was saved in `My Drive/SmartMuseumGuide/runs/detect/artwork_detection_hf_run1/weights/`.*

## 🚀 My Deployment to Render.com

Deploying this application was a significant step, and I chose Render.com for its ease of use with Streamlit apps.

### 1. Local File Preparation & GitHub Push

First, I prepared my local project folder and pushed it to my GitHub repository.

1.  **Creating a local project folder:** I set up a new folder on my computer, like `my-artwork-detector-render`.
2.  **Downloading `best.pt`:** I downloaded the `best.pt` file from my Google Drive (from the `runs/detect/artwork_detection_hf_run1/weights/` folder) and placed it directly into my local project folder.
3.  **Creating `streamlit_app.py`:** I created `streamlit_app.py` in my local project folder and pasted my Streamlit application code into it.
4.  **Creating `requirements.txt`:** I created `requirements.txt` with the following essential dependencies, which I refined during the deployment process to ensure compatibility:
    ```
    streamlit==1.46.1
    ultralytics==8.0.196
    pillow==10.3.0
    torch==2.3.1
    torchvision==0.18.1
    ```
5.  **Initializing Git & Pushing to GitHub:** I set up Git in my local project folder and pushed my code to my GitHub repository.
    * I created a new, empty public repository named `met-artwork-detector` under my `TejaReddy1402` account on GitHub.
    * Then, from my terminal, navigating to my local project folder, I ran:
        ```bash
        git init
        git config --global user.email "reddyteja2022@gmail.com"
        git config --global user.name "TejaReddy1402"
        git add .
        git commit -m "Initial commit of Streamlit app and YOLOv8 model"
        git remote add origin [https://github.com/TejaReddy1402/met-artwork-detector.git](https://github.com/TejaReddy1402/met-artwork-detector.git)
        git branch -M main
        git push -u origin main
        ```
    * I verified that `streamlit_app.py`, `best.pt`, and `requirements.txt` were all visible in my GitHub repository.

### 2. Render.com Service Creation

With my code on GitHub, I proceeded to set up the web service on Render.com:

1.  **Signing Up/Logging In to Render:** I went to [render.com](https://render.com/) and logged in using my GitHub account.
2.  **Creating a New Web Service:** From my dashboard, I selected **"New"** -> **"Web Service"**.
3.  **Connecting GitHub:** I authorized Render to access my GitHub repositories and selected my `met-artwork-detector` repository.
4.  **Configuring the Service:** I set up the deployment details as follows:
    * **Name:** `met-artwork-detector-app`
    * **Region:** I chose a region close to me.
    * **Branch:** `main`
    * **Root Directory:** I left this blank as my app files were in the repository root.
    * **Runtime:** Render automatically detected **Python 3**.
    * **Build Command:** `pip install -r requirements.txt`
    * **Start Command:** `streamlit run streamlit_app.py --server.port $PORT --server.address 0.0.0.0`
    * **Instance Type:** I selected `Free` for initial testing.
    * **Environment Variables:** This was a crucial step for compatibility. I added:
        * **Key:** `PYTHON_VERSION`
        * **Value:** `3.9.19` (This helped resolve Python version conflicts during dependency installation.)
5.  **Deploying:** I clicked **"Create Web Service"**.

Render then automatically pulled my code, installed dependencies, and deployed my Streamlit application. I monitored the logs on the Render dashboard closely throughout this process.

## 🌐 How to Use My App

Once my Render service was live (I looked for the "Your service is live 🎉" message on the Render dashboard and the URL), I could access it in my web browser:

**My Live App URL:** `https://met-artwork-detector.onrender.com`

1.  **Upload an Image:** Simply click "Choose an image..." or drag and drop an image file into the designated area.
2.  **View Detections:** The app will process the image and display it with bounding boxes around any detected artworks.
3.  **Adjust Thresholds:** I included sliders in the sidebar that allow you to change the confidence and IoU thresholds to see how they affect the detections.

## 🤝 Contributing

I welcome contributions to this project! Feel free to fork this repository, suggest improvements to the model or UI, or add new features. Pull requests are always appreciated.

## 🙏 Acknowledgements

I'm grateful to the creators of these tools and resources that made this project possible:

* Ultralytics for developing YOLOv8.
* Streamlit for providing such an intuitive web app framework.
* Render.com for offering a seamless hosting platform.
* The Metropolitan Museum of Art for making their Open Access collection available, which was invaluable for training data.