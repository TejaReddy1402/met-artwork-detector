🖼️ My Met Museum Artwork Detector
I'm excited to share a project I've been working on: a custom object detection model designed to identify artworks within images. I've successfully deployed it as an interactive web application on Render.com, leveraging YOLOv8 for detection and Streamlit for the user interface. My journey involved training this model in Google Colab, managing data on Google Drive, and overcoming various deployment challenges.

✨ Features I Built
Here are the key features I implemented in this application:

Custom Object Detection: I fine-tuned a YOLOv8n model specifically to detect "artwork" objects, tailoring it to my specific needs.

Interactive Web UI: I built the user interface using Streamlit, making it easy for anyone to upload images and see the results.

Real-time Inference: The application processes uploaded images and displays bounding boxes and confidence scores in real-time.

Configurable Thresholds: I included sliders that allow users to adjust the confidence and Intersection over Union (IoU) thresholds, giving them control over detection sensitivity.

Cloud Deployment: I deployed the final application on Render.com, making it publicly accessible for others to use.

🚀 Technologies I Leveraged
Throughout this project, I utilized a range of powerful tools and frameworks:

YOLOv8 (Ultralytics): This was the core object detection framework I chose for its efficiency and accuracy.

PyTorch: As the underlying deep learning library for YOLOv8, PyTorch was fundamental to my model's development.

Streamlit: I used Streamlit to quickly build and deploy the interactive web interface.

Google Colab: This cloud-based environment provided me with the necessary GPU acceleration for model training.

Google Drive: I used Google Drive for storing my dataset and the outputs of my training runs.

Render.com: This platform became my chosen solution for hosting the web service.

Git \& GitHub: These were essential for managing my code, tracking changes, and preparing for deployment.

📁 My Project Structure
I organized my project files across two main environments: Google Drive for the larger training assets, and my local machine (which then synced with GitHub) for the application's deployment.

Google Drive Structure (for Training)
This is how I structured my data and training outputs on Google Drive:

My Drive/

└── SmartMuseumGuide/

├── data/

│   ├── images/

│   │   ├── train/

│   │   └── val/

│   ├── labels/

│   │   ├── train/

│   │   └── val/

│   └── artwork\_dataset.yaml  # My dataset configuration for YOLOv8

├── runs/                     # This is where YOLOv8 saved my training results

│   └── detect/

│       └── artwork\_detection\_hf\_run1/ # My specific training run folder

│           └── weights/

│               └── best.pt   # The trained model weights I used for deployment

└── yolo\_weights/             # I also kept base YOLO models here
Local Project Structure (for Render Deployment)
For deployment, I prepared a clean local folder that I then pushed to GitHub:

my-artwork-detector-hf/ 

├── streamlit\_app.py    # My Streamlit web application code

├── best.pt             # The trained YOLOv8 model weights I downloaded from Drive

└── requirements.txt    # The Python dependencies my Streamlit app needed
⚙️ My Setup and Training Process

1. Google Drive Preparation
   My journey began by organizing my data on Google Drive. I created a SmartMuseumGuide folder, and within it, a data folder to house my dataset. I made sure my images and their corresponding YOLO-formatted .txt labels were correctly placed in images/train, images/val, labels/train, and labels/val respectively. Crucially, I placed my artwork\_dataset.yaml file directly in the data folder, defining my dataset paths, number of classes, and class names.
2. Model Training in Google Colab
   Next, I moved to Google Colab for model training, leveraging its GPU capabilities.

Opening a new Google Colab Notebook: I started a fresh notebook.

Mounting Google Drive: I connected my Google Drive to access my data:

from google.colab import drive
from pathlib import Path
import os

drive.mount('/content/drive')
GOOGLE\_DRIVE\_PROJECT\_ROOT = Path('/content/drive/My Drive/SmartMuseumGuide')
DATASET\_ROOT = GOOGLE\_DRIVE\_PROJECT\_ROOT / 'data'
YOLO\_RUNS\_DIR = GOOGLE\_DRIVE\_PROJECT\_ROOT / 'runs'
YOLO\_RUNS\_DIR.mkdir(parents=True, exist\_ok=True)
DATA\_YAML\_FILE = DATASET\_ROOT / 'artwork\_dataset.yaml'

print(f"Dataset root: {DATASET\_ROOT}")
print(f"Data YAML file: {DATA\_YAML\_FILE}")

After running this cell, I authorized Drive access.

Installing \& Upgrading Ultralytics: I ensured I had the latest version of Ultralytics:

%pip install ultralytics --upgrade --quiet

I ran this cell to get the necessary libraries.

Training my YOLOv8 Model: This was the core training step. I used a pre-trained yolov8n.pt as my base:

from ultralytics import YOLO

model = YOLO('yolov8n.pt') # Loads pre-trained nano model
results = model.train(
data=DATA\_YAML\_FILE.as\_posix(),
epochs=50,
imgsz=640,
batch=-1,
name='artwork\_detection\_hf\_run1', # I named this specific training run
project=YOLO\_RUNS\_DIR.as\_posix()
)

I monitored the training progress, and once completed, my best.pt model was saved in My Drive/SmartMuseumGuide/runs/detect/artwork\_detection\_hf\_run1/weights/.

🚀 My Deployment to Render.com
Deploying this application was a significant step, and I chose Render.com for its ease of use with Streamlit apps.

1. Local File Preparation \& GitHub Push
   First, I prepared my local project folder and pushed it to my GitHub repository.

Creating a local project folder: I set up a new folder on my computer, like my-artwork-detector-render.

Downloading best.pt: I downloaded the best.pt file from my Google Drive (from the runs/detect/artwork\_detection\_hf\_run1/weights/ folder) and placed it directly into my local project folder.

Creating streamlit\_app.py: I created streamlit\_app.py in my local project folder and pasted my Streamlit application code into it.

Creating requirements.txt: I created requirements.txt with the following essential dependencies, which I refined during the deployment process to ensure compatibility:

streamlit==1.46.1
ultralytics==8.0.196
pillow==10.3.0
torch==2.3.1
torchvision==0.18.1

Initializing Git \& Pushing to GitHub: I set up Git in my local project folder and pushed my code to my GitHub repository.

I created a new, empty public repository named met-artwork-detector under my TejaReddy1402 account on GitHub.

Then, from my terminal, navigating to my local project folder, I ran:

git init
git config --global user.email "reddyteja2022@gmail.com"
git config --global user.name "TejaReddy1402"
git add .
git commit -m "Initial commit of Streamlit app and YOLOv8 model"
git remote add origin https://github.com/TejaReddy1402/met-artwork-detector.git
git branch -M main
git push -u origin main

I verified that streamlit\_app.py, best.pt, and requirements.txt were all visible in my GitHub repository.

2. Render.com Service Creation
   With my code on GitHub, I proceeded to set up the web service on Render.com:

Signing Up/Logging In to Render: I went to render.com and logged in using my GitHub account.

Creating a New Web Service: From my dashboard, I selected "New" -> "Web Service".

Connecting GitHub: I authorized Render to access my GitHub repositories and selected my met-artwork-detector repository.

Configuring the Service: I set up the deployment details as follows:

Name: met-artwork-detector-app

Region: I chose a region close to me.

Branch: main

Root Directory: I left this blank as my app files were in the repository root.

Runtime: Render automatically detected Python 3.

Build Command: pip install -r requirements.txt

Start Command: streamlit run streamlit\_app.py --server.port $PORT --server.address 0.0.0.0

Instance Type: I selected Free for initial testing.

Environment Variables: This was a crucial step for compatibility. I added:

Key: PYTHON\_VERSION

Value: 3.9.19 (This helped resolve Python version conflicts during dependency installation.)

Deploying: I clicked "Create Web Service".

Render then automatically pulled my code, installed dependencies, and deployed my Streamlit application. I monitored the logs on the Render dashboard closely throughout this process.

🌐 How to Use My App
Once my Render service was live (I looked for the "Your service is live 🎉" message on the Render dashboard and the URL), I could access it in my web browser:

My Live App URL: https://met-artwork-detector.onrender.com

Upload an Image: Simply click "Choose an image..." or drag and drop an image file into the designated area.

View Detections: The app will process the image and display it with bounding boxes around any detected artworks.

Adjust Thresholds: I included sliders in the sidebar that allow you to change the confidence and IoU thresholds to see how they affect the detections.

🤝 Contributing
I welcome contributions to this project! Feel free to fork this repository, suggest improvements to the model or UI, or add new features. Pull requests are always appreciated.

🙏 Acknowledgements
I'm grateful to the creators of these tools and resources that made this project possible:

Ultralytics for developing YOLOv8.

Streamlit for providing such an intuitive web app framework.

Render.com for offering a seamless hosting platform.

The Metropolitan Museum of Art for making their Open Access collection available, which was invaluable for training data.

