import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import io

# --- Custom CSS for Background Image ---
# Replace 'YOUR_MUSEUM_IMAGE_URL_HERE' with the actual URL of your museum image
# Example placeholder URL: https://placehold.co/1920x1080/f0f0f0/333333/png?text=Museum+Background
# For a real image, find a high-resolution, publicly accessible URL.
# Consider images from Wikimedia Commons or the Met Museum's open access collection if linkable.
background_image_url = "c:\Users\tejak\my-artwork-detector-hf\Background.png" 

st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("{background_image_url}");
        background-size: cover; /* Cover the entire area */
        background-position: center; /* Center the image */
        background-repeat: no-repeat; /* Do not repeat the image */
        background-attachment: fixed; /* Keep the image fixed when scrolling */
    }}
    /* Optional: Add a semi-transparent overlay to improve text readability */
    .stApp::before {{
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-color: rgba(0, 0, 0, 0.3); /* Black overlay with 30% opacity */
        z-index: -1; /* Place behind content */
    }}
    /* Optional: Adjust text color for better contrast on a dark background */
    body {{
        color: #333333; /* Dark text for light background */
    }}
    .stSidebar, .stFileUploader, .stSlider, .stButton button {{
        background-color: rgba(255, 255, 255, 0.8); /* Slightly transparent white for sidebar elements */
        border-radius: 10px;
        padding: 10px;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# --- Your existing Streamlit app code starts here ---
st.title("🖼️ Met Museum Artwork Detector")
# ... rest of your streamlit_app.py code ...import streamlit as st
from ultralytics import YOLO
from PIL import Image
import io
import numpy as np
import os
import sys
import traceback # Import the traceback module for detailed error logging

# --- Configuration ---
# MODEL_PATH is now relative to the app.py file
MODEL_PATH = "best.pt" 

# --- Load Model ---
@st.cache_resource # Cache the model loading to improve performance
def load_yolo_model():
    try:
        model = YOLO(MODEL_PATH)
        st.success("YOLOv8 model loaded successfully!") # Add a success message
        return model
    except Exception as e:
        # Print a user-friendly error message to the Streamlit app
        st.error(f"Error loading model: {e}. Please ensure 'best.pt' is in the same directory and is a valid YOLOv8 model.")
        
        # Print the full traceback to the Streamlit app for debugging
        st.error("Full traceback for model loading error:") 
        st.code(traceback.format_exc()) 
        
        st.stop() # Stop the app if model loading fails

model = load_yolo_model()

# --- Streamlit App Layout ---
st.set_page_config(
    page_title="Met Museum Artwork Detector",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🖼️ Met Museum Artwork Detector")
st.subheader("Upload an image and detect artworks using your custom YOLOv8 model.")

# Sidebar for options
st.sidebar.header("Configuration")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)
iou_threshold = st.sidebar.slider("IoU Threshold (NMS)", 0.0, 1.0, 0.7, 0.05)

st.sidebar.markdown("---")
st.sidebar.info("This app uses a YOLOv8n model trained on a subset of the Met Museum Open Access collection to detect 'artwork' objects.")

# Main content area
st.write("### Upload an Image")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image_bytes = uploaded_file.read()
    image = Image.open(io.BytesIO(image_bytes))

    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("")
    st.write("Detecting objects...")

    img_np = np.array(image)

    # Initialize current_boxes to None for safety, before the try block
    current_boxes = None 

    try:
        # Perform inference using the loaded model
        inference_results = model(img_np, conf=confidence_threshold, iou=iou_threshold)

        # Ensure we have valid results before proceeding
        if inference_results and len(inference_results) > 0 and inference_results[0] is not None:
            first_result = inference_results[0] 
            
            # Assign current_boxes from the first result's boxes attribute
            current_boxes = first_result.boxes 

            # Plot detections on the image
            annotated_image = first_result.plot()
            st.image(annotated_image, caption="Image with Detections", use_column_width=True)

            st.write("### Detection Details:")
            
            # Display details only if detections exist
            if current_boxes is not None and len(current_boxes) > 0:
                st.write(f"Detected {len(current_boxes)} object(s):")
                for i, box in enumerate(current_boxes):
                    conf = box.conf.item()
                    cls = int(box.cls.item())
                    # model.names will contain the class names from your training (e.g., 'artwork')
                    class_name = model.names[cls] 
                    xyxy = box.xyxy.tolist()[0] 
                    st.markdown(f"- **{class_name}** (Confidence: {conf:.2f}) at coordinates: ({int(xyxy[0])},{int(xyxy[1])}) - ({int(xyxy[2])},{int(xyxy[3])})")
            else: 
                st.info("No artworks detected in this image with the current confidence threshold.")
        else:
            st.warning("No valid detection results were returned from the model.")
    except Exception as e:
        st.error(f"An unexpected error occurred during inference: {e}")
        st.error("Full traceback for inference error:")
        st.code(traceback.format_exc())

st.markdown("---")
st.markdown("Developed by Teja Reddy")
