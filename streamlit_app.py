# Import necessary libraries
import streamlit as st
from PIL import Image
from ultralytics import YOLO
import io
import sys
import traceback # Import the traceback module for detailed error logging

# --- App Configuration ---
# Set the page title and icon
st.set_page_config(
    page_title="Met Museum Artwork Detector",
    page_icon="🖼️",
    layout="wide"
)

# Set the title for the main page
st.title("🖼️ My Personal Met Museum Artwork Detector")
st.markdown(
    """
    This Streamlit application uses a YOLOv8 model to detect artworks in images.
    Upload an image, and the model will highlight the artworks it finds.
    """
)

# --- Sidebar for user settings ---
st.sidebar.header("Model Settings")

# Load the trained model.
# This assumes 'best.pt' is in the same directory as this script.
try:
    model = YOLO("best.pt")
    st.sidebar.success("Model loaded successfully!")
except Exception as e:
    # Print a user-friendly error message
    st.sidebar.error(f"Error loading model: {e}")
    # Print the full traceback to the Streamlit logs for debugging
    st.error("Full traceback for model loading error:")
    st.code(traceback.format_exc())
    # This will stop the app from running further, which is intended in this case
    st.stop()

# Create sliders for confidence and IoU thresholds
confidence = st.sidebar.slider(
    "Confidence Threshold", min_value=0.0, max_value=1.0, value=0.25, step=0.05
)
iou = st.sidebar.slider(
    "IoU Threshold", min_value=0.0, max_value=1.0, value=0.45, step=0.05
)

# --- Main Content Area ---
# Create a file uploader widget
uploaded_file = st.file_uploader(
    "Choose an image...", type=["jpg", "jpeg", "png"]
)

# Process the image if one is uploaded
if uploaded_file is not None:
    try:
        # Display the uploaded image
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        st.write("")
        st.write("Detecting...")

        # Open the image using Pillow and run the model
        image = Image.open(uploaded_file)

        # Convert image to bytes
        img_bytes = io.BytesIO()
        image.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        # Perform detection with the specified thresholds
        results = model.predict(
            source=img_bytes,
            conf=confidence,
            iou=iou,
            save=False  # Do not save the results to disk
        )

        # Get the detected image with bounding boxes
        if results and results[0].boxes:
            detected_image = Image.fromarray(results[0].plot()[:, :, ::-1])
            st.image(detected_image, caption="Artwork Detected!", use_column_width=True)
        else:
            st.warning("No artworks were detected in the image with the current settings.")
            st.image(image, caption="Uploaded Image", use_column_width=True)

        st.success("Detection complete!")
    except Exception as e:
        # Catch and log any errors that happen during the detection process
        st.error(f"An unexpected error occurred during inference: {e}")
        st.error("Full traceback for inference error:")
        st.code(traceback.format_exc())

