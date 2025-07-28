import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import io

# --- Custom CSS for Background Image ---
# IMPORTANT: This MUST be a publicly accessible URL, not a local file path.
# Example: If you upload Background.png to a folder named 'assets' in your GitHub repo,
# the URL might look like: "https://raw.githubusercontent.com/TejaReddy1402/met-artwork-detector/main/assets/Background.png"
# For now, using a placeholder:
background_image_url = r"https://github.com/TejaReddy1402/met-artwork-detector/blob/main/Background.png"

st.markdown(
    # The 'r' here is important to handle any potential backslashes in the CSS itself,
    # though less common for URLs. It's good practice.
    rf"""
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
st.title("🖼️ My Personal Met Museum Artwork Detector")
# ... rest of your streamlit_app.py code ...
