import streamlit as st
import torch
from PIL import Image

# Function to load the YOLOv5 model from a local file path
@st.cache(allow_output_mutation=True)
def load_model():
    model = torch.hub.load('ultralytics/yolov5', 'custom', path="weights/best.pt", force_reload=True)
    return model

# Load the YOLOv5 model
model = load_model()

# Streamlit app layout
st.title('Fire Detection')

# Function to detect fire in an image
def detect_fire(image):
    results = model(image)  # Perform detection
    return results

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Detect fire in the uploaded image
    results = detect_fire(image)

    # Display detection results
    st.subheader('Detection Results:')
    st.write(results)

# Example of detecting fire in a sample image
demo_img = "fire.9.png"
demo_image = Image.open(demo_img)

if st.button('Detect Fire in Sample Image'):
    st.image(demo_image, caption='Sample Image', use_column_width=True)
    results = detect_fire(demo_image)
    st.subheader('Detection Results:')
    st.write(results)
