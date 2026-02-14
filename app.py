import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image

st.title("Object Detector")

# Load YOLO model
model = YOLO("yolov8n.pt")

# File uploader for images
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # Open image with PIL
    image = Image.open(uploaded_file)

    # Ensure image is in RGB
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Convert to numpy array for YOLO
    frame = np.array(image)

    # Run YOLO detection
    results = model(frame)
    boxes = results[0].boxes

    # Count all objects
    counts = {}
    for box in boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        counts[label] = counts.get(label, 0) + 1

    # Annotate image with detected boxes
    annotated_frame = results[0].plot()  # Returns a numpy array

    # Display annotated image
    st.image(annotated_frame, channels="BGR")

    # Display all counts
    for obj, cnt in counts.items():
        st.success(f"Detected {cnt} {obj}(s)")