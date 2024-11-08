import streamlit as st
import torch
import cv2
import tempfile
import numpy as np
from pathlib import Path

# Load YOLOv5 model (ensure 'best.pt' is in the same repo or provide the correct path)
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)

# Streamlit app UI
st.title('Tennis Player Detection App')
st.write('Upload a tennis video to detect players in real-time.')

# File uploader for video input
uploaded_video = st.file_uploader("Choose a video file...", type=["mp4", "avi", "mov"])

if uploaded_video is not None:
    # Save the uploaded video to a temporary file
    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    temp_video.write(uploaded_video.read())
    temp_video.close()

    # Open video capture
    cap = cv2.VideoCapture(temp_video.name)
    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Perform detection
        results = model(frame)
        frame = np.squeeze(results.render())  # Draw the detection boxes on the frame

        # Convert BGR to RGB for display
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Display the frame in Streamlit
        stframe.image(frame, channels='RGB', use_column_width=True)

    cap.release()
    st.success('Video processing complete!')

st.write("Ensure 'best.pt' is present in the repository to load the model.")
