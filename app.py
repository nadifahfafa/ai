import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile  # very important for video upload

st.title("🚨 Real-Time Human and Animal Detection")

# Load YOLOv8 model
model = YOLO('yolov8s.pt')

# Choose input source
option = st.selectbox(
    "Select input source:",
    ("Webcam", "Upload a Video")
)

# Labels (optional, for your info)
labels = ['person', 'dog', 'cat', 'cow', 'horse', 'sheep', 'bird']

if option == "Webcam":
    cap = cv2.VideoCapture(0)
    stframe = st.empty()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, verbose=False)

        for result in results:
            plotted_img = result.plot()

        stframe.image(plotted_img, channels="RGB")

elif option == "Upload a Video":
    uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "mov", "avi"])

    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())

        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model.predict(frame, verbose=False)

            for result in results:
                plotted_img = result.plot()

            stframe.image(plotted_img, channels="RGB")

        cap.release()
