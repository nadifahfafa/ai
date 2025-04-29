import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode

import av
from ultralytics import YOLO
import cv2
import numpy as np

st.title("🚨 Real-Time Human and Animal Detection (Cloud Webcam)")

# Load YOLOv8 model
model = YOLO("yolov8s.pt")

# Set target classes
TARGET_CLASSES = ['person', 'cat', 'dog', 'horse', 'sheep', 'cow']

class YOLOProcessor(VideoProcessorBase):
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        results = model.predict(img, verbose=False)

        boxes = results[0].boxes
        if boxes is not None:
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = model.names[cls_id]

                if label in TARGET_CLASSES and conf > 0.5:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img, f"{label} {conf:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Launch the webcam
webrtc_streamer(
    key="yolo-webcam",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=YOLOProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)
