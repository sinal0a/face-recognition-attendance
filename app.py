import streamlit as st
import cv2
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av

st.title("Face Recognition Attendance System")

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


class FaceDetection(VideoTransformerBase):

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0), 2)

        return img


webrtc_streamer(
    key="face-detection",
    video_transformer_factory=FaceDetection
)