import streamlit as st
import cv2
import os
from datetime import datetime
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

st.title("AI Face Attendance System")

# create dataset folder
if not os.path.exists("dataset"):
    os.makedirs("dataset")

# create attendance file
if not os.path.exists("attendance.csv"):
    with open("attendance.csv","w") as f:
        f.write("Name,Time\n")


face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


# attendance function
def mark_attendance(name):
    with open("attendance.csv","a") as f:
        time_now = datetime.now().strftime("%H:%M:%S")
        f.write(f"{name},{time_now}\n")


mode = st.sidebar.selectbox(
    "Menu",
    ["Face Detection","Register Face","Attendance Log"]
)


# =====================
# FACE DETECTION
# =====================

if mode == "Face Detection":

    class FaceDetection(VideoTransformerBase):

        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(gray,1.3,5)

            for (x,y,w,h) in faces:
                cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

            return img

    webrtc_streamer(
        key="face-detect",
        video_transformer_factory=FaceDetection
    )


# =====================
# REGISTER FACE
# =====================

elif mode == "Register Face":

    name = st.text_input("Enter Name")

    capture = st.button("Capture Face")

    class RegisterFace(VideoTransformerBase):

        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")

            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(gray,1.3,5)

            for (x,y,w,h) in faces:

                cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

                if capture and name!="":

                    face_img = img[y:y+h,x:x+w]

                    file_path = f"dataset/{name}.jpg"

                    cv2.imwrite(file_path,face_img)

                    mark_attendance(name)

            return img

    webrtc_streamer(
        key="register",
        video_transformer_factory=RegisterFace
    )


# =====================
# ATTENDANCE LOG
# =====================

elif mode == "Attendance Log":

    st.subheader("Attendance Log")

    if os.path.exists("attendance.csv"):
        st.dataframe(
            st.session_state.get("attendance",None)
        )

        import pandas as pd
        df = pd.read_csv("attendance.csv")
        st.dataframe(df)