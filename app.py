import streamlit as st
import cv2
import os
import numpy as np
import pandas as pd
from datetime import datetime
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

st.set_page_config(page_title="AI Face Attendance",layout="wide")

st.title("🚀 AI Face Recognition Attendance System")

# create folders
if not os.path.exists("dataset"):
    os.makedirs("dataset")

if not os.path.exists("attendance.csv"):
    with open("attendance.csv","w") as f:
        f.write("Name,Time\n")

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades+"haarcascade_frontalface_default.xml"
)

# =========================
# LOAD DATASET
# =========================

def load_faces():

    encodings=[]
    names=[]

    for file in os.listdir("dataset"):

        img=cv2.imread(f"dataset/{file}")

        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        hist=cv2.calcHist([gray],[0],None,[256],[0,256])
        hist=cv2.normalize(hist,hist).flatten()

        encodings.append(hist)
        names.append(file.split(".")[0])

    return encodings,names


# =========================
# ATTENDANCE
# =========================

def mark_attendance(name):

    with open("attendance.csv","a") as f:

        now=datetime.now().strftime("%H:%M:%S")

        f.write(f"{name},{now}\n")


menu=st.sidebar.selectbox(
    "Menu",
    ["Register Face","Face Recognition","Dashboard"]
)

# =========================
# REGISTER
# =========================

if menu=="Register Face":

    st.subheader("Register New Face")

    name=st.text_input("Enter Name")

    capture=st.button("Capture Face")

    class Register(VideoTransformerBase):

        def transform(self,frame):

            img=frame.to_ndarray(format="bgr24")

            gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

            faces=face_cascade.detectMultiScale(gray,1.3,5)

            for (x,y,w,h) in faces:

                cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

                if capture and name!="":

                    face=img[y:y+h,x:x+w]

                    cv2.imwrite(f"dataset/{name}.jpg",face)

            return img

    webrtc_streamer(
        key="register",
        video_transformer_factory=Register
    )


# =========================
# FACE RECOGNITION
# =========================

elif menu=="Face Recognition":

    st.subheader("Face Recognition")

    encodings,names=load_faces()

    class Recognition(VideoTransformerBase):

        def transform(self,frame):

            img=frame.to_ndarray(format="bgr24")

            gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

            faces=face_cascade.detectMultiScale(gray,1.3,5)

            for (x,y,w,h) in faces:

                face=gray[y:y+h,x:x+w]

                hist=cv2.calcHist([face],[0],None,[256],[0,256])
                hist=cv2.normalize(hist,hist).flatten()

                scores=[]

                for enc in encodings:

                    score=cv2.compareHist(hist,enc,cv2.HISTCMP_CORREL)

                    scores.append(score)

                if len(scores)>0:

                    best=np.argmax(scores)

                    name=names[best]

                    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

                    cv2.putText(
                        img,
                        name,
                        (x,y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0,255,0),
                        2
                    )

                    mark_attendance(name)

            return img

    webrtc_streamer(
        key="recognition",
        video_transformer_factory=Recognition
    )


# =========================
# DASHBOARD
# =========================

elif menu=="Dashboard":

    st.subheader("Attendance Dashboard")

    df=pd.read_csv("attendance.csv")

    st.dataframe(df)

    if len(df)>0:

        st.subheader("Attendance Count")

        chart=df["Name"].value_counts()

        st.bar_chart(chart)

    st.download_button(
        "Download Attendance CSV",
        df.to_csv(index=False),
        file_name="attendance.csv"
    )