import streamlit as st
import cv2
import face_recognition
import numpy as np
import pandas as pd
import os
from datetime import datetime
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av

st.title("Face Recognition Attendance System")

# ================= LOAD DATASET =================

path = "dataset"

images = []
classNames = []

myList = os.listdir(path)

for cl in myList:
    curImg = cv2.imread(f"{path}/{cl}")
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])

def findEncodings(images):

    encodeList = []

    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(img)

        if len(encodings) > 0:
            encodeList.append(encodings[0])

    return encodeList


def markAttendance(name):


    with open("attendance.csv","a+") as f:

        f.seek(0)
        myDataList = f.readlines()
        nameList = []

        for line in myDataList:
            entry = line.split(",")
            nameList.append(entry[0])

        if name not in nameList:

            now = datetime.now()
            dtString = now.strftime("%H:%M:%S")

            f.writelines(f"\n{name},{dtString}")


encodeListKnown = findEncodings(images)

# ================= VIDEO PROCESSOR =================

class FaceRecognitionProcessor(VideoProcessorBase):

    def recv(self, frame):

        img = frame.to_ndarray(format="bgr24")

        small_img = cv2.resize(img,(0,0),None,0.25,0.25)
        small_img = cv2.cvtColor(small_img,cv2.COLOR_BGR2RGB)

        facesCurFrame = face_recognition.face_locations(small_img,model="hog")
        encodesCurFrame = face_recognition.face_encodings(small_img,facesCurFrame)

        for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):

            matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)

            matchIndex = np.argmin(faceDis)

            name = "UNKNOWN"

            if matches[matchIndex]:

                name = classNames[matchIndex].upper()
                markAttendance(name)

            y1,x2,y2,x1 = faceLoc
            y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4

            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)

            cv2.putText(img,name,(x1+6,y2-6),
                        cv2.FONT_HERSHEY_COMPLEX,1,
                        (255,255,255),2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ================= MENU =================

menu = ["Home","Attendance Log"]
choice = st.sidebar.selectbox("Menu",menu)

if choice == "Home":

    st.subheader("Real-Time Face Recognition")

    webrtc_streamer(
        key="face-recognition",
        video_processor_factory=FaceRecognitionProcessor
    )


elif choice == "Attendance Log":

    st.subheader("Attendance Log")

    if os.path.exists("attendance.csv"):

        df = pd.read_csv("attendance.csv")
        st.dataframe(df)

    else:
        st.write("Belum ada data attendance")

