import cv2
import face_recognition
import numpy as np
import os
import time
from datetime import datetime

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

    with open("attendance.csv", "r+") as f:

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

print("Encoding Complete")

cap = cv2.VideoCapture(0)

cap.set(3, 640)
cap.set(4, 480)

process_this_frame = True

face_locations = []
face_names = []

# FPS counter
prev_time = 0

while True:

    success, img = cap.read()

    if not success:
        break

    small_img = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    small_img = cv2.cvtColor(small_img, cv2.COLOR_BGR2RGB)

    if process_this_frame:

        facesCurFrame = face_recognition.face_locations(small_img, model="hog")
        encodesCurFrame = face_recognition.face_encodings(small_img, facesCurFrame)

        face_locations = facesCurFrame
        face_names = []

        for encodeFace in encodesCurFrame:

            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

            matchIndex = np.argmin(faceDis)

            name = "UNKNOWN"

            if matches[matchIndex]:
                name = classNames[matchIndex].upper()
                markAttendance(name)

            face_names.append(name)

    process_this_frame = not process_this_frame

    for (faceLoc, name) in zip(face_locations, face_names):

        y1, x2, y2, x1 = faceLoc
        y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4

        cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.rectangle(img, (x1, y2-35), (x2, y2), (0,255,0), cv2.FILLED)

        cv2.putText(img, name, (x1+6, y2-6),
                    cv2.FONT_HERSHEY_COMPLEX, 1,
                    (255,255,255), 2)

    # Hitung FPS
    current_time = time.time()
    fps = 1/(current_time - prev_time)
    prev_time = current_time

    cv2.putText(img, f'FPS: {int(fps)}', (20,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (0,255,0), 2)

    cv2.imshow("Face Recognition Attendance", img)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()