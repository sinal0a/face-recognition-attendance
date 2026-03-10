import cv2
import os
import time

name = input("Masukkan nama: ")

dataset_path = "dataset"

cap = cv2.VideoCapture(0)

cap.set(3,640)
cap.set(4,480)

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

count = 0
max_images = 20
last_capture_time = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    for (x,y,w,h) in faces:

        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

        face_crop = frame[y:y+h, x:x+w]

        current_time = time.time()

        if current_time - last_capture_time > 1 and count < max_images:

            filename = f"{name}_{count}.jpg"
            filepath = os.path.join(dataset_path, filename)

            cv2.imwrite(filepath, face_crop)

            print(f"Captured {filename}")

            count += 1
            last_capture_time = current_time

    cv2.putText(
        frame,
        f"Images: {count}/{max_images}",
        (20,40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0),
        2
    )

    cv2.imshow("Smart Face Registration", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if count >= max_images:
        break

cap.release()
cv2.destroyAllWindows()

print("Dataset berhasil dibuat!")