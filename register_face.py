import cv2
import os

name = input("Masukkan nama: ")

dataset_path = "dataset"

cap = cv2.VideoCapture(0)

count = 0

while True:

    ret, frame = cap.read()

    cv2.imshow("Register Face - Tekan S untuk capture", frame)

    key = cv2.waitKey(1)

    # tekan S untuk mengambil foto
    if key == ord("s"):

        filename = f"{name}_{count}.jpg"

        filepath = os.path.join(dataset_path, filename)

        cv2.imwrite(filepath, frame)

        print(f"Foto tersimpan: {filename}")

        count += 1

    # stop jika sudah 20 foto
    if count >= 20:
        break

    # tekan Q untuk keluar
    if key == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()

print("Dataset selesai dibuat!")