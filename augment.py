import cv2
import os
import numpy as np

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def blur_faces(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        img[y:y+h, x:x+w] = cv2.GaussianBlur(img[y:y+h, x:x+w], (45, 45), 30)
    return img

def add_fog(img):
    fog = np.ones_like(img) * 200
    return cv2.addWeighted(img, 0.7, fog, 0.3, 0)

def process_folder(folder):
    for file in os.listdir(folder):
        if file.endswith(".jpg") or file.endswith(".png"):
            path = os.path.join(folder, file)
            img = cv2.imread(path)
            if img is None:
                continue
            img = blur_faces(img)
            img = add_fog(img)
            cv2.imwrite(path, img)

process_folder("clients/client1/images")
process_folder("clients/client2/images")

print("Privacy + Weather Augmentation Done")
