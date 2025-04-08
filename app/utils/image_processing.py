import cv2
import numpy as np
from app.config import Config

def detect_and_crop_face(image: np.ndarray) -> np.ndarray | None:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    face_cascade = cv2.CascadeClassifier(Config.HAARCASCADE_PATH)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        return None  # Wajah tidak ditemukan

    # Ambil wajah pertama yang terdeteksi
    (x, y, w, h) = faces[0]
    cropped_face = image[y:y+h, x:x+w]

    return cropped_face