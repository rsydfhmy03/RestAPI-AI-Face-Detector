# import cv2
# import numpy as np
# from app.config import Config

# def detect_and_crop_face(image: np.ndarray) -> np.ndarray | None:
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     face_cascade = cv2.CascadeClassifier(Config.HAARCASCADE_PATH)

#     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

#     if len(faces) == 0:
#         return None  # Wajah tidak ditemukan

#     # Ambil wajah pertama yang terdeteksi
#     (x, y, w, h) = faces[0]
#     cropped_face = image[y:y+h, x:x+w]

#     return cropped_face
import cv2
import numpy as np
from mtcnn import MTCNN  # Tambahin ini
from app.config import Config  # Tetap sama

# Inisialisasi detektor MTCNN sekali
detector = MTCNN()

def detect_and_crop_face(image: np.ndarray) -> np.ndarray | None:
    try:
        # Konversi ke RGB karena MTCNN pakai RGB input
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Deteksi wajah
        result = detector.detect_faces(rgb)

        if len(result) == 0:
            return None  # Wajah tidak ditemukan

        # Ambil bounding box dari wajah pertama
        x, y, w, h = result[0]['box']
        x, y = max(0, x), max(0, y)  # Hindari nilai negatif

        # Crop wajah
        cropped_face = image[y:y+h, x:x+w]

        return cropped_face

    except Exception as e:
        print(f"[MTCNN] Error mendeteksi wajah: {e}")
        return None
