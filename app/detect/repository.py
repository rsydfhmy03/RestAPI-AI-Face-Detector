import numpy as np
import cv2
from keras.models import load_model
from io import BytesIO
from PIL import Image
from app.base.repository import BaseRepository
from app.utils.image_processing import detect_and_crop_face
from app.utils.lbp import apply_lbp
from app.config import Config

class OriginalDetectRepository(BaseRepository):
    def __init__(self):
        self.model = self.load_model()  # panggil abstract method yang kita override

    def load_model(self):  # ini override abstract method BaseRepository
        return load_model(Config.MODEL_ORI_PATH)
    
    # def predict(self, image_bytes: bytes, use_lbp: bool = False):
    #     # Convert to array
    #     image = Image.open(BytesIO(image_bytes)).convert("RGB")
    #     image_np = np.array(image)

    #     # Crop wajah
    #     face = detect_and_crop_face(image_np)
    #     if face is None:
    #         return None

    #     if use_lbp:
    #         face = apply_lbp(face)

    #     # Resize
    #     face_resized = cv2.resize(face, (224, 224))
    #     # Pastikan float32 dan normalisasi
    #     face_normalized = face_resized.astype(np.float32) / 255.0

    #     # Bentuk input tensor sesuai channel
    #     if use_lbp:
    #         face_normalized = face_normalized.reshape(1, 224, 224, 1)  # Grayscale
    #     else:
    #         face_normalized = face_normalized.reshape(1, 224, 224, 3)  # RGB

    #     self.class_names = ['FAKE', 'REAL']  # hasil dari le.classes_
    #     prediction = self.model.predict(face_normalized, verbose=0)[0]
    #     pred_class = np.argmax(prediction)
    #     label = self.class_names[pred_class]
    #     confidence = float(prediction[pred_class])

    #     return {
    #         "label": label,
    #         "confidence": round(confidence, 4),
    #         "probabilities": {
    #             self.class_names[0]: float(prediction[0]),
    #             self.class_names[1]: float(prediction[1])
    #         }
    #     }
    def predict(self, image_bytes: bytes, use_lbp: bool = False):
    # Convert Bytes ke array (tanpa convert RGB)
        image = Image.open(BytesIO(image_bytes))
        image_np = np.array(image)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)  # Samakan ke BGR

        img_resized = cv2.resize(image_bgr, (224, 224))
        input_tensor = np.expand_dims(img_resized, axis=0)  # shape (1, 224, 224, 3)
        self.class_names = ['FAKE', 'REAL'] 
        prediction = self.model.predict(input_tensor, verbose=0)[0]
        pred_class = np.argmax(prediction)
        label = self.class_names[pred_class]
        confidence = float(prediction[pred_class])
        
        return {
            "label": label,
            "confidence": round(confidence, 4),
            "probabilities": {
                self.class_names[0]: float(prediction[0]),
                self.class_names[1]: float(prediction[1])
            }
        }




class LBPDetectRepository(OriginalDetectRepository):
    def load_model(self):  # override method dari parent
        return load_model(Config.MODEL_LBP_PATH)