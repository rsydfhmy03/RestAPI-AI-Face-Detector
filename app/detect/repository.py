import numpy as np
import cv2
from tensorflow.keras.models import load_model
from io import BytesIO
from PIL import Image
from app.base.repository import BaseRepository
from app.utils.image_processing import detect_and_crop_face
from app.utils.lbp import apply_lbp
from app.config import Config
from app.utils.storage import upload_to_gcs
import uuid

# Helper function for image processing
def preprocess_image(image_bytes: bytes, use_lbp: bool = False):
    """Process image for face detection and resizing"""
    # Load image from bytes
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    image_np = np.array(image)

    # Convert to BGR for OpenCV
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # Detect and crop face
    cropped_face = detect_and_crop_face(image_bgr)
    if cropped_face is None:
        return None

    # Apply LBP if needed
    if use_lbp:
        cropped_face = apply_lbp(cropped_face)
        cropped_face = cv2.cvtColor(cropped_face, cv2.COLOR_GRAY2BGR)

    # Resize image and prepare for prediction
    img_resized = cv2.resize(cropped_face, (224, 224))
    return img_resized

# Helper function to upload image to GCS
def save_image_to_gcs(image_bytes: bytes):
    """Save image to GCS and return the URL"""
    image_id = str(uuid.uuid4())
    bucket_name = Config.GCS_BUCKET_NAME
    gcs_url = upload_to_gcs(bucket_name, image_bytes, f"results/{image_id}.jpg")
    return gcs_url

class OriginalDetectRepository(BaseRepository):
    def __init__(self):
        self.model = self.load_model()  # Call overridden abstract method

    def load_model(self):
        return load_model(Config.MODEL_ORI_PATH, compile=False)
    
    def predict(self, image_bytes: bytes, use_lbp: bool = False):
        # Preprocess image
        img_resized = preprocess_image(image_bytes, use_lbp)
        if img_resized is None:
            return None

        # Convert to tensor
        input_tensor = np.expand_dims(img_resized, axis=0)  # shape: (1, 224, 224, 3)
        
        # Save image to GCS
        img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        _, buffer = cv2.imencode('.jpg', img_gray)
        img_bytes = buffer.tobytes()
        gcs_url = save_image_to_gcs(img_bytes)

        # Class names for prediction
        class_names = ['FAKE', 'REAL']

        # Make prediction
        prediction = self.model.predict(input_tensor, verbose=0)[0]
        pred_class = np.argmax(prediction)
        label = class_names[pred_class]
        confidence = float(round(prediction[pred_class], 2))

        return {
            "label": label,
            "confidence": confidence,
            "image_url": gcs_url,
            "probabilities": {
                class_names[0]: float(round(prediction[0], 2)),
                class_names[1]: float(round(prediction[1], 2)),
            }
        }

class LBPDetectRepository(OriginalDetectRepository):
    def load_model(self):
        return load_model(Config.MODEL_LBP_PATH, compile=False)
