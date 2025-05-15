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
class OriginalDetectRepository(BaseRepository):
    def __init__(self):
        self.model = self.load_model()  # panggil abstract method yang kita override

    def load_model(self):  # ini override abstract method BaseRepository
        return load_model(Config.MODEL_ORI_PATH, compile=False)
    
    def predict(self, image_bytes: bytes, use_lbp: bool = False):
        import cv2
        from PIL import Image
        import numpy as np
        from io import BytesIO

        # Load image dari bytes (tanpa ubah channel)
        image = Image.open(BytesIO(image_bytes)).convert("RGB")  # Tetap RGB
        image_np = np.array(image)

        # Convert ke BGR karena Colab pakai cv2.imread()
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        # Cek wajah dan crop
        cropped_face = detect_and_crop_face(image_bgr)
        if cropped_face is None:
            return None
        
        if use_lbp:
            cropped_face = apply_lbp(cropped_face)
            cropped_face = cv2.cvtColor(cropped_face, cv2.COLOR_GRAY2BGR)
            
        # Resize dan buat batch
        img_resized = cv2.resize(cropped_face, (224, 224))
        input_tensor = np.expand_dims(img_resized, axis=0)  # shape: (1, 224, 224, 3)
        print(f"input_tensor shape: {input_tensor.shape}")
        print(f"image resized: {img_resized.shape}")
        # ----------------------
        # Simpan ke GCS
        img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        _, buffer = cv2.imencode('.jpg', img_gray)
        img_bytes = buffer.tobytes()

        image_id = str(uuid.uuid4())
        bucket_name = Config.GCS_BUCKET_NAME
        gcs_url = upload_to_gcs(bucket_name, img_bytes, f"results/{image_id}.jpg")
        # ----------------------

        self.class_names = ['FAKE', 'REAL']

        prediction = self.model.predict(input_tensor, verbose=0)[0]
        pred_class = np.argmax(prediction)
        label = self.class_names[pred_class]
        confidence = float(round(prediction[pred_class], 2))

        return {
            "label": label,
            "confidence": confidence,
            "image_url": gcs_url,
            "probabilities": {
                self.class_names[0]: float(round(prediction[0], 2)),
                self.class_names[1]: float(round(prediction[1], 2)),
            }
        }


class LBPDetectRepository(OriginalDetectRepository):
    def load_model(self):  # override method dari parent
        return load_model(Config.MODEL_LBP_PATH, compile=False)