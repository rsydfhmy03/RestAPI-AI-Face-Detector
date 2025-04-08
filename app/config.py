import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

class Config:
    MODEL_ORI_PATH = os.path.join(BASE_DIR, 'app', 'models', 'model_ai_detection_cnnVGG.h5')
    MODEL_LBP_PATH = os.path.join(BASE_DIR, 'app', 'models', 'model_ai_detection_LBPVGG.h5')
    HAARCASCADE_PATH = os.path.join(BASE_DIR, 'app' ,'haarcascades', 'haarcascade_frontalface_default.xml')
    IMAGE_SIZE = (224, 224)
    LBP_PARAMS = {
        "P": 8,
        "R": 1,
        "METHOD": "default"
    }
