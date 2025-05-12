import numpy as np
from skimage.feature import local_binary_pattern
import cv2
def apply_lbp(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    P = 8
    R = 1
    lbp = local_binary_pattern(gray, P, R, method='default')
    lbp = lbp.astype('float32')
    return lbp
