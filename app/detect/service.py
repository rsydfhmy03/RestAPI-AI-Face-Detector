from app.base.service import BaseService
from app.detect.repository import OriginalDetectRepository, LBPDetectRepository
from fastapi import UploadFile
from app.exceptions.not_found_error import NotFoundError


class OriginalDetectService(BaseService):
    def __init__(self):
        super().__init__(OriginalDetectRepository())

    async def detect(self, file: UploadFile):
        image_bytes = await file.read()
        prediction_result = self.repository.predict(image_bytes)
        
        if not prediction_result:
            raise NotFoundError("Face not detected in the uploaded image")

        return prediction_result

    def process_and_predict(self, image):
        # Implement the logic for processing and predicting using the original model
        return self.repository.predict(image)


class LBPDetectService(BaseService):
    def __init__(self):
        super().__init__(LBPDetectRepository())

    async def detect(self, file: UploadFile):
        image_bytes = await file.read()
        prediction_result = self.repository.predict(image_bytes, use_lbp=True)

        if not prediction_result:
            raise NotFoundError("Face not detected in the uploaded image")

        return prediction_result

    def process_and_predict(self, image):
        # Implement the logic for processing and predicting using the LBP model
        return self.repository.predict(image, use_lbp=True)