from fastapi import UploadFile
from app.schemas.response import ResponseModel  # Corrected import
from app.detect.service import OriginalDetectService, LBPDetectService

# Initialize services
original_service = OriginalDetectService()
lbp_service = LBPDetectService()

async def detect_cnn_original(file: UploadFile) -> ResponseModel:
    result = await original_service.detect(file)
    return ResponseModel(
        data=result,
        statusCode=201,
        message="Human Face Detected using Original CNN Model"
    )

async def detect_lbp(file: UploadFile) -> ResponseModel:
    result = await lbp_service.detect(file)
    return ResponseModel(
        data=result,
        statusCode=201,
        message="Human Face Detected using LBP + CNN Model"
    )