from fastapi import UploadFile
from app.schemas.response import ResponseModel
from app.detect.service import OriginalDetectService, LBPDetectService

# Initialize services
original_service = OriginalDetectService()
lbp_service = LBPDetectService()

async def detect_face(
    file: UploadFile,
    service,
    message: str,
    status_code: int = 201
) -> ResponseModel:
    result = await service.detect(file)
    return ResponseModel(
        statusCode=status_code,
        message=message,
        data=result
    )

async def detect_cnn_original(file: UploadFile) -> ResponseModel:
    return await detect_face(
        file=file,
        service=original_service,
        message="Human Face Detected using Original CNN Model"
    )

async def detect_lbp(file: UploadFile) -> ResponseModel:
    return await detect_face(
        file=file,
        service=lbp_service,
        message="Human Face Detected using LBP + CNN Model"
    )
