from fastapi import APIRouter, UploadFile, File
from app.detect.controller import detect_cnn_original, detect_lbp
from app.schemas.response import ResponseModel

router = APIRouter()

@router.post("/detect")
async def detect_image(file: UploadFile = File(...)) -> ResponseModel:
    return await detect_cnn_original(file)

@router.post("/detectLBP")
async def detect_lbp_image(file: UploadFile = File(...)) -> ResponseModel:
    return await detect_lbp(file)
