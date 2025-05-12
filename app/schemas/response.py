from pydantic import BaseModel
from typing import Optional, Dict

class Probabilities(BaseModel):
    FAKE: float
    REAL: float

class DetectionResult(BaseModel):
    label: str
    confidence: float
    image_url: str
    probabilities: Probabilities

class ResponseModel(BaseModel):
    statusCode: int
    message: str
    data: Optional[DetectionResult] = None
