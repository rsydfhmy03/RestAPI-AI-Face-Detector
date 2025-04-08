from pydantic import BaseModel
from typing import Optional, Dict, Any


class ResponseModel(BaseModel):
    data: Optional[Dict[str, Any]] = None
    statusCode: int
    message: str
