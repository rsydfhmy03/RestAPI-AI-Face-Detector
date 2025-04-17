from pydantic import BaseModel
from typing import Optional, Dict, Any
class ResponseModel(BaseModel):
    statusCode: int
    message: str
    data: Optional[Dict[str, Any]] = None
