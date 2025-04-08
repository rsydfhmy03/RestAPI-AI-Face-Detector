from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from app.routes.detect import router as detect_router
from app.exceptions.client_error import ClientError
from app.exceptions.not_found_error import NotFoundError
from app.exceptions.invariant_error import InvariantError

app = FastAPI(
    title="AI Image FACE Detection API",
    version="1.0.0"
)

# Include route
app.include_router(detect_router, prefix="/api/v1", tags=["Detection"])

# Custom Error Handler
@app.exception_handler(ClientError)
async def client_error_handler(request: Request, exc: ClientError):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "data": None,
            "statusCode": exc.status_code,
            "message": exc.message
        }
    )

@app.exception_handler(NotFoundError)
async def not_found_error_handler(request: Request, exc: NotFoundError):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "data": None,
            "statusCode": exc.status_code,
            "message": exc.message
        }
    )

@app.exception_handler(InvariantError)
async def invariant_error_handler(request: Request, exc: InvariantError):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "data": None,
            "statusCode": exc.status_code,
            "message": exc.message
        }
    )

@app.get("/")
def root():
    return {
        "message": "Welcome to the AI FACE Detection API v1.0.0",
        "statusCode": 200,
        "data": None
    }
