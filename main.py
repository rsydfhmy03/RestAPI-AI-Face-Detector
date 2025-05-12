import traceback
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
            "statusCode": exc.status_code,
            "message": exc.message,
            "data": None,
        }
    )

@app.exception_handler(InvariantError)
async def invariant_error_handler(request: Request, exc: InvariantError):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "statusCode": exc.status_code,
            "message": exc.message,
            "data": None,
        }
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    error_detail = str(exc)
    stack_trace = traceback.format_exc()
    print(f"Error: {error_detail}\nStack Trace: {stack_trace}")
    return JSONResponse(
        status_code=500,
        content={"message": "Internal server error", "error": error_detail}
    )

@app.get("/")
def root():
    return {
        "statusCode": 200,
        "message": "Welcome to the AI FACE Detection API v1.0.0",
        "data": None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8080)
