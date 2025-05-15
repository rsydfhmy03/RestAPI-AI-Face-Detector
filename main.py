from fastapi import FastAPI
from app.routes.detect import router as detect_router
from app.exceptions.handlers import register_exception_handlers

app = FastAPI(
    title="AI Image FACE Detection API",
    version="1.0.0"
)

# Routes
app.include_router(detect_router, prefix="/api/v1", tags=["Detection"])

# Exception Handlers
register_exception_handlers(app)

# Root Endpoint
@app.get("/")
def root():
    return {
        "statusCode": 200,
        "message": "Welcome to the AI FACE Detection API v1.0.0",
        "data": None
    }

# Optional: For local run
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8080)
