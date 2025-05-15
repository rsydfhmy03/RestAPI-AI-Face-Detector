from fastapi import Request
from fastapi.responses import JSONResponse
from app.exceptions.client_error import ClientError
from app.exceptions.not_found_error import NotFoundError
from app.exceptions.invariant_error import InvariantError
import traceback

def generate_error_response(status_code: int, message: str):
    return JSONResponse(
        status_code=status_code,
        content={
            "statusCode": status_code,
            "message": message,
            "data": None
        }
    )

def register_exception_handlers(app):
    @app.exception_handler(ClientError)
    async def client_error_handler(request: Request, exc: ClientError):
        return generate_error_response(exc.status_code, exc.message)

    @app.exception_handler(NotFoundError)
    async def not_found_error_handler(request: Request, exc: NotFoundError):
        return generate_error_response(exc.status_code, exc.message)

    @app.exception_handler(InvariantError)
    async def invariant_error_handler(request: Request, exc: InvariantError):
        return generate_error_response(exc.status_code, exc.message)

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        error_detail = str(exc)
        stack_trace = traceback.format_exc()
        print(f"Error: {error_detail}\nStack Trace: {stack_trace}")
        return JSONResponse(
            status_code=500,
            content={"message": "Internal server error", "error": error_detail}
        )
