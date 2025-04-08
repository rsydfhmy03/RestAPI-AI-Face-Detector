from app.exceptions.client_error import ClientError

class NotFoundError(ClientError):
    def __init__(self, message: str):
        super().__init__(message, status_code=404)
