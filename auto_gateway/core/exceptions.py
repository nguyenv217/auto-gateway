import httpx
from ..strategies.adaptive import ErrorType


def classify_exception(e: Exception) -> ErrorType:
    if isinstance(e, httpx.HTTPStatusError):
        code = e.response.status_code
        if code == 429:
            return ErrorType.RATE_LIMIT
        if code in (401, 403):
            return ErrorType.AUTH
        if code == 402:
            return ErrorType.QUOTA
        if code >= 500:
            return ErrorType.PROVIDER_ERROR
    
    if isinstance(e, (httpx.ConnectError, httpx.ConnectTimeout)):
        return ErrorType.NETWORK
    
    if isinstance(e, httpx.TimeoutException):
        return ErrorType.TIMEOUT
        
    return ErrorType.UNKNOWN


class AllProvidersExhaustedError(Exception):
    """Exception raised when all providers fail to handle a request.
    
    This exception is designed to be converted to an OpenAI-compatible error response
    that is compatible with openai.RateLimitError for proper SDK handling.
    """
    
    def __init__(
        self,
        message: str = "All providers exhausted",
        error_type: str = "rate_limit_error",
        code: str = "rate_limit_exceeded",
        param: str | None = None,
    ):
        super().__init__(message)
        self.message = message
        self.error_type = error_type
        self.code = code
        self.param = param
    
    def to_openai_error_response(self) -> dict[str, any]:
        """Convert to OpenAI-compatible error response format."""
        return {
            "error": {
                "message": self.message,
                "type": self.error_type,
                "param": self.param,
                "code": self.code,
            }
        }
