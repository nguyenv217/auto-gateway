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