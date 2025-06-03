class ApiErrorRetry(Exception):
    """
    Exception raised when an API call fails but should be retried.
    """
    def __init__(self, message: str):
        super().__init__(message)


class ApiErrorAbortAndResolve(Exception):
    """
    Exception raised when an API call fails and requires manual resolution.
    """
    def __init__(self, message: str):
        super().__init__(message)
