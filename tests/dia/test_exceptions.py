from fifo_dev_dsl.dia.runtime.exceptions import ApiErrorRetry, ApiErrorAbortAndResolve


def test_api_error_retry_construction():
    message = "retry"
    exc = ApiErrorRetry(message)
    assert isinstance(exc, ApiErrorRetry)
    assert str(exc) == message


def test_api_error_abort_and_resolve_construction():
    message = "abort"
    exc = ApiErrorAbortAndResolve(message)
    assert isinstance(exc, ApiErrorAbortAndResolve)
    assert str(exc) == message
