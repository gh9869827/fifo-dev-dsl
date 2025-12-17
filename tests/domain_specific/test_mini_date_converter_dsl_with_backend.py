"""Tests for parse_natural_date_expression_with_backend function."""
from datetime import datetime
import pytest
from fifo_dev_dsl.common.llm_abstraction import LlmBackend, LlmRequest
from fifo_dev_dsl.domain_specific.mini_date_converter_dsl.core import (
    parse_natural_date_expression_with_backend,
    parse_natural_date_expression
)


class MockLlmBackend:
    """Mock LLM backend for testing."""
    
    def __init__(self, response: str):
        self.response = response
        self.last_request = None
    
    def complete(self, req: LlmRequest) -> str:
        """Return the configured response."""
        self.last_request = req
        return self.response


def test_parse_natural_date_expression_with_backend_basic():
    """Test basic functionality with mock backend."""
    # Mock backend that returns a simple DSL expression
    backend = MockLlmBackend("TODAY")
    
    dsl_code, dt = parse_natural_date_expression_with_backend(
        "today",
        backend=backend
    )
    
    assert dsl_code == "TODAY"
    assert dt.date() == datetime.now().date()
    assert dt.hour == 0
    assert dt.minute == 0


def test_parse_natural_date_expression_with_backend_offset():
    """Test with OFFSET expression."""
    backend = MockLlmBackend("OFFSET(TODAY, 1, DAY)")
    
    dsl_code, dt = parse_natural_date_expression_with_backend(
        "tomorrow",
        backend=backend
    )
    
    assert dsl_code == "OFFSET(TODAY, 1, DAY)"
    # Check the date is tomorrow
    expected = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    from datetime import timedelta
    expected += timedelta(days=1)
    assert abs((dt - expected).total_seconds()) < 1


def test_parse_natural_date_expression_with_backend_set_time():
    """Test with SET_TIME expression."""
    backend = MockLlmBackend("SET_TIME(TODAY, 17, 30)")
    
    dsl_code, dt = parse_natural_date_expression_with_backend(
        "today at 5:30pm",
        backend=backend
    )
    
    assert dsl_code == "SET_TIME(TODAY, 17, 30)"
    assert dt.hour == 17
    assert dt.minute == 30


def test_parse_natural_date_expression_with_backend_custom_now():
    """Test with custom now parameter."""
    backend = MockLlmBackend("TODAY")
    custom_now = datetime(2025, 1, 15, 10, 30)
    
    dsl_code, dt = parse_natural_date_expression_with_backend(
        "today",
        custom_now,
        backend=backend
    )
    
    assert dsl_code == "TODAY"
    assert dt == datetime(2025, 1, 15, 0, 0)


def test_parse_natural_date_expression_with_backend_max_new_tokens():
    """Test that max_new_tokens parameter is passed to LlmRequest."""
    backend = MockLlmBackend("TODAY")
    
    parse_natural_date_expression_with_backend(
        "today",
        backend=backend,
        max_new_tokens=512
    )
    
    assert backend.last_request.max_new_tokens == 512


def test_parse_natural_date_expression_with_backend_temperature():
    """Test that temperature parameter is passed to LlmRequest."""
    backend = MockLlmBackend("TODAY")
    
    parse_natural_date_expression_with_backend(
        "today",
        backend=backend,
        temperature=0.5
    )
    
    assert backend.last_request.temperature == 0.5


def test_parse_natural_date_expression_with_backend_invalid_dsl():
    """Test that invalid DSL raises ValueError."""
    backend = MockLlmBackend("INVALID_FUNCTION()")
    
    with pytest.raises(ValueError) as exc_info:
        parse_natural_date_expression_with_backend(
            "some invalid input",
            backend=backend
        )
    
    assert "INVALID_FUNCTION" in str(exc_info.value)


def test_parse_natural_date_expression_with_backend_system_prompt():
    """Test that system prompt is correctly set."""
    backend = MockLlmBackend("TODAY")
    
    parse_natural_date_expression_with_backend(
        "today",
        backend=backend
    )
    
    # System prompt should mention temporal parser and DSL functions
    assert "temporal parser" in backend.last_request.system_prompt.lower()
    assert "OFFSET" in backend.last_request.system_prompt
    assert "DATE_FROM_MONTH_DAY" in backend.last_request.system_prompt


def test_parse_natural_date_expression_with_backend_user_prompt():
    """Test that user prompt is the question."""
    backend = MockLlmBackend("TODAY")
    question = "next Monday"
    
    parse_natural_date_expression_with_backend(
        question,
        backend=backend
    )
    
    assert backend.last_request.user_prompt == question


def test_parse_natural_date_expression_deprecated_warning():
    """Test that the old function emits a deprecation warning."""
    with pytest.warns(DeprecationWarning, match="parse_natural_date_expression is deprecated"):
        # This will fail because we don't have a real backend, but we're just
        # checking for the warning
        try:
            parse_natural_date_expression(
                "today",
                container_name="test",
                adapter="test-adapter"
            )
        except Exception:
            # Expected to fail without real backend
            pass
